"""serve/test_slot_alloc.py - which batch slot a request gets (StrataEngine.pick_slot), no engine needed.

    python -m unittest serve.test_slot_alloc -v
"""
from __future__ import annotations

import random
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import (  # noqa: E402
    StrataEngine,
    effective_max_context,
    protected_slots_of,
    session_of,
    slot_caps_of,
    slot_protect_seconds_of,
    zone_of,
)


def engine(batch, groups, pipelined=True, slot_cap=None, protected=None):
    e = object.__new__(StrataEngine)          # pick_slot reads only the slot tables
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    e.slot_gs = gs if pipelined and groups > 1 else 0
    e.slot_groups = groups
    e.slot_group = [b // gs if gs else 0 for b in range(batch)]   # each slot's pipeline group (as __init__ sets)
    e.slot_cap = list(slot_cap) if slot_cap else [0] * batch   # 0 = no capacity limit (homogeneous)
    # "protected_slots" in the config: the slots reserved for checkpointed conversations; the rest take one-shot calls
    e.slot_zone = ["protected" if b in set(protected or ()) else "guest" for b in range(batch)] if protected else []
    return e


class RequestZone(unittest.TestCase):
    """Which zone a request belongs to: a one-shot call (strata_checkpoint false, ckpt=0) is a transient guest,
    every conversation that later turns extend belongs in the protected slots.  The zone is a property of the
    request, so the #1994 slot refusal recomputes the same one and cannot move a conversation into another zone."""

    def test_one_shot_call_is_the_guest_zone(self):
        self.assertEqual(zone_of({"strata_checkpoint": False}), "guest")

    def test_conversation_is_the_protected_zone(self):
        self.assertEqual(zone_of({}), "protected")
        self.assertEqual(zone_of({"temperature": 0.7}), "protected")
        self.assertEqual(zone_of({"strata_checkpoint": True}), "protected")

    def test_no_sampling_defaults_to_protected(self):
        self.assertEqual(zone_of(None), "protected")


class Partition(unittest.TestCase):
    """"protected_slots": the slots a checkpointed conversation may occupy.  A one-shot call never takes one, so
    it can neither evict a conversation's chain nor be the request that rewrites the solo session."""

    def test_conversation_lands_only_in_the_protected_slots(self):
        e = engine(4, 1, pipelined=False, protected=[0, 1, 2])
        self.assertEqual(e.pick_slot(list(range(100)), "protected"), 0)
        e.slot_busy[0] = True
        self.assertEqual(e.pick_slot(list(range(100)), "protected"), 1)

    def test_one_shot_lands_only_in_the_guest_slots(self):
        e = engine(4, 1, pipelined=False, protected=[0, 1, 2])
        self.assertEqual(e.pick_slot(list(range(100)), "guest"), 3)

    def test_all_protected_slots_busy_queues_rather_than_taking_a_guest_slot(self):
        e = engine(4, 1, pipelined=False, protected=[0, 1, 2])
        for b in (0, 1, 2):
            e.slot_busy[b] = True
        self.assertIsNone(e.pick_slot(list(range(100)), "protected"))   # guest slot 3 stays the one-shot's

    def test_guest_slots_full_lets_the_one_shot_queue_not_invade_the_protected_slots(self):
        e = engine(4, 1, pipelined=False, protected=[0, 1, 2])
        e.slot_busy[3] = True
        self.assertIsNone(e.pick_slot(list(range(100)), "guest"))

    def test_no_partition_keeps_the_homogeneous_behaviour(self):
        # the default path: no "protected_slots" - every slot serves every request, as before the partition
        e = engine(4, 1, pipelined=False)
        self.assertEqual(e.pick_slot(list(range(100)), "guest"), 0)
        self.assertEqual(e.pick_slot(list(range(100)), "protected"), 0)

    def test_reuse_stays_inside_the_zone(self):
        # a conversation's next turn returns to the protected slot that holds its start, never to a guest slot
        e = engine(4, 1, pipelined=False, protected=[0, 1, 2])
        prompt = list(range(1000))
        e.slot_held[3] = prompt[:900]                     # a guest slot happens to hold the start
        e.slot_held[1] = prompt[:500]
        self.assertEqual(e.pick_slot(prompt, "protected"), 1)

    def test_a_hot_protected_slot_is_never_evicted_for_a_new_conversation(self):
        # inside the partition the lock still rules: the hot protected slot keeps its chain, the cold one gives way
        e = engine(4, 1, pipelined=False, protected=[0, 1])
        e.slot_held[0] = list(range(200000))
        e.slot_used[0] = time.time() - 60                 # hot: locked
        e.slot_held[1] = list(range(200000))
        e.slot_used[1] = time.time() - (StrataEngine.SLOT_PROTECT_IDLE_GRACE_S + 1)   # cold: evictable
        self.assertEqual(e.pick_slot(list(range(3000)), "protected"), 1)


class Capacity(unittest.TestCase):
    def test_long_prompt_only_lands_in_a_slot_that_fits(self):
        e = engine(4, 1, pipelined=False, slot_cap=[204800, 100000, 50000, 50000])
        self.assertEqual(e.pick_slot(list(range(150000))), 0)   # only slot 0 (204800) fits 150K

    def test_short_prompt_takes_the_smallest_slot_that_fits(self):
        # best-fit: a 40K prompt lands in a 50K slot, keeping the 200K slot free for a long conversation
        e = engine(4, 1, pipelined=False, slot_cap=[204800, 100000, 50000, 50000])
        self.assertEqual(e.pick_slot(list(range(40000))), 2)

    def test_no_slot_fits_returns_none(self):
        e = engine(4, 1, pipelined=False, slot_cap=[204800, 100000, 50000, 50000])
        self.assertIsNone(e.pick_slot(list(range(300000))))     # nothing holds 300K: not a queue, the caller rejects

    def test_missing_capacity_is_homogeneous(self):
        # slot_cap all 0 = no per-slot limit: behaviour is the pre-heterogeneous one (no capacity filter)
        e = engine(4, 1, pipelined=False)
        self.assertEqual(e.pick_slot(list(range(300000))), 0)

    def test_capacity_beats_a_longer_held_prefix_in_a_too_small_slot(self):
        # a slot holds a prefix that continues this prompt, but the slot is too small for the full prompt: skip it
        e = engine(4, 1, pipelined=False, slot_cap=[204800, 50000, 204800, 204800])
        e.slot_held[1] = list(range(40000))                     # slot 1 holds a start, but caps at 50000
        self.assertEqual(e.pick_slot(list(range(120000))), 0)   # 120K does not fit slot 1: take slot 0


class Locking(unittest.TestCase):
    """A slot holding a long, recently-used conversation is protected: the pick orders it after the unlocked ones,
    so a short request evicts a cold slot first.  The lock is a weight, never a veto - see LockIsAWeight for why
    refusing every slot starves the request instead of protecting the conversation."""

    def test_all_slots_protected_takes_the_least_recently_used(self):
        # the lock is a weight: when every free slot holds a hot chain the pick still returns one (the least
        # recently used), because queueing the request starves the conversation whose next turn it is
        e = engine(2, 1, pipelined=False, slot_cap=[204800, 204800])
        e.slot_held[0] = list(range(200000))
        e.slot_held[1] = list(range(200000))
        e.slot_used[0] = time.time()
        e.slot_used[1] = time.time() - 60          # both inside the idle grace: both locked
        self.assertEqual(e.pick_slot(list(range(3000))), 1)

    def test_lock_expires_when_idle_past_grace(self):
        # an idle long conversation past the grace is dead: its slot is evictable again
        e = engine(1, 1, pipelined=False, slot_cap=[204800])
        e.slot_held[0] = list(range(200000))
        e.slot_used[0] = time.time() - (StrataEngine.SLOT_PROTECT_IDLE_GRACE_S + 1)   # past the grace: not protected
        self.assertEqual(e.pick_slot(list(range(3000))), 0)

    def test_cold_slot_evicted_before_a_protected_one(self):
        # one hot (protected) and one idle-past-grace slot of the same width: eviction takes the cold one, not the hot
        e = engine(2, 1, pipelined=False, slot_cap=[204800, 204800])
        e.slot_held[0] = list(range(200000))
        e.slot_held[1] = list(range(200000))
        e.slot_used[0] = time.time() - 60                                     # hot: protected
        e.slot_used[1] = time.time() - (StrataEngine.SLOT_PROTECT_IDLE_GRACE_S + 1)   # cold: evictable
        self.assertEqual(e.pick_slot(list(range(3000))), 1)


class Pipelined(unittest.TestCase):
    def test_spreads_over_the_groups_first(self):
        e = engine(6, 3)
        got = []
        for _ in range(6):
            b = e.pick_slot([1, 2, 3])
            e.slot_busy[b] = True
            got.append(b)
        self.assertEqual(got, [0, 2, 4, 1, 3, 5])

    def test_fills_a_leading_hole_first(self):
        # #793's report: a group left as [_ B] costs a whole row; the next request goes into that hole
        e = engine(6, 3)
        e.slot_busy[3] = True                       # group 1 = [_ B]
        e.slot_used[0] = 0.0                        # least recently used: what the LRU fallback picked
        e.slot_used[2] = 99.0
        self.assertEqual(e.pick_slot([1]), 2)

    def test_never_opens_a_hole_below_the_pick(self):
        for batch, groups in ((6, 3), (8, 4), (8, 2), (4, 2)):
            gs = batch // groups
            e, rng = engine(batch, groups), random.Random(batch * 10 + groups)
            for _ in range(2000):
                busy = [b for b in range(batch) if e.slot_busy[b]]
                if busy and (rng.random() < 0.45 or len(busy) == batch):
                    e.slot_busy[rng.choice(busy)] = False          # a request ends, anywhere
                    continue
                b = e.pick_slot([7])
                g0 = b - b % gs
                with self.subTest(batch=batch, groups=groups):
                    self.assertTrue(all(e.slot_busy[x] for x in range(g0, b)),
                                    f"slot {b} picked with a free slot below it in its group: {e.slot_busy}")
                e.slot_busy[b] = True

    def test_held_tokens_are_ignored(self):
        # the engine keeps no pipelined slot as a cache: a held prefix must not pull a request into a hole-making slot
        e = engine(6, 3)
        e.slot_held[5] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)


class OneGroup(unittest.TestCase):
    def test_reuse_then_least_recently_used(self):
        e = engine(4, 1, pipelined=False)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)        # its conversation's next turn (the longest shared start)
        e.slot_held = [[5], [], [9], [6]]
        self.assertEqual(e.pick_slot([1, 2, 3]), 1)        # nothing shares a token: the empty slot
        # nothing shares a token and none is empty: the least recently used, and the idle age decides it
        e.slot_held = [[5], [7], [9], [6]]
        e.slot_used = [5.0, 3.0, 9.0, 1.0]
        self.assertEqual(e.pick_slot([1, 2, 3]), 3)


class InfoParse(unittest.TestCase):
    """The engine reports the per-slot capacities in its INFO line (slot_caps=a,b,c); the server parses them
    into the table pick_slot filters on.  A homogeneous engine says nothing: the table is empty."""

    def test_parses_the_comma_list(self):
        self.assertEqual(slot_caps_of({"slot_caps": "204800,100000,50000,50000"}),
                         [204800, 100000, 50000, 50000])

    def test_missing_is_empty(self):
        self.assertEqual(slot_caps_of({}), [])
        self.assertEqual(slot_caps_of({"slot_caps": ""}), [])

    def test_drops_malformed_entries(self):
        self.assertEqual(slot_caps_of({"slot_caps": "204800,bad,50000"}), [204800, 50000])

    def test_pads_to_the_slot_count(self):
        # the cross-process contract: pick_slot indexes slot_cap by slot id, so the table is always batch long.
        # A short slot_caps (the engine fitted fewer slots than it listed) pads with 0 = no per-slot limit.
        self.assertEqual(slot_caps_of({"slot_caps": "204800,50000"}, 4), [204800, 50000, 0, 0])

    def test_truncates_past_the_slot_count(self):
        self.assertEqual(slot_caps_of({"slot_caps": "204800,100000,50000,50000,99"}, 3), [204800, 100000, 50000])


class EffectiveContext(unittest.TestCase):
    """With heterogeneous slots --max-context is dead: the context the server reports and refuses against is the
    widest slot.  A prompt past it fails at the request entry (the 400), never queued on a table where no slot fits."""

    def test_homogeneous_keeps_max_context(self):
        self.assertEqual(effective_max_context(4096, []), 4096)

    def test_heterogeneous_ignores_a_too_large_max_context(self):
        # the tooth: --max-context 300000 beside a 204800 widest slot is DEAD, not the larger of the two
        self.assertEqual(effective_max_context(300000, [204800, 100000, 50000]), 204800)

    def test_heterogeneous_raises_to_the_widest_slot(self):
        self.assertEqual(effective_max_context(32768, [204800, 50000]), 204800)


class LongestPrefix(unittest.TestCase):
    """The pick is by shared prefix length, not by a whole-array match: a slot that shares 30,000 tokens with the
    prompt is worth more than one that shares none, even when neither holds the prompt's entire start.  A prompt
    that shares only part of a slot's chain still returns it - the engine reads the rest, which is the cost of a
    change the server cannot see (a client that rewrites its system prompt between turns)."""

    def test_a_partial_prefix_returns_the_slot(self):
        # the v0.1.44 case: the slot holds 30,293, the prompt shares its first 30,116 - the old whole-array match
        # scored 0 and the request went to an empty slot, so the protected slots filled up one turn at a time
        e = engine(3, 1, pipelined=False, protected=[0, 1, 2])
        e.slot_held[0] = list(range(30293))
        e.slot_used[0] = time.time()
        self.assertEqual(e.pick_slot(list(range(30116)) + [9999] * 100, "protected"), 0)

    def test_the_longest_shared_prefix_wins(self):
        e = engine(3, 1, pipelined=False, protected=[0, 1, 2])
        prompt = list(range(1000))
        e.slot_held[1] = prompt[:900]          # the longest start of this prompt
        e.slot_held[2] = prompt[:100]
        self.assertEqual(e.pick_slot(prompt, "protected"), 1)

    def test_a_prefix_beats_a_never_used_slot(self):
        # a never-used slot is the best eviction target, but not a better match than a longer prefix
        e = engine(3, 1, pipelined=False, protected=[0, 1, 2])
        e.slot_held[0] = list(range(30293))
        e.slot_used[0] = time.time() - 30
        self.assertEqual(e.pick_slot(list(range(30116)), "protected"), 0)

    def test_a_whole_prefix_is_a_match(self):
        # a slot holding exactly the prompt is the perfect match, never "no match" (the old test used < not <=)
        e = engine(2, 1, pipelined=False, protected=[0, 1])
        e.slot_held[0] = list(range(1000))
        self.assertEqual(e.pick_slot(list(range(1000)), "protected"), 0)


class LockIsAWeight(unittest.TestCase):
    """The lock orders the free slots, it never refuses them.  A locked slot is taken when nothing else fits the
    prompt, because queueing the request starves the conversation whose next turn it is (the hang of v0.1.44:
    three turns of one conversation, three locked slots, the fourth turn cancelled at 114 s)."""

    def test_lcp_beats_the_lock(self):
        # a slot sharing 30,000 tokens is worth more than protecting a hot chain that shares none
        e = engine(2, 1, pipelined=False, protected=[0, 1])
        e.slot_held[0] = list(range(30116))                 # shares this prompt's whole start
        e.slot_used[0] = time.time() - 1                    # hot: locked
        e.slot_held[1] = list(range(200000))                # a hot chain sharing nothing
        e.slot_used[1] = time.time() - 1                    # hot: locked too
        self.assertEqual(e.pick_slot(list(range(30116)), "protected"), 0)

    def test_a_free_unlocked_slot_is_taken_before_a_locked_one(self):
        # neither shares anything: the lock still keeps the hot chain
        e = engine(2, 1, pipelined=False, protected=[0, 1])
        e.slot_held[0] = list(range(200000))
        e.slot_used[0] = time.time() - 1                    # hot: locked
        e.slot_held[1] = list(range(200000))
        e.slot_used[1] = time.time() - (StrataEngine.SLOT_PROTECT_IDLE_GRACE_S + 1)   # cold: not locked
        self.assertEqual(e.pick_slot(list(range(3000)), "protected"), 1)

    def test_every_slot_locked_never_returns_none(self):
        e = engine(3, 1, pipelined=False, protected=[0, 1, 2])
        now = time.time()
        for b, held in enumerate((95277, 96869, 100708)):
            e.slot_held[b] = list(range(held))
            e.slot_used[b] = now - 30 * b                   # all inside the grace: all locked
        self.assertIsNotNone(e.pick_slot(list(range(102124)), "protected"))

    def test_the_grace_is_two_minutes(self):
        # a conversation idle past the grace is dead: its slot gives way without waiting half an hour
        self.assertEqual(StrataEngine.SLOT_PROTECT_IDLE_GRACE_S, 120.0)

    def test_lock_expires_after_the_grace(self):
        e = engine(1, 1, pipelined=False, protected=[0])
        e.slot_held[0] = list(range(200000))
        e.slot_used[0] = time.time() - 130                  # past the two-minute grace
        self.assertFalse(e._slot_locked(0))


class ZoneConfig(unittest.TestCase):
    """The partition is a config setting, a top-level "protected_slots" like the parallel one: it routes the
    server's own slots, and the engine takes any free slot BGEN names, so it never reaches the engine's
    command line (an unknown argument there exits the engine)."""

    @staticmethod
    def zones_said(batch, cfg):
        """protected_slots_of's zones and what it printed - a config that asks for a partition and does not get
        one is the silent failure: the conversation's chain is evicted and the log says nothing about why."""
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            zones = protected_slots_of(batch, cfg)
        return zones, buf.getvalue()

    def test_reads_the_config_list(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 1, 2]}),
                         ["protected", "protected", "protected", "guest"])

    def test_a_partition_that_takes_says_nothing(self):
        # a working config is not chatty: only a partition the user asked for and did not get is worth a line
        zones, said = self.zones_said(4, {"protected_slots": [0, 1, 2]})
        self.assertEqual(zones, ["protected", "protected", "protected", "guest"])
        self.assertEqual(said, "")

    def test_accepts_a_comma_string(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": "0,1,2"}),
                         ["protected", "protected", "protected", "guest"])

    def test_absent_is_no_partition(self):
        self.assertEqual(protected_slots_of(4, {}), [])
        self.assertEqual(protected_slots_of(4, {"protected_slots": []}), [])

    def test_a_slot_past_the_engines_says_so(self):
        # the engine reports fewer slots than the config was written for (a layer split, a --fit cap): the slots
        # past it are dropped, and the user has to be told the partition is not what they wrote
        zones, said = self.zones_said(4, {"protected_slots": [0, 1, 9]})
        self.assertEqual(zones, ["protected", "protected", "guest", "guest"])
        self.assertIn("protected_slots", said)
        self.assertIn("9", said)

    def test_slots_past_every_slot_leave_no_partition_and_say_so(self):
        # "parallel" 2 with protected [5]: every asked slot is out of range and the partition closes; closing it
        # silently left a conversation rebuilt by a guest turn with nothing in the log to trace
        zones, said = self.zones_said(2, {"protected_slots": [5]})
        self.assertEqual(zones, [])
        self.assertIn("protected_slots", said)
        self.assertIn("no partition", said)

    def test_every_slot_protected_is_no_partition_and_says_so(self):
        # all slots protected = the default routing: a partition that constrains nothing is not said to be on,
        # and the config that asked for it is told (the engine reporting fewer slots than the config reaches
        # this through the same door: [0,1,2] asked, batch 2 left)
        zones, said = self.zones_said(4, {"protected_slots": [0, 1, 2, 3]})
        self.assertEqual(zones, [])
        self.assertIn("protected_slots", said)
        self.assertIn("no partition", said)
        zones, said = self.zones_said(2, {"protected_slots": [0, 1, 2]})
        self.assertEqual(zones, [])
        self.assertIn("no partition", said)

    def test_out_of_range_and_duplicates_are_dropped(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 0, 9, -1]}),
                         ["protected", "guest", "guest", "guest"])


class OwnershipConfig(unittest.TestCase):
    """"slot_protect_seconds": how long a slot belongs to the conversation that last used it.  Absent or 0: no
    ownership, every free slot serves every request (the default, byte-identical to the routing before)."""

    def test_absent_is_off(self):
        self.assertEqual(slot_protect_seconds_of({}), 0.0)
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": None}), 0.0)

    def test_reads_a_number_and_a_string(self):
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": 120}), 120.0)
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": "120"}), 120.0)
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": 90.5}), 90.5)

    def test_off_words_and_zero(self):
        for off in (0, "0", False, "false", "off", "none", ""):
            self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": off}), 0.0, off)

    def test_garbage_and_negative_are_off(self):
        for bad in ("abc", -5, True, {"a": 1}):
            self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": bad}), 0.0, bad)

    def test_the_former_name_is_read_so_a_renamed_config_does_not_go_off(self):
        # a setup before the rename wrote "slot_ownership_s": reading only the new name would silently turn the
        # protection off - the whole reason for the fallback (the same silent-off Bug #1 was about)
        self.assertEqual(slot_protect_seconds_of({"slot_ownership_s": 120}), 120.0)
        self.assertEqual(slot_protect_seconds_of({"slot_ownership_s": "90"}), 90.0)

    def test_the_new_name_wins_when_a_config_has_both(self):
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": 60, "slot_ownership_s": 120}), 60.0)

    def test_the_former_name_off_stays_off(self):
        self.assertEqual(slot_protect_seconds_of({"slot_ownership_s": 0}), 0.0)
        self.assertEqual(slot_protect_seconds_of({"slot_ownership_s": "off"}), 0.0)

    def test_the_conversation_may_name_itself(self):
        # an agent that rewrites its system prompt mid-session changes its prompt's start and would lose the slot
        # its chain lives in; "strata_session" keeps it by its own id
        self.assertEqual(session_of({"strata_session": "task-42"}), "task-42")
        self.assertIsNone(session_of({}))
        self.assertIsNone(session_of({"strata_session": ""}))
        self.assertIsNone(session_of(None))


class SessionOwnership(unittest.TestCase):
    """A slot belongs to the conversation that holds its chain until the ownership grace passes.  The lock of
    the eviction ordering is a weight and needs 16384 held tokens; a 12.7K agent conversation reaches neither,
    so its chain was rebuilt by a neighbour every turn.  Ownership is a veto: with every slot owned by a live
    conversation the pick queues (None) rather than evicting, and it asks no token floor."""

    BATCH, PROTECTED = 4, [0, 1, 2]
    APPENDED = 80      # the tokens one turn adds to its conversation's start

    def engine(self, ownership_s):
        e = object.__new__(StrataEngine)
        e.batch = self.BATCH
        e.slot_order = list(range(self.BATCH))
        e.slot_busy = [False] * self.BATCH
        e.slot_held = [[] for _ in range(self.BATCH)]
        e.slot_used = [0.0] * self.BATCH
        e.slot_gs = 0
        e.slot_groups = 1
        e.slot_group = [0] * self.BATCH
        e.slot_cap = [0] * self.BATCH
        e.slot_zone = ["protected" if b in self.PROTECTED else "guest" for b in range(self.BATCH)]
        e.slot_protect_seconds = ownership_s
        e.slot_owner = [None] * self.BATCH
        return e

    def prompt(self, session, turn=0, tokens=12692):
        """A conversation's prompt: a start every later turn carries, plus what this turn appends.  Every
        appended token carries the turn number, so the shared prefix of two turns is exactly the start."""
        return [1000 + session] * tokens + [7000 + session + turn] * self.APPENDED

    def claim(self, e, slot, prompt, at, skey=None):
        """What an admission and its BDONE leave behind: the tokens the slot now holds, the time it was last
        used, and the conversation that owns it; the slot itself is free for the next turn."""
        e.slot_held[slot] = list(prompt)
        e.slot_used[slot] = at
        e.slot_owner[slot] = e._session_key(prompt, skey)

    def turn(self, e, session, slot, at, skey=None):
        """One turn of a conversation: the pick, the slot busy while it runs, the chain it leaves."""
        prompt = self.prompt(session)
        self.assertEqual(e.pick_slot(prompt, "protected", skey), slot)
        e.slot_busy[slot] = True
        self.claim(e, slot, prompt, at, skey)
        e.slot_busy[slot] = False
        return prompt

    def fill(self, e, at):
        """Every protected slot owned by its own live conversation."""
        for s in self.PROTECTED:
            self.turn(e, s, s, at)

    def test_off_is_the_old_routing(self):
        # 6 conversations over 3 protected slots, ownership off: every one of them gets a slot, so each slot
        # serves several conversations and rebuilds the chain of the one it displaced
        e = self.engine(0.0)
        now = time.time()
        seen = {}
        for session in range(6):
            slot = e.pick_slot(self.prompt(session), "protected")
            self.assertIsNotNone(slot, session)
            seen.setdefault(slot, set()).add(session)
            self.claim(e, slot, self.prompt(session), now)
        self.assertGreater(max(len(v) for v in seen.values()), 1, seen)

    def test_the_first_conversations_take_the_empty_slots(self):
        e = self.engine(120.0)
        now = time.time()
        for s in self.PROTECTED:
            self.assertEqual(e.pick_slot(self.prompt(s), "protected"), s)
            self.claim(e, s, self.prompt(s), now)
        self.assertIsNone(e.pick_slot(self.prompt(3), "protected"))

    def test_a_fourth_conversation_queues_rather_than_taking_an_owned_slot(self):
        e = self.engine(120.0)
        self.fill(e, time.time())
        self.assertIsNone(e.pick_slot(self.prompt(3), "protected"))
        self.assertIsNone(e.pick_slot(self.prompt(4), "protected"))

    def test_a_turn_returns_to_its_own_slot_and_its_own_chain(self):
        e = self.engine(120.0)
        self.turn(e, 0, 0, time.time())
        again = self.prompt(0, turn=1)
        slot = e.pick_slot(again, "protected")
        self.assertEqual(slot, 0)
        held = e.slot_held[0]
        shared = 0
        while shared < min(len(held), len(again)) and held[shared] == again[shared]:
            shared += 1
        self.assertEqual(shared, len(again) - self.APPENDED)   # the chain covers all but this turn's own tokens

    def test_the_same_conversation_shares_its_slot_only_with_itself(self):
        # its slot is taken by its running turn: a second request of the same conversation waits for it, and a
        # conversation of its own takes a slot nobody owns - never the busy one
        e = self.engine(120.0)
        e.slot_busy[0] = True
        self.claim(e, 0, self.prompt(0), time.time())
        self.assertIsNone(e.pick_slot(self.prompt(0), "protected"))
        self.assertEqual(e.pick_slot(self.prompt(1), "protected"), 1)

    def test_no_token_floor(self):
        # 12,692 tokens is the size a real agent conversation measured on the panel; the eviction lock's 16384
        # floor left every conversation of that size unguarded, turn after turn
        e = self.engine(120.0)
        now = time.time()
        for s in self.PROTECTED:
            self.turn(e, s, s, now, )
        self.assertLess(len(e.slot_held[0]), StrataEngine.SLOT_PROTECT_MIN_TOKENS)
        self.assertFalse(e._slot_locked(0))                    # the lock of the ordering: off at this size
        self.assertIsNone(e.pick_slot(self.prompt(3, tokens=12692), "protected"))   # ownership: guards it

    def test_a_slot_whose_owner_went_idle_past_the_grace_is_taken(self):
        e = self.engine(120.0)
        now = time.time()
        self.fill(e, now)
        e.slot_used[0] = now - 121.0                           # 121 s idle: the conversation is gone
        self.assertEqual(e.pick_slot(self.prompt(3), "protected"), 0)

    def test_just_inside_the_grace_still_holds(self):
        e = self.engine(120.0)
        now = time.time()
        self.fill(e, now)
        e.slot_used[0] = now - 119.0
        self.assertIsNone(e.pick_slot(self.prompt(3), "protected"))

    def test_the_owner_keeps_its_slot_while_its_turn_runs(self):
        e = self.engine(120.0)
        self.fill(e, time.time())
        e.slot_busy[2] = True
        self.assertIsNone(e.pick_slot(self.prompt(2), "protected"))   # its own slot is busy: it waits, no second copy

    def test_bdone_frees_the_slot_and_keeps_the_owner(self):
        e = self.engine(120.0)
        now = time.time()
        self.turn(e, 0, 0, now)
        self.assertFalse(e.slot_busy[0])
        self.assertEqual(e.slot_owner[0], e._session_key(self.prompt(0)))

    def test_an_explicit_session_survives_a_changed_prompt_start(self):
        # the client rewrote its system prompt: the digest of the start moved, "strata_session" did not
        e = self.engine(120.0)
        now = time.time()
        mine = self.prompt(0)
        slot = e.pick_slot(mine, "protected", "task-42")
        self.claim(e, slot, mine, now, "task-42")
        rewritten = [555] * 9000 + mine                        # a new start, the same conversation
        self.assertEqual(e.pick_slot(rewritten, "protected", "task-42"), slot)
        # unnamed, the rewritten start is a new conversation: it takes a slot nobody owns and loses the chain,
        # which is why a client that edits its prompt has to name its session
        self.assertEqual(e.pick_slot(rewritten, "protected"), 1)

    def test_the_guest_zone_never_owns_a_slot(self):
        # a one-shot call has no chain to keep: it takes no protected slot and owns nothing when it lands
        e = self.engine(120.0)
        self.fill(e, time.time())
        self.assertEqual(e.pick_slot([1, 2, 3], "guest"), 3)
        self.assertIsNone(e.slot_owner[3])
        self.assertEqual(e.pick_slot([4, 5, 6], "guest"), 3)   # the next one-shot takes it as freely

    def test_a_guest_never_takes_a_protected_slot(self):
        e = self.engine(120.0)
        self.turn(e, 0, 0, time.time())
        self.assertIsNone(e.pick_slot(self.prompt(3), "guest")) if self.PROTECTED == [0] else None
        self.assertEqual(e.pick_slot(self.prompt(3), "guest"), 3)

    def test_the_key_of_two_conversations_differs(self):
        e = self.engine(120.0)
        self.assertNotEqual(e._session_key(self.prompt(0)), e._session_key(self.prompt(1)))
        self.assertEqual(e._session_key(self.prompt(0)), e._session_key(self.prompt(0, turn=1)))

    def test_the_key_ignores_what_a_turn_appends(self):
        # the start is what identifies a conversation: 512 tokens of it is far past the system prompt and the
        # first turns, and every later turn carries them unchanged
        e = self.engine(120.0)
        self.assertEqual(e._session_key(self.prompt(0, tokens=12692)), e._session_key(self.prompt(0, tokens=90000)))


class SpawnCarriesTheRouting(unittest.TestCase):
    """"protected_slots", "solo_path" and "slot_protect_seconds" ride in spawn: restart() re-runs __init__ with
    *self.spawn and so must the start retry - an engine that comes back without them routes nothing, and a
    conversation silently loses the slot its chain lives in."""

    def test_spawn_carries_all_three(self):
        e = StrataEngine("noop", ["--max-context", "4096"], lazy=True,
                         protected_slots=[0, 1], solo_path=False, slot_protect_seconds=120)
        self.assertEqual(e.spawn[8:], ([0, 1], False, 120))

    def test_a_lazy_engine_can_be_picked_against(self):
        # pick_slot runs against an engine that is restarting: the ownership tables exist from the lazy return,
        # not only after READY (a missing attribute would raise inside a request)
        e = StrataEngine("noop", ["--max-context", "4096"], lazy=True, slot_protect_seconds=120)
        self.assertEqual(e.slot_protect_seconds, 120.0)
        self.assertEqual(e.slot_owner, [])

    def test_the_start_retry_passes_the_tail_through(self):
        # a dropped positional tail is a silent routing loss, and only a start retry (a GPU runtime that was not
        # ready, START_RETRIES) takes this path - no test harness reaches it without a GPU
        source = Path(__file__).resolve().parents[1].joinpath("serve/server.py").read_text(encoding="utf-8")
        self.assertIn("_attempt + 1, *self.spawn[8:]", source)


class OwnershipVisible(unittest.TestCase):
    """The panel is how a slot's owner is diagnosed: the fourth conversation not running is invisible unless the
    slots say who holds them.  The server states the fact (who owns the slot, and whether that is inside the
    window); strata_top decides how to render it.  Without "slot_protect_seconds" the view is exactly as it was."""

    def engine(self, ownership_s):
        e = object.__new__(StrataEngine)
        e.batch = 3
        e.slot_order = list(range(3))
        e.slot_busy = [False] * 3
        e.slot_held = [[] for _ in range(3)]
        e.slot_used = [0.0] * 3
        e.slot_live = [None] * 3
        e.slot_gs = 0
        e.slot_groups = 1
        e.slot_group = [0] * 3
        e.slot_cap = [0] * 3
        e.slot_zone = ["protected", "protected", "guest"]
        e.slot_protect_seconds = ownership_s
        e.slot_owner = [None] * 3
        return e

    def test_no_window_adds_nothing_to_the_view(self):
        e = self.engine(0.0)
        e.slot_owner[0] = "s:conv-0"           # a stale table with the window off must not leak into the view
        v = e.slots_view()
        self.assertNotIn("owner", v[0])
        self.assertNotIn("owned", v[0])
        self.assertEqual(v[0], {"slot": 0, "state": "idle", "held_tokens": 0, "zone": "protected",
                                "cap": 0, "locked": False, "idle_s": v[0]["idle_s"]}, v[0])

    def test_the_view_says_who_holds_each_slot(self):
        now = time.time()
        e = self.engine(120.0)
        e.slot_owner[0], e.slot_used[0] = "s:conv-0", now
        e.slot_owner[1], e.slot_used[1] = "s:conv-1", now - 300.0     # past the window: a dead conversation
        v = e.slots_view()
        self.assertEqual((v[0]["owner"], v[0]["owned"]), ("s:conv-0", True), v[0])
        self.assertEqual((v[1]["owner"], v[1]["owned"]), ("s:conv-1", False), v[1])
        self.assertEqual((v[2]["owner"], v[2]["owned"]), (None, False), v[2])

    def test_a_running_turn_owns_its_slot(self):
        e = self.engine(120.0)
        e.slot_owner[0] = "s:conv-0"
        e.slot_used[0] = time.time()
        e.slot_busy[0] = True
        e.slot_live[0] = {"slot": 0, "state": "decoding", "prompt_tokens": 9, "generated": 4,
                          "started": time.time() - 2, "first_token": time.time() - 1}
        v = e.slots_view()
        self.assertEqual((v[0]["state"], v[0]["owner"], v[0]["owned"]), ("decoding", "s:conv-0", True), v[0])


if __name__ == "__main__":
    unittest.main()
