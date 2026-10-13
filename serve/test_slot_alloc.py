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
    slot_caps_of,
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

    def test_reads_the_config_list(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 1, 2]}),
                         ["protected", "protected", "protected", "guest"])

    def test_accepts_a_comma_string(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": "0,1,2"}),
                         ["protected", "protected", "protected", "guest"])

    def test_absent_is_no_partition(self):
        self.assertEqual(protected_slots_of(4, {}), [])
        self.assertEqual(protected_slots_of(4, {"protected_slots": []}), [])

    def test_out_of_range_and_duplicates_are_dropped(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 0, 9, -1]}),
                         ["protected", "guest", "guest", "guest"])

    def test_every_slot_protected_is_no_partition(self):
        # all slots protected = the default routing: a partition that constrains nothing is not said to be on
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 1, 2, 3]}), [])


if __name__ == "__main__":
    unittest.main()
