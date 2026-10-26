"""serve/test_slot_zones.py - the slot partition and the hard lock, no engine needed.

    python -m pytest serve/test_slot_zones.py -v

Three things are checked here:

  * "protected_slots" splits the batch slots: a request that names its conversation ("strata_session") is
    admitted only to a protected slot, and a request that names nothing is never admitted to one.
  * "slot_protect_seconds" locks a slot to its conversation: while the window is open no other request enters
    it - not a named one, and not an unnamed one (the plain pick looks only at the tokens and used to walk
    straight into a locked slot and destroy the chain its conversation will come back to).
  * "reject_when_full" turns "no slot can take this" into an answer rather than a queue: the same pick the
    admission uses, read once without waiting, so the HTTP layer can answer 429 before it writes headers.
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import (
    SlotFull,
    StrataEngine,
    protected_slots_of,
    reject_when_full_of,
    slot_zone_of,
)

NOW = time.time()


def engine(batch, zones=(), protect=0.0, groups=1, pipelined=False):
    e = object.__new__(StrataEngine)
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_group = [b // gs if gs else 0 for b in range(batch)]
    e.slot_groups = groups
    e.slot_gs = gs if pipelined and groups > 1 else 0
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [NOW] * batch
    e.slot_owner = [None] * batch
    e.slot_zone = list(zones)
    e.slot_protect_seconds = protect
    e.slot_reject_full = False
    e.slot_live = [None] * batch
    return e


def used_ago(e, slot, seconds):
    e.slot_used[slot] = NOW - seconds


# ---- the partition: which slots a request may take ----------------------------------------------------------------

class ProtectedSlots(unittest.TestCase):
    def test_absent_is_no_partition(self):
        for raw in (None, [], "", "off", False):
            with self.subTest(raw=raw):
                self.assertEqual(protected_slots_of(4, {"protected_slots": raw}), [])

    def test_a_count_is_the_first_slots(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": 2}),
                         ["protected", "protected", "public", "public"])

    def test_a_list_names_the_slots(self):
        for raw in ([0, 3], "0,3", "0 3", [3, 0, 0]):
            with self.subTest(raw=raw):
                self.assertEqual(protected_slots_of(4, {"protected_slots": raw}),
                                 ["protected", "public", "public", "protected"])

    def test_every_slot_protected_is_no_partition(self):
        self.assertEqual(protected_slots_of(4, {"protected_slots": 4}), [])
        self.assertEqual(protected_slots_of(4, {"protected_slots": [0, 1, 2, 3]}), [])

    def test_indices_outside_the_batch_are_dropped(self):
        self.assertEqual(protected_slots_of(2, {"protected_slots": [0, 5, -1, "x"]}),
                         ["protected", "public"])

    def test_a_value_that_is_not_slots_is_no_partition(self):
        for raw in ("abc", {"a": 1}, -3):
            with self.subTest(raw=raw):
                self.assertEqual(protected_slots_of(4, {"protected_slots": raw}), [])

    def test_a_named_conversation_wants_the_protected_slots(self):
        self.assertEqual(slot_zone_of("conv-1", True), "protected")
        self.assertEqual(slot_zone_of(None, True), "public")

    def test_no_partition_puts_no_zone_on_the_request(self):
        self.assertEqual(slot_zone_of("conv-1", False), "")
        self.assertEqual(slot_zone_of(None, False), "")


class ThePartitionRoutesRequests(unittest.TestCase):
    def test_a_conversation_never_lands_in_a_public_slot(self):
        e = engine(4, zones=["protected", "public", "public", "public"])
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 0)
        e.slot_busy[0] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], "conv-b"),
                          "a conversation took the public slot 1, 2 or 3 with the protected slot free")

    def test_an_unnamed_request_never_lands_in_a_protected_slot(self):
        e = engine(4, zones=["protected", "public", "public", "public"])
        self.assertEqual(e.pick_slot([1, 2, 3], None), 1)
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([1, 2, 3], None), 2)
        e.slot_busy[2] = True
        self.assertEqual(e.pick_slot([1, 2, 3], None), 3)
        e.slot_busy[3] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], None),
                          "an unnamed request took the protected slot 0, which is a conversation's")

    def test_two_conversations_take_the_two_protected_slots(self):
        e = engine(4, zones=["protected", "protected", "public", "public"])
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 0)
        e.slot_busy[0] = True
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-b"), 1)
        e.slot_busy[1] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], "conv-c"))
        self.assertEqual(e.pick_slot([1, 2, 3], None), 2)

    def test_the_partition_prefers_the_slot_holding_the_prompts_start(self):
        e = engine(4, zones=["protected", "protected", "public", "public"])
        e.slot_held[1] = list(range(1000, 1090))
        self.assertEqual(e.pick_slot(list(range(1000, 1100)), "conv-a"), 1)

    def test_with_no_partition_every_slot_serves_every_request(self):
        e = engine(3)
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 0)
        e.slot_busy[0] = True
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-b"), 1)
        e.slot_busy[1] = True
        self.assertEqual(e.pick_slot([1, 2, 3], None), 2)

    def test_the_partition_is_ignored_on_pipelined_groups(self):
        e = engine(4, zones=["protected", "public", "public", "public"], groups=2, pipelined=True)
        self.assertIn(e.pick_slot([1, 2, 3], "conv-a"), (0, 1, 2, 3))

    def test_the_lock_and_the_partition_change_the_filling_of_pipelined_groups_nothing(self):
        """#793's hole filling, with both slot settings on: a pipelined group's window spans its slots up to its last
        busy one, so a leading hole costs a whole row, and the engine writes an idle slot below a busy one with its
        pad row - there is no chain to keep and nothing to reserve.  The pick must stay the pick of the plain path,
        slot for slot, whoever owns what and whichever side of the partition a request is on."""
        prompt = list(range(3000, 3100))
        off = engine(6, groups=3, pipelined=True)
        on = engine(6, groups=3, pipelined=True, zones=["protected", "public"] * 3, protect=120.0)
        on.slot_owner = ["conv-a"] * 6
        for b in range(6):
            on.slot_used[b] = NOW
            off.slot_used[b] = NOW
        for busy in ([], [3], [1, 4], [0, 3], [2, 3, 4], [0, 1, 2, 3, 4]):
            with self.subTest(busy=busy):
                for e in (off, on):
                    e.slot_busy = [b in busy for b in range(6)]
                for session in (None, "conv-a", "conv-b"):
                    self.assertEqual(on.pick_slot(prompt, session), off.pick_slot(prompt, None),
                                     "a slot setting changed how a pipelined group is filled")


# ---- the hard lock: no other request enters a locked slot --------------------------------------------------------

class TheLockIsAbsolute(unittest.TestCase):
    def test_an_unnamed_request_is_kept_out_of_a_locked_slot(self):
        """The gap this closes: the plain pick looks only at the prompt's tokens, so an unnamed request used to
        walk into a slot that is idle and locked, and the chain its conversation will come back to was gone."""
        e = engine(2, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 5.0)
        e.slot_held[1] = list(range(500, 600))
        self.assertEqual(e.pick_slot(list(range(500, 600)), None), 1,
                         "an unnamed request took slot 0, locked to conv-a")

    def test_a_named_request_is_kept_out_of_a_locked_slot_of_the_other_kind(self):
        """No partition, one slot, locked to another conversation: there is nowhere for this one to go."""
        e = engine(1, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 5.0)
        self.assertIsNone(e.pick_slot([1, 2, 3], "conv-b"))
        self.assertIsNone(e.pick_slot([1, 2, 3], None))

    def test_its_own_locked_slot_is_still_the_first_choice(self):
        e = engine(3, protect=120.0)
        e.slot_owner[1] = "conv-a"
        used_ago(e, 1, 5.0)
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 1)

    def test_the_lock_opens_when_the_window_passes(self):
        e = engine(2, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 121.0)
        used_ago(e, 1, 1.0)
        self.assertEqual(e.pick_slot([1, 2, 3], None), 0)
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-b"), 0)

    def test_its_own_busy_slot_is_a_queue_not_a_second_copy(self):
        """A second request of a conversation whose turn is decoding waits for that slot: a copy elsewhere puts
        the conversation in the engine twice and evicts the slot it lands on."""
        e = engine(3, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 5.0)
        e.slot_busy[0] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], "conv-a"))
        self.assertEqual(e.pick_slot([1, 2, 3], None), 1)

    def test_with_the_window_off_a_slot_is_everyones_as_it_was(self):
        e = engine(2)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 1.0)
        self.assertEqual(e.pick_slot(list(range(500, 600)), None), 0)
        e.slot_held[0] = []
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-b"), 0)


# ---- the ledger: an admission always rewrites who owns the slot it took ------------------------------------------------

class TheLedgerIsTheTruth(unittest.TestCase):
    """The invariant every routing rule reads: after an admission takes a slot, slot_owner[slot] names the request
    that took it - or nobody.  A stale name on a busy slot made the pick answer None to a conversation with a free
    slot elsewhere ("its own slot is decoding"), and let a conversation re-enter a slot holding somebody else's
    chain.  Both poisons came from one line: an unnamed request took a slot and the ledger kept the previous owner."""

    def test_an_admission_always_rewrites_the_ledger(self):
        e = engine(2, protect=120.0)
        for session in (None, "conv-a", "conv-b"):
            with self.subTest(session=session):
                e.slot_owner[0] = "stale-conv"
                e._note_owner(0, session)
                self.assertEqual(e.slot_owner[0], session or None,
                                 "an admission left the previous owner on the ledger")

    def test_the_ledger_is_frozen_with_the_window_off(self):
        e = engine(2)
        e.slot_owner[0] = "conv-a"
        e._note_owner(0, "conv-b")
        e._note_owner(1, None)
        self.assertEqual(e.slot_owner[0], "conv-a")
        self.assertIsNone(e.slot_owner[1])

    def test_a_stale_owner_on_a_busy_slot_no_longer_refuses_a_free_conversation(self):
        """The production poison, end to end: conv-a's window lapsed, an unnamed request took its slot, conv-b now
        decodes there.  conv-a's next turn used to read the stale ledger, see "my slot is busy", and get refused
        with a free slot sitting right next to it."""
        e = engine(2, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 121.0)                       # conv-a's window is gone
        self.assertEqual(e.pick_slot([1, 2, 3], None), 0)
        e._note_owner(0, None)                      # the unnamed admission wipes the stale name
        e.slot_owner[0] = "conv-b"                  # conv-b takes the slot and decodes
        e.slot_busy[0] = True
        e.slot_held[1] = [9, 9, 9]
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 1,
                         "conv-a was refused while slot 1 was free: the stale ledger said slot 0 was 'its own'")

    def test_a_reclaimed_slot_cannot_steal_another_conversations_chain(self):
        """The other poison: conv-a's name stayed on the ledger after a stranger overwrote the chain, so conv-a's
        next turn walked back into a slot now holding conv-b's cache and destroyed it.  After the wipe conv-b is
        routed by its own name instead."""
        e = engine(2, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 121.0)
        self.assertEqual(e.pick_slot([1, 2, 3], None), 0)
        e._note_owner(0, None)
        e.slot_owner[0] = "conv-b"                  # conv-b's admission, its chain now in slot 0
        e.slot_held[0] = list(range(700, 800))
        self.assertEqual(e.pick_slot(list(range(700, 810)), "conv-b"), 0)
        self.assertEqual(e.pick_slot([1, 2, 3], "conv-a"), 1,
                         "conv-a walked back into the slot conv-b's chain now lives in")


class RejectWhenFull(unittest.TestCase):
    def test_absent_is_the_queue_as_it_was(self):
        for raw in (None, "", False, "off", "false", 0, "no"):
            with self.subTest(raw=raw):
                self.assertFalse(reject_when_full_of({"reject_when_full": raw}))

    def test_true_and_its_words_are_true(self):
        for raw in (True, "true", "TRUE", "yes", "on", 1):
            with self.subTest(raw=raw):
                self.assertTrue(reject_when_full_of({"reject_when_full": raw}))

    def test_the_pick_is_read_without_waiting(self):
        e = engine(2, zones=["protected", "public"], protect=120.0)
        self.assertIsNone(e.slot_rejection(None))
        self.assertIsNone(e.slot_rejection("conv-a"))
        e.slot_busy[1] = True
        # the public side is full: a request that names nothing is refused - a reserved slot is not its side of the
        # batch, and filling it would leave the public side with nothing
        self.assertIsNotNone(e.slot_rejection(None))
        self.assertIsNone(e.slot_rejection("conv-a"))
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 5.0)
        self.assertIsNotNone(e.slot_rejection("conv-b"))
        self.assertIsNotNone(e.slot_rejection(None))
        self.assertIsNone(e.slot_rejection("conv-a"))

    def test_a_busy_own_slot_is_a_refusal_not_a_queue(self):
        e = engine(2, zones=["protected", "public"], protect=120.0)
        e.slot_owner[0] = "conv-a"
        e.slot_busy[0] = True
        used_ago(e, 0, 5.0)
        self.assertIsNotNone(e.slot_rejection("conv-a"))

    def test_the_refusal_says_when_to_come_back(self):
        e = engine(1, protect=120.0)
        e.slot_owner[0] = "conv-a"
        e.slot_busy[0] = True
        used_ago(e, 0, 20.0)
        reason, retry_after = e.slot_rejection("conv-b")
        self.assertIn("slot", reason)
        self.assertLessEqual(retry_after, 100.0)
        self.assertGreater(retry_after, 0.0)

    def test_a_locked_idle_slot_says_when_the_lock_opens(self):
        e = engine(1, protect=120.0)
        e.slot_owner[0] = "conv-a"
        used_ago(e, 0, 100.0)
        reason, retry_after = e.slot_rejection(None)
        self.assertIn("slot", reason)
        self.assertLessEqual(retry_after, 21.0)
        self.assertGreater(retry_after, 0.0)

    def test_every_slot_busy_says_soonest(self):
        e = engine(2)
        e.slot_busy[0] = e.slot_busy[1] = True
        reason, retry_after = e.slot_rejection(None)
        self.assertIn("slot", reason)
        self.assertGreaterEqual(retry_after, 1.0)

    def test_the_read_answers_the_same_with_the_setting_off(self):
        e = engine(2)
        e.slot_busy[0] = e.slot_busy[1] = True
        self.assertIsNotNone(e.slot_rejection(None))

    def test_the_queued_path_raises_with_the_setting_on(self):
        self.assertTrue(issubclass(SlotFull, Exception))
        self.assertIn("slot", str(SlotFull("every batch slot is taken or locked")))


if __name__ == "__main__":
    unittest.main()
