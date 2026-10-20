"""serve/test_slot_owner.py - which batch slot a request gets when it names its conversation
    (StrataEngine.pick_slot / _note_owner), no engine needed.

    python -m unittest serve.test_slot_owner -v
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.server import (
    SLOT_PROTECT_MAX_S,
    StrataEngine,
    session_of,
    slot_protect_seconds_of,
)


def engine(batch, groups=1, pipelined=False, protect=0.0):
    e = object.__new__(StrataEngine)
    gs = batch // groups
    e.batch = batch
    e.slot_order = [g * gs + t for t in range(gs) for g in range(groups)]
    e.slot_group = [b // gs if gs else 0 for b in range(batch)]
    e.slot_groups = groups
    e.slot_busy = [False] * batch
    e.slot_held = [[] for _ in range(batch)]
    e.slot_used = [0.0] * batch
    e.slot_gs = gs if pipelined and groups > 1 else 0
    e.slot_owner = [None] * batch
    e.slot_protect_seconds = protect
    e.slot_live = [None] * batch
    return e


def live(e, slot, when=None):
    e.slot_used[slot] = time.time() if when is None else when


class SessionOfARequest(unittest.TestCase):
    def test_a_named_session_is_its_trimmed_text(self):
        self.assertEqual(session_of({"strata_session": "  sess-7  "}), "sess-7")

    def test_a_request_that_names_nothing_is_none(self):
        for sampling in (None, {}, {"strata_session": ""}, {"strata_session": "   "},
                         {"strata_session": None}, {"strata_session": 42},
                         {"strata_session": ["a"]}, {"strata_checkpoint": False}):
            with self.subTest(sampling=sampling):
                self.assertIsNone(session_of(sampling))


class TheWindow(unittest.TestCase):
    def test_absent_means_no_ownership(self):
        self.assertEqual(slot_protect_seconds_of({}), 0.0)
        self.assertEqual(slot_protect_seconds_of(None), 0.0)

    def test_off_words_are_off(self):
        for raw in (0, 0.0, False, True, None, "0", "false", "OFF", " no ", "", "none"):
            with self.subTest(raw=raw):
                self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": raw}), 0.0)

    def test_a_number_of_seconds_is_a_number(self):
        for raw, want in ((120, 120.0), ("120", 120.0), ("12.5", 12.5), (120.5, 120.5)):
            with self.subTest(raw=raw):
                self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": raw}), want)

    def test_a_value_that_is_not_seconds_turns_the_mechanism_off(self):
        for raw in (-5, -1.5, "abc", "12s", {}, []):
            with self.subTest(raw=raw):
                self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": raw}), 0.0)

    def test_a_window_that_never_passes_is_clamped(self):
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": 10 ** 9}), SLOT_PROTECT_MAX_S)


class AnOwnedSlot(unittest.TestCase):
    def test_its_own_slot_beats_a_slot_holding_a_longer_prefix(self):
        e = engine(3, protect=60.0)
        prompt = list(range(1000, 1100))
        e.slot_held[0] = list(range(1000, 1090))
        e.slot_owner[1] = "X"
        live(e, 1)
        self.assertEqual(e.pick_slot(prompt, "X"), 1)

    def test_it_queues_behind_its_own_slot_while_that_slot_is_busy(self):
        e = engine(3, protect=60.0)
        e.slot_busy[1] = True
        e.slot_owner[1] = "X"
        live(e, 1)
        self.assertIsNone(e.pick_slot(list(range(1000, 1100)), "X"))

    def test_its_second_turn_returns_to_the_slot_after_that_slot_frees(self):
        e = engine(3, protect=60.0)
        hist = list(range(1000, 1500))
        e.slot_owner[2] = "X"
        e.slot_held[2] = hist
        live(e, 2)
        self.assertEqual(e.pick_slot(hist + [1500, 1501], "X"), 2)

    def test_its_first_turn_takes_a_slot_nobody_owns(self):
        e = engine(3, protect=60.0)
        e.slot_owner[0] = "X"
        live(e, 0)
        slot = e.pick_slot(list(range(2000, 2100)), "Z")
        self.assertIn(slot, (1, 2))
        self.assertFalse(e._slot_owned_live(slot, time.time()))

    def test_a_slot_held_by_a_live_conversation_is_not_taken(self):
        e = engine(2, protect=60.0)
        e.slot_owner = ["X", "Y"]
        live(e, 0)
        live(e, 1)
        self.assertIsNone(e.pick_slot(list(range(3000, 3100)), "Z"))

    def test_a_slot_whose_conversation_is_past_the_window_is_taken(self):
        e = engine(2, protect=60.0)
        e.slot_owner = ["X", "Y"]
        live(e, 0, time.time() - 120.0)
        live(e, 1)
        self.assertEqual(e.pick_slot(list(range(3000, 3100)), "Z"), 0)

    def test_the_window_is_counted_from_the_slots_last_use(self):
        e = engine(2, protect=60.0)
        self.assertFalse(e._slot_owned_live(0, time.time()))
        e.slot_owner[0] = "X"
        live(e, 0)
        self.assertTrue(e._slot_owned_live(0, time.time()))
        self.assertFalse(e._slot_owned_live(0, time.time() + 61.0))

    def test_pipelined_groups_of_several_slots_keep_their_own_pick(self):
        e = engine(6, 3, pipelined=True, protect=60.0)
        self.assertEqual(e.slot_gs, 2)
        e.slot_owner = ["X"] * 6
        for b in range(6):
            live(e, b)
        self.assertEqual(e.pick_slot(list(range(3000, 3100)), "Z"), 0)


class ARequestWithoutASession(unittest.TestCase):
    def test_it_takes_the_slot_the_default_pick_gives_it(self):
        e = engine(4, protect=60.0)
        e.slot_held[2] = [1, 2]
        self.assertEqual(e.pick_slot([1, 2, 3]), 2)
        self.assertEqual(e.pick_slot([1, 2, 3], None), 2)

    def test_it_is_refused_when_every_slot_is_locked(self):
        """The lock is absolute: a slot whose window is open is closed to a request that names no conversation too.
        The plain pick reads only the prompt's tokens, so before the lock it walked into an idle locked slot and
        overwrote the chain its owner will come back to."""
        e = engine(2, protect=60.0)
        e.slot_owner = ["X", "Y"]
        live(e, 0)
        live(e, 1)
        self.assertIsNone(e.pick_slot(list(range(3000, 3100))))

    def test_it_takes_a_slot_no_conversation_owns(self):
        e = engine(2, protect=60.0)
        e.slot_owner = ["X", None]
        live(e, 0)
        self.assertEqual(e.pick_slot(list(range(3000, 3100)), None), 1)

    def test_two_conversations_sharing_one_system_prompt_are_not_queued(self):
        e = engine(2, protect=60.0)
        prompt = list(range(1000, 1700))
        first = e.pick_slot(prompt)
        e.slot_busy[first] = True
        e.slot_held[first] = prompt
        e.slot_used[first] = time.time()
        self.assertIsNotNone(e.pick_slot(prompt))

    def test_an_owner_table_with_the_window_off_changes_nothing(self):
        """The default path: with no window an owner table is a table of nothing, and every free slot serves every
        request exactly as it always did."""
        e = engine(4)
        e.slot_owner = ["X", "Y", "Z", "W"]
        for b in range(4):
            live(e, b)
        self.assertEqual(e.pick_slot([1, 2, 3]), 0)

    def test_the_pick_with_the_window_off_is_the_plain_pick(self):
        prompt = list(range(4000, 4100))
        off, on = engine(4), engine(4, protect=60.0)
        for e in (off, on):
            e.slot_held[3] = list(range(4000, 4050))
            e.slot_used = [5.0, 3.0, 9.0, 1.0]
        self.assertEqual(off.pick_slot(prompt), on.pick_slot(prompt))
        self.assertEqual(off.pick_slot(prompt, "X"), on.pick_slot(prompt, "X"))


class TheOwnerOfASlot(unittest.TestCase):
    def test_only_an_admission_that_names_its_conversation_writes_one(self):
        e = engine(2, protect=60.0)
        e._note_owner(0, "X")
        self.assertEqual(e.slot_owner[0], "X")
        e._note_owner(1, None)
        self.assertIsNone(e.slot_owner[1])

    def test_with_the_window_off_no_admission_writes_one(self):
        e = engine(2)
        e._note_owner(0, "X")
        self.assertIsNone(e.slot_owner[0])

    def test_a_slot_frees_for_the_next_turn_and_stays_its_conversations(self):
        e = engine(2, protect=60.0)
        e._note_owner(1, "X")
        e.slot_busy[1] = False
        self.assertEqual(e.slot_owner[1], "X")
        self.assertEqual(e.pick_slot(list(range(9, 20)), "X"), 1)


class ThePanelView(unittest.TestCase):
    def test_the_view_is_the_plain_view_without_the_window(self):
        e = engine(2, protect=0.0)
        e.slot_held[1] = [1, 2, 3]
        self.assertEqual(e.slots_view(),
                         [{"slot": 0, "state": "idle", "held_tokens": 0},
                          {"slot": 1, "state": "idle", "held_tokens": 3}])

    def test_the_view_names_the_owner_and_whether_the_window_is_open(self):
        e = engine(2, protect=60.0)
        e._note_owner(0, "X")
        live(e, 0)
        view = {v["slot"]: v for v in e.slots_view()}
        self.assertEqual(view[0]["owner"], "X")
        self.assertTrue(view[0]["owned"])
        self.assertIsNone(view[1]["owner"])
        self.assertFalse(view[1]["owned"])

    def test_the_view_hides_the_fields_with_the_window_off(self):
        e = engine(1, protect=60.0)
        e.slot_protect_seconds = 0.0
        self.assertNotIn("owner", e.slots_view()[0])
        self.assertNotIn("owned", e.slots_view()[0])


class ARestartKeepsTheWindow(unittest.TestCase):
    """restart() runs __init__(*self.spawn), and a start-up retry that lost the window starts an engine that
    routes nothing: the conversation silently loses the slot its chain lives in.  spawn's lazy slot is False (a
    restart starts the engine for real), so the restarts here are replayed with that slot set back to True: the
    tail is what a restart carries, and that is what these tests read."""

    def engine_with(self, window, protected=None, reject=None):
        return StrataEngine("missing-executable", ["--batch", "3"], lazy=True, slot_protect_seconds=window,
                            protected_slots=protected, reject_when_full=reject)

    def restarted(self, e):
        e.__init__(*e.spawn[:5], True, *e.spawn[6:])
        return e

    def test_the_window_rides_in_spawn(self):
        e = self.engine_with(120)
        self.assertEqual(e.spawn[8], 120)
        self.assertEqual(e.slot_protect_seconds, 120.0)

    def test_a_restart_started_from_spawn_keeps_it(self):
        e = self.engine_with(120)
        e.slot_owner = ["X", "X", "X"]
        e = self.restarted(e)
        self.assertEqual(e.slot_protect_seconds, 120.0)
        self.assertEqual(e.slot_owner, [], "a restart left the owners of the old engine's slots in place")

    def test_the_start_up_retry_gets_the_tail_from_spawn(self):
        """The retry path passes self.spawn[8:] on: read what it would pass and check it is this engine's routing
        settings (the window, the partition, the refusal).  A lazy engine sizes no slot tables, so the partition is
        read as what a restart is handed, not as a filled table (the filled table is protected_slots_of's own
        test)."""
        e = self.engine_with(120, 2, True)
        self.assertEqual(e.spawn[8:], (120, 2, True))
        e = self.restarted(e)
        self.assertEqual(e.spawn[8:], (120, 2, True), "a restart lost the routing tail")
        self.assertTrue(e.slot_reject_full, "a restart lost the refusal: conversations queued again")

    def test_the_window_off_survives_a_restart_as_off(self):
        e = self.engine_with(None)
        e = self.restarted(e)
        self.assertEqual(e.slot_protect_seconds, 0.0)

    def test_a_request_against_a_lazy_engine_does_not_raise(self):
        """The ownership tables exist from __init__ (the lazy path returns before the slot tables are made), so a
        request that arrives then reads the window and an empty owner table rather than a missing attribute."""
        e = self.engine_with(120)
        self.assertEqual(e.slot_owner, [])
        self.assertEqual(e.slot_protect_seconds, 120.0)
        self.assertFalse(e._slot_owned_live(0, time.time()))

    def test_a_request_against_an_engine_that_is_restarting_does_not_raise(self):
        """__init__ writes the owner table where the slot tables are sized, so a request that picks while an engine
        is restarting sees the old engine's slot tables and an owner table that is not them: it routes by the plain
        pick, it does not index a table shorter than the slots (and a slot whose conversation is gone owns nothing)."""
        e = self.engine_with(120)
        e.slot_owner = []
        e.batch = 3
        e.slot_order = [0, 1, 2]
        e.slot_group = [0, 0, 0]
        e.slot_groups = 1
        e.slot_gs = 0
        e.slot_busy = [False] * 3
        e.slot_held = [[], [1, 2], []]
        e.slot_used = [0.0, 0.0, 0.0]
        self.assertEqual(e.pick_slot([1, 2, 3], "X"), 1)
        e.slot_owner = ["X", "Y", "Z"]      # the table this engine sizes for real: now ownership routes
        e.slot_busy[0] = True
        self.assertIsNone(e.pick_slot([1, 2, 3], "X"))
        self.assertEqual(e.pick_slot([1, 2, 3], "W"), 1)


if __name__ == "__main__":
    unittest.main()
