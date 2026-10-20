"""The three slot routing settings as one answer: what a server says about them, and what it says about a value it
is handed.

    python -m pytest serve/test_slot_config.py -v

What the settings do to a pick is tested in test_slot_zones.py and test_slot_owner.py.  What is checked here is the
part that faces the operator:

  * one summary of the settings a server runs with, the same answer for a single engine and for a replica server,
    naming only what is set (an absent setting is not reported, because "off" and "not read" must not look alike),
    and the startup line built from it;
  * the operator's own flags for the three settings: a value outside what a slot can mean is an error, and a flag
    left out says nothing, so a saved value stays in place.
"""
from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.replicas import ReplicaEngine
from serve.server import (
    protected_slots_of,
    reject_when_full_of,
    slot_protect_seconds_of,
    slot_routing_note,
    slot_routing_summary,
    slot_run_flags,
)


class FakeEngine:
    """The three settings, as an engine would answer them."""

    def __init__(self, batch=2, zones=(), protect=0.0, reject=False):
        self.batch = batch
        self.slot_zone = list(zones)
        self.slot_protect_seconds = protect
        self.slot_reject_full = reject


class Summary(unittest.TestCase):
    def test_nothing_set_names_nothing(self):
        self.assertEqual(slot_routing_summary(FakeEngine()), {})

    def test_an_engine_with_all_three(self):
        got = slot_routing_summary(FakeEngine(4, ("protected", "protected", "public", "public"), 120.0, True))
        self.assertEqual(got, {"slot_protect_seconds": 120.0, "protected_slots": 2, "reject_when_full": True})

    def test_a_window_alone_does_not_claim_slots(self):
        # the window and the split are separate switches: a server with only the window must not look as if it
        # reserved slots, and a server with only the split must not look as if it holds slots open
        self.assertEqual(slot_routing_summary(FakeEngine(2, (), 120.0)), {"slot_protect_seconds": 120.0})
        self.assertEqual(slot_routing_summary(FakeEngine(2, ("protected", "public"))), {"protected_slots": 1})

    def test_a_window_of_zero_is_off_not_reported(self):
        self.assertEqual(slot_routing_summary(FakeEngine(2, (), 0.0)), {})

    def test_a_replica_server_totals_the_slots_of_every_replica(self):
        # the panel shows every replica's slots, so the count of reserved slots is of all of them; asking one
        # engine is what under-reported the split of a two-card server
        both = ReplicaEngine([FakeEngine(2, ("protected", "public")), FakeEngine(2, ("protected", "public"))])
        self.assertEqual(slot_routing_summary(both), {"protected_slots": 2})

    def test_a_replica_server_reports_a_setting_of_any_replica(self):
        mixed = ReplicaEngine([FakeEngine(2, (), 0.0, False), FakeEngine(2, (), 120.0, True)])
        self.assertEqual(slot_routing_summary(mixed), {"slot_protect_seconds": 120.0, "reject_when_full": True})

    def test_a_replica_server_of_engines_without_the_settings(self):
        # an engine of an older build answers none of these; the summary of such a server is empty, not a row of
        # zeroes that reads as a setting turned off on purpose
        self.assertEqual(slot_routing_summary(ReplicaEngine([object(), object()])), {})


class StartupNote(unittest.TestCase):
    """The line the server prints about the slot routing it runs with: the operator set the switches and should
    read once, at the start, what the server understood - and a server running the plain pick prints nothing."""

    def test_nothing_set_prints_nothing(self):
        self.assertIsNone(slot_routing_note(FakeEngine()))

    def test_all_three_are_said_in_their_own_words(self):
        note = slot_routing_note(FakeEngine(4, ("protected", "protected", "public", "public"), 120.0, True))
        self.assertIn("120 s", note)
        self.assertIn("2 slots kept", note)
        self.assertIn("429", note)

    def test_a_window_clamped_to_the_ceiling_is_reported_as_the_engine_read_it(self):
        # a config asking for a day and a half is run as a day; the note has to say the day, not the ask
        note = slot_routing_note(FakeEngine(2, (), slot_protect_seconds_of({"slot_protect_seconds": 200000})))
        self.assertIn("86400 s", note)
        self.assertNotIn("200000", note)

    def test_one_slot_of_a_split_is_not_slots(self):
        note = slot_routing_note(FakeEngine(2, ("protected", "public")))
        self.assertIn("1 slot kept", note)
        self.assertNotIn("1 slots", note)

    def test_the_note_reports_the_table_the_config_words_resolved_to(self):
        # the note is built from what the engine resolved, not from the words in the config: "2" on a batch of 3 is
        # two slots, and the note has to say two - an operator reading the startup line is reading the routing
        note = slot_routing_note(FakeEngine(3, protected_slots_of(3, {"protected_slots": 2}),
                                            slot_protect_seconds_of({"slot_protect_seconds": 120})))
        self.assertIn("2 slots kept", note)
        self.assertIn("120 s", note)


class RunFlags(unittest.TestCase):
    """The operator's own flags for the three settings (serve/server.py: --slot-protect-seconds, --protected-slots,
    --reject-when-full).  Checked as flags, not as config values: a value typed at a terminal is worth an error
    while it can still be fixed, and a flag that names nothing routes nothing."""

    @staticmethod
    def flags(protect=None, slots=None, reject=None):
        return argparse.Namespace(slot_protect_seconds=protect, protected_slots=slots, reject_when_full=reject)

    @staticmethod
    def cfg(batch=4):
        return {"args": ["--batch", str(batch)]}

    def test_the_flags_are_what_the_engine_is_told(self):
        self.assertEqual(slot_run_flags(self.flags("120", "2", "on"), self.cfg()),
                         {"slot_protect_seconds": 120.0, "protected_slots": "2", "reject_when_full": True})

    def test_a_bare_switch_is_on(self):
        # a switch flag typed at a terminal means the operator wants the switch: nobody asks for what the default is
        self.assertEqual(slot_run_flags(self.flags(reject="on"), self.cfg()), {"reject_when_full": True})

    def test_a_flag_left_out_says_nothing(self):
        # the config's value stays in place: the answer covers only what was asked of it, and main() puts that
        # answer over the config - an absent flag must not write an off value over a saved window
        self.assertEqual(slot_run_flags(self.flags(), self.cfg()), {})
        self.assertEqual(slot_run_flags(self.flags(protect="60"), self.cfg()), {"slot_protect_seconds": 60.0})

    def test_the_off_word_cancels_a_saved_value(self):
        # a profile saved a window and a split; this run wants the plain queue and the plain pick
        self.assertEqual(slot_run_flags(self.flags("0", "off", False), self.cfg()),
                         {"slot_protect_seconds": 0.0, "protected_slots": "off", "reject_when_full": False})

    def test_a_window_above_the_ceiling_is_clamped_as_the_config_window_is(self):
        # the same rule in both doors: a day and a half is run as a day, and the engine is handed the day
        self.assertEqual(slot_run_flags(self.flags(protect="200000"), self.cfg()),
                         {"slot_protect_seconds": slot_protect_seconds_of({"slot_protect_seconds": 200000})})

    def test_values_that_cannot_name_a_slot_are_refused(self):
        for protect in ("-1", "nan", "1e400", "two minutes"):
            with self.assertRaises(ValueError, msg=protect):
                slot_run_flags(self.flags(protect=protect), self.cfg())
        for slots in ("0,a", "2.5", "1,1", "one"):
            with self.assertRaises(ValueError, msg=slots):
                slot_run_flags(self.flags(slots=slots), self.cfg())

    def test_a_split_of_every_slot_is_refused_with_the_reason(self):
        # every slot protected leaves no slot for a request that names nothing: not a split, a server that serves
        # nobody, and the message has to say so
        for slots in ("4", "0,1,2,3"):
            with self.assertRaises(ValueError) as got:
                slot_run_flags(self.flags(slots=slots), self.cfg(4))
            self.assertIn("names nothing", str(got.exception))

    def test_a_slot_this_run_has_none_of_is_refused_with_the_range(self):
        for slots in ("9", "4,7"):
            with self.assertRaises(ValueError) as got:
                slot_run_flags(self.flags(slots=slots), self.cfg(4))
            self.assertIn("0..3", str(got.exception))

    def test_the_batch_the_split_is_checked_against_is_the_one_the_engine_gets(self):
        # --parallel is the requested slot count of this run as the config carries it: the split counts those
        # slots, so a split of 6 of a run asking 4 names slots that will not exist
        self.assertEqual(slot_run_flags(self.flags(slots="6"), {"args": [], "parallel": 8}),
                         {"protected_slots": "6"})
        for cfg in ({"args": [], "parallel": 4}, {"args": ["--batch", "4"]}, {"args": ["--slots", "4"]}):
            with self.assertRaises(ValueError, msg=cfg):
                slot_run_flags(self.flags(slots="6"), cfg)

    def test_a_server_without_batch_slots_refuses_a_split(self):
        # one request at a time: there is no slot to keep, and a split there routes nothing
        for cfg in ({"args": []}, {"args": ["--batch", "auto"]}):
            with self.assertRaises(ValueError, msg=cfg):
                slot_run_flags(self.flags(slots="1"), cfg)

    def test_the_flag_says_the_two_words_it_names(self):
        # the help of the flag names on and off, and the flag accepts exactly those: what is typed at a terminal is
        # checked while it can still be fixed.  The config is looser ("true", "yes", 1, the off words) because
        # setup and a hand-written file are answered with a warning, not with a refusal
        self.assertEqual(slot_run_flags(self.flags(reject="off"), self.cfg()), {"reject_when_full": False})
        for word in ("yes", "true", "1", "no", "0", "none", ""):
            with self.assertRaises(ValueError, msg=word):
                slot_run_flags(self.flags(reject=word), self.cfg())
        self.assertEqual(reject_when_full_of({"reject_when_full": "yes"}), True)

    def test_the_config_said_is_said_not_refused(self):
        # the config is written by setup and by an operator by hand: a bad value there warns and the plain routing
        # runs, as every other config value does; only a flag the operator typed is an error.  A flag given wins
        # over the config's unreadable value, which is the point of checking flags at all
        out = slot_run_flags(self.flags(protect="60"), {"args": [], "slot_protect_seconds": "abc"})
        self.assertEqual(out, {"slot_protect_seconds": 60.0})
        self.assertEqual(slot_protect_seconds_of({"slot_protect_seconds": "abc"}), 0.0)
        self.assertEqual(reject_when_full_of({"reject_when_full": "abc"}), False)


if __name__ == "__main__":
    unittest.main()
