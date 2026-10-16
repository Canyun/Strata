"""#465: setup's recommendation for "parallel" (several requests decoded together in the engine's batch slots).
No GPU or download needed: python tools/test_setup_parallel.py"""
import argparse
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import setup as S  # noqa: E402
from test_setup_golden import PROFILES, install  # noqa: E402


class Parallel(unittest.TestCase):
    def test_slot_size(self):
        # measured: a slot's session at 32K, int8 KV: 0.56 GiB = 0.60 GB
        self.assertAlmostEqual(S.parallel_slot_gb(32768, "int8", False), 0.59, delta=0.03)
        # KV streaming keeps only the attention's 32K positions in VRAM, whatever the context
        self.assertEqual(S.parallel_slot_gb(262144, "int8", True), S.parallel_slot_gb(32768, "int8", False))
        self.assertLess(S.parallel_slot_gb(32768, "q4_0", False), S.parallel_slot_gb(32768, "int8", False))

    def test_recommendation_only_where_the_experts_mostly_fit(self):
        q2, iq3s = S.MODELS["Q2_0"]["arena_gb"], S.MODELS["IQ3_S"]["arena_gb"]
        # most experts on the CPU: slots cost speed per request (measured on the 12 GB 5070) - none recommended
        self.assertEqual(S.parallel_recommend(12, q2, 32768, "int8", False), 0)
        self.assertEqual(S.parallel_recommend(16, q2, 32768, "int8", False), 0)
        self.assertEqual(S.parallel_recommend(24, iq3s, 32768, "int8", False), 0)
        # the cache holds most of the experts beside the slots: recommended
        self.assertEqual(S.parallel_recommend(24, q2, 32768, "int8", False), 3)
        self.assertEqual(S.parallel_recommend(48, q2, 32768, "int8", False), 4)       # capped at 4
        self.assertEqual(S.parallel_recommend([24, 24], iq3s, 32768, "int8", False), 4)   # a layer split: every card
        self.assertEqual(S.parallel_recommend([16, 16], q2, 32768, "int8", False), 4)
        self.assertEqual(S.parallel_recommend(32, q2, 262144, "int8", False), 0)      # a 262K KV per slot: no
        self.assertEqual(S.parallel_recommend(32, q2, 262144, "int8", True), 4)       # ... unless it streams

    def test_notes_recommend_never_force(self):
        q2 = S.MODELS["Q2_0"]["arena_gb"]
        # 12 GB: left at one, with the reason
        note = S.parallel_note(None, 12, q2, 32768, "int8", False)
        self.assertEqual(len(note), 1)
        self.assertIn("one at a time", note[0])
        self.assertIn("costs about 10-25% speed per request on this card", note[0])
        self.assertIn("--parallel 3", S.parallel_note(None, 24, q2, 32768, "int8", False)[0])
        # asked on 12 GB: kept, with the cost said
        lines = S.parallel_note(4, 12, q2, 32768, "int8", False)
        self.assertIn("4 at once", lines[0])
        self.assertTrue(any("one at a time" in x and "kept as you chose" in x and "10-25%" in x for x in lines))
        # asked above the recommendation where it pays
        lines = S.parallel_note(4, 24, q2, 32768, "int8", False)
        self.assertTrue(any("recommended for this card: 3" in x for x in lines))
        self.assertTrue(any("at most 8" in x for x in S.parallel_note(12, 48, q2, 32768, "int8", False)))
        self.assertEqual(len(S.parallel_note(3, 24, q2, 32768, "int8", False)), 1)   # as recommended: no warning


class SlotOwnership(unittest.TestCase):
    def test_default_partition_leaves_one_guest_slot(self):
        self.assertEqual(S.protected_slots_default(4), [0, 1, 2])
        self.assertEqual(S.protected_slots_default(2), [0])
        self.assertEqual(S.protected_slots_default(1), [])
        self.assertEqual(S.protected_slots_default(0), [])
        self.assertEqual(S.protected_slots_default(None), [])
        self.assertEqual(S.protected_slots_default(True), [])

    def test_window_off_leaves_the_old_behaviour(self):
        for asked in (0, "0", None, -5, True):
            seconds, notes = S.slot_protect_seconds_choice(asked, 4)
            self.assertEqual(seconds, 0.0)
            self.assertIn("off", notes[0])

    def test_window_needs_two_slots(self):
        seconds, notes = S.slot_protect_seconds_choice(120, 1)
        self.assertEqual(seconds, 0.0)
        self.assertIn("needs", notes[0])
        self.assertIn("parallel", notes[0])

    def test_window_written_with_the_pair_it_needs(self):
        seconds, notes = S.slot_protect_seconds_choice(120, 4)
        self.assertEqual(seconds, 120.0)
        self.assertIn("3 of the 4 slots", notes[0])
        self.assertTrue(any("solo_path" in x for x in notes))
        self.assertTrue(any("run at once" in x for x in notes))

    def test_main_writes_the_three_keys_together(self):
        # the mechanism is three keys as one: a window without the partition and the solo path off is half a
        # mechanism (serve/server.py: the solo path takes no slot, a full partition is no partition)
        source = Path(__file__).resolve().parent.parent.joinpath("setup.py").read_text(encoding="utf-8")
        self.assertIn('cfg["slot_protect_seconds"]', source)
        self.assertIn('cfg["protected_slots"]', source)
        self.assertIn('cfg["solo_path"] = False', source)

    def test_a_rerun_keeps_the_mechanism_the_user_wrote_by_hand(self):
        # the three keys are not SETUP_KEYS: setup writes them only when asked, so a setup run that does not
        # touch the mechanism must carry the user's own values over, never delete them (the #629 rule)
        old = {"parallel": 4, "slot_protect_seconds": 120, "protected_slots": [0, 1, 2], "solo_path": False}
        cfg = {"exe": "x"}
        kept = S.carry_over(old, cfg)
        self.assertEqual(cfg.get("slot_protect_seconds"), 120)
        self.assertIn("slot_protect_seconds", kept)

    def test_turning_the_mechanism_off_survives_the_rerun(self):
        # setup.py's help says 0 turns it off.  The off branch pops the window, but the three keys are not
        # SETUP_KEYS, so carry_over moved the old ones back and the switch never turned off (the review's
        # finding #1, confirmed with a probe).  apply() now drops all three, and carry_over honours the drop.
        a = SimpleNamespace(slot_protect_seconds=0, parallel=4)
        cfg = {"parallel": 4}
        drop = S.slot_protection_apply(a, cfg, {}, lambda m: None, lambda m: None)
        self.assertEqual(drop, {"slot_protect_seconds", "protected_slots", "solo_path"})
        for k in drop:
            self.assertNotIn(k, cfg)
        old = {"parallel": 4, "slot_protect_seconds": 120, "protected_slots": [0, 1, 2], "solo_path": False}
        S.carry_over(old, cfg, drop=drop)
        self.assertNotIn("slot_protect_seconds", cfg)
        self.assertNotIn("solo_path", cfg)

    def test_slots_fall_back_to_the_installed_parallel(self):
        # a re-run that names only --slot-protect-seconds: "parallel" is in the installed config, not in the
        # fresh cfg yet (it is merged at write time).  Reading the fresh cfg alone returned 0 and the
        # mechanism was silently refused with "needs parallel of 2+" on a 4-slot install (finding #2).
        self.assertEqual(S.slots_for_protection(None, {}, {"parallel": 4}), 4)
        self.assertEqual(S.slots_for_protection(None, {"parallel": 3}, {"parallel": 4}), 3)   # this run wins
        self.assertEqual(S.slots_for_protection(5, {}, {"parallel": 4}), 5)                  # asked wins
        self.assertEqual(S.slots_for_protection(None, {}, {}), 0)

    def test_own_path_is_told_the_mechanism_needs_parallel(self):
        # the own-path install has no batch slots (no --parallel): --slot-protect-seconds there must warn, not
        # write the keys silently (finding #3: accepted by argparse, never reached the config)
        a = SimpleNamespace(slot_protect_seconds=120, parallel=None)
        cfg, notes = {}, []
        S.slot_protection_apply(a, cfg, {}, lambda m: notes.append(m), lambda m: notes.append(m))
        self.assertNotIn("slot_protect_seconds", cfg)
        self.assertTrue(any("parallel" in m for m in notes), notes)

    def test_asking_for_a_window_it_cannot_size_leaves_the_installed_mechanism_alone(self):
        # "off" is the only run that removes the mechanism.  A run that asks for a window the slot count cannot
        # size (no parallel of 2) is refused with a warning - and must not delete what the user wrote by hand:
        # the installed mechanism keeps running as it was, which is what the warning is about
        a = SimpleNamespace(slot_protect_seconds=120, parallel=1)
        drop = S.slot_protection_apply(a, {"parallel": 1}, {}, lambda m: None, lambda m: None)
        self.assertEqual(drop, frozenset())

    def test_only_an_explicit_off_returns_the_keys_to_drop(self):
        for asked in (0, "off", -1):
            a = SimpleNamespace(slot_protect_seconds=S.slot_protect_seconds_arg(asked), parallel=4)
            drop = S.slot_protection_apply(a, {"parallel": 4}, {}, lambda m: None, lambda m: None)
            self.assertEqual(drop, S.SLOT_PROTECTION_KEYS, str(asked))

    def test_args_without_the_slot_fields_are_tolerated(self):
        # install_own_path calls apply() with its own argparse.Namespace, which has no --slot-protect-seconds /
        # --parallel on it (the own-path family writes no batch slots).  Reading a.slot_protect_seconds straight
        # raised AttributeError and broke every Qwen3.6 install (the full suite caught it): read the fields
        # defensively, as the engine-attribute reads do (getattr).
        bare = SimpleNamespace(model="qwen36")
        self.assertEqual(S.slot_protection_apply(bare, {"exe": "x"}, {}, lambda m: None, lambda m: None),
                         frozenset())
        no_parallel = SimpleNamespace(slot_protect_seconds=120)   # asked, no parallel anywhere -> refused, keep
        cfg = {"exe": "x", "slot_protect_seconds": 90}
        self.assertEqual(S.slot_protection_apply(no_parallel, cfg, {}, lambda m: None, lambda m: None),
                         frozenset())
        self.assertEqual(cfg["slot_protect_seconds"], 90)

    def test_the_window_switch_takes_the_off_the_help_advertises(self):
        # the help advertises that a word turns the window off, but type=int made argparse exit 2 on the word
        # before main() ever saw it (finding #5).  The converter takes the word, so the help is the truth.
        for word in ("off", "OFF", "off ", "none", "no", "false", "disable"):
            self.assertEqual(S.slot_protect_seconds_arg(word), 0)
        self.assertEqual(S.slot_protect_seconds_arg("120"), 120)
        self.assertEqual(S.slot_protect_seconds_arg("0"), 0)
        self.assertEqual(S.slot_protect_seconds_arg(120), 120)      # argparse passes a non-string through
        for bad in ("abc", "10s", "120.5", ""):
            with self.assertRaises(ValueError):
                S.slot_protect_seconds_arg(bad)

    def test_the_converter_is_what_argparse_hands_the_value_to(self):
        # setup builds its flag inside main(), so the converter is given to an equivalent parser: the same
        # contract - the word parses to 0, and a non-number stays argparse's usage error.  WindowInstalled
        # below runs the word through setup's own parser and config file, end to end.
        ap = argparse.ArgumentParser()
        ap.add_argument("--slot-protect-seconds", type=S.slot_protect_seconds_arg, metavar="N")
        self.assertEqual(ap.parse_args(["--slot-protect-seconds", "off"]).slot_protect_seconds, 0)
        self.assertEqual(ap.parse_args(["--slot-protect-seconds", "120"]).slot_protect_seconds, 120)
        with self.assertRaises(SystemExit):
            ap.parse_args(["--slot-protect-seconds", "10s"])


class WindowInstalled(unittest.TestCase):
    """The window through setup's own parser and its own config file: on, off by the word, off for good."""

    def setUp(self):
        self.ram, self.found = PROFILES["64GB-1x32GB"]

    def run_setup(self, argv, installed=None):
        configs = [("strata-q2_0.json", installed)] if installed else ()
        return install(self.ram, self.found, ["--family", "qwen", "--model", "Q2_0", "--no-start", *argv],
                       configs=configs)

    def test_on_then_off_by_the_word_the_help_advertises(self):
        # on: the window, its partition and the solo path off, written as the one mechanism they are
        code, out, first, _ = self.run_setup(["--parallel", "4", "--slot-protect-seconds", "120"])
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(first["slot_protect_seconds"], 120)
        self.assertEqual(first["protected_slots"], [0, 1, 2])
        self.assertIs(first["solo_path"], False)
        # off: the word the help has always advertised.  type=int made argparse exit 2 on it before main() saw
        # it (finding #5), and the off branch's pop() was undone by carry_over (finding #1).  The re-run names
        # no --parallel either: the installed config is what sizes the window (finding #2)
        code, out, off, _ = self.run_setup(["--slot-protect-seconds", "off"], installed=first)
        self.assertEqual(code, 0, out[-3000:])
        self.assertIn("off", out)
        for key in S.SLOT_PROTECTION_KEYS:
            self.assertNotIn(key, off, key)
        # a later plain setup run has nothing to carry back: the switch stayed off
        code, out, again, _ = self.run_setup(["--context", "65536"], installed=off)
        self.assertEqual(code, 0, out[-3000:])
        for key in S.SLOT_PROTECTION_KEYS:
            self.assertNotIn(key, again, key)

    def test_a_hand_written_window_survives_a_run_that_names_nothing(self):
        # the mirror of the case above: setup writes these keys only when asked, so a run that does not ask
        # leaves a user's own window in place (the #629 rule) - dropping it here would delete a hand edit
        installed = {"args": ["--mmap-experts"], "exe": "x", "parallel": 4, "slot_protect_seconds": 90,
                     "protected_slots": [0, 1], "solo_path": False}
        code, out, cfg, _ = self.run_setup(["--context", "65536"], installed=installed)
        self.assertEqual(code, 0, out[-3000:])
        self.assertEqual(cfg.get("slot_protect_seconds"), 90)
        self.assertEqual(cfg.get("protected_slots"), [0, 1])
        self.assertIs(cfg.get("solo_path"), False)


if __name__ == "__main__":
    unittest.main()
