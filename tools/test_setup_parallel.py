"""#465: setup's recommendation for "parallel" (several requests decoded together in the engine's batch slots).
No GPU or download needed: python tools/test_setup_parallel.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import setup as S  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
