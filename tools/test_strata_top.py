"""tools/test_strata_top.py - the /metrics dict as a screen: the batch slot's MTP drafts, the idle-slot
guard, the color layer and what an older server's shorter /metrics still renders.  No server, no GPU.

    python -m unittest tools.test_strata_top
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import strata_top as ST  # noqa: E402

NEVER = 1.79e9        # slot_used starts at 0.0 (server.py:935), so idle_s is the whole Unix clock


def metrics():
    """The shape serve/server.py's metrics() returns: one decoding batch slot, an engine running
    --batch-mtp, two cards, and finished requests whose BDONE / DONE carried their draft counts."""
    return {
        "time": 1800000000.0,
        "engine": {"model": "qwen3.8-flash-next-iq3_s", "engine": "0.1.44.1", "max_context": 204800,
                   "context": 204800, "kv": "int8", "kv_resident": 0, "vram_free_mib": 403,
                   "vram_elastic": 0, "expert_slots": 20246, "expert_cache_mib": 39424,
                   "expert_slots_primary": 10576, "expert_cache_primary_mib": 19558,
                   "arena_mib": 15564, "pool_workers": 6, "spec": 6, "mtp_max": 4, "lookup": 3,
                   "spec_min_p": 0.5, "pcie_frac": 0.37, "batch_slots": 4, "slot_cache": 1,
                   "batch_groups": 1, "batch_mtp": 1},
        "live": {"state": "generating", "queued": 0, "tok_s": 80.8, "tok_s_mean": 76.0,
                 "generated": 119, "max_tokens": 5000, "elapsed_s": 1.5, "phase": "answering",
                 "parallel": 4, "running": 1, "waiting": 0, "admitting": 0, "outside_slots": 0,
                 "slots": [
                     {"slot": 0, "state": "decoding", "zone": "protected", "cap": 204800, "locked": True,
                      "prompt_tokens": 59355, "generated": 119, "elapsed_s": 1.5, "tok_s": 80.8,
                      "idle_s": 1.5, "held_tokens": 64981, "drafts_accepted": 97, "drafts_offered": 113},
                     {"slot": 1, "state": "idle", "zone": "protected", "cap": 204800, "locked": False,
                      "idle_s": 240.0, "held_tokens": 64650},
                     {"slot": 2, "state": "idle", "zone": "protected", "cap": 204800, "locked": False,
                      "idle_s": NEVER, "held_tokens": 0},
                     {"slot": 3, "state": "idle", "zone": "guest", "cap": 204800, "locked": False,
                      "idle_s": 1.0, "held_tokens": 3263}]},
        "hardware": {"gpus": [{"index": 0, "util": 96, "mem_used": 34e9, "mem_total": 34.4e9, "temp": 55,
                               "power": 268},
                              {"index": 1, "util": 0, "mem_used": 34.3e9, "mem_total": 34.4e9, "temp": 55,
                               "power": 89}],
                     "pcie_gen": 4, "pcie_gen_max": 4, "pcie_width": 16, "gpu_pcie_rx_mb": 1.0,
                     "gpu_pcie_tx_mb": 1.0, "cpu": 12, "ram_used": 28.7e9, "ram_total": 32.4e9,
                     "disk_read_mb": 0, "disk_write_mb": 0},
        "hardware_static": {"gpu_name": "NVIDIA GeForce RTX 4080 SUPER", "gpu_count": 2,
                            "cpu_name": "AMD Ryzen 9 5900X", "cores": 12, "threads": 24},
        "history": {"gpu_util": [10, 20, 90, 96], "tok_s": [1, 60, 80], "ram_used": [1e10, 2.8e10]},
        "conversation_cache": {"enabled": True, "budget_mib": 5120, "slots": 4, "parked": 1, "bytes": 2.3e9,
                               "parks": 21, "restores": 0, "evictions": 3, "last_event": "parked",
                               "last_tokens": 63294, "reused_tokens": 4121494, "prompt_tokens": 4606326,
                               "last_reused": 0, "last_prompt": 3145},
        "totals": {"since": 1800000000.0 - 5220, "requests": 76, "prompt_tokens": 4606326, "reused": 4121494,
                   "output_tokens": 65052, "prompt_ms": 2.3e6, "decode_ms": 8.1e5,
                   "drafts_offered": 12000, "drafts_accepted": 9000},
        "requests": [
            {"time": 1800000000.0, "finish": "stop", "prompt_tokens": 3145, "prompt_read": 3145,
             "reused": 0, "output_tokens": 119, "decode_tok_s": 80.8, "duration_s": 66.0,
             "hit_rate": 0.922, "pcie_share": 0.01, "ram_blobs": 42, "file_blobs": 0, "file_mb": 0,
             "drafts_offered": 113, "drafts_accepted": 97},
            {"time": 1800000000.0 - 30, "finish": "length", "prompt_tokens": 4946, "prompt_read": 0,
             "reused": 4946, "output_tokens": 5000, "decode_tok_s": 81.7, "duration_s": 81.6,
             "hit_rate": 0.923, "pcie_share": 0.002, "ram_blobs": 38, "file_blobs": 0, "file_mb": 0,
             "drafts_offered": None, "drafts_accepted": None}],
    }


def opts(**kw):
    """The render options, one overridden per test; compact turns the table and the trends off, as --compact does."""
    base = {"compact": False, "history": 6, "spark": 44, "color": False}
    base.update(kw)
    if base["compact"]:
        base["history"] = base["spark"] = 0
    return SimpleNamespace(**base)


def screen(m, **kw):
    return ST.render(m, opts(**kw))


class DraftCounts(unittest.TestCase):
    """The MTP drafts a batch slot ran are the engine's BDONE fields; the panel shows them per request,
    in the totals and in the drafter line.  A leg that ran no drafts is a dash, never a 0/0."""

    def test_the_column_shows_accepted_of_offered(self):
        self.assertEqual(ST.draft_text(metrics()["requests"][0]), "97/113 85.8%")

    def test_the_share_of_the_accepted(self):
        self.assertIn("85.8%", screen(metrics()))

    def test_a_leg_without_drafts_is_a_dash(self):
        self.assertEqual(ST.draft_text(metrics()["requests"][1]), "-")

    def test_zero_offered_is_a_dash_too(self):
        self.assertEqual(ST.draft_text({"drafts_accepted": 0, "drafts_offered": 0}), "-")

    def test_the_totals_show_their_own_counts(self):
        self.assertIn("9,000/12,000", screen(metrics()))

    def test_the_drafter_line_names_batch_mtp(self):
        self.assertIn("batch-mtp on", screen(metrics()))

    def test_the_drafter_line_says_off_without_it(self):
        m = metrics()
        m["engine"].pop("batch_mtp")
        self.assertIn("batch-mtp off", screen(m))


class IdleGuard(unittest.TestCase):
    """A slot the engine has never used reports idle_s as the whole clock, because slot_used starts at
    0.0: that reads "never", not 497765h, and it keeps the column's width."""

    def test_the_idle_of_a_never_used_slot(self):
        self.assertEqual(ST.idle_text(NEVER), "never")

    def test_a_real_idle_still_shows(self):
        self.assertEqual(ST.idle_text(240.0), "4m")
        self.assertEqual(ST.idle_text(1.0), "1s")

    def test_the_screen_never_shows_the_era(self):
        rows = {ln.split()[1]: ln.rstrip() for ln in screen(metrics()).splitlines() if ln.startswith("  slot ")}
        self.assertIn("never", rows["2"])
        self.assertNotIn("497765h", rows["2"])
        # a real idle ("4m") and a never-used slot ("never") share the idle column, so the flags column starts at
        # the same place on every row: the width the table had before the guard, not the clock's width
        start = {n: len(row) - len(row.split()[-1]) for n, row in rows.items()}
        self.assertEqual(len(set(start.values())), 1, start)


class ColorLayer(unittest.TestCase):
    """Color is the same text with escapes around it: stripping it returns exactly the plain screen, so
    a log, a pipe and --once keep the byte-identical output they have always had."""

    def test_the_plain_screen_has_no_escapes(self):
        self.assertNotIn("\x1b", screen(metrics()))

    def test_the_colored_screen_has_them(self):
        self.assertIn("\x1b[", screen(metrics(), color=True))

    def test_stripping_gives_the_plain_screen(self):
        self.assertEqual(ST.strip(screen(metrics(), color=True)), screen(metrics()))

    def test_tone_follows_the_thresholds(self):
        self.assertEqual(ST.tone(0.95, good=0.9, warn=0.5), "green")
        self.assertEqual(ST.tone(0.7, good=0.9, warn=0.5), "yellow")
        self.assertEqual(ST.tone(0.2, good=0.9, warn=0.5), "red")

    def test_an_unknown_reading_has_no_color(self):
        self.assertEqual(ST.tone(None, good=0.9, warn=0.5), "")


class OlderServer(unittest.TestCase):
    """An older server's /metrics has less in it: no partition, no batch, no cache, no hardware.  Every
    panel reads through .get(), so a short payload renders and says what it can."""

    def test_a_minimal_payload_renders(self):
        body = screen({"engine": {"model": "qwen-mini", "engine": "0.1.20"}, "live": {"state": "idle"},
                       "hardware": {}, "hardware_static": {}, "history": {}})
        self.assertIn("qwen-mini", body)
        self.assertIn("idle", body)

    def test_no_batch_says_one_at_a_time(self):
        body = screen({"engine": {"model": "m"}, "live": {"state": "idle", "parallel": 0}, "hardware": {}})
        self.assertIn("one request at a time", body)

    def test_compact_drops_the_slots_and_the_table(self):
        body = screen(metrics(), compact=True)
        self.assertNotIn("slot 0", body)
        self.assertIn("experts", body)

    def test_none_everywhere_still_renders(self):
        m = metrics()
        for key in ("totals", "conversation_cache", "requests"):
            m[key] = None
        for key in m["live"]["slots"][0]:
            m["live"]["slots"][0][key] = None
        self.assertIn("Strata", screen(m))


class Ownership(unittest.TestCase):
    """"slot_protect_seconds": a slot belongs to a conversation, and the panel is the only place that shows it.  An
    owned slot reads "idle, free" while it is closed to every conversation but its owner - which is what makes
    the fourth conversation not run, and what a user has to be able to see."""

    def payload(self, window=120.0, owners=(("s:task-42", None), ("s:task-7", 240.0), None, None)):
        """Four slots, a 120 s window: slot 0 runs its conversation, slot 1 holds one idle past the window (its
        chain is dead and the slot is open), slots 2-3 have none.  owners: (key, idle_s) per slot."""
        m = metrics()
        m["live"]["slot_protect_seconds"] = window
        m["live"]["slot_holders"] = sum(1 for owner in owners
                                        if owner and (owner[1] is None or owner[1] <= window))
        for s, owner in zip(m["live"]["slots"], owners):
            s.pop("owner", None)
            s.pop("owned", None)
            if owner is None:
                continue
            key, idle = owner if isinstance(owner, tuple) else (owner, s.get("idle_s"))
            s["owner"], s["owned"] = key, (idle is not None and idle <= window)
        return m

    def rows(self, m):
        """The slot rows by slot number, from a /metrics payload."""
        return {ln.split()[1]: ln.rstrip() for ln in screen(m).splitlines() if ln.startswith("  slot ")}

    def test_the_running_slots_owner_shows_and_the_head_names_the_window(self):
        m = self.payload()
        self.assertIn("task-42", self.rows(m)["0"])
        self.assertIn(f"window {ST.dur_text(120.0)}", screen(m))

    def test_an_idle_owned_slot_shows_the_window_left(self):
        m = self.payload()
        m["live"]["slots"][1]["owner"], m["live"]["slots"][1]["owned"] = "s:task-7", True
        m["live"]["slots"][1]["idle_s"] = 30.0          # 90 s of the 120 s window left
        row = self.rows(m)["1"]
        self.assertIn("task-7", row)
        self.assertIn("owns", row)
        self.assertIn(ST.dur_text(90.0), row)

    def test_a_slot_whose_owner_went_idle_past_the_window_is_marked(self):
        row = self.rows(self.payload())["1"]
        self.assertIn("task-7", row)
        self.assertIn("stale", row)

    def test_the_screen_with_a_window_shows_owns_not_the_eviction_lock(self):
        # with the window on, the eviction lock's LOCKED flag ranks nothing the pick obeys (a conversation's
        # pick returns at the ownership veto): showing it next to "stale"/"owns" is a second, dead flag for
        # the same slot - the contradiction the user's panel showed as "LOCKED stale task-42"
        m = self.payload()
        m["live"]["slots"][0]["owned"] = True       # decoding: used this instant, the window is open on it
        rows = self.rows(m)
        self.assertIn("owns", rows["0"])
        self.assertNotIn("LOCKED", screen(m))

    def test_an_unowned_slot_says_nothing_about_ownership(self):
        rows = self.rows(self.payload())
        self.assertNotIn("owns", rows["3"])
        self.assertNotIn("stale", rows["3"])

    def test_no_window_is_the_screen_as_it_was(self):
        # an older server (no owner fields) and an engine with the window off: the flags column is what it was
        m = self.payload(window=0.0, owners=(None, None, None, None))
        m["live"].pop("slot_protect_seconds")
        for row in self.rows(m).values():
            self.assertNotIn("owns", row)
            self.assertNotIn("stale", row)
        rows = self.rows(m)
        self.assertIn("LOCKED", rows["0"])
        self.assertIn("free", rows["2"])


class Cli(unittest.TestCase):
    """main() builds its ArgumentParser before it parses anything: 001ab0e wrote a quoted "-" between two
    adjacent epilog literals, which Python reads as str - str, and every call - --help, the panel, --once -
    died with a TypeError before doing anything.  A screen the user cannot start is the bug the panel's own
    tests (render on a dict) never see, so the entry point itself is what gets tested here."""

    def run_cli(self, argv):
        """main() with sys.argv set: the help exits 0 through SystemExit, and the TypeError (the bug) escapes."""
        out = io.StringIO()
        with mock.patch.object(sys, "argv", ["strata_top.py"] + argv), contextlib.redirect_stdout(out):
            try:
                code = ST.main()
            except SystemExit as e:               # --help: print, exit 0
                code = e.code
        return code, out.getvalue()

    def test_the_entry_point_starts_and_its_help_shows_the_legend(self):
        code, out = self.run_cli(["--help"])
        self.assertEqual(code, 0)
        self.assertIn("drafts = that slot", " ".join(out.split()))

    def test_the_epilog_keeps_its_hyphen_literal(self):
        # the drafts legend quotes the "-" a row shows: it has to survive into the printed help.  argparse
        # wraps the epilog at word boundaries at any width, so the phrase is matched with the wrap flattened.
        code, out = self.run_cli(["--help"])
        self.assertIn('"-" on every row', " ".join(out.split()))


if __name__ == "__main__":
    unittest.main()
