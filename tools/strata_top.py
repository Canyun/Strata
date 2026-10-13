"""A top-like live view of a running Strata server, built on its own GET /metrics.

It shows the engine's live state (decode tok/s, prefill, prompt read, queued, per-slot) beside the hardware
(every GPU's load, VRAM, temperature, power, plus CPU and RAM) and refreshes like top.  It needs only Python and
the server's URL - no jq, no nvidia-smi, no psutil on the machine running this script.  The same readings the web
Monitor tab shows (serve/telemetry.py samples them once a second).

    python3 tools/strata_top.py [URL] [--api KEY] [--interval S] [--slots]

URL defaults to http://127.0.0.1:8080.  With an api_key on the server, pass --api (or set STRATA_API_KEY).
--slots prints one line per batch slot.  Ctrl-C exits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.request


def fetch(url: str, key: str) -> dict:
    """This server's /metrics as a dict (Bearer key when one is set)."""
    req = urllib.request.Request(url.rstrip("/") + "/metrics")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def gib(b) -> str:
    """A byte count as GiB (telemetry's mem_used/mem_total/ram_* are bytes); None stays a dash."""
    return "-" if b is None else f"{b / 1024 ** 3:.1f}G"


def pct(x) -> str:
    return "-" if x is None else f"{x:.0f}%"


def val(x, unit: str = "") -> str:
    return "-" if x is None else f"{x}{unit}"


def bar(x, width: int = 24) -> str:
    """A 0..100 (a percent) as a bar; None is an empty bar."""
    if x is None:
        return "[" + " " * width + "]"
    f = max(0, min(width, round(x / 100.0 * width)))
    return "[" + "#" * f + " " * (width - f) + "]"


def render(m: dict, slots: bool) -> str:
    """The /metrics dict as one screen of top: the engine, the live request, each GPU, CPU/RAM, the cache, slots."""
    eng, live, hw = m.get("engine", {}), m.get("live", {}), m.get("hardware", {})
    out = [(f"Strata  {eng.get('model') or '?'}   ctx {eng.get('max_context')}   "
            f"state {live.get('state')}   queued {live.get('queued', 0)}   {time.strftime('%H:%M:%S')}")]
    if live.get("state") == "generating":
        out.append(f"  decode  {val(live.get('tok_s'), ' tok/s')} (mean {val(live.get('tok_s_mean'))})"
                   f"   {val(live.get('generated'))}/{val(live.get('max_tokens'))} tok   {val(live.get('elapsed_s'))}s")
    elif live.get("state") == "reading":
        read, tot = live.get("prompt_read"), live.get("prompt_total")
        out.append(f"  prefill {val(live.get('prefill_tok_s_mean'), ' tok/s')}   read {val(read)}/{val(tot)}"
                   f"   {bar(read / tot * 100 if read and tot else None)}")
    else:
        out.append("  idle")
    if live.get("parallel"):
        out.append(f"  parallel {live.get('running', 0)}/{live.get('parallel')} running   "
                   f"waiting {live.get('waiting', 0)}   admitting {live.get('admitting', 0)}")

    gpus = hw.get("gpus")
    if gpus:
        for g in gpus:
            used, total = g.get("mem_used"), g.get("mem_total")
            vram = used / total * 100 if used is not None and total else None
            out.append(f"  GPU{g.get('index')} {bar(g.get('util'))} {pct(g.get('util')):>4}"
                       f"  VRAM {gib(used)}/{gib(total)}  {bar(vram, 10)}"
                       f"  {val(g.get('temp'), 'C'):>5}  {val(g.get('power'), 'W')}")
    else:
        used, total = hw.get("gpu_mem_used"), hw.get("gpu_mem_total")
        vram = used / total * 100 if used is not None and total else None
        out.append(f"  GPU   {bar(hw.get('gpu_util'))} {pct(hw.get('gpu_util')):>4}"
                   f"  VRAM {gib(used)}/{gib(total)}  {bar(vram, 10)}"
                   f"  {val(hw.get('gpu_temp'), 'C')}  {val(hw.get('gpu_power'), 'W')}")

    cpu, ru, rt = hw.get("cpu"), hw.get("ram_used"), hw.get("ram_total")
    ram = ru / rt * 100 if ru is not None and rt else None
    out.append(f"  CPU   {bar(cpu)} {pct(cpu):>4}   RAM {gib(ru)}/{gib(rt)}  {bar(ram, 10)}")

    cc = m.get("conversation_cache") or {}
    if cc.get("enabled"):
        reused, prompt = cc.get("reused_tokens") or 0, cc.get("prompt_tokens") or 0
        out.append(f"  conv-cache  budget {val(cc.get('budget_mib'), ' MiB')}  slots {val(cc.get('slots'))}"
                   f"   reused {reused}/{prompt} tok ({pct(reused / prompt * 100 if prompt else None)})")

    if slots:
        for s in live.get("slots", []):
            if s.get("state") == "idle":
                out.append(f"    slot {s.get('slot')}: idle   {val(s.get('held_tokens'))} tok held")
            else:
                out.append(f"    slot {s.get('slot')}: {s.get('state')}   "
                           f"{val(s.get('generated'))} tok   {val(s.get('tok_s'), ' tok/s')}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="a top-like view of a running Strata server (reads GET /metrics)")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("--api", default=os.environ.get("STRATA_API_KEY", ""), help="the server's api_key (Bearer)")
    ap.add_argument("--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    ap.add_argument("--slots", action="store_true", help="show one line per batch slot")
    a = ap.parse_args()
    try:
        while True:
            try:
                body = render(fetch(a.url, a.api), a.slots)
            except Exception as e:  # noqa: BLE001 - a mid-restart server, a wrong key: say it, keep refreshing
                body = f"Strata  {a.url}: {e.__class__.__name__}: {e}\n  (retrying every {a.interval}s)"
            sys.stdout.write("\x1b[2J\x1b[H")
            print(body, flush=True)
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
