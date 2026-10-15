"""a top-like live view of a running Strata server: it renders the server's own GET /metrics, with the
batch slots' MTP drafts, the expert cache's hits and the hardware, colored on a TTY and plain in a log.

    python tools/strata_top.py                      # live, refreshing every second
    python tools/strata_top.py --once               # one screen, for a log or a bug report
    python -m unittest tools.test_strata_top        # the tests, no server and no GPU
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.request

HISTORY_SECONDS = 60
TREND_ROWS = (("GPU util", "gpu_util", "%", "hardware"), ("GPU PCIe rx", "gpu_pcie_rx_mb", " MB/s", "hardware"),
              ("RAM used", "ram_used", " GiB", "hardware"), ("decode", "tok_s", " tok/s", "live"))
BLOCKS = "▁▂▃▄▅▆▇█"
IDLE_NEVER_S = 365 * 86400       # slot_used starts at 0.0 (server.py:935): an unused slot reports the clock
GOOD, WARN = 0.6, 0.3            # a draft acceptance worth its rows, and the floor below which it costs them

SGR = {"red": "31", "green": "32", "yellow": "33", "cyan": "36", "bold": "1", "dim": "2"}
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip(text: str) -> str:
    """The screen with its color escapes removed: exactly the plain screen it was built from."""
    return ANSI_RE.sub("", text)


def tone(v, good: float, warn: float) -> str:
    """The color of a reading where higher is better: green at `good`, yellow at `warn`, red below it;
    a missing reading has no color, so an older server's dashes stay plain."""
    if v is None:
        return ""
    if v >= good:
        return "green"
    if v >= warn:
        return "yellow"
    return "red"


def tone_low(v, good: float, warn: float) -> str:
    """The color of a reading where lower is better (a PCIe share, a card close to its power cap)."""
    if v is None:
        return ""
    if v <= good:
        return "green"
    if v <= warn:
        return "yellow"
    return "red"


class Palette:
    """The color layer as one object built per screen: a terminal gets escapes, a pipe, a redirect and
    --once get plain text.  paint() wraps a reading in its color and changes no character of it, so
    strip() of a colored screen is byte-identical to the plain screen built in the same call - which is
    what lets one screen be both a dashboard and the log a bug report is filed from."""

    def __init__(self, on: bool):
        self.on = on

    def paint(self, name: str, text: str) -> str:
        if not self.on or not name or not text:
            return text
        return f"\x1b[{SGR[name]}m{text}\x1b[0m"

    def rpad(self, name: str, text: str, width: int) -> str:
        """A right-aligned field, colored: the padding is inside the escapes, so the field keeps its
        column in a plain log and in a colored terminal (escapes have no width, but a padding applied
        around them would count them)."""
        return self.paint(name, f"{text:>{width}}")

    def good(self, ratio, text: str) -> str:
        """A reading whose thresholds are the ones that say whether --batch-mtp pays: green above the
        acceptance worth its extra row, yellow above the break-even, red below it."""
        return self.paint(tone(ratio, GOOD, WARN), text)

    def label(self, text: str, width: int = 10) -> str:
        """A panel's leading label padded to its column: the eye reads the screen down the labels."""
        return self.paint("cyan", f"{text:<{width}}")

    def bold(self, text: str) -> str:
        return self.paint("bold", text)

    def dim(self, text: str) -> str:
        return self.paint("dim", text)


def fetch(url: str, key: str) -> dict:
    """This server's GET /metrics as a dict (a Bearer key when one is set)."""
    req = urllib.request.Request(url.rstrip("/") + "/metrics")
    if key:
        req.add_header("Authorization", "Bearer " + key)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode())


def num(n) -> str:
    """An integer with thousands separators; None is a dash."""
    return "-" if n is None else f"{n:,}"


def val(x, unit: str = "") -> str:
    """A number followed by its unit; None is a dash."""
    return "-" if x is None else f"{x}{unit}"


def rate(x, unit: str = "") -> str:
    """A reading that arrives as a float (a PCIe throughput, a wattage) rounded to a whole number, so the row it
    shares with a sparkline stays the same width second by second."""
    return "-" if x is None else f"{x:,.0f}{unit}"


def gib(nbytes) -> str:
    """Bytes as GiB (telemetry reports mem_used, mem_total and ram_* in bytes)."""
    return "-" if nbytes is None else f"{nbytes / 1024 ** 3:.1f}"


def gib_from_mib(mib) -> str:
    """A MiB count as GiB (the engine's INFO line gives its cache sizes in MiB)."""
    return "-" if mib is None else f"{mib / 1024:.1f}"


def pct(ratio, digits: int = 1) -> str:
    """A 0..1 ratio as a percentage (a cache hit rate, a PCIe share)."""
    return "-" if ratio is None else f"{ratio * 100:.{digits}f}%"


def p100(x, digits: int = 0) -> str:
    """A value already in percent (a GPU's load, a CPU's) as a string."""
    return "-" if x is None else f"{x:.{digits}f}%"


def on_off(x, absent: str = "-") -> str:
    """A 0/1 flag the engine reports as an int (vram_elastic, slot_cache) as a word.  `absent` is the word
    for a flag the engine does not report: --batch-mtp is answered by the engine, and an older one says
    nothing, which for a drafter means off."""
    return absent if x is None else ("on" if x else "off")


def share(part, whole, digits: int = 1) -> str:
    """part/whole as a percentage; a missing count is a dash."""
    return "-" if part is None or not whole else f"{part / whole * 100:.{digits}f}%"


def tok_s(tokens, ms) -> str:
    """Tokens per second from a token count and the milliseconds it took."""
    return "-" if not tokens or not ms else f"{tokens / (ms / 1000.0):,.0f}"


def accept_rate(counts: dict):
    """A run's draft acceptance as a 0..1 ratio, from counts that may be absent (an older engine, plain
    batching).  None is "this run reported no drafts", which is not 0."""
    offered, accepted = counts.get("drafts_offered"), counts.get("drafts_accepted")
    if not offered or accepted is None:
        return None
    return accepted / offered


def draft_text(r: dict) -> str:
    """A run's MTP drafts as accepted/offered with their share: the solo path's DONE and a batch slot's
    BDONE both report them.  The dash is a run that reported NO drafts (plain batching, --batch-mtp off, an
    older engine); 0/N is drafts that were offered and taken by none.  Keeping the two apart is the whole
    point of the column: a dash on every row is a flag, not a measurement."""
    offered, accepted = r.get("drafts_offered"), r.get("drafts_accepted")
    if not offered:
        return "-"
    return f"{num(accepted)}/{num(offered)} {share(accepted, offered)}"


def bar(x, width: int = 22, pal: Palette = None) -> str:
    """A 0..100 percent as a bar; None is an empty bar.  The filled cells carry the reading's color, inside
    the fixed width, so a plain and a colored bar occupy the same columns."""
    if x is None:
        return "[" + " " * width + "]"
    filled = max(0, min(width, round(x / 100.0 * width)))
    body = pal.paint(tone(x, 40, 10), "#" * filled) if pal is not None else "#" * filled
    return "[" + body + " " * (width - filled) + "]"


def spark(values: list, width: int, pal: Palette = None) -> str:
    """A series as a sparkline (min-max scaled blocks); a gap prints as a space."""
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    series = list(values)[-width:]
    if len(numbers) < 2:
        out = " " * len(series)
    else:
        low, high = min(numbers), max(numbers)
        if high <= low:
            out = "".join(BLOCKS[1] if isinstance(v, (int, float)) else " " for v in series)
        else:
            out = "".join(BLOCKS[min(7, int((v - low) / (high - low) * 7.999))]
                         if isinstance(v, (int, float)) and not isinstance(v, bool) else " " for v in series)
    return pal.dim(out) if pal is not None else out


def dur_text(seconds) -> str:
    """A duration as 45s / 12m / 3h05m."""
    if seconds is None:
        return "-"
    s = int(seconds)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


def idle_text(seconds) -> str:
    """A slot's idle time.  The server stamps a slot's clock when a request leaves it and starts it at 0.0,
    so a slot the engine has never served reports the whole Unix clock: that is a fresh slot, and it reads
    "never".  497765h54m is the clock, not an idle time, and it would widen the column every screen."""
    if seconds is None:
        return "-"
    if seconds >= IDLE_NEVER_S:
        return "never"
    return dur_text(seconds)


def owner_flag(s: dict, window) -> str:
    """The flags a slot gets for "slot_ownership_s": who owns it, and how much of the window is left on it.  A
    slot that is idle and owned reads "idle free" without this, while it is closed to every conversation but
    its owner - which is why the fourth conversation does not run.  Past the window the slot holds a dead
    conversation and the next one takes it: "stale", the state to see when a chain is about to be rebuilt.
    The id is what the client named ("strata_session"); a hashed start is a digest and reads as itself.  No
    window (an engine without the setting, an older server): no flag, the screen is as it was."""
    if not window or window <= 0 or "owner" not in s:
        return ""
    owner = s.get("owner")
    if owner is None:
        return ""
    short = owner[2:] if owner[:2] in ("s:", "p:") else owner
    if not s.get("owned"):
        return f"stale {short}"
    left = window - (s.get("idle_s") or 0.0)
    return f"owns {short} {dur_text(max(0.0, left))}"


def title_panel(m: dict, pal: Palette) -> list[str]:
    """The headline: the model, the engine's version, the context, the state and how long it has been up."""
    eng, live, totals = m.get("engine", {}), m.get("live", {}), m.get("totals") or {}
    state = live.get("state")
    state_c = pal.paint({"generating": "green", "reading": "cyan"}.get(state, "dim"), str(state))
    parts = [pal.bold(f"Strata  {eng.get('model') or '?'}"), f"engine {val(eng.get('engine') or eng.get('version'))}",
             f"ctx {num(eng.get('max_context'))}", f"state {state_c}", f"queued {live.get('queued', 0)}"]
    if totals.get("since"):
        parts.append(f"up {dur_text(m.get('time', time.time()) - totals['since'])}")
    parts.append(pal.dim(time.strftime("%H:%M:%S")))
    return ["  ".join(parts)]


def engine_panel(m: dict, pal: Palette) -> list[str]:
    """The engine's own INFO facts: the KV format, the expert cache's size and slots, the drafter - with
    --batch-mtp named, because the batch slots' drafts ride on it and the INFO line is the one place that
    says whether the engine loaded the slot drafters - and the PCIe share."""
    eng = m.get("engine", {})
    kv = f"kv {val(eng.get('kv'))}" + (f" (resident {num(eng.get('kv_resident'))})" if eng.get("kv_resident") else "")
    free = eng.get("vram_free_mib")
    mtp_on = eng.get("batch_mtp")
    rows = [(pal.label("engine") + f"{kv}   ctx {num(eng.get('context'))}"
             f"   vram free {pal.paint(tone(free, 2048, 512), val(free, ' MiB'))}"
             f"   elastic {on_off(eng.get('vram_elastic'))}"),
            (pal.label("experts") + f"{num(eng.get('expert_slots'))} slots / {gib_from_mib(eng.get('expert_cache_mib'))} GiB"
             f"  (primary {num(eng.get('expert_slots_primary'))} / {gib_from_mib(eng.get('expert_cache_primary_mib'))}"
             f" GiB)   in RAM {gib_from_mib(eng.get('arena_mib'))} GiB   workers {val(eng.get('pool_workers'))}"),
            (pal.label("drafter") + f"MTP {val(eng.get('spec'))} (max {val(eng.get('mtp_max'))})"
             f"   lookup {val(eng.get('lookup'))}   min_p {val(eng.get('spec_min_p'))}"
             f"   pcie_frac {val(eng.get('pcie_frac'))}"
             f"   {pal.paint('green' if mtp_on else 'dim', 'batch-mtp ' + on_off(mtp_on, 'off'))}"
             + (pal.dim(f"   slot drafters {val(eng.get('batch_mtp_slots'))}") if eng.get("batch_mtp_slots") else ""))]
    if eng.get("batch_slots"):
        caps = [c for c in str(eng.get("slot_caps") or "").split(",") if c.strip().isdigit()]
        row = (pal.label("batch") + f"slots {val(eng.get('batch_slots'))}   slot_cache {on_off(eng.get('slot_cache'))}"
               f"   groups {val(eng.get('batch_groups'))}")
        if caps:
            row += "   slot contexts " + ", ".join(num(int(c)) for c in caps)
        rows.append(row)
    return rows


def live_panel(m: dict, pal: Palette) -> list[str]:
    """What the engine is doing now: the decode rate and a prompt read's progress."""
    live = m.get("live", {})
    state = live.get("state")
    rows = []
    if state == "generating":
        phase = f"  [{live['phase']}]" if live.get("phase") else ""
        rows.append(pal.label("decode") + f"{pal.bold(val(live.get('tok_s'), ' tok/s'))}"
                    f"  (mean {val(live.get('tok_s_mean'))})   {num(live.get('generated'))}/{num(live.get('max_tokens'))} tok"
                    f"   {val(live.get('elapsed_s'), 's')}{phase}")
    elif state == "reading":
        read, total = live.get("prompt_read"), live.get("prompt_total")
        rows.append(pal.label("prefill") + f"{val(live.get('prefill_tok_s_mean'), ' tok/s')}   read"
                    f" {num(read)}/{num(total)}   {bar(read / total * 100 if read and total else None, pal=pal)}")
    else:
        rows.append(pal.label("idle") + "nothing running")
    if live.get("parallel"):
        rows.append(pal.label("parallel") + f"{pal.bold(str(live.get('running', 0)))}/{live.get('parallel')} running"
                    f"   waiting {live.get('waiting', 0)}   admitting {live.get('admitting', 0)}"
                    f"   outside slots {live.get('outside_slots', 0)}"
                    + (pal.dim("  <- the solo path") if live.get("outside_slots") else ""))
    else:
        rows.append(pal.label("parallel") + 'off   one request at a time ("parallel": N in the config opens'
                    " N batch slots, and the slot panel with it)")
    return rows


def slots_panel(m: dict, pal: Palette) -> list[str]:
    """Every batch slot: its zone, the context it fits, what it runs, the drafts it ran and, when idle, the
    tokens it holds for the conversation that will come back to it.  A locked slot holds a long conversation
    used inside the grace, so eviction never takes it; an idle slot past the grace holds a dead conversation
    and can be taken.  `cap` is the read capacity pick_slot routes on: a prompt wider than a slot's cap never
    lands in it, so a request that reads its whole prompt again is a request that found no slot wide enough
    for it.  `drafts` is that slot's own MTP proposals: a slot of a --batch-mtp engine that shows "-" on
    every row is a slot decoding one token per window."""
    live = m.get("live", {})
    slots = live.get("slots") or []
    if not slots:
        return []
    zones = [s.get("zone") for s in slots if s.get("zone")]
    window = live.get("slot_ownership_s") or 0
    busy = sum(1 for s in slots if s.get("state") not in (None, "idle"))
    head = pal.label("slots") + f"{len(slots)}"
    if busy:
        head += pal.paint("green", f"  {busy} running")
    if zones:
        protected = sum(1 for z in zones if z == "protected")
        head += f"  ({protected} protected / {len(zones) - protected} guest)"
    if window:
        # how many slots a live conversation holds, of the slots that can be held: the window is a semaphore on
        # the conversations that run at once, not a cap on the conversations that are connected.  "3/3 held" and
        # six conversations producing tokens is the whole picture of that, on one line
        holders = live.get("slot_holders")
        seats = sum(1 for s in slots if s.get("zone") == "protected") if zones else len(slots)
        head += f"  (owned {holders if holders is not None else '?'} of {seats}, window {dur_text(window)})"
    rows = [pal.bold(head)]
    if zones:
        rows.append(pal.dim(f"{'':<10} {'zone':<10} {'cap':>9}  {'state':<11} {'tokens':>17}  {'tok/s':>8}"
                            f"  {'idle':>7}  {'drafts':>14}  flags"))
    for s in slots:
        state = s.get("state")
        if state == "idle":
            tokens = f"holds {num(s.get('held_tokens'))}"
        elif state == "reading":
            tokens = f"reading {num(s.get('prompt_tokens'))}"
        else:
            tokens = f"gen {num(s.get('generated'))}/{num(s.get('prompt_tokens'))}"
        clock = idle_text(s.get("idle_s")) if zones else val(s.get("elapsed_s"), "s")
        state_c = pal.paint({"decoding": "green", "reading": "cyan"}.get(state, "dim"), f"{state!s:<11}")
        drafts_c = pal.rpad(tone(accept_rate(s), GOOD, WARN), draft_text(s), 14)
        if zones:
            flags = ",".join((["LOCKED"] if s.get("locked") else [])
                             + (["cache"] if state == "idle" and s.get("held_tokens")
                                else ["free"] if state == "idle" else []))
            owned = owner_flag(s, window)
            if owned:
                flags = f"{flags} {owned}" if flags else owned
            flags = pal.paint("yellow", flags) if s.get("locked") else pal.dim(flags)
            rows.append(f"{f'  slot {s.get('slot')}'!s:<10} {s.get('zone')!s:<10} {num(s.get('cap')):>9}  {state_c}"
                        f" {tokens:>17}  {val(s.get('tok_s')):>8}  {pal.paint('dim', f'{clock:>7}')}"
                        f"  {drafts_c}  {flags}")
        else:
            rows.append(f"{f'  slot {s.get('slot')}'!s:<10} {state_c} {tokens:>17}  {val(s.get('tok_s')):>8}"
                        f"  {pal.paint('dim', f'{clock:>7}')}")
    return rows


def hardware_panel(m: dict, pal: Palette) -> list[str]:
    """Every card's load, VRAM, temperature, power and PCIe link, then CPU, RAM and disk.  NVML's PCIe
    generation is the card's current link, and it drops to gen 1 while the card is idle to save power, so the
    figure to read is the throughput next to it, not the generation."""
    hw, static = m.get("hardware", {}), m.get("hardware_static", {})
    rows = []
    gpus = hw.get("gpus")
    if gpus:
        for g in gpus:
            used, total = g.get("mem_used"), g.get("mem_total")
            frac = used / total * 100 if used and total else None
            temp = g.get("temp")
            rows.append(f"{pal.label(f'GPU{g.get('index')}', 8)}{bar(g.get('util'), pal=pal)}"
                        f" {pal.rpad('cyan', p100(g.get('util')), 5)}  VRAM {gib(used)}/{gib(total)} GiB"
                        f"   {bar(frac, 8, pal)}  {pal.rpad(tone_low(temp, 70, 85), val(temp, 'C'), 4)}"
                        f"  {pal.rpad('dim', rate(g.get('power'), 'W'), 6)}")
    else:
        used, total = hw.get("gpu_mem_used"), hw.get("gpu_mem_total")
        rows.append(f"{pal.label('GPU', 9)}{bar(hw.get('gpu_util'), pal=pal)} {p100(hw.get('gpu_util')):>5}"
                    f"  VRAM {gib(used)}/{gib(total)} GiB"
                    f"   {val(hw.get('gpu_temp'), 'C')}  {rate(hw.get('gpu_power'), 'W')}")
    gen = hw.get("pcie_gen")
    link = f"gen {val(gen)} x{val(hw.get('pcie_width'))}"
    if gen is not None and hw.get("pcie_gen_max") and gen < hw["pcie_gen_max"]:
        link += f" (the card's max is gen {hw['pcie_gen_max']}: NVML reads the idle link, so trust the MB/s)"
    rows.append(f"{pal.label('PCIe', 9)}{link}   rx {rate(hw.get('gpu_pcie_rx_mb'), ' MB/s')}"
                f"   tx {rate(hw.get('gpu_pcie_tx_mb'), ' MB/s')}")
    cpu, ram_used, ram_total = hw.get("cpu"), hw.get("ram_used"), hw.get("ram_total")
    frac = ram_used / ram_total if ram_used and ram_total else None
    free = gib((ram_total or 0) - (ram_used or 0)) if ram_total and ram_used is not None else "-"
    rows.append(f"{pal.label('CPU', 9)}{bar(cpu, pal=pal)} {pal.rpad('cyan', p100(cpu), 5)}"
                f"   RAM {gib(ram_used)}/{gib(ram_total)} GiB   {bar(frac * 100 if frac else None, 8, pal)}"
                f"  free {pal.paint(tone_low(frac, 0.8, 0.92), free)} GiB"
                f"   disk r{rate(hw.get('disk_read_mb'))} w{rate(hw.get('disk_write_mb'))} MB/s")
    if static.get("gpu_name") or static.get("cpu_name"):
        host = static.get("cpu_name") or "?"
        rows.append(f"{pal.label('host', 9)}{host}   {val(static.get('cores'))} cores /"
                    f" {val(static.get('threads'))} threads"
                    f"   {static.get('gpu_count') or len(gpus or [1])} GPU: {static.get('gpu_name') or '-'}")
    if static.get("gpu_note"):
        rows.append(f"{pal.label('note', 9)}{static['gpu_note']}")
    return rows


def trend_panel(m: dict, width: int, pal: Palette) -> list[str]:
    """The last minute of four series as sparklines: the GPU's load, its PCIe traffic, RAM, the decode rate.
    The GPU series and RAM live in the hardware sample, the rates in the server's own live status."""
    hist = m.get("history") or {}
    current_of = {"hardware": m.get("hardware", {}), "live": m.get("live", {})}
    rows = [pal.label("trend") + f"the last {HISTORY_SECONDS}s, one sample a second"]
    for label, key, unit, source in TREND_ROWS:
        series = hist.get(key) or []
        if not series:
            continue
        current = current_of[source].get(key)
        shown = gib(current) if unit == " GiB" else (rate(current) if unit == " MB/s" else val(current))
        rows.append(f"          {pal.paint('cyan', f'{label:<12}')} {spark(series, width, pal)}  {shown}{unit}")
    return rows


def cache_panel(m: dict, pal: Palette) -> list[str]:
    """The conversation cache: its budget, what is parked, the parks/restores/evictions and the prompt reuse.
    The reuse counts are the server's totals, so they report the prompt read again with the cache OFF too - that
    is the number a slot's own KV is spending, which is what a partition and a lock are there to bring down."""
    cc = m.get("conversation_cache")
    if not cc:
        return []
    rows = []
    if cc.get("enabled"):
        rows.append(pal.label("conv cache", 12) + f"on   budget {gib_from_mib(cc.get('budget_mib'))} GiB"
                    f"   slots {val(cc.get('slots'))}   parked {num(cc.get('parked'))} ({gib(cc.get('bytes'))} GiB)"
                    f"   {num(cc.get('parks'))} parks / {num(cc.get('restores'))} restores"
                    f" / {num(cc.get('evictions'))} evictions")
        if cc.get("last_event"):
            rows[-1] += pal.dim(f"   last: {cc['last_event']} {num(cc.get('last_tokens'))} tok")
    else:
        rows.append(pal.label("conv cache", 12) + f"{pal.paint('red', 'OFF')}   a conversation keeps only"
                    " its slot's KV (a parked turn: 0)")
    reused, prompt = cc.get("reused_tokens"), cc.get("prompt_tokens")
    ratio = reused / prompt if reused and prompt else None
    rows.append(pal.label("reuse", 12) + f"{num(reused)}/{num(prompt)} prompt tok"
                f"  ({pal.paint(tone(ratio, 0.8, 0.5), share(reused, prompt))} given back)"
                f"   last {num(cc.get('last_reused'))}/{num(cc.get('last_prompt'))}")
    return rows


def totals_panel(m: dict, pal: Palette) -> list[str]:
    """The running totals: requests, the tokens read again, the output, the MTP drafts accepted (the solo
    path's DONE and, since the slot BDONE carried them, the batch slots' too) and the rates since the server
    started (a mean over everything it has done, so they move slowly and follow the cache, not one turn)."""
    t = m.get("totals") or {}
    prompt = t.get("prompt_tokens")
    again = (prompt or 0) - (t.get("reused") or 0)
    counts = {"drafts_offered": t.get("drafts_offered"), "drafts_accepted": t.get("drafts_accepted")}
    accepted = f"{num(t.get('drafts_accepted'))}/{num(t.get('drafts_offered'))}" \
               f" ({share(t.get('drafts_accepted'), t.get('drafts_offered'))})"
    return [(pal.label("totals") + f"{num(t.get('requests'))} requests   prompt {num(prompt)} tok"
             f"   read again {num(again)} ({pal.paint(tone(1 - again / prompt if prompt else None, 0.8, 0.5), share(again, prompt))})"
             f"   output {num(t.get('output_tokens'))} tok"),
            (f"          MTP accepted {pal.good(accept_rate(counts), accepted)}"
             f"   since start: prefill {tok_s(prompt, t.get('prompt_ms'))} tok/s"
             f"   decode {tok_s(t.get('output_tokens'), t.get('decode_ms'))} tok/s")]


def expert_row(r: dict, pal: Palette) -> str | None:
    """One request's expert-cache row: its hit rate, the PCIe share and where its experts came from.  The hit
    rate is what --batch-mtp's extra rows spend: a window over several conversations routes to more experts,
    and every one of them that misses the VRAM cache arrives over PCIe."""
    if r.get("hit_rate") is None and r.get("pcie_share") is None and r.get("file_mb") is None:
        return None
    return (f"           experts  hit {pal.paint(tone(r.get('hit_rate'), 0.9, 0.7), pct(r.get('hit_rate')))}"
            f"   pcie {pal.paint(tone_low(r.get('pcie_share'), 0.05, 0.2), pct(r.get('pcie_share')))}"
            f"   RAM blobs {num(r.get('ram_blobs'))}   file {num(r.get('file_blobs'))} blobs"
            f" / {val(r.get('file_mb'))} MB")


def history_table(m: dict, rows: int, pal: Palette) -> list[str]:
    """The finished requests as a table. `read` is what the engine read again: a full cache reuse is 0, and a
    read as wide as the prompt is a turn that lost its checkpoint - the column to watch when testing a
    partition.  `drafts` is the request's MTP drafts (accepted/offered and their share): the solo path's DONE
    and a batch slot's BDONE both report them, and the dash is a request that ran none."""
    shown = (m.get("requests") or [])[:rows]
    if not shown:
        return []
    header = (f"           {'time':<9} {'finish':<8} {'prompt':>8} {'read':>8} {'reused':>8} {'out':>6}"
              f" {'decode t/s':>10} {'hit':>7} {'drafts':>11} {'dur':>6}")
    out = [pal.label("requests", 12) + f"the last {len(shown)} (read = tokens read again, 0 = a full reuse)",
           pal.dim(header)]
    for r in shown:
        stamp = time.strftime("%H:%M:%S", time.localtime(r["time"])) if r.get("time") else "-"
        counts = {"drafts_offered": r.get("drafts_offered"), "drafts_accepted": r.get("drafts_accepted")}
        brief = (f"{num(r.get('drafts_accepted'))}/{num(r.get('drafts_offered'))}"
                 if r.get("drafts_offered") else "-")
        out.append(f"           {stamp:<9} {r.get('finish')!s:<8} {num(r.get('prompt_tokens')):>8}"
                   f" {num(r.get('prompt_read')):>8} {num(r.get('reused')):>8} {num(r.get('output_tokens')):>6}"
                   f" {val(r.get('decode_tok_s')):>10} {pal.rpad(tone(r.get('hit_rate'), 0.9, 0.7), pct(r.get('hit_rate')), 7)}"
                   f" {pal.rpad(tone(accept_rate(counts), GOOD, WARN), brief, 11)}"
                   f" {val(r.get('duration_s'), 's'):>6}")
        extra = expert_row(r, pal)
        if extra:
            out.append(extra)
    return out


def render(m: dict, opts) -> str:
    """The /metrics dict as one top screen: the engine, what runs now, the hardware, trends, the cache,
    totals and the last requests.  With opts.color the screen carries escapes and nothing else changes:
    strip(render(m, color on)) is the text render(m, color off) builds, so one screen is both a dashboard
    and the log line a bug report is filed from."""
    pal = Palette(bool(getattr(opts, "color", False)))
    screen = title_panel(m, pal) + engine_panel(m, pal) + live_panel(m, pal)
    if not opts.compact:
        screen += slots_panel(m, pal)
    screen += hardware_panel(m, pal)
    if opts.spark:
        screen += trend_panel(m, opts.spark, pal)
    screen += cache_panel(m, pal) + totals_panel(m, pal)
    if opts.history:
        screen += history_table(m, opts.history, pal)
    return "\n".join(line.rstrip() for line in screen)


def color_for(when: str, stream) -> bool:
    """Whether this run paints: always only when asked, auto only on a terminal - a pipe, a redirect and a
    log get the plain screen, which is what every script that reads this output has always parsed."""
    if when == "always":
        return True
    if when == "never":
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def main() -> int:
    ap = argparse.ArgumentParser(
        description="a top-like live view of a running Strata server (it reads the server's own GET /metrics)",
        epilog="slots: zone = the config's protected_slots partition; cap = the context that slot fits;"
               " LOCKED = a long conversation in its grace, eviction never takes it; idle = seconds since used,"
               " never = a slot the engine has not served yet; drafts = that slot's MTP proposals (accepted/offered,"
               " and their share; "-" on every row of a --batch-mtp engine is the flag that its drafts are not running)."
               " read (in the table) = the prompt tokens read again (0 = a full cache reuse)."
               " A PCIe generation below the card's max is NVML reading the idle link: read the MB/s.")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("--api", default=os.environ.get("STRATA_API_KEY", ""), help="the server's api_key (Bearer)")
    ap.add_argument("--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    ap.add_argument("--history", type=int, default=6, help="rows of finished requests (0 turns the table off)")
    ap.add_argument("--spark", type=int, default=44, help="sparkline width (0 turns the trends off)")
    ap.add_argument("--compact", action="store_true", help="the essentials only: no slots, no trends, no table")
    ap.add_argument("--once", action="store_true", help="print one screen and exit (a log, a bug report)")
    ap.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                    help="color the screen (auto: on a terminal; a pipe, a redirect and --once stay plain)")
    opts = ap.parse_args()
    if opts.compact:
        opts.history, opts.spark = 0, 0
    opts.color = color_for(opts.color, sys.stdout)
    if opts.once:
        print(render(fetch(opts.url, opts.api), opts))
        return 0
    try:
        while True:
            try:
                body = render(fetch(opts.url, opts.api), opts)
            except Exception as e:  # noqa: BLE001 - a mid-restart server, a wrong key: say it, keep refreshing
                body = f"Strata  {opts.url}: {e.__class__.__name__}: {e}\n  (retrying every {opts.interval}s)"
            sys.stdout.write("\x1b[2J\x1b[H")
            print(body, flush=True)
            time.sleep(opts.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
