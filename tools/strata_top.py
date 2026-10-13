import argparse
import json
import os
import sys
import time
import urllib.request

HISTORY_SECONDS = 60
TREND_ROWS = (("GPU util", "gpu_util", "%", "hardware"), ("GPU PCIe rx", "gpu_pcie_rx_mb", " MB/s", "hardware"),
              ("RAM used", "ram_used", " GiB", "hardware"), ("decode", "tok_s", " tok/s", "live"))
BLOCKS = "▁▂▃▄▅▆▇█"


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


def on_off(x) -> str:
    """A 0/1 flag the engine reports as an int (vram_elastic, slot_cache) as a word."""
    return "-" if x is None else ("on" if x else "off")


def share(part, whole, digits: int = 1) -> str:
    """part/whole as a percentage; a missing count is a dash."""
    return "-" if part is None or not whole else f"{part / whole * 100:.{digits}f}%"


def tok_s(tokens, ms) -> str:
    """Tokens per second from a token count and the milliseconds it took."""
    return "-" if not tokens or not ms else f"{tokens / (ms / 1000.0):,.0f}"


def bar(x, width: int = 22) -> str:
    """A 0..100 percent as a bar; None is an empty bar."""
    if x is None:
        return "[" + " " * width + "]"
    filled = max(0, min(width, round(x / 100.0 * width)))
    return "[" + "#" * filled + " " * (width - filled) + "]"


def spark(values: list, width: int) -> str:
    """A series as a sparkline (min-max scaled blocks); a gap prints as a space."""
    numbers = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    series = list(values)[-width:]
    if len(numbers) < 2:
        return " " * len(series)
    low, high = min(numbers), max(numbers)
    if high <= low:
        return "".join(BLOCKS[1] if isinstance(v, (int, float)) else " " for v in series)
    out = []
    for v in series:
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            out.append(" ")
        else:
            out.append(BLOCKS[min(7, int((v - low) / (high - low) * 7.999))])
    return "".join(out)


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


def title_panel(m: dict) -> list[str]:
    """The headline: the model, the engine's version, the context, the state and how long it has been up."""
    eng, live, totals = m.get("engine", {}), m.get("live", {}), m.get("totals") or {}
    parts = [f"Strata  {eng.get('model') or '?'}", f"engine {val(eng.get('engine') or eng.get('version'))}",
             f"ctx {num(eng.get('max_context'))}", f"state {live.get('state')}",
             f"queued {live.get('queued', 0)}"]
    if totals.get("since"):
        parts.append(f"up {dur_text(m.get('time', time.time()) - totals['since'])}")
    parts.append(time.strftime("%H:%M:%S"))
    return ["  ".join(parts)]


def engine_panel(m: dict) -> list[str]:
    """The engine's own INFO facts: the KV format, the expert cache's size and slots, the drafter, the PCIe share."""
    eng = m.get("engine", {})
    kv = f"kv {val(eng.get('kv'))}" + (f" (resident {num(eng.get('kv_resident'))})" if eng.get("kv_resident") else "")
    rows = [(f"engine    {kv}   ctx {num(eng.get('context'))}   vram free {val(eng.get('vram_free_mib'), ' MiB')}"
             f"   elastic {on_off(eng.get('vram_elastic'))}"),
            (f"experts   {num(eng.get('expert_slots'))} slots / {gib_from_mib(eng.get('expert_cache_mib'))} GiB"
             f"  (primary {num(eng.get('expert_slots_primary'))} / {gib_from_mib(eng.get('expert_cache_primary_mib'))}"
             f" GiB)   in RAM {gib_from_mib(eng.get('arena_mib'))} GiB   workers {val(eng.get('pool_workers'))}"),
            (f"drafter   MTP {val(eng.get('spec'))} (max {val(eng.get('mtp_max'))})   lookup {val(eng.get('lookup'))}"
             f"   min_p {val(eng.get('spec_min_p'))}   pcie_frac {val(eng.get('pcie_frac'))}")]
    if eng.get("batch_slots"):
        caps = [c for c in str(eng.get("slot_caps") or "").split(",") if c.strip().isdigit()]
        row = (f"batch     slots {val(eng.get('batch_slots'))}   slot_cache {on_off(eng.get('slot_cache'))}"
               f"   groups {val(eng.get('batch_groups'))}")
        if caps:
            row += "   slot contexts " + ", ".join(num(int(c)) for c in caps)
        rows.append(row)
    return rows


def live_panel(m: dict) -> list[str]:
    """What the engine is doing now: the decode rate and a prompt read's progress."""
    live = m.get("live", {})
    state = live.get("state")
    rows = []
    if state == "generating":
        phase = f"  [{live['phase']}]" if live.get("phase") else ""
        rows.append(f"decode    {val(live.get('tok_s'), ' tok/s')}  (mean {val(live.get('tok_s_mean'))})"
                    f"   {num(live.get('generated'))}/{num(live.get('max_tokens'))} tok"
                    f"   {val(live.get('elapsed_s'), 's')}{phase}")
    elif state == "reading":
        read, total = live.get("prompt_read"), live.get("prompt_total")
        rows.append(f"prefill   {val(live.get('prefill_tok_s_mean'), ' tok/s')}   read {num(read)}/{num(total)}"
                    f"   {bar(read / total * 100 if read and total else None)}")
    else:
        rows.append("idle      nothing running")
    if live.get("parallel"):
        rows.append(f"parallel  {live.get('running', 0)}/{live.get('parallel')} running"
                    f"   waiting {live.get('waiting', 0)}   admitting {live.get('admitting', 0)}"
                    f"   outside slots {live.get('outside_slots', 0)}"
                    + ("  <- the solo path" if live.get("outside_slots") else ""))
    else:
        rows.append('parallel  off   one request at a time ("parallel": N in the config opens N batch slots,'
                    " and the slot panel with it)")
    return rows


def slots_panel(m: dict) -> list[str]:
    """Every batch slot: its zone, the context it fits, what it runs and, when idle, the tokens it holds for the
    conversation that will come back to it.  A locked slot holds a long conversation used inside the grace, so
    eviction never takes it; an idle slot past the grace holds a dead conversation and can be taken.  `cap` is the
    read capacity pick_slot routes on: a prompt wider than a slot's cap never lands in it, so a request that reads
    its whole prompt again is a request that found no slot wide enough for it."""
    live = m.get("live", {})
    slots = live.get("slots") or []
    if not slots:
        return []
    zones = [s.get("zone") for s in slots if s.get("zone")]
    head = f"slots     {len(slots)}"
    if zones:
        protected = sum(1 for z in zones if z == "protected")
        head += f"  ({protected} protected / {len(zones) - protected} guest)"
    rows = [head]
    if zones:
        rows.append(f"{'':<10} {'zone':<10} {'cap':>9}  {'state':<11} {'tokens':>17}"
                    f"  {'tok/s':>8}  {'idle':>7}  flags")
    for s in slots:
        slot_id = f"  slot {s.get('slot')}"
        if s.get("state") == "idle":
            tokens = f"holds {num(s.get('held_tokens'))}"
        elif s.get("state") == "reading":
            tokens = f"reading {num(s.get('prompt_tokens'))}"
        else:
            tokens = f"gen {num(s.get('generated'))}/{num(s.get('prompt_tokens'))}"
        clock = dur_text(s.get("idle_s")) if zones else val(s.get("elapsed_s"), "s")
        flags = ""
        if zones:
            flags = ",".join((["LOCKED"] if s.get("locked") else [])
                             + (["cache"] if s.get("state") == "idle" and s.get("held_tokens")
                                else ["free"] if s.get("state") == "idle" else []))
        if zones:
            rows.append(f"{slot_id:<10} {s.get('zone')!s:<10} {num(s.get('cap')):>9}"
                        f"  {s.get('state')!s:<11} {tokens:>17}  {val(s.get('tok_s')):>8}"
                        f"  {clock:>7}  {flags}")
        else:
            rows.append(f"{slot_id:<10} {s.get('state')!s:<11} {tokens:>17}  {val(s.get('tok_s')):>8}"
                        f"  {clock:>7}")
    return rows


def hardware_panel(m: dict) -> list[str]:
    """Every card's load, VRAM, temperature, power and PCIe link, then CPU, RAM and disk.  NVML's PCIe generation is
    the card's current link, and it drops to gen 1 while the card is idle to save power, so the figure to read is
    the throughput next to it, not the generation."""
    hw, static = m.get("hardware", {}), m.get("hardware_static", {})
    rows = []
    gpus = hw.get("gpus")
    if gpus:
        for g in gpus:
            used, total = g.get("mem_used"), g.get("mem_total")
            rows.append(f"GPU{g.get('index')}    {bar(g.get('util'))} {p100(g.get('util')):>5}"
                        f"  VRAM {gib(used)}/{gib(total)} GiB"
                        f"   {bar(used / total * 100 if used and total else None, 8)}"
                        f"  {val(g.get('temp'), 'C'):>4}  {rate(g.get('power'), 'W'):>6}")
    else:
        used, total = hw.get("gpu_mem_used"), hw.get("gpu_mem_total")
        rows.append(f"GPU      {bar(hw.get('gpu_util'))} {p100(hw.get('gpu_util')):>5}"
                    f"  VRAM {gib(used)}/{gib(total)} GiB"
                    f"   {val(hw.get('gpu_temp'), 'C')}  {rate(hw.get('gpu_power'), 'W')}")
    gen = hw.get("pcie_gen")
    link = f"gen {val(gen)} x{val(hw.get('pcie_width'))}"
    if gen is not None and hw.get("pcie_gen_max") and gen < hw["pcie_gen_max"]:
        link += f" (the card's max is gen {hw['pcie_gen_max']}: NVML reads the idle link, so trust the MB/s)"
    rows.append(f"PCIe     {link}   rx {rate(hw.get('gpu_pcie_rx_mb'), ' MB/s')}"
                f"   tx {rate(hw.get('gpu_pcie_tx_mb'), ' MB/s')}")
    cpu, ram_used, ram_total = hw.get("cpu"), hw.get("ram_used"), hw.get("ram_total")
    rows.append(f"CPU      {bar(cpu)} {p100(cpu):>5}   RAM {gib(ram_used)}/{gib(ram_total)} GiB"
                f"   {bar(ram_used / ram_total * 100 if ram_used and ram_total else None, 8)}"
                f"  free {gib((ram_total or 0) - (ram_used or 0)) if ram_total and ram_used is not None else '-'} GiB"
                f"   disk r{rate(hw.get('disk_read_mb'))} w{rate(hw.get('disk_write_mb'))} MB/s")
    if static.get("gpu_name") or static.get("cpu_name"):
        host = static.get("cpu_name") or "?"
        rows.append(f"host     {host}   {val(static.get('cores'))} cores / {val(static.get('threads'))} threads"
                    f"   {static.get('gpu_count') or len(gpus or [1])} GPU: {static.get('gpu_name') or '-'}")
    if static.get("gpu_note"):
        rows.append(f"note     {static['gpu_note']}")
    return rows


def trend_panel(m: dict, width: int) -> list[str]:
    """The last minute of four series as sparklines: the GPU's load, its PCIe traffic, RAM, the decode rate.
    The GPU series and RAM live in the hardware sample, the rates in the server's own live status."""
    hist = m.get("history") or {}
    current_of = {"hardware": m.get("hardware", {}), "live": m.get("live", {})}
    rows = [f"trend     the last {HISTORY_SECONDS}s, one sample a second"]
    for label, key, unit, source in TREND_ROWS:
        series = hist.get(key) or []
        if not series:
            continue
        current = current_of[source].get(key)
        shown = gib(current) if unit == " GiB" else (rate(current) if unit == " MB/s" else val(current))
        rows.append(f"          {label:<12} {spark(series, width)}  {shown}{unit}")
    return rows


def cache_panel(m: dict) -> list[str]:
    """The conversation cache: its budget, what is parked, the parks/restores/evictions and the prompt reuse.
    The reuse counts are the server's totals, so they report the prompt read again with the cache OFF too - that
    is the number a slot's own KV is spending, which is what a partition and a lock are there to bring down."""
    cc = m.get("conversation_cache")
    if not cc:
        return []
    rows = []
    if cc.get("enabled"):
        rows.append(f"conv cache  on   budget {gib_from_mib(cc.get('budget_mib'))} GiB"
                    f"   slots {val(cc.get('slots'))}   parked {num(cc.get('parked'))} ({gib(cc.get('bytes'))} GiB)"
                    f"   {num(cc.get('parks'))} parks / {num(cc.get('restores'))} restores"
                    f" / {num(cc.get('evictions'))} evictions")
        if cc.get("last_event"):
            rows[-1] += f"   last: {cc['last_event']} {num(cc.get('last_tokens'))} tok"
    else:
        rows.append("conv cache  OFF   a conversation keeps only its slot's KV (a parked turn: 0)")
    rows.append(f"reuse       {num(cc.get('reused_tokens'))}/{num(cc.get('prompt_tokens'))} prompt tok"
                f"  ({share(cc.get('reused_tokens'), cc.get('prompt_tokens'))} given back)"
                f"   last {num(cc.get('last_reused'))}/{num(cc.get('last_prompt'))}")
    return rows


def totals_panel(m: dict) -> list[str]:
    """The running totals: requests, the tokens read again, the output, the MTP drafts accepted and the rates since
    the server started (a mean over everything it has done, so they move slowly and follow the cache, not one turn)."""
    t = m.get("totals") or {}
    prompt = t.get("prompt_tokens")
    again = (prompt or 0) - (t.get("reused") or 0)
    return [(f"totals    {num(t.get('requests'))} requests   prompt {num(prompt)} tok"
             f"   read again {num(again)} ({share(again, prompt)})   output {num(t.get('output_tokens'))} tok"),
            (f"          MTP accepted {num(t.get('drafts_accepted'))}/{num(t.get('drafts_offered'))}"
             f" ({share(t.get('drafts_accepted'), t.get('drafts_offered'))})"
             f"   since start: prefill {tok_s(prompt, t.get('prompt_ms'))} tok/s"
             f"   decode {tok_s(t.get('output_tokens'), t.get('decode_ms'))} tok/s")]


def expert_row(r: dict) -> str | None:
    """One request's expert-cache row: its hit rate, the PCIe share and where its experts came from."""
    if r.get("hit_rate") is None and r.get("pcie_share") is None and r.get("file_mb") is None:
        return None
    return (f"           experts  hit {pct(r.get('hit_rate'))}   pcie {pct(r.get('pcie_share'))}"
            f"   RAM blobs {num(r.get('ram_blobs'))}   file {num(r.get('file_blobs'))} blobs"
            f" / {val(r.get('file_mb'))} MB")


def history_table(m: dict, rows: int) -> list[str]:
    """The finished requests as a table. `read` is what the engine read again: a full cache reuse is 0, and a read
    as wide as the prompt is a turn that lost its checkpoint - the column to watch when testing a partition."""
    shown = (m.get("requests") or [])[:rows]
    if not shown:
        return []
    header = (f"           {'time':<9} {'finish':<8} {'prompt':>8} {'read':>8} {'reused':>8} {'out':>6}"
              f" {'decode t/s':>10} {'hit':>7} {'drafts':>11} {'dur':>6}")
    out = ["requests    the last " + str(len(shown)) + " (read = tokens read again, 0 = a full reuse)", header]
    for r in shown:
        stamp = time.strftime("%H:%M:%S", time.localtime(r["time"])) if r.get("time") else "-"
        drafts = (f"{num(r.get('drafts_accepted'))}/{num(r.get('drafts_offered'))}"
                  if r.get("drafts_offered") else "-")
        out.append(f"           {stamp:<9} {r.get('finish')!s:<8} {num(r.get('prompt_tokens')):>8}"
                   f" {num(r.get('prompt_read')):>8} {num(r.get('reused')):>8} {num(r.get('output_tokens')):>6}"
                   f" {val(r.get('decode_tok_s')):>10} {pct(r.get('hit_rate')):>7} {drafts:>11}"
                   f" {val(r.get('duration_s'), 's'):>6}")
        extra = expert_row(r)
        if extra:
            out.append(extra)
    return out


def render(m: dict, opts) -> str:
    """The /metrics dict as one top screen: the engine, what runs now, the hardware, trends, the cache, totals."""
    screen = title_panel(m) + engine_panel(m) + live_panel(m)
    if not opts.compact:
        screen += slots_panel(m)
    screen += hardware_panel(m)
    if opts.spark:
        screen += trend_panel(m, opts.spark)
    screen += cache_panel(m) + totals_panel(m)
    if opts.history:
        screen += history_table(m, opts.history)
    return "\n".join(line.rstrip() for line in screen)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="a top-like live view of a running Strata server (it reads the server's own GET /metrics)",
        epilog="slots: zone = the config's protected_slots partition; cap = the context that slot fits;"
               " LOCKED = a long conversation in its grace, eviction never takes it; idle = seconds since used."
               " read (in the table) = the prompt tokens read again (0 = a full cache reuse)."
               " A PCIe generation below the card's max is NVML reading the idle link: read the MB/s.")
    ap.add_argument("url", nargs="?", default="http://127.0.0.1:8080")
    ap.add_argument("--api", default=os.environ.get("STRATA_API_KEY", ""), help="the server's api_key (Bearer)")
    ap.add_argument("--interval", type=float, default=1.0, help="refresh seconds (default 1)")
    ap.add_argument("--history", type=int, default=6, help="rows of finished requests (0 turns the table off)")
    ap.add_argument("--spark", type=int, default=44, help="sparkline width (0 turns the trends off)")
    ap.add_argument("--compact", action="store_true", help="the essentials only: no slots, no trends, no table")
    ap.add_argument("--once", action="store_true", help="print one screen and exit (a log, a bug report)")
    opts = ap.parse_args()
    if opts.compact:
        opts.history, opts.spark = 0, 0
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
