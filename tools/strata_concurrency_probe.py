#!/usr/bin/env python3
"""Strata 并发实测：在"持久会话逐轮增长 + 串行交替到达"的现场负载模型下，验证 solo path 单检查点链是否被交替会话清场。

现场证据（streaming 日志）：长会话 X 连续两轮 reused 命中，被短会话 Y 插队一次，X 下一轮 reused=0 全量 prefill 9 秒。
根因假设：parallel 请求串行到达时走 solo path（GEN），复用靠单条 live-session 检查点链；两会话交替时对方 GEN 清链。

本脚本用因果对照钉死它：
  阶段 B · 交替清场对照：同一长会话，"连续组"连发不被插队（命中）vs"交替组"每轮之间夹一发短会话（清场）。
  阶段 C · 真并发对照：把交替组会话的下一轮同时发出 → 走 batch slot 路径，隔离应良好；与阶段 B 的串行路径对照，
            暴露"并发路径隔离良好"≠"现场串行路径不清场"。

用法：python3 strata_concurrency_probe.py --url http://127.0.0.1:8080 --log /home/canyun/Strata/strata-iq3_s.log
标准库实现；checkpoints 只存在于引擎日志，reused/prefill 走 GET /metrics。
"""

import argparse
import contextlib
import io
import json
import os
import random
import re
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PROMPT_LINE = re.compile(
    r"prompt (\d+) tokens = (\d+) reused \+ (\d+)(?: of \d+)? read in ([\d.]+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in ([\d.]+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+), (\d+) checkpoints"
)

FACT_KEYS = (
    "engine", "context", "kv", "kv_resident", "expert_slots", "expert_cache_mib",
    "expert_slots_primary", "expert_cache_primary_mib", "arena_mib", "spec", "mtp_max",
    "pool_workers", "pcie_frac", "vram_free_mib", "conversation_cache_mib",
    "conversation_cache_slots", "slot_cache", "batch_slots", "batch_groups",
)

# 每 token 每会话的 KV 字节（全模型含 draft 层，int8 实测 13728 B/token，200K ≈ 2.75 GB/会话；
# layer-split 时每卡 = 该值 / 卡数）。现场 4 slot 每卡 5.49 GB 由 4×(200000×13728/1e9/2) 复现。
KV_BYTES_PER_TOKEN = {"k8v4": 10608, "q4_0": 7488, "int8": 13728}

VOCAB = (
    "router", "expert", "activation", "dense", "layer", "attention", "kernel", "gradient",
    "token", "sequence", "cache", "memory", "bandwidth", "latency", "throughput", "scheduler",
    "dispatch", "quantized", "tensor", "gather", "scatter", "prefill", "decode", "verify",
    "draft", "accepted", "rejected", "window", "chunk", "resident", "streaming", "pinned",
    "checkpoint", "snapshot", "eviction", "branch", "history", "sibling", "query", "prefix",
    "suffix", "system", "vocabulary", "hint", "budget", "quota", "admission", "slot", "group",
    "pipeline", "barrier", "mailbox", "ring", "buffer", "stream", "event", "semaphore", "mutex",
    "atomic", "volatile", "unrolled", "fused", "vector", "register", "spill", "reload",
    "policy", "thermal", "throttle", "power", "limit", "clock", "domain", "cross", "coalesce",
    "merge", "split", "partition", "shard", "replica", "quorum", "leader", "follower",
    "handshake", "probe", "heartbeat", "deadline", "cancel", "resume", "suspend", "restore",
    "migrate", "audit", "counter",
)

SYSTEM_PROMPT = "You are a precise assistant. Answer in one short sentence."


def owner_drift(samples):
    """采样序列（每个含 owners: ["slot:s:conv-0", ...]）→ 跨采样换过主人的 slot 集合。
    主人跨轮稳定 = 所有权的核心证据：会话的链一直在自己的槽里，下一轮只读新增。
    空主人（"-"）是窗口过期的瞬间（链已死、槽开放），不算漂移。"""
    seen, drift = {}, set()
    for sample in samples:
        for entry in sample.get("owners") or []:
            slot, _, owner = entry.partition(":")
            owner = owner.removeprefix("s:").removeprefix("p:")
            if owner in ("", "-"):
                continue
            if seen.get(slot) not in (None, owner):
                drift.add(slot)
            seen[slot] = owner
    return drift


def classify_line(line):
    """一条 SSE 行的角色："keepalive"（: 注释，排队期间服务端每 10s 发的 : keep-alive）、"data"、"skip"。
    所有权场景下 keepalive 是超出会话排队的直接证据：它保护会话在等一个窗口时不被客户端看门狗掐断。"""
    if line.startswith(":"):
        return "keepalive"
    if line.startswith("data:"):
        return "data"
    return "skip"


def human(value):
    return "-" if value is None else (f"{value:,.1f}" if isinstance(value, float) else f"{value:,}")


def spread(values, suffix=""):
    if not values:
        return "-"
    return " / ".join(f"{human(value)}{suffix}"
                      for value in (min(values), statistics.median(values), max(values)))


def build_prompt(target_words, rng, nonce=""):
    """确定性伪词序列：seed 相同即同文本，让前缀检查点跨轮稳定命中；nonce 标记会话，供离线 mock 归属。"""
    words = rng.sample(VOCAB, len(VOCAB))
    out = [f"[sess {nonce}]" if nonce else "#init"]
    while len(out) < target_words:
        out.extend(rng.choice(words) for _ in range(16))
        out.append(f"#{rng.randint(0, 999999)}")
    return " ".join(out[:target_words])


def seed_messages(prompt_text):
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt_text}]


def one_shot_messages(nonce, words, rng):
    """一次性请求：唯一 nonce、只发一轮、发完即走，自身 reused=0 正常（它不追求复用）。"""
    return seed_messages(build_prompt(words, rng, nonce=nonce))


def grow_messages(messages, reply_text, followup_words, rng):
    """会话逐轮增长：追加上一轮回复 + 新的追问，复现 agent 每轮拼接历史重发。"""
    tail = " ".join(rng.choice(VOCAB) for _ in range(followup_words))
    out = list(messages)
    out.append({"role": "assistant", "content": reply_text or "Noted."})
    out.append({"role": "user", "content": f"{tail} Reply with one word."})
    return out


def interleave_plan(long_names, short_names, rounds):
    """交替计划：每轮先用短会话插队（清链），再发长会话（观察是否被清场）。返回 [(会话, 角色)]。"""
    plan = []
    for _ in range(rounds - 1):
        plan += [(name, "filler") for name in short_names]
        plan += [(name, "probe") for name in long_names]
    return plan


def sequential_plan(long_names, rounds):
    """连续计划：只发长会话，组内不插队。作为交替组的对照组，控制变量 = 是否有其它会话插入。"""
    return [(name, "probe") for _ in range(rounds - 1) for name in long_names]


def assign_sessions(records, expected_tokens):
    """把并发新增的 per-request 记录按 prompt 长度贪心配对到各会话，区间重叠也不串号。"""
    pairs = sorted(
        (abs(record["prompt_total"] - tokens), name, index)
        for index, record in enumerate(records)
        for name, tokens in expected_tokens.items()
        if record.get("prompt_total") is not None
    )
    claimed, assigned = set(), {}
    for _gap, name, index in pairs:
        if name in assigned or index in claimed:
            continue
        assigned[name] = records[index]
        claimed.add(index)
    return assigned


class EngineProbe:
    """Strata server 的 HTTP 观测面：/metrics 的 engine 块即启动 INFO 行的解析结果。"""

    def __init__(self, base_url, api_key=None, timeout=20.0):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def get_json(self, path):
        request = urllib.request.Request(self.base_url + path, method="GET")
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def engine_facts(self):
        return self.get_json("/metrics").get("engine") or {}

    def cache_view(self):
        return self.get_json("/metrics").get("conversation_cache") or {}

    def history_snapshot(self):
        return self.get_json("/metrics?requests=all")

    def history_count(self):
        try:
            return int(self.history_snapshot().get("requests_kept") or 0)
        except Exception:  # noqa: BLE001
            return 0

    def slot_snapshot(self):
        try:
            live = self.get_json("/metrics").get("live") or {}
        except Exception:  # noqa: BLE001
            live = {}
        slots = live.get("slots")
        if slots is None:
            try:
                slots = self.get_json("/slots")
            except Exception:  # noqa: BLE001
                slots = []
        active = sum(1 for slot in slots if slot.get("state", "processing") != "idle")
        return active, slots, live

    def watch(self, stop_event, samples, interval):
        while not stop_event.is_set():
            try:
                active, slots, live = self.slot_snapshot()
                samples.append({
                    "active": active,
                    "running": live.get("running"),
                    "outside": live.get("outside_slots"),
                    "waiting": live.get("waiting"),
                    "holders": live.get("slot_holders"),
                    "window": live.get("slot_protect_seconds"),
                    "states": [f"{slot.get('slot')}:{slot.get('state', 'processing')}" for slot in slots],
                    "zones": [f"{slot.get('slot')}:{slot.get('zone', '-')}" for slot in slots],
                    "owners": [f"{slot.get('slot')}:{slot.get('owner', '-')}" for slot in slots],
                })
            except Exception:  # noqa: BLE001
                samples.append({"active": -1, "running": None, "outside": None,
                                "waiting": None, "holders": None, "window": None,
                                "states": [], "zones": [], "owners": []})
            time.sleep(interval)


class LogTailer:
    """增量解析引擎日志的 prompt 行：checkpoints 数只存在于这里（/metrics 没有该字段）。"""

    def __init__(self, log_path):
        self.log_path = log_path
        self.lock = threading.Lock()
        self.offset = os.path.getsize(log_path) if log_path and os.path.exists(log_path) else 0
        self.records = []
        self.claimed = set()

    def drain(self):
        if not self.log_path or not os.path.exists(self.log_path):
            return
        with self.lock:
            if os.path.getsize(self.log_path) < self.offset:
                self.offset = 0
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        for line in chunk.splitlines():
            match = PROMPT_LINE.search(line)
            if not match:
                continue
            fields = match.groups()
            self.records.append({
                "prompt_tokens": int(fields[0]),
                "reused": int(fields[1]),
                "read": int(fields[2]),
                "read_ms": float(fields[3]),
                "read_tok_s": float(fields[4]),
                "generated": int(fields[5]),
                "decode_ms": float(fields[6]),
                "decode_tok_s": float(fields[7]),
                "checkpoints": int(fields[10]),
            })

    def claim_new(self, count, deadline_s=25.0):
        def take():
            with self.lock:
                pending = [i for i in range(len(self.records)) if i not in self.claimed]
                if len(pending) >= count:
                    out = []
                    for index in pending[-count:]:
                        self.claimed.add(index)
                        out.append(dict(self.records[index]))
                    return out
                return None
        got = take()
        if got is not None:
            return got
        deadline = time.monotonic() + deadline_s
        while time.monotonic() < deadline:
            time.sleep(0.2)
            self.drain()
            got = take()
            if got is not None:
                return got
        return []

    def checkpoint_counts(self):
        self.drain()
        with self.lock:
            return [record["checkpoints"] for record in self.records]


class Turn:
    """一次请求：客户端测 TTFT/墙钟，引擎侧记录 reused/prefill/decode/checkpoints。"""

    def __init__(self, name, role="probe", session=None):
        self.name = name
        self.role = role          # probe=被观察的长会话轮次；filler=用来插队清链的短会话轮次
        self.session = session    # 请求自声明的 "strata_session"（所有权场景下会话保住槽的身份）
        self.keepalive = None     # 排队期间收到的 : keep-alive 数（超出受保护槽的会话的证据）
        self.ttft_s = None
        self.wall_s = None
        self.reply_words = 0
        self.http_error = None
        self.engine_record = {}
        self.log_record = {}

    def bind(self, engine_record=None, log_record=None):
        self.engine_record = engine_record or {}
        self.log_record = log_record or {}

    def _pick(self, key):
        if self.engine_record.get(key) is not None:
            return self.engine_record.get(key)
        return self.log_record.get(key)

    @property
    def prompt_tokens(self):
        return self._pick("prompt_total")

    @property
    def reused(self):
        return self._pick("reused")

    @property
    def prefix_hit_rate(self):
        """检查点复用率：/metrics 的 hit_rate 是专家缓存命中率，语义不同，必须自算。"""
        prompt, reused = self.prompt_tokens, self.reused
        if prompt and reused is not None:
            return reused / prompt
        return None

    @property
    def expert_hit_rate(self):
        rate = self.engine_record.get("hit_rate")
        return float(rate) if isinstance(rate, (int, float)) else None

    @property
    def prefill_tok_s(self):
        read_ms = self.log_record.get("read_ms")
        if read_ms:
            return self.log_record.get("read_tok_s")
        prompt_ms, prompt = self._pick("prompt_ms"), self.prompt_tokens
        reused = self.reused or 0
        if prompt_ms and prompt:
            return (prompt - reused) / (prompt_ms / 1000.0)
        return None

    @property
    def read_ms(self):
        return self.log_record.get("read_ms")

    @property
    def decode_tok_s(self):
        return self._pick("decode_tok_s")

    @property
    def checkpoints(self):
        return self.log_record.get("checkpoints")

    def summary(self):
        hit = "-" if self.prefix_hit_rate is None else f"{self.prefix_hit_rate * 100:.0f}%"
        ttft = "-" if self.ttft_s is None else f"{self.ttft_s:.2f}s"
        wall = "-" if self.wall_s is None else f"{self.wall_s:.1f}s"
        read = "-" if self.read_ms is None else f"{self.read_ms / 1000.0:.1f}s"
        return (f"  {self.name:<6} prompt={human(self.prompt_tokens):>9} reused={human(self.reused):>9} "
                f"hit={hit:>4} prefill_read={read:>6} prefill={human(self.prefill_tok_s):>9} tok/s "
                f"decode={human(self.decode_tok_s):>7} tok/s ckpt={self.checkpoints!s:>4} "
                f"ttft={ttft:>6} wall={wall:>6}")


class StreamClient:
    """OpenAI 兼容流式客户端：返回回复文本，并记录首个 content 增量到达时间（TTFT）。"""

    def __init__(self, base_url, model, api_key=None, extra_body=None, timeout=1800.0):
        self.url = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.api_key = api_key
        self.extra_body = extra_body or {}
        self.timeout = timeout

    def stream(self, messages, max_new_tokens, temperature, session=None):
        payload = {"model": self.model, "messages": messages, "max_tokens": max_new_tokens,
                   "temperature": temperature, "stream": True}
        payload.update(self.extra_body)
        # 所有权场景的钥匙：请求自声明会话身份（"strata_session"），服务端据此把槽判给这个会话并在窗口内独占；
        # 客户端会改写 system prompt（注入时间戳），前缀哈希会断，自声明是保住槽的唯一方式。extra_body 可覆盖。
        if session and "strata_session" not in payload:
            payload["strata_session"] = session
        request = urllib.request.Request(
            self.url, data=json.dumps(payload).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"})
        if self.api_key:
            request.add_header("Authorization", f"Bearer {self.api_key}")
        pieces = []
        started = time.monotonic()
        first_token_s = None
        keepalive = 0
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for raw in response:
                line = raw.decode("utf-8", "replace").strip()
                kind = classify_line(line)
                if kind == "keepalive":
                    keepalive += 1        # 排队保活：超出受保护槽的会话在等一个窗口
                    continue
                if kind != "data":
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
                    if piece:
                        if first_token_s is None:
                            first_token_s = time.monotonic() - started
                        pieces.append(piece)
        return "".join(pieces), first_token_s, time.monotonic() - started, keepalive


class MetricsHistory:
    """GET /metrics 的 requests 数组：引擎 DONE 行的结构化记录，逐轮累积。"""

    def __init__(self, probe):
        self.probe = probe
        self.baseline = 0

    def mark(self):
        self.baseline = self.probe.history_count()
        return self.baseline

    def collect(self, expected, deadline_s=25.0):
        deadline = time.monotonic() + deadline_s
        records = []
        while True:
            try:
                payload = self.probe.history_snapshot()
                kept = int(payload.get("requests_kept") or 0)
                records = list(payload.get("requests") or [])[:max(0, kept - self.baseline)]
            except Exception:  # noqa: BLE001
                records = []
            if len(records) >= expected or time.monotonic() >= deadline:
                return records
            time.sleep(0.4)


class Session:
    """一个持久会话：messages 逐轮增长，复现 agent 每轮拼接历史重发。nonce 标记会话身份。"""

    def __init__(self, name, prompt_text):
        self.name = name
        self.messages = seed_messages(prompt_text)
        self.last_reply = ""
        self.turns = []                 # list[Turn]，index 0 是首轮建立

    def next_messages(self, followup_words, rng):
        return grow_messages(self.messages, self.last_reply, followup_words, rng)


class Probe:
    def __init__(self, args):
        self.args = args
        self.probe = EngineProbe(args.url, args.api_key)
        self.tailer = LogTailer(args.log)
        self.history = MetricsHistory(self.probe)
        self.client = StreamClient(args.url, args.model, args.api_key, args.extra_body)
        self.rng = random.Random(args.seed)
        self.tokens_per_word = 1.4
        self.s = {}
        self.c_turns = []

    def rule(self, title):
        print()
        print("=" * 122)
        print(title)
        print("=" * 122)

    def report_facts(self):
        self.rule("阶段 A · 引擎自报事实（GET /metrics 的 engine 块 = 启动 INFO 行；日志文件里没有这行）")
        try:
            facts = self.probe.engine_facts()
            cache = self.probe.cache_view()
        except Exception as error:  # noqa: BLE001
            print(f"  读取 /metrics 失败: {error}")
            print("  → 确认端口/前缀：Strata 裸端口无需 /v1；经 new-api 网关需带渠道前缀")
            return {}
        for key in FACT_KEYS:
            value = facts.get(key, "（引擎未报）")
            print(f"  {key:<30} {value}")
        print(f"  {'conversation_cache':<30} enabled={cache.get('enabled')} "
              f"parked={cache.get('parked')} budget_mib={cache.get('budget_mib')}")
        print()
        asked, batch = self.args.parallel_asked, facts.get("batch_slots")
        print(f"  配置 parallel={asked} → 引擎实得 batch_slots={batch}")
        if asked is not None and batch is not None:
            if int(batch) < int(asked):
                print("  x 引擎削减了槽数（装不下）：实际并发低于配置")
            elif int(batch) == int(asked):
                print("  + 引擎按配置给满槽数")
        if facts.get("slot_cache") == 0:
            print("  x slot_cache=0：检查点缓存关闭，每个请求都从零 prefill")
        if not cache.get("enabled"):
            print("  ! parking 关闭：交替会话断链后无 host RAM 快照兜底，只能全量 prefill")
        context = facts.get("context")
        kv_bytes = KV_BYTES_PER_TOKEN.get(facts.get("kv"), 13728)
        if context:
            session_gb = context * kv_bytes / 1e9
            gpus = max(1, self.args.gpus)
            slots = int(batch) if batch else max(1, self.args.parallel_asked or 1)
            print(f"  · 单会话 KV ≈ {session_gb:.2f} GB（context={human(context)}, --kv {facts.get('kv')}）；"
                  f"每卡 ≈ {session_gb / gpus:.2f} GB")
            print(f"  · {slots} slot 全模型 KV ≈ {session_gb * slots:.1f} GB；每卡 ≈ {session_gb * slots / gpus:.1f} GB")
        return facts

    def run_one(self, name, role, messages, max_new_tokens, session=None):
        turn = Turn(name, role, session=session)
        try:
            reply, first_token_s, wall_s, keepalive = self.client.stream(
                messages, max_new_tokens, self.args.temperature, session=session)
            turn.ttft_s, turn.wall_s, turn.keepalive = first_token_s, wall_s, keepalive
            turn.reply_words = len(reply.split())
            return turn, reply
        except Exception as error:  # noqa: BLE001
            turn.http_error = str(error)
            return turn, ""

    def _send_and_bind(self, session, role, messages, max_new_tokens):
        """发一轮：串行到达时每个请求独占控制线（solo path），逐条认领引擎记录与日志记录。"""
        self.history.mark()
        turn, reply = self.run_one(session.name, role, messages, max_new_tokens, session=session.name)
        engine_records = self.history.collect(1)
        log_records = self.tailer.claim_new(1)
        turn.bind(engine_records[-1] if engine_records else None,
                  log_records[-1] if log_records else None)
        session.messages = messages
        session.last_reply = reply
        session.turns.append(turn)
        print(turn.summary())
        if turn.http_error:
            print(f"    HTTP 错误: {turn.http_error}")
        return turn

    def establish(self, session):
        """首轮建立：reused=0 是冷加载（正常）。建立后用首条引擎记录校准词/token 系数。"""
        turn = self._send_and_bind(session, "probe", list(session.messages), self.args.build_new_tokens)
        if turn.prompt_tokens:
            words = len(session.messages[-1]["content"].split())
            self.tokens_per_word = turn.prompt_tokens / max(1.0, words)
        return turn

    def advance(self, session, role):
        messages = session.next_messages(self.args.followup_words, self.rng)
        if self.args.arrive_gap > 0:
            time.sleep(self.args.arrive_gap)
        return self._send_and_bind(session, role, messages, self.args.decode_new_tokens)

    def make_session(self, name, words, seed_offset):
        prompt = build_prompt(words, random.Random(self.args.seed + seed_offset), nonce=name)
        return Session(name, prompt)

    def interleave_experiment(self):
        """阶段 B · 交替清场因果对照：连续组（不被插队）vs 交替组（每轮夹一发短会话）。

        两组各自建立自己需要的会话，避免建立阶段互相清链——现场 B 链正是"建立后独占连发"才连续命中，
        连续组必须自包含才能复现"链被保住"。
        """
        args = self.args
        self.rule(f"阶段 B · 交替清场因果对照（{args.rounds} 轮；控制变量 = 长会话每轮之间是否被短会话插队）")
        print("        连续组：建立 Xc 后独占连发，链不被清 → 每轮应 reused>0（命中）")
        print("        交替组：建立 Xa/Ya，Xa 每轮之前夹一发 Ya → 若 solo 单检查点链被 Ya 清场，Xa 每轮 reused=0 全量 prefill")

        xc = self.make_session("Xc", args.long_words, 11)
        self.s["Xc"] = xc
        print(f"  连续组 · 建立 Xc（≈{args.long_words}词；首轮 reused=0 为冷加载，正常）后连发 {args.rounds - 1} 轮:")
        self.establish(xc)
        seq_turns = [self.advance(xc, "probe") for _ in range(args.rounds - 1)]

        xa = self.make_session("Xa", args.long_words, 21)
        ya = self.make_session("Ya", args.short_words, 22)
        self.s["Xa"], self.s["Ya"] = xa, ya
        print(f"\n  交替组 · 建立 Xa（≈{args.long_words}词）/ Ya（≈{args.short_words}词）后，Xa 每轮夹一发 Ya"
              f"（{args.rounds - 1} 个插队周期，到达间隔 {args.arrive_gap}s）:")
        self.establish(xa)
        self.establish(ya)
        inter_turns = [self.advance(self.s[name], role)
                       for name, role in interleave_plan(["Xa"], ["Ya"], args.rounds)]
        print(f"  实测词/token 系数 {self.tokens_per_word:.3f}")
        return seq_turns, inter_turns

    def concurrent_round(self, label, names, max_new_tokens):
        """同一时刻发出 names 各会话的下一轮：alone 不成立 → 进各自 slot（并发路径）。按长度归属各轮记录。"""
        stop_event = threading.Event()
        samples = []
        watcher = threading.Thread(target=self.probe.watch,
                                   args=(stop_event, samples, self.args.sample_every), daemon=True)
        watcher.start()
        self.history.mark()
        started = time.monotonic()
        pending = {name: self.s[name].next_messages(self.args.followup_words, self.rng) for name in names}
        expected = {name: int(self.tokens_per_word * len(pending[name][-1]["content"].split())) for name in names}
        turns = {}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {pool.submit(self.run_one, name, "probe", pending[name], max_new_tokens, session=name): name
                       for name in names}
            for future, name in futures.items():
                turn, reply = future.result()
                turns[name] = turn
                self.s[name].messages = pending[name]
                self.s[name].last_reply = reply
                self.s[name].turns.append(turn)
        batch_wall = time.monotonic() - started
        stop_event.set()
        watcher.join(timeout=4.0)

        records = self.history.collect(len(names), deadline_s=12.0)
        assigned = assign_sessions(records, expected)
        log_records = self.tailer.claim_new(len(names), deadline_s=12.0)
        log_assigned = assign_sessions(
            [{"prompt_total": r["prompt_tokens"], **r} for r in log_records], expected)
        for name, turn in turns.items():
            turn.bind(assigned.get(name), log_assigned.get(name))

        for name in sorted(turns):
            print(turns[name].summary())
            if turns[name].http_error:
                print(f"    HTTP 错误: {turns[name].http_error}")
        peak = max((sample["active"] for sample in samples), default=0)
        reading = sum(1 for smp in samples for state in smp["states"] if ":reading" in state)
        decoding = sum(1 for smp in samples for state in smp["states"] if ":decoding" in state)
        print()
        print(f"  [{label}] 批次墙钟 {batch_wall:.1f}s；采样 {len(samples)} 次；活跃槽峰值 {peak}；"
              f"reading 观测 {reading} 次 / decoding 观测 {decoding} 次")
        for sample in samples[:16]:
            print(f"    active={sample['active']} running={sample.get('running')} "
                  f"outside={sample.get('outside')} waiting={sample.get('waiting')} "
                  f"{' '.join(sample['states'])}")
        return list(turns.values()), batch_wall, peak

    def concurrent_probe(self):
        """阶段 C · 真并发对照：同时发 Xa/Ya 下一轮 → 进各自 batch slot，隔离应良好。

        先 warm 一轮：Xa/Ya 此前一直 solo 跑、从没进过 slot，首次并发进空 slot 必然 reused=0（冷加载）。
        warm 把会话放进 slot 后，measure 轮才测得到"slot 隔离让每会话保住自己的链"。
        """
        self.rule("阶段 C · 真并发对照（同时发 Xa/Ya 下一轮 → batch slot 路径；与阶段 B 串行路径对照）")
        print("        同时到达时 alone 不成立，请求进独立 slot；此路径隔离良好 ≠ 现场串行路径不清场")
        print("        （warm 轮让会话进 slot，measure 轮测隔离；首轮冷加载 reused=0 不计入判定）")
        print("        [warm]")
        self.concurrent_round("C-warm", ["Xa", "Ya"], self.args.decode_new_tokens)
        print("        [measure]")
        return self.concurrent_round("C", ["Xa", "Ya"], self.args.decode_new_tokens)

    def send_one_shot(self, index):
        """发一发一次性请求并绑定引擎/日志记录；不入任何常驻会话，测它对常驻会话链与 cache 的影响。"""
        self.history.mark()
        turn, _reply = self.run_one(
            f"O{index}", "oneshot",
            one_shot_messages(f"O{index}", self.args.short_words, self.rng), self.args.build_new_tokens)
        engine_records = self.history.collect(1)
        log_records = self.tailer.claim_new(1)
        turn.bind(engine_records[-1] if engine_records else None,
                  log_records[-1] if log_records else None)
        print(turn.summary())
        if turn.http_error:
            print(f"    HTTP 错误: {turn.http_error}")
        return turn

    def one_shot_experiment(self):
        """阶段 D · 一次性请求是否占用/清掉常驻会话 cache。

        基线：常驻长会话 X 建立链后连发，reused 命中（链活着）。注入：每轮之前夹一发一次性请求
        （唯一 nonce、发完即走、自身 reused=0）紧接 X 下一轮——若 X reused=0，一次性请求抢占了 solo
        单检查点链。开 parking 时再看 cache 占用增量：一次性请求若带 checkpoint 会被 park，挤占常驻会话预算。
        """
        args = self.args
        self.rule(f"阶段 D · 一次性请求占用/清掉常驻会话 cache 测试（{args.rounds - 1} 次注入）")
        print("        基线：常驻长会话 X 建立链后连发，reused 应命中（链活着）")
        print("        注入：基线每轮之前发一发一次性请求 O_i（唯一 nonce，发完即走，自身 reused=0）→ 看 X 下一轮 reused")
        print("        reused=0 → 一次性请求抢占 solo 单检查点；开 parking 时再看 cache 占用增量（bytes/parked）")
        x = self.make_session("X", args.long_words, 41)
        self.s["X"] = x
        self.establish(x)
        print(f"  基线 · X 连发 {args.rounds - 1} 轮（无注入，链应命中）:")
        base = [self.advance(x, "probe") for _ in range(args.rounds - 1)]
        print(f"\n  注入 · 每轮前夹一发一次性请求（≈{args.short_words}词，唯一 nonce）紧接 X 下一轮:")
        injected = []
        for index in range(args.rounds - 1):
            before = self.probe.cache_view()
            one = self.send_one_shot(index)
            after = self.probe.cache_view()
            x_turn = self.advance(x, "probe")
            injected.append((one, x_turn, before, after))
            print(f"      cache 占用（注入 O{index} 后）parked={after.get('parked')} bytes={after.get('bytes')}")
        return base, injected

    def ownership_round(self, names, label, max_new_tokens, first=False):
        """阶段 E · 全部会话同时发下一轮（每发自带 "strata_session"）：采样槽 owner/zone/state、收集每会话
        排队期间的 keepalive。首轮用种子 messages（冷加载 reused=0 正常）；提示词长度按会话错开，
        让 assign_sessions 能按长度把引擎 DONE 记录归属到会话。"""
        stop_event = threading.Event()
        samples = []
        watcher = threading.Thread(target=self.probe.watch,
                                   args=(stop_event, samples, self.args.sample_every), daemon=True)
        watcher.start()
        self.history.mark()
        started = time.monotonic()
        if first:
            pending = {name: list(self.s[name].messages) for name in names}
        else:
            pending = {name: self.s[name].next_messages(self.args.followup_words, self.rng) for name in names}
        expected = {name: int(self.tokens_per_word * len(pending[name][-1]["content"].split())) for name in names}
        turns = {}
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {pool.submit(self.run_one, name, "probe", pending[name], max_new_tokens, session=name): name
                       for name in names}
            for future, name in futures.items():
                turn, reply = future.result()
                turns[name] = turn
                self.s[name].messages = pending[name]
                self.s[name].last_reply = reply
                self.s[name].turns.append(turn)
        wall = time.monotonic() - started
        stop_event.set()
        watcher.join(timeout=4.0)
        records = self.history.collect(len(names), deadline_s=12.0)
        assigned = assign_sessions(records, expected)
        for name, turn in turns.items():
            turn.bind(assigned.get(name))
        for name in sorted(turns):
            print(turns[name].summary() + f"  keepalive={turns[name].keepalive!s:>4}")
            if turns[name].http_error:
                print(f"    HTTP 错误: {turns[name].http_error}")
        peak = max((sample["active"] for sample in samples), default=0)
        holders = [sample.get("holders") for sample in samples if sample.get("holders") is not None]
        protected = sum(1 for entry in (samples[-1].get("zones") or []) if entry.endswith(":protected"))
        print(f"\n  [{label}] 墙钟 {wall:.1f}s；采样 {len(samples)} 次；活跃槽峰值 {peak}；受保护槽 {protected}；"
              f"holder 峰值 {human(max(holders, default=0))}")
        return turns, samples, wall, peak, protected

    def ownership_experiment(self):
        """阶段 E · slot-protect 场景自证：会话数（默认 6）刻意多于受保护槽（现场 3），每轮全并发。
        期望拓扑：活跃槽峰值 = 受保护数（超出者挤不进来）；受保护槽的 owner 跨轮不漂移（链在一个地方）；
        会话第 2..N 轮 reused>0（只读新增）；每轮落选者排队并收到 keepalive（队列活着，不被看门狗掐断）。"""
        args = self.args
        names = [f"P{i}" for i in range(args.sessions)]
        self.rule(f"阶段 E · 槽位所有权自证（{args.sessions} 会话全并发；现场拓扑 6 会话 > 3 受保护槽）")
        print("        期望：活跃槽峰值=受保护数；owner 跨轮稳定；每会话 reused 命中；落选者排队收 keepalive")
        for i, name in enumerate(names):
            words = args.ownership_words + 250 * i     # 长度指纹：DONE 记录按 prompt 长度归属会话
            self.s[name] = self.make_session(name, words, 51 + i)
        self.ownership_round(names, "E-建立", args.build_new_tokens, first=True)
        return [self.ownership_round(names, f"E-轮{r}", args.decode_new_tokens)
                for r in range(1, args.rounds)]

    def _verdict_ownership(self, rounds):
        """阶段 E 判定：四个读数各自自证 slot-protect 拓扑的一环，缺一个就对应一个配置没生效。
        建立轮是冷加载（reused=0 正常），不列入命中判定。"""
        self.rule("判定 · 阶段 E（slot-protect 自证）")
        if not rounds:
            return
        samples = [sample for _t, sample_list, _w, _p, _c in rounds for sample in sample_list]
        later = rounds
        windows = {sample.get("window") for sample in samples if sample.get("window")}
        if not windows:
            print("  x /metrics 没有 slot_protect_seconds：所有权未生效——检查配置三字段 "
                  "slot_protect_seconds / protected_slots / solo_path: false（setup --slot-protect-seconds 一次写全）")
            return
        print(f"  + 引擎开启所有权窗口 window={max(windows):.0f}s（超出会话排队要等满它，会话活跃则永不让槽）")
        protected = max((c for _t, _s, _w, _p, c in later if c), default=0)
        peaks = [p for _t, _s, _w, p, _c in later]
        if protected:
            capped = max(peaks, default=0) <= protected
            print(f"  · 活跃槽峰值 {max(peaks, default=0)} / 受保护槽 {protected} → "
                  + ("超出会话挤不进来，并发被所有权封住" if capped
                     else "活跃槽超过受保护数：保护没封住（查 solo_path 是否 false——solo 的 GEN 不占槽）"))
        drift = owner_drift(samples)
        print(f"  · 跨采样换过主人的槽：{sorted(drift) if drift else '无'} → "
              + ("owner 稳定，每会话的链在一个地方"
                 if not drift else "owner 漂移：窗口过期太快（会话间隔>window）或 strata_session 没带上（哈希断）"))
        all_turns = [turn for turns, _s, _w, _p, _c in later for turn in turns.values()]
        hits = [turn for turn in all_turns if turn.reused]
        print(f"  · reused>0（命中自己链、只读新增）：{len(hits)}/{len(all_turns)} 轮")
        queued = [turn for turn in all_turns if (turn.keepalive or 0) > 0]
        if queued:
            beats = sum(turn.keepalive for turn in queued)
            q_ttft = [turn.ttft_s for turn in queued if turn.ttft_s is not None]
            n_ttft = [turn.ttft_s for turn in all_turns if not turn.keepalive and turn.ttft_s is not None]
            print(f"  · 排队轮 {len(queued)}（keepalive 共 {beats} 发，约每 {10.0:.0f}s 一发 = #1619 保活）→ "
                  "超出会话在等一个窗口，队列活着、没被客户端掐断")
            if q_ttft and n_ttft:
                print(f"  · TTFT 对比：当轮者 {spread(n_ttft)} s vs 排队者 {spread(q_ttft)} s"
                      f"（排队者 = 等满 window，这是会话数>受保护槽数的既定代价）")
        else:
            print("  · 本负载没有会话排队：会话数 ≤ 受保护槽数，全员有槽")

    def verdict(self, facts, seq_turns, inter_turns, wall, peak, one_shot=None):
        self.rule("判定")
        batch = facts.get("batch_slots")
        if batch is None:
            print("  · 引擎未报 batch_slots：--batch 未传入，或引擎早于该 INFO 字段")
        elif int(batch) >= 2:
            print(f"  + 引擎持有 {batch} 个 batch slot（并发路径存在）")
        else:
            print("  x 引擎只有 1 个 slot：并发路径不存在，所有请求串行")

        def probe_reused(turns):
            return [t.reused for t in turns if t.role == "probe" and t.reused is not None]

        def probe_prompt(turns):
            return [t.prompt_tokens for t in turns if t.role == "probe" and t.prompt_tokens]

        seq, inter = probe_reused(seq_turns), probe_reused(inter_turns)
        if seq:
            hit = sum(1 for r in seq if r > 0)
            label = "链被保住（命中）" if hit == len(seq) else "出现清场（异常）"
            print(f"  · 连续组 Xc 第 2..N 轮 reused>0：{hit}/{len(seq)} → {label}")
        if inter:
            cleared = sum(1 for r in inter if r == 0)
            if cleared == len(inter):
                label = "交替清场坐实（solo 单检查点链被 Ya 清）"
            elif cleared:
                label = "部分清场"
            else:
                label = "交替下链也保住（与假设相反）"
            print(f"  · 交替组 Xa 第 2..N 轮 reused=0：{cleared}/{len(inter)} → {label}")
        if seq and inter:
            seq_rate = sum(seq) / max(1, sum(probe_prompt(seq_turns)))
            inter_rate = sum(inter) / max(1, sum(probe_prompt(inter_turns)))
            label = "插队即清链，根因坐实" if seq_rate - inter_rate > 0.3 else "两组接近，交替清场不成立"
            print(f"  · 复用率对比 连续 {seq_rate * 100:.0f}% vs 交替 {inter_rate * 100:.0f}% → {label}")

        inter_read = [t.read_ms / 1000.0 for t in inter_turns
                      if t.role == "probe" and t.read_ms is not None and t.reused == 0]
        if inter_read:
            print(f"  · 交替组被清场轮的 prefill 墙钟 s {spread(inter_read)}（这就是现场 9 秒全量读）")

        c_turns = [t for t in self.c_turns if t.role == "probe"]
        c_hits = [t.prefix_hit_rate for t in c_turns if t.prefix_hit_rate is not None]
        if c_hits:
            label = "slot 隔离良好" if min(c_hits) > 0.5 else "并发路径也清场"
            print(f"  · 阶段 C 并发轮复用率 {spread([h * 100 for h in c_hits], '%')} → {label}")
        if peak >= 2:
            print(f"  + 并发实测证据：活跃槽峰值 {peak} → 真并发（非排队假象）")
        elif peak == 1:
            print("  x 并发实测证据：活跃槽峰值 1 → 同时发也没进多槽")
        decodes = [t.decode_tok_s for t in c_turns if t.decode_tok_s]
        if decodes:
            print(f"  · 并发 decode tok/s {spread(decodes)}")
        ttfts = [t.ttft_s for t in c_turns if t.ttft_s]
        if ttfts:
            print(f"  · 并发 TTFT s {spread(ttfts)}")
        experts = [t.expert_hit_rate for t in c_turns if t.expert_hit_rate is not None]
        if experts:
            print(f"  · 专家缓存命中率 {spread([e * 100 for e in experts], '%')}")
        counts = self.tailer.checkpoint_counts()
        if counts:
            print(f"  · 日志检查点数最近 12 次 = {counts[-12:]}")
        print(f"  · 阶段 C 批次墙钟 {wall:.1f}s（对比阶段 B 串行同等轮次即为并发收益/代价）")
        self._verdict_one_shot(one_shot)

    def _verdict_one_shot(self, one_shot):
        """一次性请求占用 cache 的判定：清链维度（solo）+ park 占用维度（开 parking 时）。"""
        if not one_shot:
            return
        base, injected = one_shot
        base_reused = [t.reused for t in base if t.reused is not None]
        if base_reused:
            ok = sum(1 for r in base_reused if r > 0)
            label = "链活着（命中基线）" if ok == len(base_reused) else "基线链不稳（环境可能有外部请求干扰）"
            print(f"  · 阶段 D 基线 · X 连发 reused>0：{ok}/{len(base_reused)} → {label}")
        after_reused = [x_turn.reused for _one, x_turn, _b, _a in injected if x_turn.reused is not None]
        cleared = sum(1 for r in after_reused if r == 0)
        if after_reused:
            if cleared == len(after_reused):
                label = "一次性请求清掉常驻会话链（抢占 solo 单检查点）"
            elif cleared:
                label = "部分清链"
            else:
                label = "一次性请求未占用常驻会话链（隔离到位）"
            print(f"  · 注入一次性请求后 X reused=0：{cleared}/{len(after_reused)} → {label}")
        probe_cache = self.probe.cache_view()
        if not probe_cache.get("enabled"):
            print("  · parking 关闭：cache 占用维度不可测，清链维度已覆盖一次性请求的影响")
            return
        first_bytes = injected[0][2].get("bytes") if injected and injected[0][2] else None
        last_bytes = injected[-1][3].get("bytes") if injected and injected[-1][3] else None
        if first_bytes is not None and last_bytes is not None:
            growth = (last_bytes or 0) - (first_bytes or 0)
            label = "一次性请求被 park、挤占常驻会话 cache 预算" if growth > 0 else "一次性请求未被 park（不占预算）"
            print(f"  · 一次性请求 cache 占用增量 {growth:,} bytes → {label}")

    def run(self):
        facts = self.report_facts()
        if self.args.ownership_only:
            self._verdict_ownership(self.ownership_experiment())
            return
        seq_turns, inter_turns = self.interleave_experiment()
        self.c_turns, wall, peak = self.concurrent_probe()
        one_shot = self.one_shot_experiment()
        self.verdict(facts, seq_turns, inter_turns, wall, peak, one_shot)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strata 并发实测：在持久会话串行交替负载下验证 solo 单检查点链是否被交替会话清场（因果对照 + 并发对照）")
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--log", default="/home/canyun/Strata/strata-iq3_s.log",
                        help="引擎日志路径：checkpoints 数只在这里")
    parser.add_argument("--parallel-asked", type=int, default=4, help="配置里的 parallel，用于对比 batch_slots")
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--long-words", type=int, default=23000,
                        help="长会话首轮词数（现场 X≈32K token，用 23000 词×1.4 复现）")
    parser.add_argument("--short-words", type=int, default=2800,
                        help="短会话首轮词数（现场 Y≈4K token，用来插队清链）")
    parser.add_argument("--rounds", type=int, default=4, help="每个长会话的观察轮数（含首轮建立）")
    parser.add_argument("--sessions", type=int, default=6,
                        help="阶段 E 所有权自证的并发会话数（现场 6；超过受保护槽数时超出者排队）")
    parser.add_argument("--ownership-only", action="store_true",
                        help="只跑阶段 E（slot-protect 自证）：保护已开启时 solo 清场阶段（B/C/D）的判定无意义")
    parser.add_argument("--ownership-words", type=int, default=9000,
                        help="阶段 E 每会话首轮词数（现场约 12.6K token，用 9000 词×1.4 复现）")
    parser.add_argument("--build-new-tokens", type=int, default=24)
    parser.add_argument("--decode-new-tokens", type=int, default=160, help="后续轮输出长度（拉长解码窗口便于采样）")
    parser.add_argument("--followup-words", type=int, default=40)
    parser.add_argument("--arrive-gap", type=float, default=1.0,
                        help="串行交替时两次到达的间隔秒（现场约 2s；0=背靠背，都走 solo）")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--sample-every", type=float, default=0.3)
    parser.add_argument("--extra-body", default=None, help="附加请求字段，JSON 对象")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.extra_body:
        args.extra_body = json.loads(args.extra_body)
    return args


def self_test():
    line = ("strata serve: prompt 52537 tokens = 0 reused + 52537 read in 22908 ms (2293.4 tok/s), "
            "1 generated in 12 ms (83.6 tok/s), drafts accepted 0 of 0, 4 checkpoints")
    fields = PROMPT_LINE.search(line).groups()
    assert fields[0:3] == ("52537", "0", "52537"), fields
    assert fields[3:5] == ("22908", "2293.4"), fields
    assert fields[5:8] == ("1", "12", "83.6"), fields
    assert fields[10] == "4", fields

    tail = LogTailer(None)
    tail.records = [{"prompt_tokens": 100, "reused": 0, "read_tok_s": 10.0,
                     "read_ms": 10.0, "decode_tok_s": 80.0, "checkpoints": 3}]
    assert tail.claim_new(1, deadline_s=0.1)[0]["checkpoints"] == 3
    assert tail.claim_new(1, deadline_s=0.1) == []
    assert tail.checkpoint_counts() == [3]

    records = [{"prompt_total": 17000, "reused": 16900, "hit_rate": 0.99},
               {"prompt_total": 18700, "reused": 0, "hit_rate": 0.0},
               {"prompt_total": 17900, "reused": 17800, "hit_rate": 0.99}]
    expected = {"S0": 17050, "S1": 17950, "S2": 18650}
    assigned = assign_sessions(records, expected)
    assert {name: rec["prompt_total"] for name, rec in assigned.items()} == \
        {"S0": 17000, "S1": 17900, "S2": 18700}, assigned
    assert len(assign_sessions(records[:1], expected)) == 1
    assert assign_sessions([], expected) == {}

    plan = interleave_plan(["Xa"], ["Ya"], 4)
    assert plan == [("Ya", "filler"), ("Xa", "probe")] * 3, plan
    seq = sequential_plan(["Xc"], 4)
    assert seq == [("Xc", "probe")] * 3, seq

    rng = random.Random(1)
    session = Session("X", build_prompt(40, rng, nonce="X"))
    assert len(session.messages) == 2
    session.last_reply = "a"
    grown = session.next_messages(10, rng)
    assert len(grown) == 4 and grown[2]["role"] == "assistant" and grown[3]["role"] == "user"

    one = one_shot_messages("O0", 30, random.Random(1))
    assert len(one) == 2 and "[sess O0]" in one[1]["content"]
    assert one[1]["content"] != one_shot_messages("O1", 30, random.Random(1))[1]["content"]

    turn = Turn("X")
    assert turn.reused is None and turn.prefix_hit_rate is None and turn.summary().count("-") >= 4
    turn.bind({"prompt_total": 100, "reused": 50, "hit_rate": 0.72, "decode_tok_s": 80.0, "prompt_ms": 100.0},
              {"prompt_tokens": 100, "reused": 50, "read_ms": 50.0, "read_tok_s": 1000.0, "checkpoints": 3})
    assert abs(turn.prefix_hit_rate - 0.5) < 1e-9, turn.prefix_hit_rate
    assert abs(turn.expert_hit_rate - 0.72) < 1e-9
    assert turn.prefill_tok_s == 1000.0 and turn.decode_tok_s == 80.0 and turn.checkpoints == 3
    assert turn.read_ms == 50.0 and "hit= 50%" in turn.summary(), turn.summary()

    fallback = Turn("Y")
    fallback.bind({"prompt_total": 200, "reused": 100, "prompt_ms": 100.0})
    assert abs(fallback.prefix_hit_rate - 0.5) < 1e-9 and fallback.prefill_tok_s == 1000.0
    assert fallback.expert_hit_rate is None and fallback.checkpoints is None

    assert build_prompt(200, random.Random(1)) != build_prompt(200, random.Random(2))
    assert len(build_prompt(500, random.Random(1)).split()) == 500
    assert "[sess X]" in build_prompt(50, random.Random(1), nonce="X")
    assert human(None) == "-" and human(1234) == "1,234" and human(1.5) == "1.5"
    assert spread([1.0, 2.0, 3.0]) == "1.0 / 2.0 / 3.0"
    assert spread([50.0, 100.0], "%") == "50.0% / 75.0% / 100.0%"
    assert spread([]) == "-"

    # 阶段 E（slot-protect 自证）的判定逻辑：SSE 行分类、owner 漂移检测、判定四条读数
    assert classify_line(": keep-alive") == "keepalive"
    assert classify_line('data: {"x":1}') == "data"
    assert classify_line("") == "skip"

    stable = [{"owners": ["0:s:conv-0", "1:s:conv-1"]}, {"owners": ["0:s:conv-0", "1:s:conv-1"]}]
    assert owner_drift(stable) == set(), stable
    moved = [{"owners": ["0:s:conv-0"]}, {"owners": ["0:s:conv-5"]}]
    assert owner_drift(moved) == {"0"}, moved
    gap = [{"owners": ["0:s:conv-0"]}, {"owners": ["0:-"]}]
    assert owner_drift(gap) == set(), gap        # 窗口过期的空主不算漂移
    taken = [{"owners": ["0:s:conv-0"]}, {"owners": ["0:p:abcd"]}]
    assert owner_drift(taken) == {"0"}, taken    # 命名会话没了槽、被哈希新会话接管：正是要抓的签名
    hashed = [{"owners": ["0:p:abcd"]}, {"owners": ["0:p:ffff"]}]
    assert owner_drift(hashed) == {"0"}, hashed

    def e_turn(name, reused, keepalive, ttft):
        turn = Turn(name)
        turn.bind({"reused": reused, "prompt_total": max(reused, 1)})
        turn.role, turn.keepalive, turn.ttft_s = "probe", keepalive, ttft
        return turn

    owners = ["0:s:P0", "1:s:P1", "2:s:P2", "3:-"]
    zones = ["0:protected", "1:protected", "2:protected", "3:guest"]
    def e_sample(peak, holders):
        return {"active": peak, "holders": holders, "window": 120.0, "zones": zones, "owners": owners}

    def e_round(peak, turns_map):
        return (turns_map, [e_sample(peak, sum(1 for o in owners if o != "-"))], 3.0, peak, 3)

    turns_map = {"P0": e_turn("P0", 12000, 0, 0.4), "P1": e_turn("P1", 11800, 0, 0.5),
                 "P2": e_turn("P2", 11500, 0, 0.5), "P3": e_turn("P3", 0, 4, 21.0)}
    io_ = io.StringIO()
    with contextlib.redirect_stdout(io_):
        probe = Probe.__new__(Probe)
        probe._verdict_ownership([e_round(3, turns_map), e_round(3, turns_map)])
    text = io_.getvalue()
    assert "window=120s" in text, text
    assert "超出会话挤不进来" in text, text
    assert "owner 稳定" in text, text
    assert "reused>0（命中自己链、只读新增）：6/8" in text, text
    assert "排队轮 2" in text, text

    with contextlib.redirect_stdout(io.StringIO()):
        no_window = [(turns_map, [{"active": 4, "holders": None, "window": None,
                                   "zones": zones, "owners": ["0:-", "1:-", "2:-", "3:-"]}], 1.0, 4, 0)]
        off = Probe.__new__(Probe)
        io2 = io.StringIO()
        with contextlib.redirect_stdout(io2):
            off._verdict_ownership(no_window)
    assert "所有权未生效" in io2.getvalue(), io2.getvalue()

    drift_map = {"P0": e_turn("P0", 0, 0, 0.4), "P1": e_turn("P1", 0, 0, 0.5)}
    drift_round = (drift_map, [{"active": 4, "holders": 4, "window": 120.0, "zones": zones,
                                "owners": ["0:s:P0", "1:s:P1", "2:s:P2", "3:s:P3"]},
                               {"active": 4, "holders": 4, "window": 120.0, "zones": zones,
                                "owners": ["0:s:P3", "1:s:P0", "2:s:P1", "3:s:P2"]}], 2.0, 4, 3)
    io3 = io.StringIO()
    with contextlib.redirect_stdout(io3):
        Probe.__new__(Probe)._verdict_ownership([drift_round])
    assert "owner 漂移" in io3.getvalue(), io3.getvalue()
    assert "活跃槽超过受保护数" in io3.getvalue(), io3.getvalue()
    print("self-test ok")


def main():
    args = parse_args()
    if args.self_test:
        self_test()
        return
    Probe(args).run()


if __name__ == "__main__":
    main()
