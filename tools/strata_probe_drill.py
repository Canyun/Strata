#!/usr/bin/env python3
"""离线演练：mock 一个实现 solo 单检查点链语义的 Strata server，跑通探测脚本的阶段 A/B/C，
证明判定逻辑对现场模式敏感——串行交替清场（交替组 reused=0）、串行连续命中（连续组 reused 高）、
并发 batch slot 隔离（阶段 C reused 高）。

mock 核心：inflight==1 走 solo 单链（每个新 prompt 清掉别人的链），inflight>=2 走 batch 每会话独立链（隔离）。
"""

import importlib.util
import io
import json
import os
import re
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = importlib.util.spec_from_file_location("probe", os.path.join(HERE, "strata_concurrency_probe.py"))
probe_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe_module)

TOKENS_PER_WORD = 1.4
NONCE = re.compile(r"\[sess (\w+)\]")
ENGINE_FACTS = {
    "engine": "0.1.43", "context": 200000, "kv": "int8", "kv_resident": 0,
    "expert_slots": 25000, "expert_cache_mib": 45000, "arena_mib": 51507,
    "spec": 4, "mtp_max": 4, "pool_workers": 6, "pcie_frac": 0.37,
    "vram_free_mib": 403, "conversation_cache_mib": 0, "conversation_cache_slots": 0,
    "slot_cache": 1, "batch_slots": 2, "batch_groups": 1,
}
STATE = {
    "lock": threading.Lock(), "log_path": None, "inflight": 0,
    "requests": [],                      # 历史，newest last
    "live_chain": (None, 0),             # solo 单检查点链：(会话名, 该会话上一轮 prompt tokens)
    "slot_chains": {},                   # batch 每会话独立链：name -> 上次 prompt tokens
    "ckpt": {},                          # 每会话检查点数（命中累积，清场归 0）
}


def estimate_tokens(messages):
    words = sum(len(str(item.get("content", "")).split()) for item in messages)
    return int(words * TOKENS_PER_WORD)


def session_name(messages):
    for item in messages:
        found = NONCE.search(str(item.get("content", "")))
        if found:
            return found.group(1)
    return "anon"


def resolve_reused(name, prompt_tokens, batch_mode):
    """返回 (reused, checkpoints)：solo 单链被别的会话清场则 0；batch 每会话独立链恒命中。"""
    with STATE["lock"]:
        if batch_mode:
            chain = STATE["slot_chains"].get(name, 0)
            reused = min(chain, prompt_tokens - 1) if chain else 0
            STATE["slot_chains"][name] = prompt_tokens
            ckpt = STATE["ckpt"].get(name, 0) + (1 if reused > 0 else 1)
        else:
            last, chain = STATE["live_chain"]
            reused = min(chain, prompt_tokens - 1) if last == name and chain else 0
            STATE["live_chain"] = (name, prompt_tokens)
            ckpt = (STATE["ckpt"].get(name, 0) + 1) if reused > 0 else 0
        STATE["ckpt"][name] = ckpt
        return reused, ckpt


def append_log_line(prompt_tokens, reused, generated, checkpoints):
    read = prompt_tokens - reused
    line = (f"strata serve: prompt {prompt_tokens} tokens = {reused} reused + {read} read in "
            f"{read / 2.4:.0f} ms (2400.0 tok/s), {generated} generated in {generated * 12:.0f} ms "
            f"(83.3 tok/s), drafts accepted 0 of 0, {checkpoints} checkpoints\n")
    with open(STATE["log_path"], "a", encoding="utf-8") as handle:
        handle.write(line)


def live_slots():
    inflight = STATE["inflight"]
    states = ["decoding"] * inflight + ["idle"] * max(0, 2 - inflight)
    return [{"slot": index, "state": state, "held_tokens": 0,
             "prompt_tokens": 900 if state == "decoding" else 0,
             "generated": 3 if state == "decoding" else 0, "elapsed_s": 0.05,
             "tok_s": 83.3 if state == "decoding" else None}
            for index, state in enumerate(states)]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path.split("?")[0]
        with STATE["lock"]:
            history = list(reversed(STATE["requests"]))     # newest first（与 server.py 的 hist[::-1] 一致）
            payload = None
            if path in ("/metrics", "/status"):
                payload = {"engine": ENGINE_FACTS,
                           "live": {"state": "generating" if STATE["inflight"] else "idle",
                                    "running": STATE["inflight"], "outside_slots": 0, "waiting": 0,
                                    "slots": live_slots()},
                           "requests": history, "requests_kept": len(history),
                           "conversation_cache": {"enabled": False, "budget_mib": 0, "parked": 0},
                           "totals": {"requests": len(history)}}
            elif path == "/slots":
                payload = live_slots()
        if payload is None:
            self.send_response(404)
            self.end_headers()
            return
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        payload = json.loads(self.rfile.read(length) or b"{}")
        messages = payload.get("messages") or []
        max_tokens = int(payload.get("max_tokens") or 16)
        prompt_tokens = estimate_tokens(messages)
        name = session_name(messages)
        self._begin()
        batch_mode = STATE["inflight"] >= 2                  # 同时到达 → batch slot 隔离
        try:
            reused, checkpoints = resolve_reused(name, prompt_tokens, batch_mode)
            time.sleep(0.02)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for index in range(4):
                chunk = {"choices": [{"delta": {"content": f"reply{index} "}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            with STATE["lock"]:
                STATE["requests"].append({
                    "time": time.time(), "prompt_tokens": prompt_tokens, "prompt_total": prompt_tokens,
                    "reused": reused, "output_tokens": max_tokens,
                    "prompt_ms": (prompt_tokens - reused) / 2.4, "decode_ms": max_tokens * 12,
                    "decode_tok_s": 83.3, "hit_rate": 0.912 if reused == 0 else 0.995, "finish": "stop",
                })
            append_log_line(prompt_tokens, reused, max_tokens, checkpoints)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        finally:
            self._end()

    @staticmethod
    def _begin():
        with STATE["lock"]:
            STATE["inflight"] += 1

    @staticmethod
    def _end():
        with STATE["lock"]:
            STATE["inflight"] -= 1


def run_drill():
    handle, log_path = tempfile.mkstemp(suffix=".log")
    os.close(handle)
    STATE["log_path"] = log_path
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    argv = ["probe", "--url", f"http://127.0.0.1:{port}", "--log", log_path,
            "--long-words", "600", "--short-words", "150", "--rounds", "3",
            "--decode-new-tokens", "8", "--followup-words", "30",
            "--arrive-gap", "0", "--sample-every", "0.02", "--parallel-asked", "2", "--gpus", "2"]
    sys.argv = argv
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            probe_module.main()
    finally:
        server.shutdown()
        os.unlink(log_path)
    return buffer.getvalue()


def check(output):
    expectations = (
        ("阶段 A 引擎版本", "0.1.43"),
        ("阶段 A batch_slots", "batch_slots=2"),
        ("阶段 A 单会话 KV 行", "单会话 KV"),
        ("阶段 A parking 关闭告警", "parking 关闭"),
        ("阶段 B 连续组命中", "链被保住"),
        ("阶段 B 交替清场坐实", "交替清场坐实"),
        ("阶段 B 复用率对比", "插队即清链，根因坐实"),
        ("阶段 B 清场轮 prefill 墙钟", "prefill 墙钟"),
        ("阶段 C 并发 slot 隔离", "slot 隔离良好"),
        ("阶段 C 活跃槽峰值 2", "活跃槽峰值 2"),
        ("阶段 D 基线链活着", "链活着（命中基线）"),
        ("阶段 D 一次性请求清链", "一次性请求清掉常驻会话链"),
        ("阶段 D parking 关闭提示", "parking 关闭：cache 占用维度不可测"),
        ("判定并发已生效", "并发路径存在"),
        ("无占位空值 None tok/s", "None tok/s"),
    )
    failed = []
    for label, needle in expectations:
        present = needle in output
        want = label != "无占位空值 None tok/s"
        if present != want:
            failed.append(f"{label}: 期望出现={want} 实际={present}")
    print(f"演练输出 {output.count(chr(10))} 行，{len(output)} 字符")
    for item in failed:
        print("  x " + item)
    if not failed:
        print("drill ok：全部断言通过（判定逻辑对现场清场/命中/隔离模式均敏感）")
    return not failed


import sys
import tempfile

if __name__ == "__main__":
    result = run_drill()
    if "-v" in sys.argv:
        print(result)
    ok = check(result)
    print("-" * 96)
    print(result)
    sys.exit(0 if ok else 1)
