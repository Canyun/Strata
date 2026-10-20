#!/usr/bin/env python3
"""按设计验收槽位路由，逐项给出证据。

设计（验收标准）：
  C1  槽位归属只由请求参数 "strata_session" 决定，且它是唯一判据（前缀匹配、strata_checkpoint 都不参与）。
  C2  没有该参数的请求逐字节走主干默认路径：不产生归属、视图字段与主干一致、准入顺序与主干一致。
  C3  带该参数的请求彻底锁定它的槽：窗口内该槽即使空闲也不允许别的请求进入（硬锁，不是软避让）。
  C4  两种槽位都满时直接 429（带 Retry-After）；开关关闭时排队，与主干一致。
  C5  分区严格：带该参数只进保留槽，无该参数永不进保留槽，两侧不交叉。
  C6  窗口过期后让位：锁撤销，槽可以被别的请求取走。
  C7  历史功能确认已废弃（源码中不存在）。

    python3 tools/test_slot_protect.py          # 全部项，退出码 0 = 设计全部满足
    python3 tools/test_slot_protect.py -k C3    # 只跑某项
    python3 tools/test_slot_protect.py -v       # 打印每项完整证据

跑真机器：真实的 StrataEngine + Service + HTTP，引擎是 serve/test_parallel.py 的批处理假引擎（serve/
test_slot_owner_e2e.py 扩展出的 "HOLDn"：该槽解码一个 token 后停留 n * 0.02 秒，使一轮足够长，能观察别的请求
到底进不进得来）。不需要 GPU，不需要模型。

未覆盖：真引擎的 GPU 行为（本机无 GPU，引擎侧结论来自读代码）；solo 路径（设计范围外）；--batch-groups 多层
拆分管线的槽位行为（该路径显式不接管路由）。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(os.environ.get("STRATA_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO))

from serve import server  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, serve  # noqa: E402
from serve.test_slot_owner_e2e import FAKE_HOLD, session_prompt  # noqa: E402

WINDOW = 30.0           # >> 一轮：别人排队期间窗口还开着
MEDIUM = 5.0            # > 一轮（3.0s）：会话结束后锁仍开着，睡 5.5s 就过期
SHORT = 0.3             # << 一轮：窗口在一轮之内就过去了
GAP = 0.3               # 请求之间的间隔，让准入顺序可判定
LONG_TURN = 150         # 一轮 150 * 0.02 = 3.0 s
SHORT_TURN = 40         # 一轮 40 * 0.02 = 0.8 s

MAIN_IDLE_KEYS = frozenset({"slot", "state", "held_tokens"})
MAIN_BUSY_KEYS = frozenset({"slot", "state", "prompt_tokens", "prompt_read", "generated", "elapsed_s", "tok_s"})

DEAD_NAMES = ("solo_path", "slot_caps", "slot_cap", "SLOT_PROTECT_MIN_TOKENS", "_slot_locked",
              "slot_ownership_s", "_session_key", "def zone_of", "guest_borrow")


class Machine:
    """一台跑着假引擎的真实服务器：起停、发请求、读它把每个请求准入到了哪个槽。"""

    def __init__(self, batch=4):
        self.batch = batch
        self.engine = None
        self.svc = None
        self.httpd = None
        self.log = None
        self.stopper = None
        self.base = ""

    def start(self, window=0.0, protected=None, reject=None):
        """起服务器。window=保护窗口秒数，protected=保留槽数，reject=满时是否 429。"""
        tmp = Path(tempfile.mkdtemp(prefix="slot-protect-"))
        script = tmp / "fake_strata.py"
        script.write_text(FAKE_HOLD, encoding="utf-8")
        self.log = tmp / "requests.log"
        self.stopper = _FakeEngine(server.subprocess, script)
        self.stopper.start()
        self.engine = StrataEngine("strata", ["--batch", str(self.batch), "--log", str(self.log)],
                                   slot_protect_seconds=window, protected_slots=protected,
                                   reject_when_full=reject)
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(REPO / "serve" / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def restart(self, window=0.0, protected=None, reject=None):
        """关掉再按给定设置重开一台全新的服务器（新的引擎、新的日志、新的端口）：一项验收要在配置层
        比较两种设置时用它，设置走的是构造参数，与服务器读配置同一条路。"""
        self.close()
        self.engine = None
        self.svc = None
        self.httpd = None
        self.start(window=window, protected=protected, reject=reject)

    def close(self):
        """停服务器与引擎，并还原 Popen（引擎重启也走假引擎，所以整段期间都生效）。"""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.engine is not None:
            self.engine.unload()
        if self.stopper is not None:
            self.stopper.stop()

    def post(self, text, session=None, extra=None):
        """发一个非流式请求，返回 (状态码, 内容或错误码, Retry-After)。"""
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1,
                "reasoning_effort": "none", **(extra or {})}
        if session is not None:
            body["strata_session"] = session
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as got:
                answer = json.loads(got.read().decode())
            return 200, answer["choices"][0]["message"]["content"], None
        except urllib.error.HTTPError as e:
            payload = e.read().decode()
            try:
                code = json.loads(payload)["error"].get("code")
            except Exception:                                  # noqa: BLE001 - 证据用，原样带出
                code = payload[:60]
            return e.code, code, e.headers.get("Retry-After")

    def admitted(self):
        """每个请求被准入到了哪个槽（引擎日志的 BGEN 行），按准入先后排列。请求排队时不发送任何东西，
        所以这个列表就是"进来了几个"。"""
        return [int(x.split()[1]) for x in self.log.read_text().splitlines() if x.startswith("BGEN ")]

    def wait_admitted(self, n, timeout=20.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.admitted()) >= n:
                return n
            time.sleep(0.05)
        return len(self.admitted())

    def run(self, specs, gap=GAP):
        """按序起线程发请求（specs: [(key, 文本, session, extra)]），返回共享答复字典与线程列表。"""
        out, runs = {}, []
        for key, text, session, extra in specs:
            worker = threading.Thread(target=lambda key=key, text=text, session=session, extra=extra:
                                      out.__setitem__(key, self._safe_post(text, session, extra)))
            worker.start()
            runs.append(worker)
            if gap:
                time.sleep(gap)
        return out, runs

    def _safe_post(self, text, session, extra):
        try:
            return self.post(text, session=session, extra=extra)
        except BaseException as e:                             # noqa: BLE001 - 线程报告，检查项断言
            return ("error", repr(e), None)

    def owned(self):
        """当前被占用的槽 -> 归属的会话 id。"""
        return {b: v for b, v in enumerate(getattr(self.engine, "slot_owner", [])) if v}

    def view(self):
        return {v["slot"]: v for v in self.engine.slots_view()}

    def locked(self):
        return sorted(b for b, v in self.view().items() if v.get("owned"))


class _FakeEngine:
    """把 server.subprocess.Popen 换成"跑假引擎脚本、其余参数原样"。"""

    def __init__(self, target, script):
        self.target, self.script = target, script
        self.saved = None

    def start(self):
        self.saved = self.target.Popen
        self.target.Popen = lambda cmd, **kw: self.saved([sys.executable, str(self.script), *cmd[1:]], **kw)

    def stop(self):
        self.target.Popen = self.saved


def quiet(fn, *args):
    """吞掉引擎的 [strata] 输出：它是运行日志，不是验收结论。"""
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


CHECKS = []


def check(number, design, batch=4):
    """注册一项验收。batch 是这项要的槽数：锁的判别力在单槽上最强（唯一的槽被锁住，软避让与硬锁的答案
    直接相反），分区要四个槽才看得出两侧。"""
    def register(fn):
        CHECKS.append((number, design, batch, fn))
        return fn
    return register


@check("C1", "归属只由 strata_session 决定，且它是唯一判据")
def c1_session_is_the_only_key(m: Machine):
    m.start(window=WINDOW)
    out, runs = m.run([("alpha", session_prompt(1, 0, SHORT_TURN), "alpha", None)])
    for t in runs:
        t.join(60)
    alpha_owners = dict(m.owned())
    probes = [
        ("同前缀、无 session（前缀匹配不得产生归属）", session_prompt(1, 0, SHORT_TURN), None, None),
        ("同前缀 + strata_checkpoint=false", session_prompt(1, 0, SHORT_TURN), None, {"strata_checkpoint": False}),
        ("session=5（非字符串）", session_prompt(2, 0, SHORT_TURN), 5, None),
        ("session='   '（空白字符串）", session_prompt(3, 0, SHORT_TURN), "   ", None),
    ]
    answers = [m.post(text, session=session, extra=extra) for _, text, session, extra in probes]
    owners = m.owned()
    names = sorted(set(owners.values()) - {None})
    ok = (out["alpha"][0] == 200 and len(alpha_owners) == 1 and names == ["alpha"] and
          all(code == 200 for code, _, _ in answers))
    detail = "；".join(f"{label} -> 状态码 {code}" for (label, _, _, _), (code, _, _) in zip(probes, answers))
    return ok, (f"alpha 准入后 owner={alpha_owners}；随后 4 个探针请求：{detail}；"
                f"探针之后 owner 表={owners}，归属者集合={names}（只有 alpha、且探针全部正常服务 = 判据唯一）")


@check("C2", "没有 strata_session 时逐字节走主干默认路径")
def c2_default_path_is_main(m: Machine):
    m.start(window=0.0)
    _, runs = m.run([(f"p{i}", session_prompt(i, 0, LONG_TURN), None, None) for i in range(3)])
    reached = m.wait_admitted(3)
    rows = quiet(lambda: [dict(v) for v in m.engine.slots_view()])
    busy_keys = {frozenset(v) for v in rows if v["state"] != "idle"}
    for t in runs:
        t.join(60)
    rows_after = quiet(lambda: [dict(v) for v in m.engine.slots_view()])
    idle_keys = {frozenset(v) for v in rows_after if v["state"] == "idle"}
    admitted, owners = m.admitted(), m.owned()
    fourth = m.post(session_prompt(9, 0, SHORT_TURN))
    ok = (reached == 3 and admitted == [0, 1, 2] and not owners and
          busy_keys == {MAIN_BUSY_KEYS} and idle_keys == {MAIN_IDLE_KEYS} and fourth[0] == 200)
    return ok, (f"3 个无 session 请求的准入顺序={admitted}（主干按槽序填空）；owner 表={owners}（空 = 无归属）；"
                f"运行中的行字段={sorted(busy_keys) or '未采到'}，主干应为 {sorted(MAIN_BUSY_KEYS)}；"
                f"结束后的空闲行字段={sorted(idle_keys) or '未采到'}，主干应为 {sorted(MAIN_IDLE_KEYS)}；"
                f"第 4 个请求状态码={fourth[0]}（窗口关闭 = 排队后进来，与主干一致）")


@check("C3", "带 session 彻底锁定槽位：空闲也不允许别的请求进入", batch=2)
def c3_lock_is_absolute(m: Machine):
    """两个会话各占一槽并结束：两个槽都空闲，且都在窗口内。硬锁下无 session 请求进不来（只能排队）；
    软避让会让它立刻进来得 200。solo 路径（--batch 1）不在此列：它不进槽位机器，设计不接管。"""
    m.start(window=WINDOW)
    _, runs = m.run([(f"c{k}", session_prompt(k, 0, LONG_TURN), f"conv-{k}", None) for k in range(2)])
    m.wait_admitted(2)
    for t in runs:
        t.join(60)
    idle = sorted(b for b, v in m.view().items() if v["state"] == "idle")
    locked = m.locked()
    refusal = quiet(lambda: m.engine.slot_rejection(None))
    before = len(m.admitted())
    answers = {}
    worker = threading.Thread(daemon=True,
                              target=lambda: answers.__setitem__("p", m.post(session_prompt(9, 0, SHORT_TURN))))
    worker.start()
    worker.join(3.0)
    entered = m.admitted()[before:]
    ok = (idle == [0, 1] and locked == [0, 1] and refusal is not None and entered == [] and worker.is_alive())
    return ok, (f"两个会话的一轮都已结束（{LONG_TURN * 0.02:.1f}s，答复已到手），slots_view 显示空闲槽={idle}"
                f"（都无请求在跑）且 owned={locked}（窗口 {WINDOW:.0f}s 未过）；引擎 slot_rejection(无 session)="
                f"{refusal[0] if refusal else 'None（可进）'}；无 session 请求等 3 秒后进入的槽={entered}、"
                f"仍在排队={worker.is_alive()}（软避让会立刻进来得 200）")


@check("C4", "两种槽位都满时直接 429（带 Retry-After）；开关关闭时排队", batch=2)
def c4_full_is_answered_not_queued(m: Machine):
    """同一场景跑两遍，两遍的设置都从配置层进来（构造参数，与服务器读配置同一条路）：开拒绝时第三个请求
    拿 429（带 Retry-After，且计入 slot_refusals）；不开时它排队（一秒内进不了引擎），窗口缩短后进来得
    200——默认路径不变。"""
    m.start(window=WINDOW, reject=True)
    _, runs = m.run([(f"conv-{k}", session_prompt(k, 0, LONG_TURN), f"conv-{k}", None) for k in range(2)])
    m.wait_admitted(2)
    refused = m.post(session_prompt(2, 0, LONG_TURN), session="conv-2")
    refusals = m.svc.metrics()["live"].get("slot_refusals")
    for t in runs:
        t.join(60)

    m.restart(window=WINDOW, reject=False)
    _, runs2 = m.run([(f"conv-{k}", session_prompt(k, 0, LONG_TURN), f"conv-{k}", None) for k in range(2)])
    m.wait_admitted(2)
    queued, runs3 = m.run([("late", session_prompt(2, 0, LONG_TURN), "conv-2", None)], gap=0)
    time.sleep(1.0)
    admitted_while_queued = len(m.admitted())
    m.engine.slot_protect_seconds = SHORT
    for t in runs2 + runs3:
        t.join(90)
    ok = (refused[0] == 429 and refused[1] == "slots_full" and refused[2] and int(refused[2]) >= 1 and
          refusals == 1 and admitted_while_queued == 2 and queued["late"][0] == 200)
    return ok, (f"配置 reject_when_full=true、2 槽占满：第三个请求 -> 状态码={refused[0]} 码={refused[1]} "
                f"Retry-After={refused[2]}，slot_refusals={refusals}；配置 reject_when_full=false（默认）"
                f"同一场景：1 秒内准入数={admitted_while_queued}（=2 表示在排队、未进引擎），"
                f"窗口缩短后进来 状态码={queued['late'][0]}")


@check("C5", "分区严格：带 session 只进保留槽，无 session 永不进保留槽")
def c5_zones_never_cross(m: Machine):
    m.start(window=WINDOW, protected=2)
    zones = quiet(lambda: [v.get("zone") for v in m.engine.slots_view()])
    _, runs = m.run([(f"s{k}", session_prompt(k, 0, SHORT_TURN), f"conv-{k}", None) for k in range(2)])
    for t in runs:
        t.join(60)
    named = m.admitted()
    _, runs2 = m.run([(f"p{k}", session_prompt(k + 5, 0, SHORT_TURN), None, None) for k in range(2)])
    for t in runs2:
        t.join(60)
    public = m.admitted()[len(named):]
    named_ok = bool(named) and all(zones[b] == "protected" for b in named)
    public_ok = bool(public) and all(zones[b] == "public" for b in public)
    ok = zones == ["protected", "protected", "public", "public"] and named_ok and public_ok
    return ok, (f"分区表={zones}（保留前 2 个）；2 个带 session 请求进入={named}，全在 protected={named_ok}；"
                f"2 个无 session 请求进入={public}，全在 public={public_ok}；owner 表={m.owned()}")


@check("C6", "窗口过期后让位：锁撤销，槽可被别的请求取走", batch=2)
def c6_window_expiry_yields(m: Machine):
    """与 C3 闭环的同一场景，只把窗口从 30s 换成 6s：两个会话一轮跑完，两槽空闲且都锁着（C3 的证据），
    睡过窗口后锁必须撤销、槽必须能被无 session 请求取走——否则槽被永久占用，别的会话永远饿死。"""
    m.start(window=MEDIUM)
    _, runs = m.run([(f"conv-{k}", session_prompt(k, 0, LONG_TURN), f"conv-{k}", None) for k in range(2)])
    m.wait_admitted(2)
    for t in runs:
        t.join(60)
    idle = sorted(b for b, v in m.view().items() if v["state"] == "idle")
    locked_in_window = m.locked()
    blocked = quiet(lambda: m.engine.slot_rejection(None))
    home = [b for b, v in m.owned().items() if v == "conv-0"]
    time.sleep(MEDIUM + 0.5)
    locked_after = m.locked()
    opened = quiet(lambda: m.engine.slot_rejection(None))
    before = len(m.admitted())
    stranger = m.post(session_prompt(9, 0, SHORT_TURN))
    took = m.admitted()[before:]
    back, runs2 = m.run([("back", session_prompt(0, 1, SHORT_TURN), "conv-0", None)], gap=0)
    for t in runs2:
        t.join(60)
    came_back = m.admitted()[len(m.admitted()) - 1:]
    ok = (idle == [0, 1] and locked_in_window == [0, 1] and blocked is not None and
          locked_after == [] and opened is None and len(home) == 1 and
          stranger[0] == 200 and len(took) == 1 and back["back"][0] == 200 and came_back == home)
    return ok, (f"两个会话一轮结束（{LONG_TURN * 0.02:.1f}s）后：空闲槽={idle}、owned={locked_in_window}"
                f"（窗口 {MEDIUM:.0f}s 内）、slot_rejection(无 session)={blocked[0] if blocked else 'None（可进）'}；"
                f"睡过窗口 {MEDIUM:.0f}s 后：owned={locked_after}（锁已撤销）、"
                f"slot_rejection={'None（可进）' if opened is None else opened[0]}；无 session 请求"
                f"（无归属者）状态码={stranger[0]} 进入的槽={took}（槽确实让出来了；取走哪个由主干 pick 定）；"
                f"conv-0 的下一轮 状态码={back['back'][0]} 进入的槽={came_back}，"
                f"它原来的槽={home}（owner 表在 BDONE 后不清，靠窗口判活让位）")


@check("C7", "历史功能确认已废弃（源码中不存在）")
def c7_dead_features_gone(m: Machine):
    roots = [REPO / "serve", REPO / "strata_cli", REPO / "strata_control"]
    hits = {}
    for name in DEAD_NAMES:
        found = []
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                if path.name.startswith("test_"):
                    continue
                for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if name in line:
                        found.append(f"{path.relative_to(REPO)}:{n}")
        if found:
            hits[name] = found[:3]
    return not hits, (f"扫描 {len(DEAD_NAMES)} 个历史标识符（solo_path、slot_caps、slot_cap、"
                      f"SLOT_PROTECT_MIN_TOKENS、_slot_locked、slot_ownership_s、前缀哈希 _session_key、"
                      f"旧 zone_of、guest 借用），命中={hits or '无'}")


def main() -> int:
    """逐项跑验收：每项一台干净的服务器，崩溃算不通过，退出码 0 = 设计全部满足。"""
    ap = argparse.ArgumentParser(description="按设计验收槽位路由")
    ap.add_argument("-k", dest="keep", default="", help="只跑编号含此串的项（如 C3）")
    ap.add_argument("-v", dest="verbose", action="store_true", help="打印每项完整证据")
    args = ap.parse_args()

    wanted = [c for c in CHECKS if args.keep.lower() in c[0].lower()]
    print(f"槽位路由设计验收 · {len(wanted)} 项 · 真 StrataEngine/Service/HTTP + 批处理假引擎（无 GPU）")
    print("=" * 100)
    failed = []
    for number, design, batch, fn in wanted:
        machine = Machine(batch=batch)
        started = time.monotonic()
        try:
            ok, evidence = quiet(lambda: fn(machine))
        except BaseException as e:                             # noqa: BLE001 - 崩溃即不通过
            ok, evidence = False, f"检查抛出异常 {type(e).__name__}: {e}"
        finally:
            quiet(machine.close)
        took = time.monotonic() - started
        print(f"[{'PASS' if ok else 'FAIL'}] {number}  {design}  [{batch} 槽]  ({took:.1f}s)")
        if args.verbose or not ok:
            print(f"      证据: {evidence}")
        if not ok:
            failed.append(number)
    print("=" * 100)
    if failed:
        print(f"设计未全部满足：{len(failed)}/{len(wanted)} 项不通过 -> {', '.join(failed)}")
        return 1
    print(f"设计全部满足：{len(wanted)}/{len(wanted)} 项通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
