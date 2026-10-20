"""serve/test_slot_owner_e2e.py - "slot_protect_seconds" end to end: a batch slot belongs to the conversation
that names itself ("strata_session"), and that conversation's next turn finds the chain in that slot instead of
landing on a neighbour's and evicting it.

The real StrataEngine and Service run over HTTP behind the fake engine of test_parallel.py, extended with
"HOLDn": a slot decodes one token and stays in it for n windows (n * 0.02 s), so a turn keeps its slot long
enough to watch the next conversations arrive.  The window that owns a slot has to be the longer of the two -
a window shorter than a turn lets the next conversation in while the first is still decoding, which is legal
behaviour and exercises nothing.  No GPU is needed:

    python -m pytest serve/test_slot_owner_e2e.py -v

Only "strata_session" is routed here: a request that names no conversation is picked exactly as before, which is
what the last test of this file checks.
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, Service, StrataEngine, serve
from serve.test_parallel import FAKE_BATCH

# "HOLDn" in a prompt: the slot decodes one token and stays in it for n windows (the fake engine's STEP is 0.02 s)
FAKE_HOLD = FAKE_BATCH.replace("import queue, sys, threading, time", "import queue, re, sys, threading, time")
FAKE_HOLD = FAKE_HOLD.replace(
    """        n = 1 if slot is not None else max_new
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            time.sleep(STEP)
""",
    """        n = 1 if slot is not None else max_new
        hold = int(re.search(rb"HOLD(\\d+)", pb).group(1)) if b"HOLD" in pb else 0
        out = 0
        for t in toks[:n]:
            if stop.is_set():
                break
            print(f"T {t}", flush=True)
            out += 1
            if slot is not None and out == 1 and hold:
                time.sleep(hold * STEP)     # the rest of this turn's decoding: the slot stays taken
            time.sleep(STEP)
""")

HOLD = 150                      # a turn: 150 * 0.02 = 3.0 s, long enough to watch a second conversation arrive
GAP = 0.3                       # conversations start 0.3 s apart
WINDOW = 30.0                   # >> a turn: the slot stays owned while the next conversation is waiting
SHORT = 0.3                     # << a turn: the window passes while the slot is still decoding
ANSWER = "o"                    # ByteTokenizer takes a byte per token, so a one-token answer is the "o" of "ok, done."
BATCH = 3


def session_prompt(n, turn=0, hold=HOLD):
    """A conversation's prompt: a start every later turn carries unchanged, and what this turn adds.  The start
    runs past the chunks the fake engine reads (CH = 32 tokens): a very short prompt is what makes it give way
    in the middle of reading, which is a different mechanism."""
    return f"S{n} context of this conversation HOLD{hold} " + "more " * 200 + f"turn {turn}"


class SlotOwnerEndToEnd(unittest.TestCase):
    def start(self, protect_seconds, slots=BATCH, protected=None, reject=None):
        from serve import server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_HOLD, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        # the mock stays on for the whole test: restart() starts the engine again, and an engine that came back as
        # a real binary would fail the one thing a restart has to keep (the settings that ride in spawn)
        real = server.subprocess.Popen
        self.popen = mock.patch.object(server.subprocess, "Popen",
                                       lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen.start()
        self.engine = StrataEngine("strata", ["--batch", str(slots), "--log", str(self.log)],
                                   slot_protect_seconds=protect_seconds, protected_slots=protected,
                                   reject_when_full=reject)
        self.tok = ByteTokenizer()
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        for attr in ("httpd", "engine", "tmp"):
            got = getattr(self, attr, None)
            if got is None:
                continue
            if attr == "httpd":
                got.shutdown()
                got.server_close()
            elif attr == "engine":
                got.unload()
            else:
                got.cleanup()
        if getattr(self, "popen", None):
            self.popen.stop()
            self.popen = None

    def chat(self, text, extra=None, max_tokens=1):
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": max_tokens,
                "reasoning_effort": "none", **(extra or {})}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        started = time.monotonic()
        with urllib.request.urlopen(req, timeout=120) as r:
            answer = json.loads(r.read().decode())
        return answer["choices"][0]["message"]["content"], time.monotonic() - started

    def admitted(self):
        """The slots each request was admitted to, in the order the admissions were sent.  A conversation that
        queues for a slot sends nothing, so this is how many got in."""
        return [int(x.split()[1]) for x in self.log.read_text().splitlines() if x.startswith("BGEN ")]

    def count_at(self, n, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.admitted()) >= n:
                return len(self.admitted())
            time.sleep(0.05)
        return len(self.admitted())

    def one(self, out, key, text, extra=None):
        try:
            out[key] = self.chat(text, extra=extra)
        except BaseException as e:        # noqa: BLE001 - the thread reports it, the test asserts on it
            out[key] = ("error", repr(e))

    def threads(self, specs, gap=0.0):
        """specs: [(key, text, extra)], started in order; the answers land in the dict the threads share."""
        out, runs = {}, []
        for key, text, extra in specs:
            t = threading.Thread(target=self.one, args=(out, key, text, extra))
            t.start()
            runs.append(t)
            if gap:
                time.sleep(gap)
        return out, runs

    def conversations(self, n, gap=GAP):
        return self.threads([(f"c{k}", session_prompt(k), {"strata_session": f"conv-{k}"}) for k in range(n)],
                            gap=gap)

    def answers(self, out, n):
        self.assertEqual({out[f"c{k}"][0] for k in range(n)}, {ANSWER}, out)

    def owned_slots(self):
        """conversation -> the slot it owns, read from the owner table by the id it names itself by."""
        return {k: self.engine.slot_owner.index(f"conv-{k}") for k in range(BATCH)
                if f"conv-{k}" in self.engine.slot_owner}

    # ---- the mechanism --------------------------------------------------------------------------------

    def test_a_conversation_owns_the_slot_it_was_admitted_to(self):
        self.start(WINDOW)
        out, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        self.answers(out, 3)
        self.assertEqual(len(self.admitted()), 3, self.admitted())
        mine = self.owned_slots()
        self.assertEqual(len(mine), 3, f"three conversations, {len(mine)} slots owned: {self.engine.slot_owner}")
        self.assertEqual(sorted(mine.values()), list(range(BATCH)), mine)

    def test_its_next_turn_returns_to_the_slot_that_holds_its_chain(self):
        self.start(WINDOW)
        _, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        mine = self.owned_slots()
        for k in sorted(mine):
            content, _ = self.chat(session_prompt(k, turn=1), extra={"strata_session": f"conv-{k}"})
            self.assertEqual(content, ANSWER)
            self.assertEqual(self.admitted()[-1], mine[k],
                             f"conversation {k}'s second turn left slot {mine[k]}, where its chain is")

    def test_a_conversation_beyond_the_slots_waits_for_a_window(self):
        """The veto: four conversations, three slots.  The fourth is admitted only once a window passes - the pick
        queues it rather than landing a second conversation on a slot whose chain belongs to a live conversation
        (which is what evicted it, turn after turn)."""
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3, self.admitted())
        late, extra = self.threads([("c3", session_prompt(3), {"strata_session": "conv-3"})])
        self.assertEqual(self.count_at(4, timeout=1.0), 3,
                         f"a fourth conversation took a slot a live conversation owns: {self.admitted()}")
        self.engine.slot_protect_seconds = SHORT      # the window passes: the queued conversation goes in
        for t in runs + extra:
            t.join(90)
        self.assertEqual(len(self.admitted()), 4, self.admitted())
        self.answers(out, 3)
        self.assertEqual(late["c3"][0], ANSWER, late)

    def generator_beats(self, text, session, seconds=6.0):
        """The heartbeats the engine yields to a request queued for a slot, driven on the generator itself: the HTTP
        layer turns every one of them into an SSE comment.  A queued request holds no control lines, so it yields no
        heartbeat of its own - _take_control has one and the slot queue had none."""
        ids = self.tok.encode(text)
        cancel = threading.Event()
        gen = self.engine.generate(ids, 1, {"strata_session": session}, cancel)
        beats, started = 0, time.monotonic()
        try:
            for t in gen:
                if t is None:
                    beats += 1
                if time.monotonic() - started >= seconds:
                    break
        finally:
            cancel.set()
            gen.close()
        return beats

    def test_the_queued_conversation_gets_heartbeats_while_it_waits(self):
        """The slot queue keeps an agent client's idle watchdog calm: a conversation beyond the slots waits a whole
        turn (16-21 s measured on the panel's numbers) for the window to pass, and the request used to be cut off
        there with nothing sent at all."""
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3, self.admitted())
        with mock.patch("serve.server.QUEUE_BEAT_S", 0.2):
            beats = self.generator_beats(session_prompt(3), "conv-3")
        self.assertGreaterEqual(beats, 2, "the conversation queued for a slot received no keep-alive")
        self.engine.slot_protect_seconds = SHORT
        for t in runs:
            t.join(60)
        self.answers(out, 3)

    def test_a_client_that_rewrites_its_prompt_start_keeps_its_slot(self):
        """The reason naming the conversation exists: what the pick matches on must be stable, and a prompt's
        start is not - a system prompt edited mid-session, a timestamp injected.  The id a conversation names
        itself by does not move."""
        self.start(WINDOW)
        _, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        mine = self.owned_slots()
        for k in sorted(mine):
            rewritten = f"S{k} a system prompt rewritten mid-session HOLD{HOLD} " + "words " * 200 + "turn 1"
            content, _ = self.chat(rewritten, extra={"strata_session": f"conv-{k}"})
            self.assertEqual(content, ANSWER)
            self.assertEqual(self.admitted()[-1], mine[k],
                             f"conversation {k} lost slot {mine[k]} to a rewritten prompt start")

    def test_the_window_passes_and_the_slot_is_the_next_conversations(self):
        self.start(WINDOW)
        _, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        for b in range(BATCH):
            self.assertTrue(self.engine._slot_owned_live(b, time.time()))
        self.engine.slot_protect_seconds = SHORT
        for b in range(BATCH):
            self.assertFalse(self.engine._slot_owned_live(b, time.time() + 1.0))

    # ---- the default path: a request that names no conversation ---------------------------------------

    def test_an_unnamed_request_is_locked_out_of_a_held_slot(self):
        """The hard lock end to end: three conversations hold the three slots, and a request that names no
        conversation does not walk into a held slot on the strength of a matching prompt start.  The plain pick
        reads only the prompt's tokens, and before the lock it overwrote the chain its owner will come back to."""
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3, self.admitted())
        plain, extra = self.threads([("p0", session_prompt(9), None)], gap=0.1)
        self.assertEqual(self.count_at(4, timeout=1.5), 3,
                         f"an unnamed request entered a slot a live conversation holds: {self.admitted()}")
        self.engine.slot_protect_seconds = SHORT      # the window passes: the locked slots open
        for t in runs + extra:
            t.join(90)
        self.answers(out, 3)
        self.assertEqual(plain["p0"][0], ANSWER, plain)
        self.assertEqual(len(self.admitted()), 4, self.admitted())

    def test_the_window_off_leaves_the_slots_everyones_as_they_were(self):
        self.start(0.0)
        out, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        self.answers(out, 3)
        self.assertEqual(len(self.admitted()), 3, self.admitted())
        self.assertEqual(self.engine.slot_owner, [None] * BATCH, "the pick wrote owners with the window off")
        self.assertNotIn("slot_protect_seconds", self.get_live(), self.get_live())

    def test_the_panel_shows_who_holds_each_slot(self):
        self.start(WINDOW)
        _, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        live = self.get_live()
        self.assertEqual(live["slot_protect_seconds"], WINDOW)
        self.assertEqual(live["slot_holders"], 3, live)
        owners = sorted(s.get("owner") for s in live["slots"])
        self.assertEqual(owners, [f"conv-{k}" for k in range(3)], live["slots"])
        self.assertTrue(all(s["owned"] for s in live["slots"]), live["slots"])

    def get_live(self):
        return self.svc.metrics()["live"]


class ReservedSlotsAndRefusal(unittest.TestCase):
    """"protected_slots" splits the batch between the conversations that name themselves and the traffic that names
    nothing, and "reject_when_full" answers a request that finds its side of the batch full with a 429 instead of
    queueing it.  Both are read from the request before the answer starts, so the 429 is a real HTTP status with a
    Retry-After, not an error inside a stream."""

    def setUp(self):
        self.base = None

    def tearDown(self):
        for attr in ("httpd", "engine", "tmp"):
            got = getattr(self, attr, None)
            if got is None:
                continue
            if attr == "httpd":
                got.shutdown()
                got.server_close()
            elif attr == "engine":
                got.unload()
            else:
                got.cleanup()
        if getattr(self, "popen", None):
            self.popen.stop()
            self.popen = None

    def start(self, protect_seconds, slots=BATCH, protected=None, reject=None):
        from serve import server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_HOLD, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        real = server.subprocess.Popen
        self.popen = mock.patch.object(server.subprocess, "Popen",
                                       lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen.start()
        self.engine = StrataEngine("strata", ["--batch", str(slots), "--log", str(self.log)],
                                   slot_protect_seconds=protect_seconds, protected_slots=protected,
                                   reject_when_full=reject)
        self.tok = ByteTokenizer()
        self.svc = Service(self.engine, self.tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def post(self, text, extra=None):
        """The HTTP status, the error code and the Retry-After of one chat request (200 and "" when it is answered)."""
        body = {"messages": [{"role": "user", "content": text}], "max_tokens": 1,
                "reasoning_effort": "none", **(extra or {})}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, "", r.headers.get("Retry-After")
        except urllib.error.HTTPError as e:
            answer = e.read().decode()
            try:
                code = ((json.loads(answer).get("error") or {}).get("code")) or ""
            except ValueError:
                code = ""
            return e.code, code, e.headers.get("Retry-After")

    def admitted(self):
        return [int(x.split()[1]) for x in self.log.read_text().splitlines() if x.startswith("BGEN ")]

    def count_at(self, n, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.admitted()) >= n:
                return len(self.admitted())
            time.sleep(0.05)
        return len(self.admitted())

    def test_the_reserved_slots_serve_the_named_conversations_only(self):
        self.start(WINDOW, slots=3, protected=2, reject=True)
        self.assertEqual(self.engine.slot_zone, ["protected", "protected", "public"],
                         "protected_slots: 2 did not reserve slots 0 and 1")
        out, runs = {}, []
        for k, key in enumerate(("c0", "c1")):
            t = threading.Thread(target=lambda k=k, key=key: out.__setitem__(
                key, self.post(session_prompt(k), {"strata_session": f"conv-{k}"})))
            t.start()
            runs.append(t)
            time.sleep(GAP)
        self.assertEqual(self.count_at(2, timeout=8.0), 2, self.admitted())
        self.assertEqual(sorted(self.admitted()), [0, 1],
                         f"two conversations, two reserved slots: they landed on {self.admitted()}")
        # a third conversation has no reserved slot left: the public slot is not its side of the batch
        code, kind, _ = self.post(session_prompt(2), {"strata_session": "conv-2"})
        self.assertEqual((code, kind), (429, "slots_full"),
                         f"a third conversation, no reserved slot: {(code, kind)}")
        # traffic that names nothing takes the public slot, and only that one
        self.assertEqual(self.post(session_prompt(9))[0], 200)
        self.assertEqual(self.admitted()[-1], 2, f"unnamed traffic entered a reserved slot: {self.admitted()}")
        self.engine.slot_protect_seconds = SHORT
        for t in runs:
            t.join(90)

    def test_a_full_batch_is_answered_429_and_the_queue_stays_otherwise(self):
        """The two answers to a full server, on the same two held slots: with "reject_when_full" a 429 with a
        Retry-After before the answer starts; without it (the default) the same request is queued, as it has
        always been."""
        self.start(WINDOW, slots=2, reject=True)
        out, runs = {}, []
        for k in range(2):
            t = threading.Thread(target=lambda k=k: out.__setitem__(
                f"c{k}", self.post(session_prompt(k), {"strata_session": f"conv-{k}"})))
            t.start()
            runs.append(t)
            time.sleep(GAP)
        self.assertEqual(self.count_at(2, timeout=8.0), 2, self.admitted())
        code, kind, retry = self.post(session_prompt(2), {"strata_session": "conv-2"})
        self.assertEqual((code, kind), (429, "slots_full"),
                         f"both slots are held and the server refuses: {(code, kind)}")
        self.assertTrue(retry and int(retry) >= 1, f"a refusal sent no Retry-After: {retry!r}")
        self.assertEqual(len(self.admitted()), 2, "the refused request reached the engine")
        self.assertEqual(self.svc.metrics()["live"].get("slot_refusals"), 1)
        self.engine.slot_reject_full = False          # the same server, the queue back on
        queued = {}
        t = threading.Thread(target=lambda: queued.__setitem__("c2", self.post(
            session_prompt(2), {"strata_session": "conv-2"})))
        t.start()
        runs.append(t)
        time.sleep(1.0)
        self.assertEqual(len(self.admitted()), 2, "the queued request reached the engine before the slots opened")
        self.engine.slot_protect_seconds = SHORT      # the window passes: the queued request goes in
        for x in runs:
            x.join(90)
        self.assertEqual(queued.get("c2", (None,))[0], 200, queued)
        self.assertEqual({out[f"c{k}"][0] for k in range(2)}, {200}, out)


    def test_a_refusal_the_precheck_let_through_is_counted_too(self):
        """The engine's own refusal counts as a refusal, not only the pre-check's.  The pre-check reads the slots
        before the headers and the engine reads them again when it admits, so a request can be told "a slot is
        free" and lose it to a conversation's lock a moment later; that answer is the same 429 for the client and
        has to be in the same number, which is what an operator watches to decide whether the batch is too small."""
        self.start(WINDOW, slots=2, reject=True)
        out, runs = {}, []
        for k in range(2):
            t = threading.Thread(target=lambda k=k: out.__setitem__(
                f"c{k}", self.post(session_prompt(k), {"strata_session": f"conv-{k}"})))
            t.start()
            runs.append(t)
            time.sleep(GAP)
        self.assertEqual(self.count_at(2, timeout=8.0), 2, self.admitted())
        with mock.patch.object(self.svc, "slot_refusal", return_value=None):
            code, kind, _ = self.post(session_prompt(2), {"strata_session": "conv-2"})
        self.assertEqual((code, kind), (429, "slots_full"), f"the engine refused a request the pre-check passed: {(code, kind)}")
        self.assertEqual(self.svc.metrics()["live"].get("slot_refusals"), 1,
                         "the refusal the pre-check passed was not counted")
        for t in runs:
            t.join(90)


if __name__ == "__main__":
    unittest.main()
