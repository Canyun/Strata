"""serve/test_slot_protect.py - "slot_protect_seconds" end to end: a batch slot belongs to the conversation that
last used it, and its next turn finds the chain in that slot instead of a neighbour's prompt.

The real StrataEngine and Service run over HTTP behind the fake engine of test_parallel.py, extended with
"HOLDn": a slot decodes one token and stays in it for n windows (n * 0.02 s), so a turn keeps its slot long
enough to watch the next conversations arrive.  The window that owns a slot has to be the longer of the two - a
window shorter than a turn lets the next conversation in while the first is still decoding, which is legal
behaviour and exercises nothing.  No GPU is needed:

    python -m pytest serve/test_slot_protect.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
import threading
import time
import unittest
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


def session_prompt(n, turn=0, hold=HOLD):
    """A conversation's prompt: a start every later turn carries unchanged, and what this turn adds.

    The start has to run past the 512 tokens the session key is taken from, so the turn is named after the
    padding (a key over the whole prompt is a key over one turn, and the next turn of the same conversation
    would be a new conversation).  Every start is the same length: a shorter prompt is what makes the fake
    engine give way in the middle of reading it, which is a different mechanism.  ByteTokenizer takes a byte
    per token, so 200 "more " words are 1000 tokens of start."""
    return f"S{n} context of this conversation HOLD{hold} " + "more " * 200 + f"turn {turn}"


class OwnershipEndToEnd(unittest.TestCase):
    """Five conversations over three protected slots and one guest slot: the first three keep their slots, the
    rest run only once a slot's conversation has gone idle past the window, and a one-shot call never waits
    behind them.  A 12.7K-token agent conversation had nothing protecting it from this: the eviction lock asks
    16384 tokens and is a weight, not a veto."""

    BATCH, PROTECTED = 4, [0, 1, 2]

    def start(self, ownership_s, slots=None, protected=None):
        import serve.server as server
        self.tmp = tempfile.TemporaryDirectory()
        script = Path(self.tmp.name) / "fake_strata.py"
        script.write_text(FAKE_HOLD, encoding="utf-8")
        self.log = Path(self.tmp.name) / "requests.log"
        # the mock stays on for the whole test: restart() starts the engine again, and an engine that came back
        # as a real binary would fail the one thing a restart has to keep (the settings that ride in spawn)
        real = server.subprocess.Popen
        self.popen = mock.patch.object(server.subprocess, "Popen",
                                       lambda cmd, **kw: real([sys.executable, str(script), *cmd[1:]], **kw))
        self.popen.start()
        extra = ["--batch", str(slots or self.BATCH), "--log", str(self.log)]
        self.engine = StrataEngine("strata", extra, solo_path=False,
                                   protected_slots=self.PROTECTED if protected is None else protected,
                                   slot_protect_seconds=ownership_s)
        tok = ByteTokenizer()
        self.svc = Service(self.engine, tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
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

    def chat(self, text, max_tokens=1, extra=None):
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
        queues for a slot sends nothing, so this count is how many got in."""
        return [int(x.split()[1]) for x in self.log.read_text().splitlines() if x.startswith("BGEN ")]

    def count_at(self, n, timeout):
        """How many admissions there are, once n of them are seen or the timeout passes."""
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
        """n conversations, started in order and named by strata_session (so a test can name the slot each one
        owns without re-deriving the tokenized prompt the server sees).  Every test joins the threads before its
        engine is unloaded: a request still running when the engine goes away ends as an error, not a result."""
        return self.threads([(f"c{k}", session_prompt(k), {"strata_session": f"conv-{k}"}) for k in range(n)],
                            gap=gap)

    def answers(self, out, n):
        self.assertEqual({out[f"c{k}"][0] for k in range(n)}, {ANSWER}, out)

    def owned_slots(self):
        """conversation -> the slot it owns, read from the owner table by the id it names itself by."""
        return {k: self.engine.slot_owner.index(f"s:conv-{k}") for k in range(self.BATCH)
                if f"s:conv-{k}" in self.engine.slot_owner}

    # ---- the mechanism --------------------------------------------------------------------------------

    def test_the_first_conversations_take_the_protected_slots(self):
        self.start(WINDOW)
        out, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        self.assertEqual(sorted(self.admitted()), sorted(self.PROTECTED), self.admitted())
        self.answers(out, 3)

    def test_a_conversation_beyond_the_slots_waits_for_a_window(self):
        """The veto: five conversations, three protected slots.  The last two are admitted only once a window
        passes - the pick queues them rather than landing a second conversation on a slot whose chain belongs to
        a live conversation (which is what evicted it, turn after turn)."""
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3, self.admitted())
        late, extra = self.threads([("c3", session_prompt(3), None), ("c4", session_prompt(4), None)], gap=GAP)
        self.assertEqual(self.count_at(4, timeout=1.0), 3,
                         f"a fourth conversation took a slot a live conversation owns: {self.admitted()}")
        self.engine.slot_protect_seconds = SHORT       # the window passes: the queued conversations go in
        for t in runs + extra:
            t.join(90)
        admitted = self.admitted()
        self.assertEqual(len(admitted), 5, admitted)
        self.assertEqual(sorted(admitted[:3]), sorted(self.PROTECTED), admitted)
        self.assertTrue(all(b in self.PROTECTED for b in admitted[3:]), admitted)
        self.answers(out, 3)
        self.assertEqual({late["c3"][0], late["c4"][0]}, {ANSWER}, late)

    def test_the_window_holds_the_slot_against_the_next_conversation(self):
        """The half the ordering never gave: it preferred to keep a hot slot and took it anyway once every slot
        was locked.  The window keeps it - inside the window a slot is not open to the next conversation at all."""
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3)
        late = {}
        fourth = threading.Thread(target=self.one, args=(late, "c9", session_prompt(9), None))
        fourth.start()
        self.assertEqual(self.count_at(4, timeout=1.5), 3, "a conversation took a slot inside the window")
        self.engine.slot_protect_seconds = SHORT       # the window passes: it goes in
        fourth.join(60)
        self.assertEqual(len(self.admitted()), 4, self.admitted())
        for t in runs:
            t.join(60)
        self.assertEqual(late["c9"][0], ANSWER, late)

    def test_a_turn_returns_to_the_slot_that_holds_its_chain(self):
        self.start(WINDOW)
        out, runs = self.conversations(3)
        for t in runs:
            t.join(60)
        admitted = self.admitted()
        self.assertEqual(sorted(admitted), sorted(self.PROTECTED), admitted)
        mine = self.owned_slots()
        self.assertEqual(len(mine), 3, f"three conversations, {len(mine)} slots owned: {self.engine.slot_owner}")
        for k in sorted(mine):                   # the same conversations, one turn later
            content, _ = self.chat(session_prompt(k, turn=1), extra={"strata_session": f"conv-{k}"})
            self.assertEqual(content, ANSWER)
            self.assertEqual(self.admitted()[-1], mine[k],
                             f"conversation {k}'s second turn left slot {mine[k]}, where its chain is")
        self.assertEqual(self.engine.slot_owner[3], None, "the guest slot took ownership of a conversation")

    def test_a_turn_of_a_conversation_named_by_strata_session_keeps_its_slot(self):
        """The client rewrites its system prompt: the digest of the prompt's start moves, the id the conversation
        names itself by does not.  Naming itself is what keeps such a turn on the slot its chain lives in - an
        unnamed one hashes to a new conversation, reads its whole prompt again, and the slot it lands on loses
        the chain that was there (serve/test_slot_alloc.py shows the hashing side of that)."""
        self.start(WINDOW)
        named = {"strata_session": "task-42"}
        content, _ = self.chat(session_prompt(0), extra=named)
        self.assertEqual(content, ANSWER)
        slot = self.admitted()[0]
        self.assertEqual(self.engine.slot_owner[slot], "s:task-42", self.engine.slot_owner)
        rewritten = session_prompt(0).replace("S0 context of this conversation", "an entirely new system prompt")
        content, _ = self.chat(rewritten, extra=named)
        self.assertEqual(content, ANSWER)
        self.assertEqual(self.admitted()[1], slot, "a rewritten prompt lost the slot its session owns")
        self.assertEqual(len(self.admitted()), 2, self.admitted())

    def test_without_the_window_conversations_share_a_slot(self):
        # the routing as it was: no ownership, so conversations land on whatever slot is free and evict the chain
        # that was there - the same slot serves several of them, which is what the window stops
        self.start(0.0)
        out, runs = self.conversations(5)
        for t in runs:
            t.join(90)
        admitted = self.admitted()
        self.assertEqual(len(admitted), 5, admitted)
        self.assertLess(len(set(admitted)), len(admitted),
                        f"without the window each slot served one conversation only: {admitted}")
        self.answers(out, 5)
        self.assertEqual(set(self.engine.slot_owner), {None}, "ownership off wrote the owner table")

    def test_a_one_shot_call_never_waits_behind_the_conversations(self):
        # leaving one slot out of the partition is what this buys: a classification or a probe is answered at
        # once while every protected slot belongs to a live conversation
        self.start(WINDOW)
        out, runs = self.conversations(3)
        self.assertEqual(self.count_at(3, timeout=8.0), 3)
        content, waited = self.chat("probe what is 2 plus 2", extra={"strata_checkpoint": False})
        self.assertEqual(content, ANSWER)
        self.assertLess(waited, HOLD * 0.02, "a one-shot call queued behind the conversations")
        self.assertEqual(self.admitted()[3], 3, self.admitted())     # the guest slot, not a protected one
        for t in runs:
            t.join(90)
        self.answers(out, 3)

    def test_a_guest_call_never_owns_a_protected_slot(self):
        self.start(WINDOW)
        self.chat("probe one", extra={"strata_checkpoint": False})
        self.chat("probe two", extra={"strata_checkpoint": False})
        self.assertEqual(self.admitted(), [3, 3], self.admitted())
        out, runs = self.conversations(3)
        for t in runs:
            t.join(90)
        self.assertEqual(sorted(self.admitted()[2:]), sorted(self.PROTECTED), self.admitted())
        self.assertEqual(self.engine.slot_owner[3], None, "a one-shot call took ownership of its slot")

    # ---- what a conversation that queues sees ---------------------------------------------------------

    def turn(self, out, key, text, skey):
        """One turn, straight through the engine (no HTTP): a test that watches the queue needs turns it can
        start on its own schedule, and answers it can read as the raw yields (tokens and heartbeats)."""
        cancel = threading.Event()
        try:
            out[key] = list(self.engine.generate_batched(self.svc.tok.encode(text), 1,
                                                        {"strata_session": skey}, cancel))
        except BaseException as e:        # noqa: BLE001
            out[key] = ("error", repr(e))

    def test_a_queued_conversation_gets_heartbeats_while_it_waits(self):
        """The path the window makes real: a conversation beyond the protected slots waits a whole turn (16-21 s
        on the panel's numbers) with the control lines given back, so it yields no heartbeat of its own and the
        queue had none.  Agent clients abort a stream that goes quiet.  #1619's QUEUE_BEAT_S is the keep-alive of
        a request waiting for the single turn - the same situation, so the same constant, shortened here.

        The three turns run to their end first: the control lines carry one prompt read at a time, so a request
        that arrives beside three reads of 1,000 tokens spends the test's first seconds waiting for those lines
        (where _take_control keeps its own beat) rather than queued for a slot."""
        import serve.server as server
        self.start(WINDOW)
        turns = {k: f"Q{k} conversation HOLD{HOLD} pad pad" for k in range(4)}
        out, runs = {}, []
        for k in range(3):
            t = threading.Thread(target=self.turn, args=(out, f"t{k}", turns[k], f"conv-{k}"))
            t.start()
            runs.append(t)
        for t in runs:
            t.join(60)
        self.assertEqual(self.admitted(), [0, 1, 2], self.admitted())
        self.assertEqual(self.engine.slot_busy, [False] * self.BATCH)    # the slots are free and still owned
        queued, cancel = {}, threading.Event()

        def queue_up():
            try:
                queued["yields"] = list(self.engine.generate_batched(
                    self.svc.tok.encode(turns[3]), 1, {"strata_session": "conv-3"}, cancel))
            except BaseException as e:     # noqa: BLE001
                queued["error"] = repr(e)

        with mock.patch.object(server, "QUEUE_BEAT_S", 1.0):
            q = threading.Thread(target=queue_up)
            q.start()
            time.sleep(2.5)                     # the slots are all owned inside the window: it has none
            self.assertEqual(len(self.admitted()), 3, "the queued conversation took a slot it does not own")
            cancel.set()
            q.join(40)
        self.assertNotIn("error", queued, queued)
        beats = [x for x in queued.get("yields", []) if x is None]
        tokens = [x for x in queued.get("yields", []) if x is not None]
        self.assertEqual(tokens, [], "a queued conversation produced tokens before it had a slot")
        self.assertGreaterEqual(len(beats), 2, f"a conversation queued for a slot sent {len(beats)} heartbeats")
        self.assertLess(len(beats), 10, f"{len(beats)} heartbeats for 2.5 s: that is a flood")

    def test_a_queued_stream_gets_sse_keep_alives_over_http(self):
        """The contract a client sees, not the one the generator sees: the heartbeat the queued conversation
        yields has to reach the wire as an SSE comment, because the wait is a turn (16-21 s on the panel's
        numbers) and an agent client aborts a stream that goes quiet.  #1619 sends the same comment to a request
        waiting for the single turn; the slot queue is that request now, so the same QUEUE_BEAT_S drives it."""
        import http.client
        import serve.server as server
        self.start(WINDOW)
        turns = {k: f"Q{k} conversation HOLD{HOLD} pad pad" for k in range(4)}
        out, runs = {}, []
        for k in range(3):
            t = threading.Thread(target=self.turn, args=(out, f"t{k}", turns[k], f"conv-{k}"))
            t.start()
            runs.append(t)
        for t in runs:
            t.join(60)
        self.assertEqual(self.admitted(), [0, 1, 2], self.admitted())
        self.assertEqual(self.engine.slot_busy, [False] * self.BATCH)    # free, and still owned
        host, port = "127.0.0.1", self.httpd.server_address[1]
        with mock.patch.object(server, "QUEUE_BEAT_S", 1.0):
            c = http.client.HTTPConnection(host, port, timeout=60)
            self.addCleanup(c.close)
            body = {"stream": True, "max_tokens": 1, "reasoning_effort": "none",
                    "messages": [{"role": "user", "content": turns[3]}], "strata_session": "conv-3"}
            c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            r = c.getresponse()
            self.assertEqual(r.status, 200)
            raw, beats, t0 = b"", 0, time.monotonic()
            while time.monotonic() - t0 < 4.0:
                ch = r.read(1)
                if not ch:
                    break
                raw += ch
                if raw.endswith(b"\n\n"):
                    beats += raw.startswith(b":")
                    raw = b""
            c.close()
        self.assertEqual(len(self.admitted()), 3, "the queued stream took a slot it does not own")
        self.assertGreaterEqual(beats, 2, f"a queued stream saw {beats} keep-alive comments in 4 s")
        self.assertLess(beats, 15, f"{beats} comments in 4 s: that is a flood")

    # ---- the switch itself ----------------------------------------------------------------------------

    def test_the_window_comes_from_the_config_and_absent_is_off(self):
        # absent: the mechanism does not run at all (the default path stays exactly as it was)
        self.start(None)
        self.assertEqual(self.engine.slot_protect_seconds, 0.0)
        self.assertEqual(self.engine.slot_owner, [None] * self.BATCH)

    def test_a_restart_keeps_the_window_and_the_partition(self):
        # restart() re-runs __init__ with spawn: an engine that comes back without the window routes as though
        # the config had never asked for one, and every conversation silently loses the slot its chain lives in
        self.start(WINDOW)
        self.assertEqual(self.engine.spawn[8], self.PROTECTED)
        self.assertEqual(self.engine.spawn[10], WINDOW)
        self.engine.restart()
        self.assertEqual(self.engine.slot_protect_seconds, WINDOW)
        self.assertEqual(self.engine.slot_owner, [None] * self.BATCH)
        self.assertEqual(self.engine.slot_zone.count("protected"), len(self.PROTECTED), self.engine.slot_zone)
        self.chat(session_prompt(0))        # the restarted engine answers, and answers in a protected slot
        self.assertIn(self.admitted()[0], self.PROTECTED, self.admitted())


if __name__ == "__main__":
    unittest.main()
