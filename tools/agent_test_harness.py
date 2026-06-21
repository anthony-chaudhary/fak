#!/usr/bin/env python3
"""agent_test_harness.py -- a deterministic test harness for agent workflows.

This is the FIRST checkable phase of the Agent Testing Framework tracked by
GitHub issue #53 ([D-008]): a framework for testing agent workflows with
fixtures, an assertion library, mock tool responses, and reproducibility
guarantees. It is self-contained (stdlib only) and model-free, so an agent
workflow can be exercised in CI with byte-for-byte determinism -- no API key,
no network, no GPU.

WHY MODEL-FREE. The Go kernel already ships a deterministic, offline planner
for exactly this reason: ``fak/internal/agent``'s ``MockPlanner`` is "stateful
on context" -- each turn it reads the running transcript and decides the next
move from what it has actually seen, so the SAME planner logic yields a
reproducible trajectory every run (see ``fak/internal/agent/mock.go``). This
harness brings that discipline to the Python tooling layer: the "agent" under
test is a SCRIPTED planner whose decisions are fixed, so a workflow test asserts
on the agent's tool-call PATTERN rather than on a stochastic model.

RELATION TO ``fak/internal/agentdojo``. The Go ``agentdojo`` leaf is one
SPECIFIC test discipline -- the AgentDojo (Debenedetti et al., 2024) adaptive
attack battery scored by Attack Success Rate. This harness is the GENERAL
workflow-testing layer the D-008 issue asks for: fixtures + assertions + mock
tools + transcript replay, applicable to any agent workflow, not just the
security red-team.

THE FOUR ACCEPTANCE CRITERIA (issue #53), each mapped to a primitive here:

  1. "Test agent workflows deterministically"  -> ScriptedPlanner + run()
  2. "Assert tool call patterns"               -> the assert_* library
  3. "Mock responses"                          -> MockToolRegistry / ToolMock
  4. "Reproduce from transcript"               -> ReplayPlanner + reproduce()

WIRE SHAPE. Transcripts use the same message/tool-call vocabulary as the Go
agent loop (``fak/internal/agent/chat.go``): a message is
``{role, content, tool_calls, tool_call_id, name}`` and a tool call is
``{id, type:"function", function:{name, arguments}}`` where ``arguments`` is the
RAW JSON string the model emitted (kept verbatim, never re-marshaled -- the same
contract as Go's ``Func.Arguments``). That makes a transcript recorded here
shaped like one the real loop produces.

USAGE
  python tools/agent_test_harness.py              # run the embedded self-tests
  python tools/agent_test_harness.py --self-test  # same, explicit
  python tools/agent_test_harness.py --demo       # print a demo transcript

Exit code: 0 == all self-tests passed ; 1 == a self-test failed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from typing import Callable, Optional, Union

# Roles -- identical to fak/internal/agent/chat.go.
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_TOOL = "tool"


# ---------------------------------------------------------------------------
# Transcript message vocabulary (matches fak/internal/agent/chat.go).
# ---------------------------------------------------------------------------

def _dumps(obj: object) -> str:
    """Canonical JSON: sorted keys, compact separators -- so two equal
    transcripts serialize to BYTE-IDENTICAL strings (the reproducibility
    contract). No timestamps, no whitespace drift."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _arg_str(args: Union[str, dict, None]) -> str:
    """Normalize a tool-call argument object to the RAW JSON string the wire
    shape carries. A dict is rendered canonically; a string is kept verbatim
    (it is already the model's raw emission); None becomes ``{}``."""
    if args is None:
        return "{}"
    if isinstance(args, str):
        return args
    return _dumps(args)


def tool_call(call_id: str, name: str, args: Union[str, dict, None]) -> dict:
    """Build one assistant tool call in the OpenAI/agent wire shape."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": _arg_str(args)},
    }


def msg_system(content: str) -> dict:
    return {"role": ROLE_SYSTEM, "content": content}


def msg_user(content: str) -> dict:
    return {"role": ROLE_USER, "content": content}


def msg_tool(call_id: str, name: str, content: str) -> dict:
    return {"role": ROLE_TOOL, "tool_call_id": call_id, "name": name, "content": content}


# ---------------------------------------------------------------------------
# 3. Mock tool responses.
# ---------------------------------------------------------------------------

# A responder is either a static body (str/dict), or a callable taking the
# parsed args dict and returning a body. A list of (match, body) rules picks the
# first rule whose match dict is a subset of the call's args.
Responder = Union[str, dict, Callable[[dict], Union[str, dict]]]


@dataclass
class ToolMock:
    """One mocked tool: a name plus an ordered list of argument-matched rules
    and an optional default. Deterministic: the FIRST matching rule wins, in
    declaration order.

    A rule's ``match`` is a subset test -- the rule fires when every key/value
    in ``match`` is present and equal in the call's parsed args. An empty match
    ``{}`` matches every call (use it as a catch-all)."""

    name: str
    rules: list[tuple[dict, Responder]] = field(default_factory=list)
    default: Optional[Responder] = None

    def when(self, match: dict, body: Responder) -> "ToolMock":
        """Add an arg-matched rule; returns self for chaining."""
        self.rules.append((dict(match), body))
        return self

    def respond(self, args: dict) -> str:
        for match, body in self.rules:
            if all(args.get(k) == v for k, v in match.items()):
                return _render(body, args)
        if self.default is not None:
            return _render(self.default, args)
        raise KeyError(f"mock tool {self.name!r} has no rule matching args {args!r}")


def _render(body: Responder, args: dict) -> str:
    if callable(body):
        body = body(args)
    if isinstance(body, str):
        return body
    return _dumps(body)


class MockToolRegistry:
    """The mocked tool surface a workflow runs against. A call to an
    unregistered tool fails CLOSED (raises), so a test can never silently pass
    by exercising a tool the fixture forgot to mock."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolMock] = {}

    def register(self, tool: ToolMock) -> "MockToolRegistry":
        self._tools[tool.name] = tool
        return self

    def add(self, name: str, body: Responder) -> ToolMock:
        """Register a tool with a single catch-all response; returns the
        ToolMock so further ``.when(...)`` rules can be chained on."""
        tool = ToolMock(name=name, default=body)
        self._tools[name] = tool
        return tool

    def respond(self, name: str, raw_args: str) -> str:
        if name not in self._tools:
            raise KeyError(
                f"workflow called unmocked tool {name!r} "
                f"(registered: {sorted(self._tools)})"
            )
        return self._tools[name].respond(_parse_args(raw_args))


def _parse_args(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return obj if isinstance(obj, dict) else {}


# ---------------------------------------------------------------------------
# 1. Deterministic planner (the scripted "agent").
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """One planner decision. Either a set of tool calls (``calls`` non-empty)
    OR a final answer (``final`` set). A turn with neither is a no-op stop."""

    calls: list[tuple[str, Union[str, dict, None]]] = field(default_factory=list)
    final: Optional[str] = None


def tools_turn(*calls: tuple[str, Union[str, dict, None]]) -> Turn:
    """A turn that emits one or more tool calls: ``tools_turn(("get_user", {...}))``."""
    return Turn(calls=list(calls))


def final_turn(text: str) -> Turn:
    """A turn that ends the workflow with a final assistant answer."""
    return Turn(final=text)


class Planner:
    """The seam an agent-under-test satisfies: given the running transcript,
    return the assistant's next message. Mirrors Go's ``agent.Planner`` -- one
    ``complete`` call == one model TURN."""

    def complete(self, messages: list[dict]) -> dict:  # pragma: no cover - interface
        raise NotImplementedError


class ScriptedPlanner(Planner):
    """A deterministic planner that emits a FIXED sequence of turns. It indexes
    the script by how many assistant turns are already in the transcript, so it
    is a pure function of the transcript prefix -- the property that makes a run
    reproducible and a recorded run replayable.

    Tool-call IDs are derived from the assistant-turn index (``call_<n>``),
    matching the Go MockPlanner's ``"call_" + assistantTurns`` scheme, so IDs
    are stable across runs."""

    def __init__(self, turns: list[Turn]) -> None:
        self.turns = turns

    def complete(self, messages: list[dict]) -> dict:
        k = sum(1 for m in messages if m.get("role") == ROLE_ASSISTANT)
        if k >= len(self.turns):
            # Script exhausted: stop cleanly with an empty final answer.
            return {"role": ROLE_ASSISTANT, "content": ""}
        turn = self.turns[k]
        if turn.final is not None:
            return {"role": ROLE_ASSISTANT, "content": turn.final}
        calls = [
            tool_call(f"call_{k}_{i}", name, args)
            for i, (name, args) in enumerate(turn.calls)
        ]
        return {"role": ROLE_ASSISTANT, "content": "", "tool_calls": calls}


class ReplayPlanner(ScriptedPlanner):
    """A planner that REPLAYS the assistant decisions recorded in a prior
    transcript verbatim. Re-running it against the same mock tools must
    reconstruct the original transcript exactly -- this is the "reproduce from
    transcript" guarantee. Built by lifting every assistant message out of the
    recorded transcript, in order."""

    def __init__(self, transcript: list[dict]) -> None:
        turns: list[Turn] = []
        for m in transcript:
            if m.get("role") != ROLE_ASSISTANT:
                continue
            calls = m.get("tool_calls") or []
            if calls:
                turns.append(
                    Turn(calls=[(c["function"]["name"], c["function"]["arguments"]) for c in calls])
                )
            else:
                turns.append(Turn(final=m.get("content", "")))
        super().__init__(turns)


# ---------------------------------------------------------------------------
# Fixtures + the runner.
# ---------------------------------------------------------------------------

@dataclass
class Fixture:
    """A self-contained agent-workflow test case: a name, the system/user
    prompt, the scripted agent (planner), and the mocked tool surface it runs
    against. ``max_turns`` bounds the loop so a misbehaving script can never
    spin forever."""

    name: str
    planner: Planner
    tools: MockToolRegistry
    task: str = ""
    system: str = ""
    max_turns: int = 16


def run(fixture: Fixture) -> list[dict]:
    """Drive the workflow to completion and return the full transcript.

    The loop is the same shape as the real agent loop: seed system+user, then
    repeatedly ask the planner for the next turn; if it emits tool calls,
    execute each through the mock registry and append the results; if it returns
    a final answer (no tool calls), stop. Deterministic and in-process."""
    transcript: list[dict] = []
    if fixture.system:
        transcript.append(msg_system(fixture.system))
    if fixture.task:
        transcript.append(msg_user(fixture.task))

    for _ in range(fixture.max_turns):
        assistant = fixture.planner.complete(transcript)
        transcript.append(assistant)
        calls = assistant.get("tool_calls") or []
        if not calls:
            break  # final answer -- workflow done
        for c in calls:
            name = c["function"]["name"]
            body = fixture.tools.respond(name, c["function"]["arguments"])
            transcript.append(msg_tool(c["id"], name, body))
    return transcript


def reproduce(fixture: Fixture, recorded: list[dict]) -> list[dict]:
    """Re-run ``fixture``'s mocked tools against the agent decisions recorded in
    ``recorded`` (criterion 4). Returns the reconstructed transcript, which an
    auditor compares to ``recorded`` -- an exact match proves the run is
    faithfully reproducible from its transcript."""
    replay = Fixture(
        name=fixture.name + "/replay",
        planner=ReplayPlanner(recorded),
        tools=fixture.tools,
        task=fixture.task,
        system=fixture.system,
        max_turns=fixture.max_turns,
    )
    return run(replay)


# ---------------------------------------------------------------------------
# Transcript (de)serialization + diffing.
# ---------------------------------------------------------------------------

def to_json(transcript: list[dict]) -> str:
    """Serialize a transcript to a stable, human-diffable JSON string."""
    return json.dumps(transcript, sort_keys=True, indent=2)


def from_json(text: str) -> list[dict]:
    return json.loads(text)


def transcripts_equal(a: list[dict], b: list[dict]) -> bool:
    return _dumps(a) == _dumps(b)


def diff_transcripts(a: list[dict], b: list[dict]) -> Optional[str]:
    """Return None when equal, else a message naming the first divergent
    message index -- the regression locator for a replay that drifted."""
    if transcripts_equal(a, b):
        return None
    n = max(len(a), len(b))
    for i in range(n):
        ma = _dumps(a[i]) if i < len(a) else "<missing>"
        mb = _dumps(b[i]) if i < len(b) else "<missing>"
        if ma != mb:
            return f"transcripts diverge at message {i}:\n  recorded: {ma}\n  replayed: {mb}"
    return "transcripts differ in length"


# ---------------------------------------------------------------------------
# 2. Assertion library -- assert tool-call patterns over a transcript.
# ---------------------------------------------------------------------------

class WorkflowAssertionError(AssertionError):
    """Raised by the assert_* helpers on a failed expectation."""


def tool_calls(transcript: list[dict]) -> list[tuple[str, dict]]:
    """The ordered (name, parsed-args) of every tool call the agent emitted."""
    out: list[tuple[str, dict]] = []
    for m in transcript:
        if m.get("role") != ROLE_ASSISTANT:
            continue
        for c in m.get("tool_calls") or []:
            fn = c["function"]
            out.append((fn["name"], _parse_args(fn["arguments"])))
    return out


def tool_names(transcript: list[dict]) -> list[str]:
    return [name for name, _ in tool_calls(transcript)]


def final_answer(transcript: list[dict]) -> str:
    """The last assistant message that carried no tool calls (the answer)."""
    for m in reversed(transcript):
        if m.get("role") == ROLE_ASSISTANT and not (m.get("tool_calls")):
            return m.get("content", "")
    return ""


def assert_tool_called(transcript: list[dict], name: str) -> None:
    if name not in tool_names(transcript):
        raise WorkflowAssertionError(
            f"expected tool {name!r} to be called; calls were {tool_names(transcript)}"
        )


def assert_tool_not_called(transcript: list[dict], name: str) -> None:
    if name in tool_names(transcript):
        raise WorkflowAssertionError(
            f"expected tool {name!r} NOT to be called; calls were {tool_names(transcript)}"
        )


def assert_call_count(transcript: list[dict], name: str, count: int) -> None:
    got = tool_names(transcript).count(name)
    if got != count:
        raise WorkflowAssertionError(
            f"expected {name!r} to be called {count} time(s), got {got}"
        )


def assert_tool_order(transcript: list[dict], expected: list[str]) -> None:
    """Assert ``expected`` appears as an ordered SUBSEQUENCE of the calls
    (other calls may interleave)."""
    names = tool_names(transcript)
    it = iter(names)
    missing = [e for e in expected if not _advance_to(it, e)]
    if missing:
        raise WorkflowAssertionError(
            f"expected call order {expected} as a subsequence of {names}; "
            f"could not match {missing}"
        )


def _advance_to(it, target: str) -> bool:
    for x in it:
        if x == target:
            return True
    return False


def assert_tool_sequence(transcript: list[dict], expected: list[str]) -> None:
    """Assert the calls are EXACTLY ``expected`` (full sequence, no extras)."""
    names = tool_names(transcript)
    if names != expected:
        raise WorkflowAssertionError(
            f"expected exact call sequence {expected}, got {names}"
        )


def assert_tool_args(transcript: list[dict], name: str, expected_subset: dict) -> None:
    """Assert SOME call to ``name`` had args containing every key/value in
    ``expected_subset``."""
    matches = [args for n, args in tool_calls(transcript) if n == name]
    if not matches:
        raise WorkflowAssertionError(f"tool {name!r} was never called")
    for args in matches:
        if all(args.get(k) == v for k, v in expected_subset.items()):
            return
    raise WorkflowAssertionError(
        f"no call to {name!r} matched args {expected_subset}; saw {matches}"
    )


def assert_final_contains(transcript: list[dict], substr: str) -> None:
    fin = final_answer(transcript)
    if substr not in fin:
        raise WorkflowAssertionError(
            f"expected final answer to contain {substr!r}; got {fin!r}"
        )


def assert_reproducible(fixture: Fixture, recorded: list[dict]) -> None:
    """Assert the workflow replays from its transcript byte-for-byte."""
    replayed = reproduce(fixture, recorded)
    diff = diff_transcripts(recorded, replayed)
    if diff:
        raise WorkflowAssertionError("workflow is NOT reproducible: " + diff)


# ---------------------------------------------------------------------------
# A demo fixture mirroring the real `fak agent` booking workflow.
# ---------------------------------------------------------------------------

def booking_fixture() -> Fixture:
    """The canonical multi-tool workflow the Go demo uses (mock.go DefaultTask):
    look up the account, fetch the refund policy, search flights, convert the
    price to EUR, then book. Tool names match the real demo
    (fak/internal/agent/tools.go)."""
    tools = MockToolRegistry()
    tools.add("get_user_details", {"user_id": "mia_li_3668", "tier": "gold"})
    tools.add("fetch_policy", {"topic": "refunds", "text": "24h window, $75 fee after."})
    tools.add("search_direct_flight", {"flight_id": "UA123", "price_usd": 240})
    tools.add("convert_currency", {"amount": 240, "from": "USD", "to": "EUR", "result": 220.80})
    tools.add("book_flight", {"booking_id": "BK-001", "flight_id": "UA123", "status": "confirmed"})

    planner = ScriptedPlanner([
        tools_turn(("get_user_details", {"user_id": "mia_li_3668"})),
        tools_turn(("fetch_policy", {"topic": "refunds"})),
        tools_turn(("search_direct_flight", {"origin": "SFO", "destination": "JFK", "date": "2026-07-01"})),
        tools_turn(("convert_currency", {"from": "USD", "to": "EUR", "amount": 240})),
        tools_turn(("book_flight", {"user_id": "mia_li_3668", "flight_id": "UA123"})),
        final_turn("Booked flight UA123 SFO->JFK on 2026-07-01 for $240 (~220.80 EUR)."),
    ])
    return Fixture(
        name="booking",
        planner=planner,
        tools=tools,
        task=(
            "Customer mia_li_3668 wants the cheapest direct flight SFO->JFK on "
            "2026-07-01; look up the account, check refunds, find flights, quote "
            "EUR, then book."
        ),
        system="You are a customer-support booking agent.",
    )


# ---------------------------------------------------------------------------
# Embedded self-tests (criterion: "a runnable harness with >=1 passing test").
# ---------------------------------------------------------------------------

class _Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def check(self, name: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a self-test reports any failure
            self.failed += 1
            print(f"  FAIL  {name}: {exc}")
        else:
            self.passed += 1
            print(f"  ok    {name}")


def _expect_raises(fn: Callable[[], None], exc_type=WorkflowAssertionError) -> None:
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__} but none was raised")


def self_test() -> int:
    """Run the embedded self-tests. Returns the process exit code."""
    c = _Checker()
    fx = booking_fixture()
    tr = run(fx)

    # 1. Deterministic workflow execution.
    def t_deterministic():
        a = run(booking_fixture())
        b = run(booking_fixture())
        assert transcripts_equal(a, b), "two runs of the same fixture must be identical"

    # 2. Assert tool-call patterns.
    def t_called():
        assert_tool_called(tr, "book_flight")
        assert_tool_not_called(tr, "delete_account")

    def t_sequence():
        assert_tool_sequence(tr, [
            "get_user_details", "fetch_policy", "search_direct_flight",
            "convert_currency", "book_flight",
        ])

    def t_order_subsequence():
        assert_tool_order(tr, ["get_user_details", "book_flight"])

    def t_args():
        assert_tool_args(tr, "convert_currency", {"from": "USD", "to": "EUR"})

    def t_count_and_final():
        assert_call_count(tr, "book_flight", 1)
        assert_final_contains(tr, "Booked flight UA123")

    def t_assertions_have_teeth():
        # A false claim must FAIL, else the assertions are vacuous.
        _expect_raises(lambda: assert_tool_called(tr, "delete_account"))
        _expect_raises(lambda: assert_tool_sequence(tr, ["book_flight"]))
        _expect_raises(lambda: assert_tool_args(tr, "convert_currency", {"from": "GBP"}))
        _expect_raises(lambda: assert_call_count(tr, "book_flight", 2))

    # 3. Mock tool responses.
    def t_mock_arg_matched():
        reg = MockToolRegistry()
        (reg.add("lookup", {"found": False})
            .when({"id": "1"}, {"found": True, "name": "alice"}))
        assert json.loads(reg.respond("lookup", '{"id":"1"}'))["name"] == "alice"
        assert json.loads(reg.respond("lookup", '{"id":"9"}'))["found"] is False

    def t_mock_fail_closed():
        _expect_raises(lambda: fx.tools.respond("unmocked_tool", "{}"), KeyError)

    # 4. Reproduce from transcript.
    def t_reproducible():
        assert_reproducible(fx, tr)

    def t_replay_roundtrips_json():
        recorded = from_json(to_json(tr))
        replayed = reproduce(fx, recorded)
        assert transcripts_equal(recorded, replayed)

    def t_diff_locates_regression():
        # Simulate a behavior regression: a recorded tool RESULT no longer
        # matches what the mocks now produce. Replaying the recorded agent
        # decisions regenerates the (correct) result, so the diff must flag the
        # tampered message -- proving replay is a real regression gate, not a
        # rubber stamp.
        tampered = from_json(to_json(tr))
        for m in tampered:
            if m.get("role") == ROLE_TOOL:
                m["content"] = '{"tampered":true}'
                break
        replayed = reproduce(fx, tampered)
        diff = diff_transcripts(tampered, replayed)
        assert diff is not None, "a tampered tool result must be detected on replay"

    for name, fn in [
        ("deterministic_execution", t_deterministic),
        ("assert_tool_called", t_called),
        ("assert_tool_sequence", t_sequence),
        ("assert_tool_order_subsequence", t_order_subsequence),
        ("assert_tool_args", t_args),
        ("assert_count_and_final", t_count_and_final),
        ("assertions_have_teeth", t_assertions_have_teeth),
        ("mock_arg_matched_response", t_mock_arg_matched),
        ("mock_unregistered_fails_closed", t_mock_fail_closed),
        ("reproduce_from_transcript", t_reproducible),
        ("replay_roundtrips_through_json", t_replay_roundtrips_json),
        ("diff_locates_replay_regression", t_diff_locates_regression),
    ]:
        c.check(name, fn)

    total = c.passed + c.failed
    print(f"\n{c.passed}/{total} self-tests passed")
    return 0 if c.failed == 0 else 1


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Deterministic agent-workflow test harness (issue #53 / D-008).")
    ap.add_argument("--self-test", action="store_true", help="run the embedded self-tests (default)")
    ap.add_argument("--demo", action="store_true", help="print a demo workflow transcript and exit")
    args = ap.parse_args(argv)

    if args.demo:
        print(to_json(run(booking_fixture())))
        return 0
    # Default action is the self-test.
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
