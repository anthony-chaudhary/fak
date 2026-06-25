#!/usr/bin/env python3
"""
Tiny adapter from fak's native HTTP verdicts to OpenAI Agents SDK tool-guardrail
concepts.

The module intentionally depends only on the Python standard library so the example
can run without an OpenAI API key or the Agents SDK installed. In a real Agents SDK
app, call these helpers from a ToolInputGuardrail and ToolOutputGuardrail function,
then translate `GuardrailDecision.behavior` into the SDK behavior:

  allow          -> allow normal tool execution / result use
  reject_content -> reject the call or output but continue the run with a message
  raise_exception -> halt or route to approval/witness handling

If the Agents SDK tracing package is installed, `emit_agents_trace_if_available`
also records a custom span carrying the fak verdict metadata. Otherwise it returns
False and the caller can log the same metadata directly.
"""
from __future__ import annotations

import importlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class GuardrailDecision:
    phase: str
    behavior: str
    verdict: dict[str, Any]
    trace_id: str = ""
    message: str = ""
    output_info: dict[str, Any] = field(default_factory=dict)
    repaired_arguments: dict[str, Any] | None = None
    model_visible_output: Any = None

    def trace_metadata(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "behavior": self.behavior,
            "trace_id": self.trace_id,
            "fak_verdict": self.verdict,
            "message": self.message,
            "output_info": self.output_info,
        }


class FakGuardrailClient:
    def __init__(self, base_url: str, bearer_token: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.bearer_token = bearer_token

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.bearer_token:
            headers["Authorization"] = "Bearer " + self.bearer_token
        req = urllib.request.Request(self.base_url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"fak {path} HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(f"could not reach fak at {self.base_url}: {exc}") from exc

    def adjudicate(
        self,
        tool: str,
        arguments: dict[str, Any] | str | None = None,
        *,
        read_only: bool = False,
        trace_id: str = "",
        witness: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "tool": tool,
            "arguments": {} if arguments is None else arguments,
            "read_only": read_only,
        }
        if trace_id:
            payload["trace_id"] = trace_id
        if witness:
            payload["witness"] = witness
        return self._post("/v1/fak/adjudicate", payload)

    def admit(
        self,
        tool: str,
        result: Any,
        *,
        trace_id: str = "",
        witness: str = "",
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"tool": tool, "result": result}
        if trace_id:
            payload["trace_id"] = trace_id
        if witness:
            payload["witness"] = witness
        return self._post("/v1/fak/admit", payload)


def input_guardrail_decision(response: dict[str, Any]) -> GuardrailDecision:
    verdict = response.get("verdict") or {}
    kind = verdict.get("kind")
    reason = verdict.get("reason", "")
    disposition = verdict.get("disposition", "")
    trace_id = response.get("trace_id", "")
    info = {"kind": kind, "reason": reason, "disposition": disposition}

    if kind == "ALLOW":
        return GuardrailDecision(
            phase="tool_input",
            behavior="allow",
            verdict=verdict,
            trace_id=trace_id,
            output_info=info,
        )
    if kind == "TRANSFORM":
        repaired = response.get("repaired_arguments") or {}
        return GuardrailDecision(
            phase="tool_input",
            behavior="allow",
            verdict=verdict,
            trace_id=trace_id,
            message="fak repaired malformed tool arguments",
            output_info={**info, "transformed": True},
            repaired_arguments=repaired,
        )
    if kind == "REQUIRE_WITNESS":
        return GuardrailDecision(
            phase="tool_input",
            behavior="raise_exception",
            verdict=verdict,
            trace_id=trace_id,
            message="fak requires an external witness before this tool can run",
            output_info=info,
        )

    return GuardrailDecision(
        phase="tool_input",
        behavior="reject_content",
        verdict=verdict,
        trace_id=trace_id,
        message=f"fak denied tool call: {reason or kind} ({disposition or 'no-disposition'})",
        output_info=info,
        model_visible_output=f"[fak] tool call denied: {reason or kind}",
    )


def output_guardrail_decision(response: dict[str, Any]) -> GuardrailDecision:
    verdict = response.get("verdict") or {}
    kind = verdict.get("kind")
    reason = verdict.get("reason", "")
    disposition = verdict.get("disposition", "")
    trace_id = response.get("trace_id", "")
    result = response.get("result") or {}
    info = {
        "kind": kind,
        "reason": reason,
        "disposition": disposition,
        "admit": (result.get("meta") or {}).get("admit", ""),
    }

    if kind in ("ALLOW", "DEFER"):
        return GuardrailDecision(
            phase="tool_output",
            behavior="allow",
            verdict=verdict,
            trace_id=trace_id,
            output_info=info,
            model_visible_output=result.get("content"),
        )
    if kind == "QUARANTINE":
        return GuardrailDecision(
            phase="tool_output",
            behavior="reject_content",
            verdict=verdict,
            trace_id=trace_id,
            message=f"fak quarantined tool result: {reason or 'quarantined'}",
            output_info=info,
            model_visible_output=f"[fak] tool result quarantined: {reason or 'quarantined'}",
        )

    return GuardrailDecision(
        phase="tool_output",
        behavior="raise_exception",
        verdict=verdict,
        trace_id=trace_id,
        message=f"fak refused tool result: {reason or kind}",
        output_info=info,
    )


def guarded_call(
    client: FakGuardrailClient,
    tool: str,
    arguments: dict[str, Any],
    tool_fn: Callable[..., Any],
    *,
    trace_id: str = "",
    witness: str = "",
    read_only: bool = False,
) -> dict[str, Any]:
    before = input_guardrail_decision(
        client.adjudicate(
            tool,
            arguments,
            read_only=read_only,
            trace_id=trace_id,
            witness=witness,
        )
    )
    if before.behavior != "allow":
        return {"input": before, "output": None, "tool_executed": False, "tool_result": None}

    call_args = before.repaired_arguments or arguments
    tool_result = tool_fn(**call_args)
    after = output_guardrail_decision(
        client.admit(tool, tool_result, trace_id=trace_id, witness=witness)
    )
    return {
        "input": before,
        "output": after,
        "tool_executed": True,
        "tool_result": tool_result,
    }


def emit_agents_trace_if_available(name: str, metadata: dict[str, Any]) -> bool:
    """Best-effort custom span emission for real Agents SDK runs.

    The example does not require the SDK. When it is installed, current Agents SDK docs
    expose a `custom_span()` helper for custom trace data; this function tries both
    documented import locations and returns False if neither is available.
    """
    for module_name in ("agents", "agents.tracing"):
        try:
            module = importlib.import_module(module_name)
            custom_span = getattr(module, "custom_span")
        except (ImportError, AttributeError):
            continue
        try:
            with custom_span(name, data=metadata):
                pass
            return True
        except TypeError:
            with custom_span(name):
                pass
            return True
    return False
