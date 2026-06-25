# fak + OpenAI Agents SDK tool guardrail adapter

This example is the OpenAI-side adoption bridge from
[`docs/VENDOR-ADOPTION-SCORECARD.md`](../../docs/VENDOR-ADOPTION-SCORECARD.md):
call `fak_adjudicate` before a tool runs, call `fak_admit` after the tool returns,
and translate the `fak` verdict into the OpenAI Agents SDK's tool-guardrail
behaviors.

It is deliberately dependency-free. The runnable proof does not import the Agents SDK,
does not call OpenAI, and needs no model, key, network, or GPU. In a real Agents SDK
app, the same mapping belongs inside a tool input guardrail and tool output guardrail.

## Run it

From the repository root:

```bash
python examples/openai-agents-guardrail/demo.py
```

Expected shape:

```text
input guardrail blocks git_push: behavior=reject_content verdict=DENY reason=POLICY_BLOCK executed=false
input guardrail allows git_status: behavior=allow verdict=ALLOW reason= executed=true
output guardrail admits git_status result: behavior=allow verdict=DEFER reason=
output guardrail quarantines web_fetch result: behavior=reject_content verdict=QUARANTINE reason=SECRET_EXFIL
summary: PASS
```

Captured output is in [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Mapping

The OpenAI Agents SDK docs describe tool guardrails as checks that wrap function tools
before and after execution, with behaviors that allow normal execution, reject content
while continuing, or raise a tripwire. `guarded_tool.py` maps `fak` verdicts onto that
shape:

| fak phase | fak verdict | Agents SDK behavior | Meaning |
|---|---|---|---|
| Tool input | `ALLOW` | `allow` | Run the tool as proposed. |
| Tool input | `TRANSFORM` | `allow` with `repaired_arguments` | Run the canonical arguments returned by the kernel. |
| Tool input | `DENY` | `reject_content` | Do not run the tool; continue the agent run with a safe refusal message. |
| Tool input | `REQUIRE_WITNESS` | `raise_exception` | Route to approval, witness collection, or a human-in-the-loop path. |
| Tool output | `ALLOW` / `DEFER` | `allow` | Let the result enter model-visible context. |
| Tool output | `QUARANTINE` | `reject_content` | Replace the result with a safe message; preserve the held bytes in fak's audit path. |
| Tool output | `DENY` | `raise_exception` | Halt or escalate the unsafe result. |

The `GuardrailDecision.trace_metadata()` payload contains `trace_id`, verdict kind,
reason, disposition, and behavior. Pass that payload into an Agents SDK custom span or
attach it as tool-span metadata. If `agents.tracing.custom_span` is importable,
`emit_agents_trace_if_available()` can emit the custom span directly.

## Where it plugs into a real Agents SDK app

Use the SDK's current guardrail APIs for the exact decorator/class names in your
version. The integration pattern is stable:

```python
from guarded_tool import (
    FakGuardrailClient,
    input_guardrail_decision,
    output_guardrail_decision,
)

fak = FakGuardrailClient("http://127.0.0.1:8080")


async def tool_input_guardrail(tool_name: str, arguments: dict):
    response = fak.adjudicate(tool_name, arguments)
    decision = input_guardrail_decision(response)
    return decision  # translate behavior to the SDK's ToolGuardrailFunctionOutput


async def tool_output_guardrail(tool_name: str, result):
    response = fak.admit(tool_name, result)
    decision = output_guardrail_decision(response)
    return decision  # translate behavior to allow / reject_content / raise_exception
```

The demo keeps the last translation step as a plain dataclass so it can run on a
machine that has not installed `openai-agents`.

## Sources

- OpenAI Agents SDK: tool guardrails, MCP integration, and tracing:
  <https://openai.github.io/openai-agents-python/guardrails/>,
  <https://openai.github.io/openai-agents-python/mcp/>,
  <https://openai.github.io/openai-agents-python/tracing/>.
- Codex manual, current as fetched by `openai-docs` on 2026-06-25: Codex supports MCP
  server configuration in CLI and IDE, and Codex can also run as an MCP server for
  Agents SDK workflows.
- fak native wire reference: [`docs/mcp-tool-result.md`](../../docs/mcp-tool-result.md)
  and [`internal/gateway/wire.go`](../../internal/gateway/wire.go).
