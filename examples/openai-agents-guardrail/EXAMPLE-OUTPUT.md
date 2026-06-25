# Example output

Captured with:

```bash
python examples/openai-agents-guardrail/demo.py
```

Output shape:

```text
fak + OpenAI Agents SDK guardrail adapter demo
  kernel=http://127.0.0.1:<port> policy=examples\dev-agent-policy.json
  input guardrail blocks git_push: behavior=reject_content verdict=DENY reason=POLICY_BLOCK executed=false
    trace_metadata={"behavior": "reject_content", "fak_verdict": {"by": "monitor", "disposition": "TERMINAL", "kind": "DENY", "reason": "POLICY_BLOCK"}, "message": "fak denied tool call: POLICY_BLOCK (TERMINAL)", "output_info": {"disposition": "TERMINAL", "kind": "DENY", "reason": "POLICY_BLOCK"}, "phase": "tool_input", "trace_id": "agents-deny-1"}
  input guardrail allows git_status: behavior=allow verdict=ALLOW reason= executed=true
    trace_metadata={"behavior": "allow", "fak_verdict": {"by": "monitor", "kind": "ALLOW"}, "message": "", "output_info": {"disposition": "", "kind": "ALLOW", "reason": ""}, "phase": "tool_input", "trace_id": "agents-allow-1"}
  output guardrail admits git_status result: behavior=allow verdict=DEFER reason=
    trace_metadata={"behavior": "allow", "fak_verdict": {"by": "normgate", "kind": "DEFER"}, "message": "", "output_info": {"admit": "", "disposition": "", "kind": "DEFER", "reason": ""}, "phase": "tool_output", "trace_id": "agents-allow-1"}
  output guardrail quarantines web_fetch result: behavior=reject_content verdict=QUARANTINE reason=SECRET_EXFIL
    trace_metadata={"behavior": "reject_content", "fak_verdict": {"by": "normgate", "kind": "QUARANTINE", "reason": "SECRET_EXFIL"}, "message": "fak quarantined tool result: SECRET_EXFIL", "output_info": {"admit": "quarantined", "disposition": "", "kind": "QUARANTINE", "reason": "SECRET_EXFIL"}, "phase": "tool_output", "trace_id": "agents-quarantine-1"}
  agents_sdk_custom_span_emitted=false
summary: PASS - denied call did not run, clean call ran, poisoned result was quarantined
```
