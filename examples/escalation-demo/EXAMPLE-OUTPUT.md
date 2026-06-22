# Example output

A captured run of `./run.sh` (the kernel enforcing `examples/customer-support-readonly-policy.json`).
The harness proposes a denied call, the kernel refuses it, and the harness escalates the deny
along the policy's declared `safe_sink` — `transfer_to_human_agents` — carrying a redacted
ticket. A `✓` means the step held. Reproduce: `./examples/escalation-demo/run.sh`.

```
fak kernel — human-in-the-loop escalation demo  kernel=http://127.0.0.1:8080  policy=customer-support-readonly-policy.json
  a denied call must not dead-end — it escalates along the policy's declared safe_sink

  ✓ 1 kernel DENIED the call            refund_payment → DENY (POLICY_BLOCK/TERMINAL)
  ✓ 2 harness routed to the safe_sink    transfer_to_human_agents → ALLOW (declared safe_sink + on the allow-list)
  ✓ 3 human queue got a redacted ticket HUMAN-ORD-4471: original=refund_payment reason=POLICY_BLOCK ssn/token=[REDACTED]
  ✓ 4 user-facing reply stays helpful   "I can't issue that refund myself — payment changes are routed to…"

ticket delivered to the human queue:
{
  "ticket_id": "HUMAN-ORD-4471",
  "queue": "transfer_to_human_agents",
  "original_call": {
    "tool": "refund_payment",
    "arguments": {
      "order_id": "ORD-4471",
      "amount_usd": 500,
      "customer_email": "dana@example.com",
      "ssn": "[REDACTED]",
      "token": "[REDACTED]"
    }
  },
  "reason_code": "POLICY_BLOCK",
  "kernel_disposition": "TERMINAL",
  "summary": "Kernel refused a payment-mutating call at the capability boundary; a human teammate must decide."
}

summary: escalation test passed  ·  deny → catch → route → redacted ticket → helpful reply, all 4 steps held
  the load-bearing result: the kernel decided the deny; the harness routed it to the
  policy's DECLARED safe_sink; the same redact_fields scrubbed the escalation payload, so
  a refused call became a graceful human handoff with no secret leaked.
```

Note that the original call carried `ssn` and `token` values; the ticket that reaches the
human queue shows `[REDACTED]` for both. The kernel refuses a *denied* call before decoding
its arguments, so it does **not** redact a denied call's args — the harness does, using the
same `redact_fields` the policy already declares. That is the discipline this demo asserts:
the escalation path obeys the same redaction rule as the call boundary.
