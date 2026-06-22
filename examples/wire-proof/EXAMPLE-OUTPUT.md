# Captured run — `examples/wire-proof/verify.py`

A real run on a clean checkout (offline mock planner; no model, key, or GPU). The port
is chosen freely each run, so it varies; everything else is deterministic.

```
$ python3 examples/wire-proof/verify.py --no-color
fak — over-the-wire adjudication proof  offline mock planner · no model, key, or GPU
  serving: engine=inkernel planner=mock on http://127.0.0.1:57919

  ✓ A  OpenAI wire carries inline adjudication  get_user_details → ALLOW (admitted=True)
  ✓ B  unsanctioned tool refused by structure  refund_payment → DENY (POLICY_BLOCK/TERMINAL)
  ✓ C  allow-listed tool permitted  search_kb → ALLOW

summary: PASS  ·  the gate adjudicated every proposed call over the wire, with no model, key, or GPU.
  swap the offline mock for your real engine by adding --base-url; nothing else changes.

$ echo $?
0
```

The raw response behind check **A** (abridged) — a standard OpenAI chat completion whose
proposed tool call carries the kernel's verdict in a top-level `fak` block:

```json
{
  "choices": [{ "message": { "tool_calls": [
    { "id": "call_0", "type": "function",
      "function": { "name": "get_user_details", "arguments": "{\"user_id\":\"mia_li_3668\"}" } }
  ] }, "finish_reason": "tool_calls" }],
  "fak": { "adjudications": [
    { "tool_call_id": "call_0", "tool": "get_user_details", "admitted": true,
      "verdict": { "kind": "ALLOW", "by": "monitor" } }
  ] }
}
```
