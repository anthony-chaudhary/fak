# fak kernel — deny-as-value: a refusal carries a disposition

**A refusal is not just "no". It carries a derived, loop-consumable *disposition* —
`RETRYABLE` / `WAIT` / `ESCALATE` / `TERMINAL` — that tells the agent loop what to do
next *without* spending another model turn.** A SOTA loop treats "no" as an error to
argue with: it feeds the denial back to the model and fires another turn. A
deny-loopback treats "no" as a structured value — the disposition is a pure function of
the refusal's reason code, so the loop branches on it directly.

```
  agent ──proposes a tool call──▶  fak kernel  (capability floor)
                                       │  adjudicate(policy, call) → DENY + reason
                                       │  Disposition(reason) → one of four classes
                                       ▼
  loop reads the disposition and branches — no model turn:
     RETRYABLE → fix the call and re-submit          (MISROUTE / MALFORMED)
     WAIT      → back off, then retry the same call   (RATE_LIMITED / LEASE_HELD)
     ESCALATE  → stop and hand to a human             (SELF_MODIFY / TRUST_VIOLATION)
     TERMINAL  → abandon this path                    (POLICY_BLOCK / DEFAULT_DENY / …)
```

This demo walks the **full disposition vocabulary** — one `fak preflight` call per class —
so an adopter reading [`POLICY.md`](../../POLICY.md) sees how each refusal reason drives a
concrete loop behavior. (`escalation-demo/` shows the *harness* side of one disposition,
`ESCALATE`; this is the four-way map.)

## Prerequisites

A strict subset of [`adjudication-demo/`](../adjudication-demo/README.md): a built `fak`
binary. **No model, API key, GPU, or network** — every refusal is a pure function of
`(policy, the proposed call)`, so the demo is deterministic and CI-usable.

## Run it

The witness is `fak preflight --explain`: it folds the call through the kernel and prints
the verdict, the reason code, **and the derived disposition** on one line. The
[`policy.json`](policy.json) here maps one demo tool to each representative reason.

```bash
# RETRYABLE — MISROUTE: the call shape is wrong but model-fixable; re-submit a corrected call.
fak preflight --policy examples/deny-as-value/policy.json --tool fix_my_args       --args '{}' --explain

# WAIT — RATE_LIMITED: throttled; back off and retry the SAME call (LEASE_HELD is the other WAIT).
fak preflight --policy examples/deny-as-value/policy.json --tool rate_limited_call --args '{}' --explain

# ESCALATE — SELF_MODIFY: the agent tried to edit its own kernel; hand to a human (TRUST_VIOLATION too).
fak preflight --policy examples/deny-as-value/policy.json --tool write_kernel      --args '{"path":"internal/kernel/kernel.go"}' --explain

# TERMINAL — POLICY_BLOCK on an irreversible action; give up the goal (DEFAULT_DENY etc. are TERMINAL too).
fak preflight --policy examples/deny-as-value/policy.json --tool delete_account    --args '{}' --explain
```

Add `--json` instead of `--explain` to get the machine-readable verdict a loop actually
consumes — a `"disposition": "WAIT"` field alongside the `"reason"`:

```bash
fak preflight --policy examples/deny-as-value/policy.json --tool rate_limited_call --args '{}' --json
```

Full captured run of all four: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What you see

Each refusal prints both the **reason code** and the **derived disposition**:

```
verdict: DENY   reason: MISROUTE       by: monitor   disposition: RETRYABLE
verdict: DENY   reason: RATE_LIMITED   by: monitor   disposition: WAIT
verdict: DENY   reason: SELF_MODIFY    by: monitor   disposition: ESCALATE
verdict: DENY   reason: POLICY_BLOCK   by: monitor   disposition: TERMINAL
```

The `disposition` is not authored in the policy — the policy only names the **reason**.
The kernel *derives* the disposition from the reason's category (`kernel.Disposition`), so
an adopter cannot author a refusal whose disposition contradicts its reason.

## The mapping is closed

Every reason in the closed refusal vocabulary (the 12 core codes in
[`POLICY.md`](../../POLICY.md), defined in `internal/abi/reasons.go`) maps to **exactly one**
disposition. An adopter authoring a policy cannot invent a free-text reason that breaks the
mapping — `fak policy --check` rejects any deny reason outside the vocabulary, so a deny is
always loopback-consumable. The full map (`internal/kernel/kernel.go`, `Disposition`):

| disposition | what the loop does | reason codes that map to it |
|---|---|---|
| **RETRYABLE** | the call is malformed but model-fixable — re-submit a corrected call | `MISROUTE`, `MALFORMED` |
| **WAIT** | transient — back off, then retry the *same* call | `RATE_LIMITED`, `LEASE_HELD` |
| **ESCALATE** | a human must look — do not retry, do not give up silently | `SELF_MODIFY`, `TRUST_VIOLATION` |
| **TERMINAL** | give up this path — retrying cannot help | `DEFAULT_DENY`, `POLICY_BLOCK`, `SECRET_EXFIL`, `UNWITNESSED`, `OVERSIZE`, `UNKNOWN_TOOL`, `RESULT_SECRET_DISCOVERED` |

`TERMINAL` is the **default arm** of the mapping: any reason not explicitly placed in the
first three rows is terminal. That is the safe default — a refusal whose category the loop
does not specifically know how to recover from is one it must not blindly retry.

> **Honest note on the docs.** This table is the *code's* mapping. The issue that asked for
> this demo (#315) illustrates `UNWITNESSED=ESCALATE`; the shipped `Disposition` switch maps
> only `SELF_MODIFY` and `TRUST_VIOLATION` to `ESCALATE`, and `UNWITNESSED` falls to the
> `TERMINAL` default. The witness above is ground truth — this demo documents what the kernel
> actually does, not what a ticket proposed.

## How a loop consumes it

The disposition is the branch key. A deny-loopback reads it from the `--json` verdict (or
the `/v1/fak/adjudicate` wire field of the same name) and acts — with **no model turn**:

```python
v = adjudicate(policy, call)          # one kernel fold; no model
if v["verdict"] != "DENY":
    dispatch(call)
elif v["disposition"] == "RETRYABLE":
    call = repair(call, v["reason"])  # fix the shape, re-submit
elif v["disposition"] == "WAIT":
    backoff(v.get("retry_after"));  retry(call)
elif v["disposition"] == "ESCALATE":
    handoff_to_human(call, v["reason"])
else:  # TERMINAL
    abandon(call, v["reason"])
```

The branch the SOTA loop spends a whole model turn on (re-prompting the model with the
denial text and hoping it reconsiders) collapses into a `dict` lookup. That saved model
turn is exactly what `fak turntax` prices (issue #235): the deny-loopback's structural "no"
versus SOTA's "argue with the no".

## Scope — what this does **not** claim

This demonstrates the **reason→disposition derivation** and how a loop branches on it. It
does **not** prove the denials themselves are evasion-proof (that is
[`adjudication-demo/`](../adjudication-demo/README.md), which drives a real model into the
gate), nor a production escalation backend (that is
[`escalation-demo/`](../escalation-demo/README.md)). The `policy.json` here uses the
name-level `deny` map to pin each demo tool to one representative reason so the four-way map
is legible in one place; a real floor reaches several of these reasons through runtime rungs
(the rate limiter for `RATE_LIMITED`, the self-modify guard for `SELF_MODIFY`, the IFC sink
gate for `TRUST_VIOLATION`) rather than a static label.

## Files

| file | what it is |
|---|---|
| `README.md` | this walkthrough |
| `policy.json` | the demo capability floor — one tool pinned to each representative reason |
| `EXAMPLE-OUTPUT.md` | a captured run of all four `preflight` witnesses |

Related: [`../../POLICY.md`](../../POLICY.md) (the closed refusal vocabulary);
[`../../CLAIMS.md`](../../CLAIMS.md) (the "deny-as-value" claim); issue #235 (`fak turntax`,
the model-turn cost story); `internal/kernel/kernel.go` (`Disposition`, the derivation).
