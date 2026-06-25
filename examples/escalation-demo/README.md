# fak kernel — human-in-the-loop escalation demo

**A denied call should not dead-end. It should escalate to a human — along the path the
policy itself declares.** Every shipped policy template carries a `safe_sinks` field; until
now no example exercised it. This is the one that does: it shows what a well-built harness
does *after* the kernel says no.

`adjudication-demo/` proves the deny happens (the call is refused at the boundary, before its
arguments are even decoded). This demo picks up there and shows the **graceful deny** — the
thing that makes the capability lock tolerable in production:

```
  harness ──POST /v1/fak/adjudicate {refund_payment}──▶  fak serve  (the kernel; capability floor)
      │                                                      │  adjudicate(policy, call):
      │   ◀── DENY (POLICY_BLOCK / TERMINAL) ────────────────┘    refund_payment is in the deny-map
      │
      ├─ 1. CATCH the verdict (don't surface a bare "no")
      ├─ 2. ROUTE to the declared safe_sink from the SAME policy  (transfer_to_human_agents)
      ├─ 3. build a structured TICKET: original call · reason code · REDACTED args
      ├──POST /v1/fak/adjudicate {transfer_to_human_agents}──▶ kernel ── ALLOW ──┐
      │       the escalation route is itself adjudicated — it is part of the policy,
      │       not a side-channel (an un-sanctioned "human queue" tool would be denied too)
      ▼
  4. user sees: "I can't refund this myself — I've routed it to a human agent with ticket #…"
```

The four steps are **asserted**, and they gate the exit code (0 iff all four hold).

## Scope — what this does **not** claim

This demo shows the *graceful-deny* harness pattern (catch → route → redact → escalate); it
does **not** claim to be a production escalation system. It does not prove the deny itself
(that is `adjudication-demo/`), does not demonstrate the kernel-side IFC taint exemption (that
is covered in `internal/ifc`), and does not exercise auth, rate-limiting, or a real ticketing
backend — the "human queue" is a stand-in. It is a deterministic, model-free walkthrough of one
discipline: the escalation route is itself adjudicated, so it cannot become a side-channel.

## The discipline: the escalation path is part of the policy

Three rules this demo makes concrete — and the reason it is a runnable artifact and not prose:

1. **The kernel decides what is denied.** `refund_payment` is refused because the policy's
   deny-map says so (`POLICY_BLOCK`). The harness does not get to second-guess that.
2. **The harness routes to the *declared* sink — not an ad-hoc one.** It reads `safe_sinks`
   from the same manifest the kernel is enforcing, and the escalation call is *itself*
   adjudicated: `transfer_to_human_agents` returns `ALLOW` because it is on the allow-list. A
   "human queue" tool that the policy didn't sanction would be denied like any other call.
   The escalation is part of the policy, **not a side-channel around it**.
3. **Redaction applies to the escalation payload too.** A *denied* call is refused before the
   kernel decodes its arguments, so the kernel does **not** redact a denied call's args. The
   harness must — using the same `redact_fields` the policy already declares. In the captured
   run the original call carried `ssn` and `token`; the ticket that reaches the human queue
   shows `[REDACTED]` for both. No secret rides out through the handoff.

## Why no model (and why that is honest)

The deny is a pure function of `(policy, the proposed call)` — it does not depend on *why* the
call was proposed. So this demo drives the kernel's fak-native `/v1/fak/adjudicate` surface
directly (no model, key, or GPU), which keeps it deterministic and CI-usable. It also makes
the redaction step *faithful*: the OpenAI-wire deny refuses before arg-decode and never hands
the arguments back, so the harness that needs the original args to redact-and-escalate is
exactly the "client runs its own tools" path — it holds the call it proposed. `adjudication-demo/`
covers the real-model proposal side; this covers what happens **after** a deny.

## Prerequisites

A strict **subset** of [`adjudication-demo/`](../adjudication-demo/README.md): **Go** (to
build `fak`) and **Python 3** (stdlib only). No model, API key, GPU, or ollama.

## Run it

```bash
./examples/escalation-demo/run.sh
```

`run.sh` tears down everything *it* started. The demo itself **runs in a few seconds**
once `fak` is built — no model, no network. Full captured run: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## What you see

The script prints the refused `refund_payment` adjudication, the policy-declared
`safe_sink`, the redacted escalation ticket, and the second adjudication that allows
`transfer_to_human_agents`. A correct run ends with all four graceful-deny checks
passing: catch the denial, route to the declared sink, redact the sensitive fields, and
adjudicate the escalation path itself instead of using a side channel.

Windows users: run the `.sh` launcher from WSL or Git Bash; the demo itself is
plain `fak serve` plus stdlib Python, and there is no native `.ps1` wrapper yet.

## `safe_sinks` is no longer decorative

The field is declared in every shipped policy template; this demo is the first to exercise it.
Each of these manifests names a `safe_sink` — the tool a denied or tainted flow escalates to
instead of dead-ending:

| policy template | declared `safe_sink` |
|---|---|
| [`customer-support-readonly-policy.json`](../customer-support-readonly-policy.json) *(exercised here)* | `transfer_to_human_agents` |
| [`policy.example.json`](../policy.example.json) | `transfer_to_human_agents` |
| [`flight-booking-agent-policy.json`](../flight-booking-agent-policy.json) | `transfer_to_human_agents` |
| [`healthcare-phi-policy.json`](../healthcare-phi-policy.json) | `transfer_to_human_clinician` |
| [`sql-analyst-policy.json`](../sql-analyst-policy.json) | `transfer_to_analyst_queue` |
| [`devops-dryrun-policy.json`](../devops-dryrun-policy.json) | `create_change_request` |

To adapt this demo to another policy, point `run.sh` at it and the harness escalates to *that*
policy's declared sink:

```bash
FAK_DEMO_POLICY=examples/healthcare-phi-policy.json ./examples/escalation-demo/run.sh
```

> Note: `safe_sinks` has a second, kernel-side role beyond this harness pattern — it also
> tells the result-side IFC layer which egress tools are exempt from taint-gating (a human
> handoff is the safe response to an injection). That exemption is **destination-checked
> first**, so naming a sink does not launder an exfil to an external address. This demo
> exercises the call-side escalation discipline; the IFC behavior is covered in the kernel
> tests (`internal/ifc`).

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build kernel → serve the policy → run the demo → teardown |
| `demo.py` | the escalation harness (stdlib only); asserts all four steps; CI-usable exit code |
| `EXAMPLE-OUTPUT.md` | a captured run |
| `../customer-support-readonly-policy.json` | the policy the kernel enforces — declares the `safe_sink` exercised here |

Related: [`../adjudication-demo/README.md`](../adjudication-demo/README.md) shows the deny side
(a real model proposing calls the kernel refuses); the [`../wire-proof/`](../wire-proof/README.md) demo
proves the same gate over HTTP with no model.
