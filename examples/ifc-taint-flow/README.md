# fak kernel — IFC taint-flow / provenance demo

**An untrusted source stamps its bytes; a later call that would flow those bytes into a
privileged sink is refused at adjudication time — `DENY TRUST_VIOLATION`, rank-30, before
the call runs.** Not because a detector recognized an attack, but because the taint ledger
says *these bytes came from an untrusted source, you can't put them there.* This is the
**taint-tracking floor**: the same structural-floor logic as the capability gate, applied
to *data flow*.

```
  demo.py                                  fak serve  (the kernel; IFC floor)
    │                                          │
    ├─POST /v1/fak/admit  fetch_url ──────────▶│  StampGate (data plane):
    │   (an untrusted-source RESULT)           │    source=untrusted → ifc_taint="tainted"
    │                                          │    + raise this session's taint high-water mark
    │                                          │
    ├─POST /v1/fak/admit  read_corp_kb ───────▶│  StampGate: source=trusted_local
    │   (a trusted-local RESULT, fresh trace)  │    → ifc_taint="trusted"  (mark stays clean)
    │                                          │
    └─POST /v1/fak/adjudicate send_email ─────▶│  SinkGate (control plane, rank 30):
        (the SAME call on each trace)          │    EGRESS sink + tainted session → DENY TRUST_VIOLATION
                                               │    EGRESS sink + clean   session → DEFER → ALLOW
```

The `sources` field is declared in every policy template (`"fetch_url": "untrusted"`,
`"read_corp_kb": "trusted_local"`, …). This demo is the **walkthrough of what those
declarations do**: an untrusted result tries to flow into a sink, and the flow is gated.

## The load-bearing witness: same call, opposite verdict

The whole point is in two adjudications of the **identical** call:

| trace | session provenance | `send_email(to=partner@external.example.com)` |
|---|---|---|
| `ifc-demo-tainted` | read an **untrusted** `fetch_url` result | **DENY `TRUST_VIOLATION`** (`by: ifc-sink`) |
| `ifc-demo-clean` | read only a **trusted-local** `read_corp_kb` result | **ALLOW** |

Same tool, same arguments. Nothing inspected the `send_email` for badness. The only
variable is the taint of the bytes already in the session — so the refusal is a property
of the *data path*, not a judgment call about the call. That is what "structural, not
detection" means here.

## Why this is structural (provenance), not heuristic (detection)

Every content detector is sound-but-evadable: a semantic paraphrase with no marker word
walks straight through a lexical screen. IFC keys on **provenance** instead, which a
paraphrase cannot launder:

- **Structural (load-bearing, gates the exit code):** once an untrusted source has entered
  a session, the session's taint high-water mark is raised, and the rank-30 sink-gate
  refuses an egress / destructive sink on that session — regardless of *how* the would-be
  exfiltration is phrased. A successful injection's payload is "send the data to
  attacker.example.com"; this makes that egress impossible once untrusted content is in
  flight. The injection text can still be in context; it just can't *act*.
- **Heuristic (NOT load-bearing):** the *source labeling* itself — deciding a given tool's
  channel is `untrusted` vs `trusted_local` — is best-effort, host-authored configuration.
  The demo does not claim the labeler is complete. It claims that **once a label exists,
  the flow rule over it is structural.**

## Kernel-authored trust — the model can't self-assert

Authorship of trust belongs to the kernel, never to the model. The model emits the
`ToolCall` (its tool, its args, its open `Meta` map), so any trust signal carried *inside
the call* is a self-report an injected agent can forge. fak's `internal/provenance` derives
the taint label from **two kernel-controlled facts only** — the kernel-stamped result state
and the tool's **host-registered** source class — and ignores `ToolCall.Meta` entirely. A
poisoned `fetch_url` cannot mint itself `trusted` to skip the session taint; the legacy
`Meta["provenance"]="trusted_local"` self-tag is recognized solely to *surface the forgery
attempt* for an auditor, never to change a verdict.

## Honest scope

- The **flow rule** (tainted → gated sink) is the structural part. The **source labeling**
  is best-effort: a tool the host never registered defaults to `untrusted` (fail-closed),
  which is sound but coarse — a legitimate egress after reading *any* untrusted page is
  blocked until an explicit authorization (`safe_sinks` / a policy `Authorize` escape)
  relieves it. That false-positive is the price of having *no false negatives on the exfil
  channel* — the property a buyer underwrites.
- This demo gates `send_email` (an **EGRESS** sink) — gated by default. The **EXEC** sink
  (shell/code) is *not* gated on session taint by default, because that denies normal `Bash`
  after any untrusted read; declaring `sources` in a policy opts into the strict set that
  gates EXEC too (`internal/ifc` `StrictGatedSinks`). See the policy comment in
  [`research-sink-policy.json`](research-sink-policy.json).

## Prerequisites

- **Go** (to build `fak`) **or** a prebuilt binary via `FAK_BIN`, and **Python 3** (stdlib only).
- **No model.** The IFC floor is a pure function of the bytes' provenance, so the
  load-bearing verdict fires with the deterministic kernel alone.

## Run it

```bash
./examples/ifc-taint-flow/run.sh              # build, serve (with the demo policy), run, teardown
./examples/ifc-taint-flow/run.sh --no-color   # plain output
FAK_BIN=./fak ./examples/ifc-taint-flow/run.sh  # use a prebuilt binary
```

`run.sh` tears down everything *it* started. Windows users: run the `.sh` launcher from
WSL or Git Bash; the demo itself is plain `fak serve` plus stdlib Python.

> **Determinism.** The kernel's verdict is a pure function of the source class and the
> session's taint ledger, so the same sequence always yields the same stamp / refusal. No
> model is in the loop to vary.

## What you see

> **Reading the output:** a `✓` means *the kernel verdict matched expectation* — a `✓` on
> the refusal row means the kernel **correctly refused** the tainted → sink flow.

```
  ✓ untrusted source admitted + stamped     fetch_url → ifc_taint=tainted  (source=untrusted)
  ✓ trusted-local source admitted + stamped read_corp_kb → ifc_taint=trusted (source=trusted_local)
    session marks: ifc-demo-tainted=tainted (dangerous)   ifc-demo-clean=trusted (clean)
  ✓ tainted → sink REFUSED at adjudication  send_email → DENY TRUST_VIOLATION (by ifc-sink); rank-30, pre-call
  ✓ clean → SAME sink ALLOWED               send_email → ALLOW (identical call & args; only the taint differs)

summary: taint-flow test passed
```

Full captured run, including the raw wire JSON for each step:
[`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## Why the exit code gates on provenance, not the call's content

A *model* declining to forward untrusted data (its safety training) is **not** the
guarantee — it depends on the model. The guarantee is that the egress is **barred at the
kernel** once untrusted content has touched the session. So the run fails only if the
kernel let the tainted flow through (no `TRUST_VIOLATION`), or wrongly refused the clean
session's identical call. Same discipline as the sibling demos: an honest test of **fak**,
not of a model's manners.

## This demo's place in the layered model

fak's three security layers are kept separate on purpose. This demo is the **data-flow**
gate, distinct from the call-side capability gate and the result-side containment:

1. **Capability gate** *(call-side, structural)* — refuse a *call* because the tool is not
   sanctioned. Demo: [`../adjudication-demo/`](../adjudication-demo/README.md).
2. **Containment** *(result-side, structural)* — hold a flagged *result*'s bytes out of the
   model's context. Demo: [`../quarantine-demo/`](../quarantine-demo/README.md).
3. **IFC taint flow** *(this demo — data-flow, structural)* — refuse a *call into a sink*
   because the bytes in flight came from an untrusted source. Complements containment: even
   if an injection's text is in context, it cannot exfiltrate.

These compose — a quarantined result also raises the IFC taint mark, so the same sink-gate
that this demo drives directly is what bars a later exfil after a containment event.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build kernel → serve with the demo policy → run the demo → teardown |
| `demo.py` | the demo itself (fak-native HTTP client, stdlib only); CI-usable exit code |
| `research-sink-policy.json` | the capability floor: ALLOW-lists `send_email` (so only the IFC flow rule gates it) and declares the source taint of `fetch_url` / `read_corp_kb` |
| `EXAMPLE-OUTPUT.md` | a captured run, including the raw wire JSON for every step |

Related: [`../../CLAIMS.md`](../../CLAIMS.md) (#72 information-flow control, #73
kernel-authored trust/provenance), the closed refusal vocabulary's `TRUST_VIOLATION`
([`../../internal/abi/reasons.go`](../../internal/abi/reasons.go)), and the Go witnesses
behind the floor: [`../../internal/ifc/`](../../internal/ifc/),
[`../../internal/provenance/`](../../internal/provenance/).
