# fak kernel — wire-side result quarantine demo (`POST /v1/fak/admit`)

**A non-Go client POSTs a tool *result* it produced; the kernel screens that result
server-side before it can re-enter the model's context.** The property this proves:
**the client does not have to be trusted to screen its own output.** The same client
that produced a poisoned result is the one whose result gets walled off — server-side,
with no client-supplied trust.

```
  demo.py ──POST /v1/fak/admit──▶  fak serve  (the kernel; result-side floor)
          (a tool RESULT the            │  AdmitResult(result):
           CLIENT executed)             │    secret-shaped?     → QUARANTINE (page the bytes out)
                                        │    injection-shaped?  → QUARANTINE (page the bytes out)
                                        ▼    otherwise           → ADMIT  (pass through unchanged)
                                   IFC taint ledger raised on the call's trace
```

This drives `fak serve` with **no model, no API key, no GPU** — the offline mock planner.
The result-side admission floor (context-MMU / normgate quarantine + IFC source-stamp) is
armed regardless of whether a model is wired up, because it screens *results*, not calls.

## Where this sits — the three quarantine surfaces

fak runs the **same** result-side containment in three places. This demo is the wire one:

| surface | path | who drives it | demo |
|---|---|---|---|
| **call-side** | `POST /v1/fak/syscall` · `/v1/fak/adjudicate` | gates a *call* before it runs | [`../adjudication-demo/`](../adjudication-demo/README.md) · [`../wire-proof/`](../wire-proof/README.md) |
| **in-process** | the kernel's own Reap / recall path | fak runs the tool itself | issue [#210](https://github.com/anthony-chaudhary/fak/issues/210) |
| **wire-side** | **`POST /v1/fak/admit`** *(this demo)* | a non-Go client POSTs a *result* | **here** |

The call-side gate refuses a *call* before it runs and never sees a result. This demo is
the **result-side** sibling: the call already ran (on the client), and the kernel contains
what it returned. CLAIMS.md **#85** is the shipped capability that exposes the in-process
quarantine **over the wire**, so an adopting client that isn't fak itself gets the same floor.

## Run it

```bash
./examples/wire-quarantine-demo/run.sh             # build kernel, serve (no model), run the demo
./examples/wire-quarantine-demo/run.sh --no-color  # plain output
FAK_BIN=./fak ./examples/wire-quarantine-demo/run.sh   # use a prebuilt binary
```

`run.sh` builds `fak` (or honors a prebuilt `FAK_BIN`), starts `fak serve` on `127.0.0.1:8080`
(override with `FAK_DEMO_PORT`), runs `demo.py` over the wire, and tears down what it started.

> **Determinism.** No model is involved, so there is nothing to vary: the kernel's
> result-side verdicts are **deterministic** and **reproducible** — a pure function of
> the bytes you POST, so the same result always yields the same ALLOW/QUARANTINE on every
> run. The whole demo builds, serves, runs, and tears down in a few seconds.

## What you see

> **Reading the output:** a `✓` means *the verdict matched expectation* — so a `✓` on a
> `QUARANTINE` means the kernel **correctly contained** the poisoned result.

```
  ✓ CLEAN result admitted        read_file → DEFER  content passed through  ifc_taint=trusted
  ✓ SECRET result quarantined    fetch_url → QUARANTINE (SECRET_EXFIL)  paged_out=ng-q1  ifc_taint=quarantined (rose)
  ✓ INJECTION result quarantined fetch_url → QUARANTINE (TRUST_VIOLATION)  paged_out=ng-q2

summary: wire quarantine test passed
```

Full captured run: [`EXAMPLE-OUTPUT.md`](EXAMPLE-OUTPUT.md).

## The three things the demo asserts — all server-side

Each is read from the kernel's own response to `POST /v1/fak/admit`; the client decides
nothing:

1. **`verdict.kind`** — `QUARANTINE` for a poisoned result; `DEFER` (admitted, no objection)
   for a clean one. *Note:* the result-side admitter chain returns `DEFER` — not `ALLOW` —
   for a clean result, because it never raises an objection rather than affirmatively
   permitting; the contrast that matters is **admitted-and-intact vs quarantined**.
2. **The paged-out pointer (`result.meta.quarantine_id`)** — the offending bytes are held
   in a side ledger and the in-context `result.content` is replaced with an opaque
   `{"_quarantined":true,…}` stub. The `sk-live…` secret and the injection string **never
   appear** in the content the model would read.
3. **The IFC source-stamp (`result.meta.ifc_taint`)** — a clean trusted-local read stays
   `trusted`; an untrusted-source read that is quarantined is stamped `quarantined`, which
   is the kernel raising the **trace's IFC taint ledger**. That high-water mark is what the
   already-wired sink-gate reads later: an exfil call on a tainted session is refused.

## Why this is the load-bearing property

A client screening its **own** output is not a security boundary — a compromised or
confused client can simply not screen. The point of `/v1/fak/admit` is that the screen
runs **on the server**, on the path the result must cross to reach the model. So:

- the secret-bearing result is paged out **before** its bytes enter model-visible context;
- the session that produced it is marked tainted, **independent of what the client claims**;
- and none of it required the client to be trusted, instrumented, or even cooperative.

## This demo's scope

It exercises exactly the **result-side containment** layer over the wire. It does **not**
demonstrate the call-side capability gate (see [`../adjudication-demo/`](../adjudication-demo/README.md)
and [`../wire-proof/`](../wire-proof/README.md)), and it makes **no claim** that the
content screen is a complete poison *detector* — detection is heuristic and deliberately
non-load-bearing (see [`../../CLAIMS.md`](../../CLAIMS.md) and the repo `README.md`). The
load-bearing guarantee here is structural: a flagged result is **held out of context** and
the trace is stamped, server-side, on a wire any client can speak.

## Files

| file | what it is |
|---|---|
| `run.sh` | one-command launcher: build kernel → serve (no model) → run the demo → teardown |
| `demo.py` | the demo itself (stdlib-only Python client, POSTs to `/v1/fak/admit`); CI-usable exit code |
| `EXAMPLE-OUTPUT.md` | a captured run, including the raw wire JSON |

Related: [`../../CLAIMS.md`](../../CLAIMS.md) #85 (the shipped capability), issue
[#210](https://github.com/anthony-chaudhary/fak/issues/210) (the in-process sibling),
[`../../internal/gateway/admit_test.go`](../../internal/gateway/admit_test.go) (the Go
witnesses behind this endpoint).
