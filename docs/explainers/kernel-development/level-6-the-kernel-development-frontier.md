---
title: "Level 6 — The kernel-development frontier: what is still open"
description: "Rung 6 of the kernel-development ladder: the honest edge. Which seams are frozen and which are not, which gates are blocking and which are advisory, and what the repo does NOT yet prove about kernel development — read against the SHIPPED / SIMULATED / STUB honesty ledger."
slug: kernel-development-level-6
keywords:
  - kernel development frontier
  - honesty ledger
  - SHIPPED SIMULATED STUB
  - frozen ABI
  - advisory vs blocking gate
  - open seams
  - what fak does not prove
  - kernel-development ladder
date: 2026-11-18
---

# Level 6 — The kernel-development frontier: what is still open

*You are on **Level 6 of 6** of the [kernel-development ladder](README.md).*

> **What this is.** The last rung is not a new mechanism — it is the **honest edge**.
> Rungs 1–5 taught you the three gates and the two kernels. This rung teaches you to
> read the repo's own honesty ledger, so you can tell a *frozen* seam from a *moving*
> one, a *blocking* gate from an *advisory* one, and a *shipped* claim from a *stub*.
> The skill is not "know everything"; it is "know what is proven, and say so."

## Questions this page answers

- Which seams are frozen, and which are still moving?
- Which gates actually block a change, and which only advise?
- What does the repo *not* yet prove about kernel development?
- What do the `SHIPPED` / `SIMULATED` / `STUB` tags mean, and how do I read one?
- What would move a claim up a tag?
- Where do I go to actually build something?

## The whole ladder in one picture

```
   L1  What is kernel development?      ── the three senses of "kernel"
        │                                   (OS-metaphor / compute / CUDA)
        ▼
   L2  Where does my code attach?       ── the Register* seam; architest layering gate
        │                                   (Gate 1: plug in, don't fork)
        ▼
   L3  Does it preserve behavior?       ── the correctness class + a deterministic witness
        │                                   (Gate 2: prove correct)
        ▼
   L4  Is it actually a win?            ── the non-forgeable keep-bit
        │                                   (Gate 3: prove faster)
        ▼
   L5  The other kernel                 ── the adjudicator rung; the verdict fold
        │                                   (the same three gates, security-shaped)
        ▼
   L6  The frontier                     ── what is frozen, what is open, what is unproven
                                            (this page: read the ledger honestly)
```

Every rung below this one ended with a checkpoint you could *answer*. This rung ends with
a question you can only *scope*: what is the edge of what is proven?

## Which seams are frozen, and which are still moving?

A **frozen** seam is one you can build against without fear: the ABI will not move under
you. A **moving** seam is one where the interface is real but the implementation behind it
is still being built. The distinction is the whole reason `EXTENDING.md` exists.

| Seam | Status | What that means for you |
|---|---|---|
| `internal/abi` (the spine) | **frozen** | the registration table, verdict kinds, reason codes — never edited after freeze |
| `compute.Register` (the HAL) | **frozen** | a new backend is a new file + one `init()`; the registry picks which runs |
| `abi.RegisterAdjudicator` | **frozen** | a new rung is one call; the fold orders it |
| `abi.RegisterResultAdmitter` | **frozen** | the write-time result-admission seam (the context-MMU) |
| `abi.RegisterVerdictKind` | **frozen** | a new verdict kind for `k > 1023`, no core edit |
| `abi.RegisterRegionBackend` / `RegisterPageOutBackend` | **frozen** | the cross-engine KV co-residence ABI |
| the *transport* behind `RegisterRegionBackend` | **moving** | the ABI is frozen; the CUDA-IPC import of a real engine's KV region is a backend plug-in still to build |
| the fine-tuned adjudication model | **moving** | the fused model is a *stock* reference forward pass; a tuned head is unbuilt |

The rule of thumb: **the seam is frozen, the backend behind it may not be.** You can
always write a backend against a frozen ABI; you cannot assume the backend that ships
today is the one that ships next quarter.

```
   your code ──►  ┌──────────────────────────────┐   FROZEN: never moves under you
                  │  the seam (Register* / ABI)  │   ── you can build against it today
                  └──────────────┬───────────────┘
                                 │
                  ┌──────────────▼───────────────┐   MOVING: real interface, backend
                  │  the backend behind the seam │   ── still being built; do not
                  │  (transport / tuned model)   │      assume today's ships next quarter
                  └──────────────────────────────┘
```

## Which gates block, and which only advise?

Not every check stops a change. Some are **blocking** (a red gate fails the build or the
land), and some are **advisory** (they report, and a human decides). Knowing which is
which is how you avoid both false confidence and false alarm.

| Gate | Kind | What it does |
|---|---|---|
| `internal/architest` layering gate | **blocking** | fails the build on an upward/cross-tier import |
| the deterministic witness (`go test`) | **blocking** | a kernel whose witness is red cannot land |
| the keep-bit (`shipgate.Evaluate` via `cmd/rsicycle`) | **blocking** | KEEP only on a measured strict gain *and* green *and* clean |
| the claims lint (`TestCLAIMSLedger`) | **blocking** | every `- [` line needs exactly one tag, and an exposure marker on every gated `SHIPPED` line |
| the `fak preflight` fold | **advisory at authoring time** | prints a verdict; it does not itself stop a commit |
| the capability floor at runtime | **blocking at runtime** | a `DENY` never reaches dispatch |
| the detector feeding the result-side gate | **advisory** | it flags; the *decision* to quarantine is structural |

The subtlety: a gate can be blocking in one context and advisory in another. The
capability floor is advisory when you run `fak preflight` by hand (it prints and exits),
and blocking when the same fold runs on the live call path (a `DENY` never dispatches).
Read *where* the gate runs, not just its name.

## What do the tags mean, and how do I read one?

`CLAIMS.md` is the repo's honesty ledger. Every claim line carries exactly one tag, and
the tag answers one question: **is it real?**

| Tag | Meaning | On the critical path? |
|---|---|---|
| `[SHIPPED]` | real code, closed by a mechanical witness (a `go test`, a `go build`, a benchmark field, a file read-back) | yes |
| `[SIMULATED]` | modeled with labeled stand-in data (no GPU / no live engine on the build box); the seam is real, the numbers are illustrative | no — parked by its tag |
| `[STUB]` | plumbing present, behavior deferred; clearly labeled, returns a STUB/no-op result | no — parked by its tag |

A `[SHIPPED]` line also carries an **exposure** marker — the orthogonal question "is it ON
for an operator who configured nothing?":

- `[exposure: default-on]` — on with no flag set. A `[SHIPPED]` line with no marker
  asserts this, so a missing marker is a claim, not silence.
- `[exposure: gated — <reason>]` — shipped but behind an opt-in or default-off path.

The lint witness is `go test -v -run TestCLAIMSLedger ./internal/claimcheck` (the
`claims-lint` gate, in CI on every push). Run it:

```bash
go test -run TestCLAIMSLedger ./internal/claimcheck
```

**What actually happened** (run on this host, 2026-11-18): `ok
github.com/anthony-chaudhary/fak/internal/claimcheck 0.062s`. The ledger is consistent —
every tagged line has exactly one tag, and every gated `SHIPPED` line declares its
exposure.

## What the repo does NOT yet prove

This is the honest residue, read straight from the ledger. Each row is a claim, its tag,
and what would move it up.

| Claim | Tag | What would move it up |
|---|---|---|
| The in-process adjudication path (the reference monitor) | `[SHIPPED]` | — (already closed by `TestNoOsExecOnHotPath` and the A/B sentinel) |
| The pre-flight ladder + grammar rung | `[SHIPPED]` (gated) | un-gate the opt-in paths, or document each gate's default |
| The context-MMU (write-time result admission) | `[SHIPPED]` (gated) | same — the mechanism is real; the exposure is gated |
| A LIVE transport attaching a real external engine's KV region | `[STUB]` | build the CUDA-IPC / shared-memory import behind the frozen `RegisterRegionBackend` ABI |
| A fine-tuned *syscall/adjudication* model | `[STUB]` | train/graft a tuned head onto the fused model (GPU + base weights + multi-hour training) |
| token-per-watt telemetry | `[SIMULATED]` | a watt source on the box (there is none) |
| The in-process microagent host | `[SIMULATED]` | run the acceptance against a live engine, not the Mock engine |
| The live wall-clock validation of the >100k context floor | `[SIMULATED]` | a resident-GPU bench node (a 100k prefill is intractable on a CPU host) |

The pattern to internalize: **a `[STUB]` is not a failure — it is a labeled gap.** The
repo's discipline is that a gap is *named*, not hidden. A `[STUB]` that says exactly what
is missing is a first-class, mergeable result; a `[SHIPPED]` that overstates is the thing
the ledger exists to prevent.

## What is genuinely open about kernel development itself?

Step back from the individual claims. What does the repo *not* yet prove about the
practice of adding a kernel?

- **The three gates are proven for the compute HAL; the adjudicator rung is proven by
  analogy.** Rungs 1–4 have a mechanical witness per gate. Rung 5's rungs are witnessed
  too (the circuit-breaker test, the fold tests), but the *"add a rung"* path is younger
  than the *"add a backend"* path — expect the seam to be more settled than the docs.
- **The keep-bit is proven for compute kernels; it is not yet a general gate for
  adjudicator rungs.** A rung's "is it faster?" question is different from a kernel's —
  a rung's cost is a per-call latency, not a throughput win. The keep-bit machinery
  (`shipgate.Evaluate`) is compute-shaped today.
- **The frontier is a moving target by design.** The caching ladder's own frontier page
  says it plainly: "assume the tree is more current than this page; the linked source
  files are the ground truth." The same rule applies here. When a number or a flag
  matters, read the source file, not this page.

## Where to go next

You have the whole ladder. Now build something.

- [`EXTENDING.md`](../../../EXTENDING.md) — the reference: the three gates, the seam
  table, `fak new-leaf`. Read this first if your goal is "I want to optimize subsystem X."
- [`ARCHITECTURE.md`](../../../ARCHITECTURE.md) — the frozen ABI, the registration table,
  the O(1) hot-path contract.
- [`CONTRIBUTING.md`](../../../CONTRIBUTING.md) — how to land a change (the landing flow,
  the DCO/CLA on an external PR).
- [`LEARNING-PATH.md`](../../../LEARNING-PATH.md) — the prerequisite-ordered curriculum
  across every fak concept, if you want the wider map.
- [`CLAIMS.md`](../../../CLAIMS.md) — the honesty ledger you just learned to read.
- [`STATUS.md`](../../../STATUS.md) — the historical evidence snapshot (read the header:
  it is a snapshot, not current authority).

Before you start, run the extension preflight — one command tells you whether your
environment and all three gate entry points are wired:

```bash
python tools/extend_preflight.py
```

**What actually happened** (run on this host, 2026-11-18): it printed a per-check report —
`[PASS] gate1-plug-in`, `[PASS] gate2-prove-correct`, `[PASS] gate3-prove-faster`,
`[PASS] golden-path-docs` — and one `[FAIL] git-guards-installed` (the trunk guard is not
installed on this host), so the overall verdict was `NOT READY`. That is the tool working
as designed: it tells you the *one* thing to fix before the golden path is wired, rather
than letting you discover it mid-land.

## Checkpoint

1. What is the difference between a frozen seam and a moving one?
2. Name one gate that is blocking and one that is advisory, and say where each runs.
3. What does `[STUB]` mean, and is it a failure?
4. What would move the "fine-tuned adjudication model" claim up a tag?
5. Why does this page tell you to read the source file rather than trust the page?

### Answers

1. A frozen seam is one you can build against without the ABI moving under you (the
   `internal/abi` spine, `compute.Register`, `RegisterAdjudicator`); a moving seam has a
   real interface but an implementation still being built (the transport behind
   `RegisterRegionBackend`, the fine-tuned model).
2. Blocking: the `internal/architest` layering gate (fails the build) or the deterministic
   witness (a red witness cannot land). Advisory: the `fak preflight` fold at authoring
   time (it prints a verdict; it does not stop a commit) or the detector feeding the
   result-side gate (it flags; the quarantine decision is structural).
3. `[STUB]` means the plumbing is present and the behavior is deferred, clearly labeled.
   It is not a failure — it is a *named gap*, and a named gap is a first-class result.
4. Training or grafting a tuned adjudication head onto the fused model — GPU, base
   weights, and multi-hour training. Today the fused model is a stock reference forward
   pass.
5. Because the frontier moves faster than any page: the linked source files are the ground
   truth, and a doc is a snapshot. When a number or a flag matters, read the source.

## See also

- [Level 5 — The adjudicator rung](level-5-the-adjudicator-rung.md): the other kernel, the
  rung before this one.
- [Level 3 — Prove correct](level-3-prove-correct.md): the correctness gate, in full.
- [The caching frontier](../caching/level-5-the-caching-frontier.md) — the sibling
  frontier page whose honesty discipline this one mirrors.
- [`CLAIMS.md`](../../../CLAIMS.md) — the ledger, with the `SHIPPED` / `SIMULATED` /
  `STUB` tags and the exposure axis.
- [`EXTENDING.md`](../../../EXTENDING.md) — the three gates and the seam table.

---

*[← Level 5](level-5-the-adjudicator-rung.md) · [full ladder](README.md)*
