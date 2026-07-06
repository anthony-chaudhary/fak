# GLM-5.2 readiness warmup-gate contract (#3051), 2026-07-06

**Status:** specification + generation classification. The runtime implementation
is **not yet shipped and not yet witnessed** — this note pins the exact contract a
follow-on serve/launcher pass implements, and classifies #3051's horizon so it can
be dispatched. It does not itself make a first turn warm; it names the witness that
would.

**Scope of this pass:** triage-only. #3051 routed to the `docs` lane and carried no
`generation` label and no milestone (intake drift). The deliverable here is the
classification + the implementable contract; the code lands under the serve /
launcher / probe lanes named in [Decomposition](#decomposition). This pass does not
have — and must not fake — the GPU server/GCP GLM-5.2 host the acceptance witness requires.

## The problem, in one line

`fak serve` reports ready (the listener binds, `/healthz` answers, the launcher's
`Wait-Url .../healthz` returns) **before the accelerator backend has finished
warmup** — weight load into VRAM, CUDA-graph capture, and DeepGEMM/torch JIT kernel
compile (bring-up log [`GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md`](GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md),
steps 14–16). So the operator's **first real turn** absorbs the whole ~500s tax,
and the direct probe's 180s client cap can even *cancel a legitimate cold request*
mid-compile. This is axis A (warmup), not axis B (KV-prefix caching), per
[`GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md`](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md):
identical 11-token `say pong` prefix, cold `≥180024ms` (cut off) vs warm `1032ms`,
so ≥99% of the cold→warm delta is warmup.

`#3051` (gate readiness on warmup) and `#3052` (persist the compile caches across
boots) are orthogonal halves of the same tax: `#3051` moves the tax off the
operator's first *real* turn onto a synthetic warmup turn; `#3052` reduces how
often the tax is paid at all. Neither subsumes the other.

## What the code already has, and the exact gap

This is the load-bearing engineering distinction, because two of the four proposed
fixes are **partly already present** — the contract is to move an existing anchor,
not to build from zero.

| Proposed fix | Current state in-tree | The gap to close |
|---|---|---|
| 1. Gate `/health` on completed warmup | `/healthz` (waited on at `scripts/dogfood-claude.ps1:752`, kernel-backend health timeout 600s) answers on **listener bind**, not warm | add a warmup-first-token gate *behind* `/healthz` (or a distinct `/ready`) so healthy ⇒ a token was produced |
| 2. Launcher sends the warmup turn | launcher waits on health, then the operator's first turn is the first inference | have the launcher (or serve boot) issue one synthetic warmup inference before declaring ready |
| 3. Align the cold client ceiling | `FAK_PLANNER_TIMEOUT_S` is already floored to **900s** for kernel/openai backends (`scripts/dogfood-claude.ps1:696`); but the local-shim wait is hardcoded `Wait-Url .../v1/models 180` (`:598`) and the direct-chat probe's `HttpClient.Timeout` canceled at **180024ms** (`experiments/agent-live/gcp-glm-night2-20260706T133856Z/direct-chat.error.json`) | raise **the direct-chat probe's own `HttpClient.Timeout`** (locate it — it is the seam the 180024ms cancel came from, *not* the 900s planner floor, which was already correct) and the `:598` shim wait to a cold ceiling ≥ the planner floor |
| 4. Expose `time_to_ready` | `fak_gateway_time_to_ready_seconds` **already exists**, anchored at boot `t0` (`cmd/fak/serve.go:191`) | re-anchor / add a second stop so it measures **boot → warmup-first-token**, not boot → listener-bind; today it reports the *smaller* number and hides the tax |

**The honest ceiling on the win.** Gating readiness on one warmup turn moves the
tax; it does not delete it. Total boot-to-usable wall time is **unchanged or
slightly longer** (the synthetic warmup turn is added before ready). The win is
entirely in *who pays*: the operator's first real turn drops from ~500s to
warm-path ~1s, and a cold request is never canceled by a client cap. Anyone who
reads acceptance bullet 2 ("first real turn is ~1s") as "boot is faster" has
mis-scoped it — call that out in the witness. Making the *boot* cheaper is #3052's
job, not this one.

## The readiness warmup-gate contract

### 1. Readiness = a produced token, not a bound port

Serve boot, after the engine reports its listener up, issues **one** synthetic
warmup inference of a **representative token shape** and only flips the
readiness signal green after it returns its **first decoded token**. `/healthz`
(or a new `/ready`) must not report healthy until that happens. The launcher's
`Wait-Url .../healthz` then already waits on the true warm signal with no launcher
change.

### 2. The launcher owns the warmup turn (belt-and-suspenders)

Even with (1), the launcher sends its own warmup turn before handing the session to
the operator, so a serve build without the gate still never lands a cold turn on a
human. This is the cheap, lane-local half (`scripts/dogfood-claude.*`).

### 3. Cold ceiling ≥ warmup, everywhere on the path

Every client on the cold path must tolerate a full cold warmup. The 900s
`FAK_PLANNER_TIMEOUT_S` floor is already correct; the leaf raises the two seams
still capped at 180s: the direct-chat probe's `HttpClient.Timeout` and the
`:598` local-shim `Wait-Url` bound.

### 4. `time_to_ready` measures the tax, not the bind

`fak_gateway_time_to_ready_seconds` stops at **warmup-first-token**, so the emitted
number is the real amortizable cost. The existing boot-phase breakdown (weight load
/ capture / compile) stays as sub-phases so an operator can see *which* term
dominated this boot.

## Decomposition

The runtime work is one issue but several leaves, in the lanes that own the files:

| Leaf | Lane | File(s) | What it does |
|---|---|---|---|
| L1 warmup-gate the readiness signal | cmd/serve · gateway | `cmd/fak/serve.go`, gateway health handler | issue the synthetic warmup turn; flip `/healthz`/`/ready` only after first token |
| L2 re-anchor `time_to_ready` | cmd/serve | `cmd/fak/serve.go` (the `t0` / `fak_gateway_time_to_ready_seconds` seam, ~`:191`, `:624`) | stop the timer at warmup-first-token; keep the phase breakdown |
| L3 launcher warmup turn | scripts | `scripts/dogfood-claude.ps1` / `.sh` | send one warmup inference before handing off; belt-and-suspenders for L1 |
| L4 raise the cold client ceiling | scripts · probe | `scripts/dogfood-claude.ps1:598`, the direct-chat probe's `HttpClient.Timeout` | ≥ the 900s planner floor so a cold warmup is never canceled |
| L5 witness | infra | `experiments/agent-live/` | fresh-boot dogfood on GLM-5.2: `/healthz` flips only after warmup-first-token; operator's first real turn is warm ~1s; cold request not canceled — the acceptance proof and the gen/next→gen/now promotion evidence |

L2/L4 are cheap and near-witnessable (a unit/probe change); L1/L3/L5 need the real
GLM-5.2 serve host.

## Generation classification

**`generation` + `gen/next`; milestone `Generation G1 - Next Gen`.**

Applied from [`docs/generation.md`](../generation.md); consistent with the sibling
#3052 classification in
[`GLM52-COMPILE-CACHE-PERSISTENCE-CONTRACT-3052-2026-07-06.md`](GLM52-COMPILE-CACHE-PERSISTENCE-CONTRACT-3052-2026-07-06.md)
(same ~500s-tax cohort, same evidence bar).

- **Why `gen/next`, not `gen/now`.** gen/next is a "near-term foundation that should
  be runnable by agents soon, but still needs a gate, dogfood run, schema, or
  default-exposure proof." #3051 is exactly that: it adds a readiness **gate** whose
  acceptance witness is a **real GPU fresh-boot dogfood** (first real turn warm),
  and **no runtime code exists yet to witness**. gen/now's bar — "improves the
  current product / operator loop with a **clear witness** and no future-architecture
  dependency" — fails on *clear witness*: there is no cheap current-path test that
  proves "first real turn is warm"; that proof is inherently a GPU-host dogfood.
- **Why not `gen/second-next`.** It needs no cross-engine compatibility contract or
  new serving architecture — it runs on the existing SGLang/vLLM serve path and only
  moves *when* the readiness signal flips. That keeps it out of the architecture-
  option horizon.
- **Promotion evidence (→ `gen/now`).** A fresh-boot dogfood witness showing
  `/healthz` reports healthy **only after** warmup-first-token; the operator's first
  real turn on that boot is warm-path (~1s class), not ~500s; a cold request is not
  canceled by any client cap; and `fak_gateway_time_to_ready_seconds` reports the
  boot→warmup-first-token cost. Any one of these landing default-on with a captured
  witness is a promotion step.
- **Demotion / retirement evidence.** Demote/park if a single representative warmup
  turn cannot be made to warm the kernels a *differently-shaped* first real turn
  hits (see the invalidating assumption) — i.e. the gate reports warm but the user
  still pays, so gating buys nothing. Retire if #3052's persistent compile cache
  makes cold warmup cheap enough that gating on it is unnecessary, **or** if upstream
  SGLang/vLLM ships a native "ready-after-warmup" health signal, making the fak-side
  gate redundant.
- **Invalidating assumption (must be measured before promotion).** The gate assumes
  **one synthetic warmup inference of a representative token shape warms the same
  kernels / CUDA-graph buckets / MoE-expert routing that the operator's first *real*
  turn dispatches.** If the real first turn hits a different sequence-length graph
  bucket or expert path than the synthetic warmup, readiness flips green while the
  real turn still JIT-compiles a *different* specialization — the gate reports warm,
  the user still pays a slice of the tax. So the **first evidence to collect** is:
  does one representative warmup turn zero the tax on a *different-shaped* real first
  turn, or must the warmup **sweep** the buckets the operator will actually hit? If a
  sweep is required, L1 grows and the ceiling on the win shrinks.

## Refs

- Axis A/B ablation: [`GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md`](GLM52-COLD-START-VS-CACHING-ABLATION-2026-07-06.md)
- Bring-up trail: [`GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md`](GLM52-GCP-DOGFOOD-BRINGUP-2026-07-05.md)
- Ablation witnesses: `experiments/agent-live/gcp-glm-night2-20260706T133856Z/`
- `time_to_ready` seam: `cmd/fak/serve.go` (`fak_gateway_time_to_ready_seconds`, `t0` at `:191`)
- Companion tickets: #3052 (persist the compile caches across boots) · #3053 (de-conflate the warmup tax from aggregate `cache_bit`)
