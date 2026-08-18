---
title: "fak glossary: core vocabulary, shared memory, preflight vs inflight"
description: "The canonical fak glossary: overloaded core terms (session, agent, context, model, memory, tool/skill, steering), shared-memory sense splits, the active memory issue map, the preflight/inflight/prefill gate vocabulary, and the vCache streaming economy (cache rebate, owner split, read rebate vs write premium, net saving, the per-turn verdict words)."
---

# Glossary: core vocabulary, shared memory, and before/during words

A few words in this codebase look related but aren't, and a couple are reused at
several layers with no shared code. That overload is the usual source of confusion.
This page pins each one down.

This is the canonical docs-lane glossary for the term-conflation audit tracked in
[#721](https://github.com/anthony-chaudhary/fak/issues/721). The dated worklist
([`docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md`](notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md))
is the source audit; this page is the stable reader-facing contract.

This page carries the public vocabulary: product terms defined without internal
shorthand. Its contributor-layer companion is the
[concept glossary](fak/concept-glossary.md), which disambiguates implementation
vocabulary — colliding Go identifiers, package names, and internal families. If
the term you met is a code symbol, resolve it there; every product term resolves
here.


## Whole-path decision terms

These terms are deliberately not interchangeable:

- **Coordination** — The bounded cross-layer fold from observations about cache/context, compute/placement, harness/runtime, serve/engine, and trust/operations into a constrained plan, followed by typed actions/effects and evidence. Coordination is broader than worker orchestration and narrower than generic “everything integrates.” The complete fold is the target of [#6042](https://github.com/anthony-chaudhary/fak/issues/6042), not a claim that one coordinator already ships.
- **Orchestration** — Managing roles, dependencies, leases, budgets, and reconciliation for a graph of agent or worker tasks. Orchestration can be one input to coordination; it does not choose every model, context, policy, or serving decision.
- **Scheduling** — Ordering and admitting work over time and capacity after its constraints are known. Scheduling answers *when and in what order*, not which provider or policy is allowed.
- **Routing** — Selecting an allowed destination or implementation for a request, such as a model, provider, engine, region, or execution route. Routing answers *where*, within policy and capability constraints.
- **Serving** — Accepting and executing model-facing requests through a runtime endpoint and engine. Serving performs the request; it does not by itself coordinate the surrounding agent, context, tools, or operator controls.

## Canonical overloaded vocabulary

| term | senses in this repo | house rule |
|------|---------------------|------------|
| **session** | token decoder (`model.Session` / `model.BatchSession`), reloaded core image (`recall.Session`), live drive state (`internal/session.Table` / `State`), wire DTO (`gateway.SessionState`), and per-session context planner (`agent.SessionPlanner`) | qualify it. Bare "session" in architecture docs means live drive state only when the surrounding path is `internal/session`; otherwise say decoder session, core-image session, gateway session state, or planner session. |
| **agent** | `fak` as kernel/reference monitor, the external untrusted guest loop, the `fak agent` demo verb, and the `internal/agent` wire/loop package | say kernel/reference monitor for `fak`; say guest or external agent for the program being mediated. Do not make "agent" name both sides of the trust boundary in the same paragraph. |
| **context** | token window, planner-resident view for this turn, and the result-admission target (`ctxmmu`) | "fits in the window" is not "entered the view", and "entered context" means the post-result admission gate allowed model-visible bytes. |
| **model** | a routed provider/engine binding, and `internal/model.Model`, fak's own in-kernel transformer | use engine, provider, or routed LLM for the first sense; reserve model for the owned transformer when precision matters. |
| **memory** | KV working memory, durable/recall memory, and procedural memory (a cached skill/context view) | "working memory" always means the KV cache. Durable memory is cross-session recall; procedural memory is a reusable skill view, not a fact store. |
| **shared memory** | shared KV/prefix reuse, shared CAS/blob refs (`abi.Resolver` / `Ref`), a typed region/window (`internal/region`, planned in #646), an external L3 shared-KV tier, or a remote provider prompt cache (vCache) | qualify the transport and ownership. "Shared memory" alone must not imply RDMA, CUDA IPC, durable recall, or trust to reuse; say shared KV prefix, CAS ref, region window, L3 tier, or provider cache. A remote provider prompt cache (vCache) is a **cost/latency lever only — never a trust or correctness boundary**: a warm provider cache saves dollars and latency, it is never authority to omit context, so no request may depend on a hit landing (see the provider prompt-cache control row). |
| **shared state** | live messages (`a2achan`), live shared objects/whiteboards, durable task/session/window handoff, disaggregated KV/state tiers, and user-editable collaboration surfaces | qualify the rung. "Shared state" alone must not imply durable memory or collaborative editing. Use the [shared-state ladder](shared-state-ladder.md): shared live, shared durable, shared disaggregated, or user-level collaborative state. |
| **tool vs skill** | a tool is an adjudicated effect-bearing call; a skill is host-side procedure/instructions that may issue tools | `fak` gates tools. It does not directly gate a skill; it gates the tool calls the skill produces. |
| **steering** | loop steering after a deny, planner bias over what goes resident, and adversarial prompt steering | reserve steer/steering for the kernel-owned loop disposition when possible; use bias/weight for planner selection and manipulate/hijack for attacks. |
| **audit vs drive** | audit is the read-only record of what happened (journal, trace, hosted control plane); drive state is the live control value that changes what a run does next (`session.Table`, budgets, pace, priority) | audit reports decisions; drive state changes execution. Do not call both "session state" without a qualifier. |

## Shared-memory issue audit

Last checked against live GitHub issue state on 2026-06-25 with `gh issue list --repo
anthony-chaudhary/fak`. This table is deliberately narrow: it says which in-flight
ticket family owns each memory/shared-memory concept and which nearby concept it does
not own.

| concept | live owner tickets | owns | does not own |
|---------|--------------------|------|--------------|
| vocabulary cleanup | [#721](https://github.com/anthony-chaudhary/fak/issues/721) | canonical sense split and docs links | code renames/trust-framing fixes, which stay in the code-lane follow-up |
| context vs durable memory | [#82](https://github.com/anthony-chaudhary/fak/issues/82), [#81](https://github.com/anthony-chaudhary/fak/issues/81), [#80](https://github.com/anthony-chaudhary/fak/issues/80) | write-time durability, as-of validity, and TTL-driven KV expiry | shared-window concurrency, provider prompt caching, or external L3 routing |
| one-sided shared window | [#654](https://github.com/anthony-chaudhary/fak/issues/654) documents the shipped `Resolver.Put`/`Resolve` pool; [#646](https://github.com/anthony-chaudhary/fak/issues/646) builds first-class `Put` / `Get` / `Accumulate` | one-sided shared-result reads/writes over `Ref` + `ShareScope`, including the planned deterministic `Accumulate` fold | RDMA, hardware zero-copy, provider cache warmth, or durable fact promotion |
| external L3 shared KV | [#53](https://github.com/anthony-chaudhary/fak/issues/53), [#54](https://github.com/anthony-chaudhary/fak/issues/54)-[#58](https://github.com/anthony-chaudhary/fak/issues/58), [#75](https://github.com/anthony-chaudhary/fak/issues/75)-[#78](https://github.com/anthony-chaudhary/fak/issues/78) | fak as the semantics/referee layer over an external shared KV tier: digest verification, ShareScope, deletion certificate, L3 region backend | base serving parity, inline data-path byte scanning, or forking the external L3 store |
| provider prompt-cache control | [#715](https://github.com/anthony-chaudhary/fak/issues/715)-[#720](https://github.com/anthony-chaudhary/fak/issues/720), [#727](https://github.com/anthony-chaudhary/fak/issues/727) | vCache: warmth belief, anchor shaping, dedicated warming, gated chain recall, governor, and provider telemetry probes | correctness/trust claims. Warmth is a cost/latency belief confirmed by telemetry, never authority to omit context. |
| shared serving spine | [#50](https://github.com/anthony-chaudhary/fak/issues/50), [#637](https://github.com/anthony-chaudhary/fak/issues/637) | base serving substrate shared by RIDE and NATIVE tracks: streaming, EngineDriver, router/residency, metrics, parity bench | L3 governance value-adds; those ride on top of the base serving spine. |
| closed memory-view slices | [#421](https://github.com/anthony-chaudhary/fak/issues/421), [#435](https://github.com/anthony-chaudhary/fak/issues/435), [#513](https://github.com/anthony-chaudhary/fak/issues/513) | historical/completed work on opencode memory reads, materialization verdicts, and procedural-memory views | active backlog ownership unless a new issue reopens a gap. |

Two audit caveats:

- Some migrated epic bodies still carry stale internal-tracker child numbers. Prefer the live
  issue numbers in the table above when dispatching work.
- A cache hit, shared prefix, or materialized view is not automatically a memory write. It
  becomes memory only after the result/admission and durability gates say it may.

## The one distinction worth memorizing

| word | timeline | can it refuse? | one-liner |
|------|----------|----------------|-----------|
| **preflight** | runs **BEFORE** a thing starts | **yes** — it's a gate | a check that decides whether to proceed |
| **inflight** | observed **WHILE** a thing runs | **no** — it only watches | a count or lease over work already in motion |
| **prefill** | a phase **INSIDE** a model run | n/a | a model's prompt-ingestion pass (not a check) |

When you hit one of these, two questions resolve it every time:

1. **Before, or during?** A gate that can say no (preflight) versus a reading of what's
   already running (inflight).
2. **Which layer?** The same word names different mechanisms in the kernel, the serving
   gateway, and the dispatch fleet. The *path* tells you which.

Everything below is just those two questions applied.

## `preflight` — one metaphor, four mechanisms, no shared code

All four mean "check before you commit," in the aviation sense. They share the
metaphor and nothing else — the two Go ones and the two Python ones are unrelated code.

| sense | layer | gates what | when it runs | refuses with |
|-------|-------|------------|--------------|--------------|
| **kernel rung ladder** | kernel | one *tool call* — rung 0 JSON-parses the args, rung 1 schema-checks required fields | inside the adjudicator chain at submit, on every call | `VerdictDeny` (else `VerdictDefer`) |
| **`fak preflight` CLI** | CLI → kernel | the *whole* pre-dispatch chain over one call, offline | at policy-authoring time (no server, model, or network) | a printed verdict |
| **serve-readiness gate** | serving | a *node* before you start an inference server (GPU arch vs the model's kernel floor, VRAM vs quant footprint, engines installed) | an operator runs it before a serve/bench job | `BLOCKED_ARCH` / `BLOCKED_MEMORY` / … |
| **dispatch spawn gate** | dispatch fleet | *spawning another worker* — host healthy? account free? under the worker cap? | before every async worker launch | `REFUSE_HOST` / `REFUSE_AT_CAP` / … |

- Kernel ladder: `internal/preflight` — `Ladder.Adjudicate`, `Ladder.caughtAt`, registered by `RegisterAdjudicator(10, …)`. Proof: [`docs/proofs/preflight.md`](proofs/preflight.md).
- CLI: `cmd/fak/main.go` — `cmdPreflight` (see the subtlety below).
- Serve-readiness: `tools/glm52_serve_preflight.py` — `evaluate_engine`; siblings `tools/extend_preflight.py` (contributor setup), `tools/qwen36_standalone_readiness.py`.
- Dispatch spawn gate/router/progress: `fak dispatch route` / `fak dispatch tick` / `fak dispatch wave` / `fak dispatch progress` — the native `internal/dispatchtick` preflight evaluator, host process guard, issue-lane router, account route, live seat-pool fold, distinct-pool wave allocator, and issue-progress snapshot; the legacy `tools/dispatch_preflight.py` / `tools/proc_resource_guard.py` / `tools/issue_lane_router.py` / `tools/fleet_accounts.py route|seats|wave` remain compatibility oracles. Walkthrough: [`docs/dispatch-loop.md`](dispatch-loop.md).

### The subtlety that trips everyone: `fak preflight` ≠ the `internal/preflight` package

They share a name, but the CLI verb is **much broader** than the package. `fak preflight`
folds the *entire* registered adjudicator chain over one call; the `internal/preflight`
package is only **one rung** of that chain (rank 10). The chain, in execution order
(taken straight from the `RegisterAdjudicator(rank, …)` calls):

| rank | rung | package | role |
|------|------|---------|------|
| 5 | grammar | `internal/grammar` | repair/normalize a malformed call (positional→named); the cheapest rung |
| 8 | rate-limit | `internal/ratelimit` | throttle call volume |
| 10 | **preflight** | `internal/preflight` | JSON parse + schema well-formedness — *this* is the package |
| 12 | engine residency | `internal/engine` | gate on engine/model residency |
| 25 | plan-CFI | `internal/plancfi` | plan control-flow integrity (`RequireApproval` for risky steps) |
| 30 | IFC sink-gate | `internal/ifc` | refuse a sensitive-sink call when tainted data is in flight |
| 35 | git-gate *(optional)* | `internal/gitgate` | git-operation shape gate; skipped when `FAK_GITGATE=off` |
| 40 | ship-gate | `internal/shipgate` | ship/commit gate |
| 100 | **monitor** | `internal/adjudicator` | the authoritative capability/policy decision (allow/deny lists, self-modify, path redaction) |

So it's **8 rungs always, 9 with git-gate on**. When a commit message or doc says
"preflight," check whether it means rung 10 (the package) or the whole `fak preflight`
fold (all of the above). The package's own proof doc notes this exact naming caveat.

Two orderings live here and they are not the same:

- **Execution rank** (5, 8, 10, … 100) decides the *order rungs run* — cheap checks first,
  so a cheap deny short-circuits the expensive ones. It is an optimization.
- **Fold rank** (`abi.FoldRank`) decides *which verdict wins* — the most-restrictive verdict
  kind, default-deny. A rung-10 `Deny` beats a later `Allow` because `Deny` outranks `Allow`
  in the lattice, **not** because rung 10 ran first. (`internal/kernel: Fold`.)

## `inflight` — one idea ("what's moving now"), several uses

| sense | layer | what's "in flight" | mechanism |
|-------|-------|--------------------|-----------|
| **gateway requests** | serving | HTTP requests accepted but not yet finished | an atomic gauge (`+1` accept / `-1` done) **plus** a live registry (route + start per request) sampled at scrape time for per-route counts and `max_age` — the only signal that catches a *wedged* request, since completion histograms can't see one that hasn't returned |
| **radix KV-cache lease** | serving | a request whose cached prefix is still being served | refcount: `Lookup` leases the node (`refs++`), `Insert` hands the lease to the leaf, `Done` releases it; a node with `refs>0` is safe from LRU eviction, so its prefix can't be reclaimed mid-serve |
| **"in-flight work"** | docs | a feature being built, not yet shipped | just prose — e.g. "the int8/Q8 SIMD lane is the active in-flight increment." No runtime meaning |

- Gateway: `internal/gateway` — `gatewayMetrics.inflight`, `beginInflight`/`endInflight`, `inflightSnapshot`; metrics `fak_gateway_inflight_requests`, `…_by_route`, `…_max_age_seconds`. Ops view: [`docs/fak/observability.md`](fak/observability.md).
- Radix lease: `internal/radixkv` — `node.refs`, `Tree.Lookup`/`Insert`/`Done`, `evictToBudget`.
- Narrative: `CLAIMS.md`, `docs/cli-reference.md`.

> **Watch for the bare idiom.** "In flight" also shows up as plain English where it names
> *nothing*: the IFC sink-gate's doc reads "refuses a sensitive-sink call when tainted data
> **is in flight**" (`internal/ifc: SinkGate`). That gate is `SinkGate` (rung 30 above), not
> an "inflight" mechanism — the words just describe tainted data currently flowing. It's a
> tidy illustration of why the path, not the word, tells you what's meant.

## `prefill` is a different word, not a typo of `preflight`

`prefill` (one `l`) is the model's **prompt-ingestion phase**: the batched forward pass that
runs the prompt tokens through the transformer in parallel to produce the first logits, as
opposed to `decode`, which emits one token at a time. It lives only in `internal/model`
(`Session.Prefill`, `attnPrefillInto`) and never crosses paths with `preflight` — a grep for
`prefill.*preflight` returns nothing. Keep them apart by role:

- **prefiLL** **fiLLs** the KV cache — arithmetic, *during* generation.
- **prefLIGHT** is a **fLIGHT** check — a gate, *before* the thing starts.

## Adjacent kernel vocabulary (so the cluster stops blurring)

- **kernel (the word itself)** — three unrelated senses share it: fak as
  OS-metaphor reference monitor (this page's **agent** row), the compute-kernel
  arithmetic paths (`internal/model/kernel.go`), and the literal CUDA
  `__global__` kernels in `internal/compute/cuda_kernels.cu`. The full
  disambiguation, with the header-to-silicon depth ladder, is
  [what is a CUDA kernel?](explainers/what-is-a-cuda-kernel.md).
- **vDSO** (`internal/vdso`) — virtual dynamic shared object, a fast, safe read path borrowed
  from the OS-kernel term. A 3-tier local cache (pure / content / static) consulted *before
  the entire adjudicator chain*. A hit answers a repeated call with no engine round-trip
  and skips every rung. It's a cache, not a gate; preflight is a gate inside the chain a
  vDSO *miss* falls through to.
- **MMU** — memory management unit, the hardware/OS unit that maps and protects memory.
  In fak, context-MMU is the software analogue for agent context: the write-time tool-result
  gate (`internal/ctxmmu`) that decides whether bytes enter the model's context
  (allow / quarantine / transform).
- **adjudicator / fold** (`internal/abi`, `internal/kernel`) — an adjudicator is one
  stackable verdict-producer (preflight is one); the kernel *folds* all of their verdicts into
  one by the most-restrictive lattice, default-deny.
- **rung** — one ordered step inside the preflight ladder (rung 0 parse, rung 1 schema). On a
  catch it stamps `(RungPassed, RungFailed)` into a hard-negative row; a clean pass stamps
  nothing.
- **monitor** (`internal/adjudicator`) — the rank-100 *authoritative* adjudicator. preflight
  does cheap structural checks and `Defer`s a well-formed call to it; the monitor makes the
  real policy decision.
- **admit / admission** (`internal/kernel: AdmitResult`, `internal/ctxmmu: MMU.Admit`) — the
  **after** to preflight's **before**. preflight screens a *call* before it fires; admit
  screens a tool *result* after it returns, deciding whether the bytes enter the model's
  context (allow / quarantine / transform). These are the project's "two gates": the
  capability floor (pre-call) and the result quarantine (post-result).

## The vCache streaming economy: what `fak manage` prints per turn

When you run `fak manage -- claude --debug-stats`, the gateway streams **one line per
turn** to stderr that prices cache-like savings by owner, and the `fak manage` exit
summary plus `fak vcache observe`, `fak cachevalue report`, and `/metrics` carry the
same owner vocabulary. The vocabulary on those surfaces is fak's own (the vCache work,
[#218](https://github.com/anthony-chaudhary/fak/issues/218) /
[#715](https://github.com/anthony-chaudhary/fak/issues/715)-[#720](https://github.com/anthony-chaudhary/fak/issues/720)).
A representative line:

```
fak-turn trace=t1 ok prov=20.9k tok (85% of prompt) fak=0 tok cache=healthy compact=ok finish=end_turn
```

The whole vocabulary rests on one accounting law, inherited from `internal/vcachestar`
and `internal/callavoid`: **cost is always booked at the full *uncached* price first, and a
confirmed cache hit refunds part of it.** That refund is the rebate. The law's one-liner —
*"an avoided call is a realized rebate, never a trust claim"* — is the discipline: a rebate
is booked only from a hit the provider *confirmed* (`cache_read_input_tokens` came back
non-zero), never from fak *believing* a prefix was warm. Belief predicts; only telemetry
rebates.

The second law is the owner split: **provider prompt-cache savings are not fak-authored
savings**. A default `fak manage` exit summary therefore prints one attribution line:

```
fak manage: avoided-spend attribution — provider ~P (p%) + fak ~F (f%) = ~T token-equiv [...]
```

`provider` is OBSERVED/provider-relayed prompt-cache economics. `fak` is WITNESSED
fak-authored token saving: compaction shed plus in-kernel KV-prefix reuse. vDSO is
reported in the same mechanism family as avoided calls, not token-equivalents, until
there is a token witness for a skipped call.

| term | what it means | the catch that the word encodes |
|------|---------------|----------------------------------|
| **(cache) rebate** | the cost refunded by a confirmed cache hit, in input-token-equivalents | booked only on a telemetry-confirmed read; warmth belief alone never rebates. `internal/vcachestar: CostBooking.RebateTokens` |
| **read rebate** | the read axis of the rebate: each `cache_read` token billed at 0.1× base instead of 1× — a 0.9×/token refund | a read is the *only* axis that pays you back; on its own it overstates the value |
| **write premium** | the first write to a cache costs *more* than uncached: 1.25× base at the 5-minute TTL, 2.0× at the 1-hour TTL | this is why caching is a net win only once reads accrue (break-even is 2 requests at 5m, 3 at 1h). `internal/gateway: CacheWrite5mMultiplier` |
| **provider prompt-cache net / `prov=` (token-equiv)** | the provider-owned net prompt-cache number: **read rebate − write premium**, in input-token-equivalents | a fresh, cold-write turn reads **negative** (`prov=-25 tok`) — a real loss the writes haven't repaid, which a read-only number would have hidden. `internal/gateway: ProviderCacheNetSavings` |
| **fak-authored token-equiv / `fak=`** | fak-owned token savings: compaction tokens dropped before send + in-kernel KV-prefix tokens reused | this is the slice that answers "what did fak author?" It excludes provider prompt-cache warmth and excludes vDSO until vDSO has a token-equivalent witness. `internal/gateway: MechanismSavings.FakTokenEquiv` |
| **owner attribution** | the default guard summary and metrics split total avoided spend into `provider` (OBSERVED/provider-relayed) and `fak` (WITNESSED/fak-authored) owners | `fak_cache_saved_by_owner{owner="provider"}` must never satisfy a fak-authored cache win; use `owner="fak"` for that. |
| **mechanism attribution** | the next split beneath owner: provider read rebate, provider write premium, compaction shed, KV-prefix reuse, and vDSO avoided calls | vDSO is an avoided-call counter (`fak_cache_avoided_calls_by_mechanism_total`), not a token gauge. |
| **cache anchor vs compaction budget** | the two knobs behind `compact=`. The **anchor** (a `cache_control` breakpoint) is WHERE the cached prefix ends and is copied verbatim; the **budget** (`--compact-history-budget`, 48k) is the token target for the middle AFTER the anchor that compaction sheds | the anchor GATES the budget — on real Claude Code traffic the breakpoint sits on a recent turn, so the protected prefix swallows the conversation and lowering the budget does nothing (`AnchorStarved`, [#1407](https://github.com/anthony-chaudhary/fak/issues/1407)). Full line drawn in the [concept glossary](fak/concept-glossary.md#cache-anchor-vs-compaction-budget---the-knobs-that-confuse-everyone). Distinct from `--ctx-view-budget` (8k), which re-plans the resident view rather than shedding turns; both budget sizes follow the [long-context defaults doctrine](long-context-defaults.md), where the advertised context window is a cap, not a target. |
| **baseline / cost / multiplier** | `baseline` = what the session *would* have cost with no cache (every token at 1×); `cost` = what it actually cost; their ratio is the **multiplier** (`7.22x`) | baseline is a projection over OBSERVED counts, not a fak-authored claim |
| **the turn verdict word** | one glance at the turn's state, folded from the net saving + prefix health | see the four values below. `internal/gateway: turnVerdict` |

The four turn-verdict words (the `ok` slot in the line above):

| word | means |
|------|-------|
| **cold** | no provider cache activity at all this turn (a first turn, or a non-cached path) |
| **warming** | cache activity, but the writes have not yet been repaid by reads — net saving still ≤ 0 |
| **ok** | a proven net saving on a healthy (or not-yet-scored) prefix |
| **degraded** | the rolling health says the prefix is decaying / stale, or a reset is recommended |

> **The one thing to internalize.** A provider prompt-cache rebate is real spend avoided,
> but it is not a WITNESSED/fak-authored cache win. The streamed `fak-turn` line, the guard exit
> summary, `fak vcache observe`, `fak cachevalue report`, and `/metrics` must therefore
> name both the owner and the mechanism: provider prompt-cache is OBSERVED/provider-relayed;
> compaction and KV-prefix reuse are WITNESSED/fak-authored; vDSO is WITNESSED as avoided
> calls. That is the same OBSERVED-vs-WITNESSED discipline the
> [conflation scorecard](CONFLATION-SCORECARD.md) enforces across all of fak's reported
> numbers.

## Mnemonics

- **preflight** — the **before** gate; it can **refuse** before the thing starts (a call
  before dispatch, a node before serve, a worker before spawn).
- **inflight** — the **during** state; what's running *right now* and not yet done (a live
  request, a held KV lease). Observed, never refused.
- **prefill** — a **model phase**, not a check; it fills the KV cache from the prompt.
- **rebate** — a **cost refund** from a *confirmed* cache hit, booked over an uncached
  baseline; the net saving subtracts the **write premium**, so a cold-write turn reads
  negative until reads repay it.

---

## Doctrine vocabulary: the recurring doc terms, defined once

Each of these recurring terms was used as a heading across dozens of docs with no
definition anywhere in the tree. They are defined here, once, with one canonical
spelling each.

### Honest fence

An explicit statement of what a claim does **not** cover, placed next to the claim so it
cannot be overread. This is the repo's core rhetorical device: every measured number and
every shipped feature carries a fence naming the boundary of the evidence. Canonical
spelling: **honest fence** — the drifted variants *honesty fence*, *honest boundary*, and
*honesty gate* all mean this concept and should converge on the canonical form.

### Honest scope

The recurring section that states what a feature deliberately does not attempt. The
scope-declaration counterpart of the honest fence: a **fence** guards one specific claim;
a **scope** declares a whole feature's deliberate limits up front.

### Trust boundary

The line between what the kernel verifies and what it merely receives from an untrusted
agent. Claims cross the boundary only by carrying evidence the kernel can re-check
(a diff, a ledger line, an artifact path) — a bare self-report never does. This is the
load-bearing idea behind `dos verify`, the witness pipeline, and every scorecard.

### Honesty ledger

The recurring doc section recording which of a feature's claims are proven and which are
still open. It is a **doc-section convention**, not a system component: no runtime
artifact is written under this name, and a doc using the heading is summarizing its own
claim status, not pointing at a file.

### Ground truth

The section pointing at the artifact or command a reader can run to verify a claim
themselves — fak's evidence-over-assertion convention. If a doc asserts a number, its
ground-truth section says where the number comes from and how to regenerate it.

### Promotion gate

The criteria a claim or feature must meet to move up a maturity rung (measured → default,
experimental → supported, simulated → shipped). Also written as *Promotion rule*; the
gate is the criteria, the rule is the sentence stating them.

### Hot path

The latency-critical code route. In fak the term is used for two routes: the model-engine
decode loop (`internal/engine`) and the gateway per-request serve path
(`internal/gateway`). A doc invoking the hot path should say which of the two it means.

### Verdict ladder

The ordered best-to-worst verdict scale a scorecard defines for its rows — for the
implicit-explicit scorecard: explicit → named-code → named-doc → hinted → latent. Every
fak scorecard names such a ladder; the ladder position doubles as the distance-from-done
used to order worst-first backlogs.

### Unmeasured-containment residual

The standing residual that fak's privacy / prompt-injection containment is a design
posture, not a measured leakage number. Research triages repeat the italicized phrase
*containment is not a measured number* — this entry names the residual so it can be
tracked instead of re-hedged.

## Code concepts the docs never named

Each of these is a load-bearing identifier or convention in `internal/` + `cmd/` that the
implicit-explicit scorecard found in a dozen-plus production files with zero doc
mentions. One-paragraph definitions, so the code term is searchable outside the code.

### `WeightBearing` — does the engine carry real weights?

The engine-interface predicate saying whether an engine actually carries model weights
(real inference: `DynamoEngine` reports true) or is a stub / proxy (`readEngine`,
`localEngine` report false). It is a trust distinction: benchmark and generation claims
are only weight-bearing when the engine is.

### `AdmissionVerdict` — why a cache entry was (or was not) admitted

The cachemeta verdict string recording whether an entry was admitted to the addressable
KV cache and why; `AdmissionFromVerdict` maps abi verdict kinds onto it.

### `EvidenceRef` — the unit of witnessable evidence

The taskmgr/loopmgr struct that points a witness at an artifact it can read back (a file
path, a git object). The witness/claims docs talk about evidence constantly; this is the
code type that evidence is.

### The `RealRunner` seam

The repo-wide convention of a `RealRunner` function: the production process-runner
injected where tests substitute a fake, independently re-declared per package
(mergepreview, modver, releasestale, and more). New packages should copy the seam
deliberately, not by osmosis.

### `ConfigureBackgroundCommand` — the windowgate rule

The `internal/windowgate` helper every spawned subprocess passes through so background
commands never flash a console window on Windows (a no-op elsewhere). Every `exec.Cmd`
in the tree must route through it — the convention is why fak spawns are silent.

### The per-verb `ParseFlags` convention

Every CLI verb declares its own `ParseFlags`/`parseFlags` helper, with reject-args and
or-help variants — the single most repeated declared name in the tree. The convention
keeps each verb's flag surface local to the verb file instead of a central flag table.

### `writeIndentedJSON` — the `--json` output contract

The per-package helper that emits the canonical two-space-indented JSON for `--json` verb
output, with a no-escape variant (`writeIndentedJSONNoEscape`) for payloads containing
HTML-meaningful characters. This is the de-facto JSON output contract of the CLI.

### `ExpandTilde` — promised `~` expansion

The `internal/pathutil` helper that expands a leading `~` in user-supplied paths — the
reason `~/foo` works in every fak flag. User-visible behavior, now promised here rather
than only inferable from the helper.

### `statusOverloaded` — provider overload 529

The non-standard HTTP 529 the Anthropic API returns when overloaded, load-bearing in the
stream-retry and account-rotation paths. `internal/agent/retry.go` names it
(`const statusOverloaded = 529`); gateway and attempt-budget sites still carry the bare
literal and should adopt the name.

## Recurring doc concepts, bound to the artifact each one names

The most-used doc headings with no code identifier behind them. Each entry says what
the term means here and which artifact (if any) is its canonical home.

### GPU server

The lab's private 8-GPU datacenter box that big-model benchmarks run on — under its
mandated scrubbed public name. `docs/gpu-server-private-boundary.md` requires this
exact phrase (no host, IP, path, or token) in anything public, so the term is a
deliberate anonymization convention, not a machine name.

### MCP server

fak's Model Context Protocol surface: `fak serve --stdio` (or `POST /mcp` over HTTP)
speaking JSON-RPC 2.0 and exposing the `fak_*` adjudication tools. An external-protocol
term applied to fak's own gateway — there is deliberately no distinct code component by
this name.

### Compatibility matrix

The per-tool table answering "can this agent tool repoint its base URL at `fak serve`"
across the OpenAI / Anthropic / MCP wires. Canonical artifact:
`docs/integrations/compatibility-matrix.md`; other docs reference that table rather than
defining their own.

### Worked example

A doc-writing convention: the section that walks one concrete scenario end-to-end with
real numbers or a real fixture (the addressable-KV-cache explainer replays an exact test
fixture, for instance). It names a writing pattern, not a system concept.

### Threat model

fak's stated attacker assumption: the model is the untrusted program, and the attacker
controls everything it reads — prompt, retrieved docs, tool results. Refusal therefore
hangs on structural gates, not on a classifier judging intent. A real system concept;
its doctrine home is `docs/fak/security.md`.

### Triage decision

The closing verdict slot of a `docs/notes/RESEARCH-*-triage` note: Adopt, defend
against, or ignore the externally-scouted paper/repo, with reasons. The structured
outcome of the research-triage workflow.

### Master ledger

The per-theorem verdict table in `docs/proofs/README.md` — every module proof's theorem,
deterministic witness test, and live verdict filled from actual test runs. Every other
`docs/proofs/*.md` page defers its current verdict to the master ledger rather than
restating it.

### Readiness score

An overloaded phrase with two senses that should not be conflated: (1) the deliberately
simple 0–100 operator blend for a fleet node in `docs/fleet.md`; (2) the readiness
dimension re-derived by the `*-READINESS-SCORECARD` family (`tools/*.py`). No single
code identifier backs either sense.

### Durable ledger

An append-only JSONL history file committed to the trunk as durable evidence — cadence
`history.jsonl` rows, nightrun `collected.jsonl` rows. Durable trunk evidence, not a
regenerable build artifact; the discipline is what separates a ledger from a report.

## Repo-wide helper conventions and other load-bearing identifiers

The implicit-explicit scorecard found each identifier below declared or used across a
dozen-plus production files with zero doc mentions. One honest line each, so the name is
searchable outside the code. Where a line says *per-package*, the identifier is a
convention independently re-declared in many packages — copy it deliberately.

| Identifier | What it is |
|---|---|
| `firstNonEmpty` | Returns the first non-blank string argument; re-declared 31 times per package instead of shared. |
| `encodeJSONOrFail` | cmd/fak: writes a value as indented JSON to stdout, else reports the encode error and returns 1. |
| `parseFloat` | cmd/dojorsi: Sscanf-based float parse tolerating a trailing format suffix; distinct from stdlib strconv.ParseFloat. |
| `ContainsAny` | strmatch: reports whether a string contains any of the variadic substrings; replaced six per-package copies. |
| `formatFloat` | fleetcap: renders integral floats as bare integers, else with the caller's fallback verb. |
| `ActiveResolver` | abi: returns the registered RegionBackend's Resolver, or nil when no backend is registered. |
| `ledgerPath` | Convention: local/flag variable holding a JSONL ledger file path across command and report packages. |
| `ResolveToken` | Per-post-package Slack bot token resolver: package env vars, then .env.slack.local, then scoreboard fallback. |
| `sortedKeys` | Per-package copy returning lexicographically sorted map keys; superseded by generic maputil.SortedKeys. |
| `numWorkers` | model: package var capping matmul parallelism (GOMAXPROCS default, FAK_WORKERS/FAK_BUDGET overrides); read via NumWorkers(). |
| `ResolveChannel` | Per-post-package Slack channel resolver from env then .env.slack.local; empty means require explicit --channel. |
| `resolveRoot` | cmd/fak: explicit --root, else `git rev-parse --show-toplevel`, else empty string. |
| `verbFlagUsage` | cmd/fak: sets FlagSet.Usage so `fak <verb> --help` prints catalog deep help above flag defaults. |
| `MarshalReport` | benchcli: marshals a bench report with the lineage stamp spliced in; required by the lineage gate. |
| `configureDispatchHelperCommand` | cmd/fak per-OS seam: hides console windows for background dispatch helper processes on Windows; no-op on Unix. |
| `requireMethod` | gateway: enforces one HTTP method on a handler, writing the standard 405 and returning false on mismatch. |
| `LoadRegistry` | Per-package registry file loader (accounts, fleetaccounts, grafanapost, loopmgr) with differing failure semantics. |
| `MarshalJSON` | Repo json.Marshaler implementations rendering closed-vocabulary types (Severity, Outcome, CheckKind, ...) as strings. |
| `ReadLedgerFile` | Per-package best-effort JSONL ledger reader returning nil when absent; cmd/fak adds generic readLedgerFile[T]. |
| `registryPath` | Convention: local/flag variable holding a registry JSON path (loop registry, accounts sessions.json). |
| `AppendLedgerLine` | Per-package pure renderer of one ledger row to its JSONL line; the caller appends the newline. |
| `findRepoRoot` | Stat-walk up to the nearest .git directory, falling back to start; three copies coexist. |
| `marshalArtifact` | turnbench: canonical indented-JSON-plus-newline artifact encoding shared by every report JSON() method. |
| `dispatchSubcommands` | cmd/fak: shared group-verb router; a missing or unknown verb prints the hint and exits 2. |
| `quantizeVecQ8` | Quantizes one activation vector to Q8_0; scalar reference in compute, hot-path copies in model. |
| `ropeRowQKInto` | model: applies RoPE rotation in place to the query and key head rows at one position. |
| `TotalBytes` | Recurring struct-field name for an accumulated byte total across agent, compute, gateway, memgate, model. |
| `defaultSource` | cmd/fak: post source label - FAK_SCOREBOARD_SOURCE env var, else the hostname, else empty. |
| `NormDurability` | Maps a durability-class string to the canonical vocabulary, failing closed to turn; declared in ctxplan and memq. |
| `WriteHeader` | gateway statusRecorder method recording the first status code written so metrics label by actual status. |
| `defaultLoopLedger` | cmd/fak: loop ledger path - FAK_LOOP_LEDGER env var, else .fak/loops.jsonl. |
| `expertName` | model: builds the GGUF tensor name for one MoE expert from layer, expert index, and suffix. |
| `FetchExistingIssues` | Per-package `gh issue list --json` fetcher used to classify create-vs-update for planned issues. |
| `HeadCommit` | Returns `git rev-parse --short HEAD` under a root, or "unknown"; five per-package copies. |
| `verdictName` | Maps abi.VerdictKind to its ALLOW/DENY/... name; seven copies across packages and commands. |
| `WriteReport` | benchcli: lineage-stamped report file write; fleet: validated status report write behind a file-safe key guard. |
| `configHome` | guardrsi: user config dir - XDG_CONFIG_HOME, APPDATA, HOME/.config, then os.UserHomeDir fallback. |
| `envFileValue` | Ten per-package one-line wrappers delegating .env.slack.local key lookup to slackenv.FileValue. |
| `firstString` | Two same-named helpers: first non-blank variadic string (cmd/fak); first string among JSON keys (fleetmon). |
| `ListenAndServe` | gateway Server method: synchronous bind measured as a boot phase, then Serve; warns on no-auth non-loopback bind. |
| `ScoreComponent` | One named scorecard axis (Name, Value, Unit); struct re-declared in four scoring packages. |
| `TrackedTree` | hooks: the `git ls-files` tracked-path set read once and shared across hygiene gates, with a file cache. |
| `UpstreamStatusError` | agent: error carrying the upstream's non-2xx status, Retry-After, and sanitized limit metadata for downstream surfacing. |
