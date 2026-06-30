---
title: "fak glossary: core vocabulary, shared memory, preflight vs inflight"
description: "The canonical fak glossary: overloaded core terms (session, agent, context, model, memory, tool/skill, steering), shared-memory sense splits, the active memory issue map, the preflight/inflight/prefill gate vocabulary, and the vCache streaming economy (cache rebate, read rebate vs write premium, net saving, the per-turn verdict words)."
---

# Glossary: core vocabulary, shared memory, and before/during words

A few words in this codebase look related but aren't, and a couple are reused at
several layers with no shared code. That overload is the usual source of confusion.
This page pins each one down.

This is the canonical docs-lane glossary for the term-conflation audit tracked in
[#721](https://github.com/anthony-chaudhary/fak/issues/721). The dated worklist
([`docs/notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md`](notes/VOCAB-DISAMBIGUATION-WORKLIST-2026-06-24.md))
is the source audit; this page is the stable reader-facing contract.

## Canonical overloaded vocabulary

| term | senses in this repo | house rule |
|------|---------------------|------------|
| **session** | token decoder (`model.Session` / `model.BatchSession`), reloaded core image (`recall.Session`), live drive state (`internal/session.Table` / `State`), wire DTO (`gateway.SessionState`), and per-session context planner (`agent.SessionPlanner`) | qualify it. Bare "session" in architecture docs means live drive state only when the surrounding path is `internal/session`; otherwise say decoder session, core-image session, gateway session state, or planner session. |
| **agent** | `fak` as kernel/reference monitor, the external untrusted guest loop, the `fak agent` demo verb, and the `internal/agent` wire/loop package | say kernel/reference monitor for `fak`; say guest or external agent for the program being mediated. Do not make "agent" name both sides of the trust boundary in the same paragraph. |
| **context** | token window, planner-resident view for this turn, and the result-admission target (`ctxmmu`) | "fits in the window" is not "entered the view", and "entered context" means the post-result admission gate allowed model-visible bytes. |
| **model** | a routed provider/engine binding, and `internal/model.Model`, fak's own in-kernel transformer | use engine, provider, or routed LLM for the first sense; reserve model for the owned transformer when precision matters. |
| **memory** | KV working memory, durable/recall memory, and procedural memory (a cached skill/context view) | "working memory" always means the KV cache. Durable memory is cross-session recall; procedural memory is a reusable skill view, not a fact store. |
| **shared memory** | shared KV/prefix reuse, shared CAS/blob refs (`abi.Resolver` / `Ref`), a typed region/window (`internal/region`, planned in #646), an external L3 shared-KV tier, or a remote provider prompt cache (vCache) | qualify the transport and ownership. "Shared memory" alone must not imply RDMA, CUDA IPC, durable recall, or trust to reuse; say shared KV prefix, CAS ref, region window, L3 tier, or provider cache. |
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

## The vCache streaming economy: what `fak guard` prints per turn

When you run `fak guard -- claude --debug-stats`, the gateway streams **one line per
turn** to stderr that prices fak's caching value, and the `fak guard` exit summary plus
`fak vcache observe` / the `fak_vcache_*` metrics report the same totals. The vocabulary
on those surfaces is fak's own (the vCache work, [#218](https://github.com/anthony-chaudhary/fak/issues/218)
/ [#715](https://github.com/anthony-chaudhary/fak/issues/715)-[#720](https://github.com/anthony-chaudhary/fak/issues/720)).
A representative line:

```
fak-turn trace=t1 ok saved=20.9k tok (85% of prompt) cache=healthy compact=ok finish=end_turn
```

The whole vocabulary rests on one accounting law, inherited from `internal/vcachestar`
and `internal/callavoid`: **cost is always booked at the full *uncached* price first, and a
confirmed cache hit refunds part of it.** That refund is the rebate. The law's one-liner —
*"an avoided call is a realized rebate, never a trust claim"* — is the discipline: a rebate
is booked only from a hit the provider *confirmed* (`cache_read_input_tokens` came back
non-zero), never from fak *believing* a prefix was warm. Belief predicts; only telemetry
rebates.

| term | what it means | the catch that the word encodes |
|------|---------------|----------------------------------|
| **(cache) rebate** | the cost refunded by a confirmed cache hit, in input-token-equivalents | booked only on a telemetry-confirmed read; warmth belief alone never rebates. `internal/vcachestar: CostBooking.RebateTokens` |
| **read rebate** | the read axis of the rebate: each `cache_read` token billed at 0.1× base instead of 1× — a 0.9×/token refund | a read is the *only* axis that pays you back; on its own it overstates the value |
| **write premium** | the first write to a cache costs *more* than uncached: 1.25× base at the 5-minute TTL, 2.0× at the 1-hour TTL | this is why caching is a net win only once reads accrue (break-even is 2 requests at 5m, 3 at 1h). `internal/gateway: CacheWrite5mMultiplier` |
| **net saving / `saved=` (token-equiv)** | the honest fak-vs-no-cache number: **read rebate − write premium**, in input-token-equivalents | a fresh, cold-write turn reads **negative** (`saved=-25 tok`) — a real loss the writes haven't repaid, which a read-only number would have hidden. `internal/gateway: ProviderCacheNetSavings` |
| **baseline / cost / multiplier** | `baseline` = what the session *would* have cost with no cache (every token at 1×); `cost` = what it actually cost; their ratio is the **multiplier** (`7.22x`) | baseline is a projection over OBSERVED counts, not a fak-authored claim |
| **the turn verdict word** | one glance at the turn's state, folded from the net saving + prefix health | see the four values below. `internal/gateway: turnVerdict` |

The four turn-verdict words (the `ok` slot in the line above):

| word | means |
|------|-------|
| **cold** | no provider cache activity at all this turn (a first turn, or a non-cached path) |
| **warming** | cache activity, but the writes have not yet been repaid by reads — net saving still ≤ 0 |
| **ok** | a proven net saving on a healthy (or not-yet-scored) prefix |
| **degraded** | the rolling health says the prefix is decaying / stale, or a reset is recommended |

> **The one thing to internalize.** fak reports whether *fak's hop* **preserved** the
> provider's cache — not the provider's raw cached-token count, which only measures whether
> Anthropic's cache was warm. That is why the streamed `fak-turn` line, the `fak_vcache_*`
> metrics, and `fak vcache observe` can never disagree: they all fold the same counters
> through one engine (`vcacheProofFromCounters` / `vcachegov.ProveTelemetrySavings`). Every
> input is OBSERVED (provider-relayed); the saving is a realized rebate, never a fak trust
> claim — the same OBSERVED-vs-WITNESSED discipline the [conflation scorecard](CONFLATION-SCORECARD.md)
> enforces across all of fak's reported numbers.

## Mnemonics

- **preflight** — the **before** gate; it can **refuse** before the thing starts (a call
  before dispatch, a node before serve, a worker before spawn).
- **inflight** — the **during** state; what's running *right now* and not yet done (a live
  request, a held KV lease). Observed, never refused.
- **prefill** — a **model phase**, not a check; it fills the KV cache from the prompt.
- **rebate** — a **cost refund** from a *confirmed* cache hit, booked over an uncached
  baseline; the net saving subtracts the **write premium**, so a cold-write turn reads
  negative until reads repay it.
