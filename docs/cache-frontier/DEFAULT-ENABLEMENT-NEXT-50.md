---
title: "vCache default enablement - next 50"
description: "A ranked plan for making fak's virtual-cache, O(1) context, provider-cache, and engine-adapter cache work useful by default across pure fak, API/provider, SGLang, vLLM, and llama.cpp contexts."
---

# vCache default enablement - next 50

Status: planning artifact, 2026-06-30. This page does not claim new shipped
behavior. It refreshes the cache concepts and names the next 50 implementation
items that make the cache system useful by default.

The product target is:

> A user should get the right cache behavior without knowing which cache plane is
> firing: pure fak uses kernel-owned KV and O(1) context where it can; API/provider
> mode preserves provider prompt-cache economics without treating them as trust; and
> SGLang/vLLM/llama integrations expose their cache capability through the same
> evidence, scoring, and guard rails.

## Concept Refresh

Use these names consistently. The old habit of saying "the cache" hides too many
different mechanisms.

| Plane | What it is | What can be default-on | What must never be inferred |
|---|---|---|---|
| **O(1) context/query** | `ctxplan`/recall/session memory: the full history is durable and queryable while the resident model view stays bounded. | Lossless store, bounded resident view, query route, staleness/fault labels. | A smaller resident view is not a quality win until task-success or faithfulness is witnessed. |
| **Pure-fak kernel KV** | `internal/model`, `radixkv`, paged KV, exact-span eviction, prefix clone, local scheduler. fak owns the bytes. | Safe local reuse, clone, eviction, pressure relief, quarantine, and preemption when the model runs in-process. | A local KV hit is not a provider-dollar saving, and it must still pass materialization/admission gates. |
| **Provider vCache** | A virtual page table over a provider prompt cache we cannot address directly. It is shaped by request bytes, timing, affinity, and telemetry. | Observe, canonicalize, budget at uncached price, score realized rebates, demote false-warm beliefs. | Provider `cached_tokens` or `cache_read` is cost/latency telemetry only, never proof a local value may be served. |
| **External engine cache** | SGLang/vLLM/llama.cpp/Ollama/LM Studio caches behind the OpenAI-compatible wire. Some expose prefix/radix/paged KV behavior; some only expose symptoms. | Capability inventory, passive observe, wire-neutral labels, adapter-specific witnesses. | "fak fronts the engine" is not the same as "fak controls that engine's cache." |
| **Cross-plane score** | The operator artifact that says which plane fired, how much value it produced, and what to do next. | Per-plane provenance, per-family detail, cold-path correctness bit, false-warm risk, net value. | A single headline hit rate must not collapse provider rebates, local KV reuse, and O(1) context value into one number. |

## Default-On Rule

"Fully enabled" does not mean "always warm everything." It means the safe parts are
on by default, and active cache behavior arms only after evidence says it is useful:

- Always observe: every `guard`, `serve`, and pure-fak run should emit cache events
  when a cache plane is available.
- Always preserve: serializers, tools, system messages, and stable prefixes should be
  arranged so provider and engine caches are not accidentally defeated.
- Always label provenance: WITNESSED kernel events, OBSERVED provider counters,
  FORECAST synthetic estimates, and DECISION verdicts stay separate.
- Always keep the cold path correct: no request may depend on a provider hit landing.
- Arm active warming only when the workload clears concentration, false-warm, secret,
  cold-correctness, and net-value gates.

## Legacy Assumptions To Remove

The current architecture already contains many fences, but the operator story still
lets old assumptions leak in. Remove these as a matter of product design:

1. **"Provider cache is 99% of the story."** Provider cache may dominate token
   rebates, but it can hide that fak's own agentic mechanisms did not fire. Score the
   provider plane and the fak-authored planes separately.
2. **"A cache hit is a cache hit."** Provider cache hits, local KV hits, vDSO hits,
   and O(1) context elisions have different trust, cost, and correctness semantics.
3. **"Synthetic Zipf is a default proof."** Synthetic concentration is a forecast
   only. Observed snapshots and measured anchor files must win by default.
4. **"Warm means known."** Provider warmth is a belief reconciled after a paid call.
   False-warm is the dangerous direction and must demote immediately.
5. **"Recall-by-rebuild is generally good."** Single-unit chain rebuild is usually a
   loss. It stays gated off except for amortized fan-out.
6. **"Base URL compatibility equals cache integration."** A proxy can front SGLang,
   vLLM, or llama.cpp without seeing or controlling their cache. Adapter support must
   state passive, observed, or active.
7. **"O(1) context is just compression."** The product claim is queryable durable
   history plus a bounded resident view. Truncation without query/fault semantics is
   not the target.

## Scoring Refresh

Replace any single "cache score" with a per-plane score and one default-usefulness
fold. The fold should be conservative and explainable:

| Facet | Weight | Evidence |
|---|---:|---|
| Net realized value | 25 | WITNESSED kernel reuse and/or OBSERVED provider rebate, net of writes and gateway cost. |
| Agentic activation | 20 | `ctxplan`, `radixkv`, local KV, cachemeta, or vCache control events actually fired, not only provider counters. |
| Cold-path correctness | 15 | The full request remains correct without a hit; materialization/admission gates pass. |
| Granularity | 15 | Per-family/per-anchor/per-session detail, not one aggregate percentage. |
| Default coverage | 10 | Works in `guard`, `serve`, pure fak, and at least one external engine context. |
| Drift resistance | 10 | False-warm, serializer drift, model/provider vary axes, TTL decay, and secret gates are measured. |
| Operator actionability | 5 | The report names the next action: canonicalize, collect telemetry, build index, arm warming, or disable. |

The default report should also expose four non-overlapping numbers:

- **provider_rebate:** provider-relayed prompt-cache savings, OBSERVED.
- **kernel_reuse:** fak-owned KV/prefix reuse, WITNESSED.
- **context_saved_work:** O(1) resident-view avoided prefill/query work, WITNESSED or FORECAST with label.
- **agentic_activation:** count/rate of fak cache decisions that actually fired.

## Next 50 Items

These are ordered for product usefulness, with pure fak and API/provider work first.

| # | Lane | Item | Default/evidence target |
|---:|---|---|---|
| 1 | Scoring | Define `fak.cache.default_usefulness.v1` with the facets above. | `fak vcache score --json` and cachevalue reports can expose per-plane fields without breaking the old schema. |
| 2 | Scoring | Add `agentic_activation` counters to the report contract. | A provider-only cache run can score provider value high while showing fak-authored cache mechanisms at zero. |
| 3 | Scoring | Split `active_source` into `provider_observed`, `kernel_witnessed`, `context_witnessed`, and `forecast`. | No report can imply provider telemetry is fak-owned reuse. |
| 4 | Scoring | Add a cold-path correctness bit to every cache score. | Active warming is refused when correctness depends on a hit. |
| 5 | Scoring | Make measured anchors outrank synthetic Zipf whenever a snapshot exists. | Synthetic is visibly `FORECAST`; observed family distribution is the default. |
| 6 | API/provider | Persist the `vcachesnapshot` live window by default from `guard` and `serve`. | A finished session leaves a bounded, replayable cache window without extra flags. |
| 7 | API/provider | Attach provider vary axes to every provider-cache row: model, endpoint, reasoning mode, tool set, serializer, region/affinity where known. | A silent mode switch becomes a distinct cold family, not a blended hit-rate drop. |
| 8 | API/provider | Emit false-warm and false-cold alarms from live traffic, not only offline observe. | A believed-warm miss demotes the family and appears in `/debug/vars` or the session footer. |
| 9 | API/provider | Make provider-cache secret classification default-deny for active warming. | Secret/regulated prefixes can still be sent normally, but no pre-warm/pin is scheduled. |
| 10 | API/provider | Add exact serialized-prefix fingerprinting to the cache event path. | Cache keys are hashes of wire bytes, not logical message structs. |
| 11 | Pure fak | Promote the O(1) session query path to a first-class CLI/API surface. | A user can query a real fak session image without using a demo binary. |
| 12 | Pure fak | Wire ctxplan bounded resident views into the live guard/serve loop in shadow mode. | The report says what would be resident, what would fault, and what query would recover. |
| 13 | Pure fak | Move the planned-elision-to-KV-eviction bridge from proof to live HTTP loop behind a gate. | When a span is elided from the resident view, local KV residency shrinks too. |
| 14 | Pure fak | Turn local radix/prefix reuse into a default in-kernel session option when `--engine inkernel` is used. | Shared stable prefixes are cloned or reused without requiring a benchmark harness. |
| 15 | Pure fak | Make paged KV the default under a memory budget once parity witnesses pass on the target models. | Pressure relief frees blocks instead of forcing whole-session reset. |
| 16 | Pure fak | Connect quarantine/result-side rejection to exact-span KV eviction in the served in-kernel path. | Poisoned tool output is removed from resident KV, not only from transcript state. |
| 17 | Pure fak | Add a public prefix-clone API for multi-agent in-kernel fan-out. | The 50-turn x 5-agent value stack becomes a product path, not only a benchmark. |
| 18 | Pure fak | Rehydrate session images into ctxplan plus cachemeta records. | Archived work resumes with queryable history and explicit cache invalidation state. |
| 19 | Pure fak | Put `cachemeta.MaterializeVerdict` in front of every local cache serve path. | Local reuse remains governed by scope, freshness, taint, and quality evidence. |
| 20 | Pure fak | Add a turn-tax adaptive planner that picks reuse, query, or cold prefill per turn. | Pure fak makes a cache decision by default and records why. |
| 21 | API/provider | Extend provider telemetry parsing across OpenAI Chat, OpenAI Responses, Anthropic Messages, Gemini, and xAI-compatible responses. | All API modes can feed the same scorecard when counters exist. |
| 22 | API/provider | Preserve Anthropic `cache_control` and OpenAI-compatible stable prefix bytes through guard/serve transformations. | The gateway's safety layer does not defeat provider reuse by rewriting stable prefixes. |
| 23 | API/provider | Add an explicit "passive only" label for providers with no safe active warm primitive. | The operator sees observe/preserve/score but no active scheduler. |
| 24 | API/provider | Implement provider constants as measured records with freshness, not hard-coded defaults. | TTL/min-prefix/read-discount are `MEASURED` or `HYPOTHESIS` with date and source. |
| 25 | API/provider | Add a cache-budget dry run to `fak guard` startup. | Before a session starts, the user sees expected uncached budget and cache rebate is not pre-credited. |
| 26 | API/provider | Join Track 2 provider-dollar rows into `fak cachevalue report`. | Provider-dollar claims become net OBSERVED economics, not token-only proxies. |
| 27 | API/provider | Add provider-cache card fields for write cost, read rebate, hit rate, false-warm, and agentic activation. | A high provider hit rate cannot hide zero fak-authored cache decisions. |
| 28 | API/provider | Add per-family cache review rows to `docs/cache-frontier/review-ledger.jsonl`. | Weekly reviews can say which families are useful, flat, drifting, or disabled. |
| 29 | API/provider | Gate dedicated warming on break-even, false-warm risk, secret class, and rate headroom. | Active warm does not arm just because a prefix is long. |
| 30 | API/provider | Add send-one-then-fan scheduling for any active provider warm/fan-out path. | Dependents wait until the first request has made the prefix readable. |
| 31 | External engines | Build a cache capability inventory for SGLang, vLLM, llama.cpp, Ollama, and LM Studio. | Each adapter row says passive observe, active warm, exact evict, prefix clone, paged KV, or unknown. |
| 32 | External engines | Add a wire-neutral `engine.CacheCapability` contract. | Gateway reports what the upstream engine can expose without importing engine-specific packages into core. |
| 33 | External engines | Add a vLLM prefix-cache observation adapter. | vLLM-fronted sessions can report observed prefix reuse or explicitly say unavailable. |
| 34 | External engines | Add an SGLang radix/prefix-cache observation adapter. | SGLang-fronted sessions can map radix cache evidence into the same score lanes. |
| 35 | External engines | Add a llama.cpp/llama-server session-cache observation adapter. | llama-backed local sessions report cache state or a passive/no-evidence label. |
| 36 | External engines | Add adapter conformance tests for "fronted" vs "cache-integrated." | A base-url proxy cannot accidentally claim active cache integration. |
| 37 | External engines | Add active warm/fan-out harnesses only for engines whose capability row proves support. | Engine-specific active cache use is opt-in by evidence, not by brand. |
| 38 | External engines | Add per-engine cold-path correctness witnesses. | A cache miss on SGLang/vLLM/llama still sends full required context. |
| 39 | External engines | Compare pure-fak, vLLM, SGLang, and llama paths on the same session geometry. | The value stack is reported as mechanism value plus engine throughput, not blended. |
| 40 | External engines | Surface external-engine cache lanes in `/debug/vars` and `fak cachevalue report`. | Operators see whether external engine cache evidence is missing, passive, or active. |
| 41 | O(1) context | Make "query old session memory" part of the default agent loop contract. | Agents stop relying on growing transcripts or stale recall for prior work. |
| 42 | O(1) context | Add faithfulness/task-success witnesses for bounded resident views. | O(1) context claims can graduate from economics to quality-preserving default. |
| 43 | O(1) context | Add cache invalidation from `fak_changes` into ctxplan/recalled pages. | Cross-agent edits tombstone stale pages before they are served. |
| 44 | O(1) context | Score resident-view decisions by saved prefill, query latency, and miss/fault rate. | Context value is granular and debuggable, not a single compression ratio. |
| 45 | O(1) context | Tie sys-prompt overlays to cache fingerprints. | System prompt changes invalidate affected prefixes instead of silently poisoning reuse. |
| 46 | Remove legacy | Add docs/claims lint for cache headlines that omit plane/provenance. | "99% cache" or "cache win" without provider/kernel/context labels fails review. |
| 47 | Remove legacy | Replace old "provider cache as trust" wording with "cost/latency only" everywhere. | The provider-not-trust invariant is impossible to miss. |
| 48 | Remove legacy | Remove stale docs that imply vCache active provider loop is fully live. | Status pages distinguish shipped observe/score from future active warming. |
| 49 | Remove legacy | Make unsupported active cache paths fail closed with a named reason. | Unknown engine/provider capabilities do not silently fall back to optimistic claims. |
| 50 | Release gate | Add a cache-default readiness gate to maturity/CI. | Regressions in per-plane scoring, cold-path correctness, or provenance labels block default-on claims. |

## First Wave

The first useful slice is items 1-10 plus 11, 12, 21, 24, 27, 31, 32, 36,
46, and 47. That slice makes the system honest by default before it tries to
warm more aggressively:

1. Reports split provider value from fak-authored cache activation.
2. `guard`/`serve` leave replayable snapshots without flags.
3. Pure-fak session memory becomes queryable as a product surface.
4. External engines get a capability vocabulary that prevents overclaiming.
5. Legacy cache language starts failing review when it hides provenance.

Only after that should dedicated warming, chain recall, or engine-specific active
cache paths be armed.

## Completion Test

This goal is complete only when a fresh user can run these surfaces and see a
coherent, per-plane cache story:

```bash
fak guard -- <agent>
fak vcache score --json
fak cachevalue report --since <date> --json
fak cachevalue review --since <date> --json
fak serve --engine inkernel
fak serve --provider openai --base-url <sglang-or-vllm-or-llama-server>/v1
```

The evidence must prove:

- provider rebate is visible but not treated as trust;
- fak-authored cache decisions are counted separately;
- O(1) context/query is available on a real session;
- pure-fak local KV reuse and eviction have live-path witnesses;
- external engine adapters say exactly what cache capability they expose;
- cold-path correctness remains true when every cache hit misses.
