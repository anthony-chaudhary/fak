---
title: "The caching frontier: where prompt & KV caching is going"
description: "Field guide to fak's cache frontier: SOTA serving optimizations, cross-engine cache capabilities, and the default-enablement backlog — honest about shipped vs planned."
slug: level-5-the-caching-frontier
keywords:
  - cache frontier
  - SOTA serving optimizations
  - prompt cache research
  - KV cache research
  - cross-engine cache capability
  - vCache default enablement
  - RadixAttention PagedAttention
  - fak caching ladder
date: 2026-07-10
---

# The caching frontier

*You are on **Level 5 of 5** of the [fak caching ladder](README.md).*

> **Short answer.** This is a living research frontier, not a settled feature. The
> caching stack — provider prompt caches, kernel-owned KV, cross-engine caches, and
> O(1) context — moves fast, and fak's design rule is to keep every plane's evidence
> separate and label *witnessed* vs *planned* honestly. This page is the map: what the
> SOTA landscape looks like, what fak proves today, and where to read next when it
> changes. Assume the tree is more current than this page; the linked source files are
> the ground truth.

## The map: SOTA serving optimizations

The "tuned SOTA" serving stack fak benchmarks against is a bundle of optimizations —
KV/prefix caching, batched and continuous inference, quantization, SIMD/fused kernels,
paged attention, tensor parallelism, speculative decoding, request routing, tool
batching ([sota-optimizations.md](../sota-optimizations.md)). Most of these live **in
the serving engine** (llama.cpp, vLLM, SGLang, Ollama), not in fak — they apply whether
or not fak fronts the engine.

fak's contribution is a **governance layer** on top: policy-driven eviction (evict by
quarantine verdict, not just LRU), `internal/radixkv` (a RadixAttention implementation,
reported at an 86.7% hit rate on an agents workload), and `internal/kvmmu` paged KV with
policy-aware invalidation. The benchmarked **1.5–4× vs tuned SOTA** is attributed to
fused serving, cross-agent prefix sharing, and cache-aware scheduling — explicitly *not*
raw model speed (fak reports near parity there) and *not* basic KV reuse (SOTA already
has it). Treat those multipliers as claims with baselines attached, never loose numbers.

## The program: five planes, one honesty rule

The cache-frontier program's core move is refusing to say "the cache" — it names five
planes and forbids blending their evidence
([DEFAULT-ENABLEMENT-NEXT-50.md](../../cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md)):

| Plane | What it is | Trust rule |
|---|---|---|
| **O(1) context/query** | Durable, queryable history with a bounded resident view. | A smaller resident view is not a quality win until faithfulness/task-success is witnessed. |
| **Pure-fak kernel KV** | `internal/model`, `radixkv`, paged KV, exact-span eviction. fak owns the bytes. | A local KV hit is **WITNESSED** reuse — not a provider-dollar saving. |
| **Provider vCache** | A virtual page table over a provider cache fak cannot address. | Provider `cached_tokens`/`cache_read` is cost/latency telemetry **only** — never proof to serve a local value. |
| **External engine cache** | SGLang/vLLM/llama.cpp caches behind the OpenAI-compatible wire. | "fak fronts the engine" ≠ "fak controls that engine's cache." |
| **Cross-plane score** | The operator artifact that says which plane fired and by how much. | One headline hit-rate must never collapse the planes into a single number. |

The scorecard schema (`fak.cache.default_usefulness.v1`) folds **seven** weighted,
non-overlapping facets and deliberately keeps their low sub-scores instead of smoothing
them into one headline: `net_realized_value`, `agentic_activation`,
`cold_path_correctness`, `granularity`, `default_coverage`, `drift_resistance`, and
`operator_actionable` ([score.go](../../../internal/vcachescore/score.go)). Beyond that
fold, the program's *target* is to also surface four plane-separated headline numbers —
provider_rebate (OBSERVED), kernel_reuse (WITNESSED), context_saved_work (WITNESSED or
FORECAST), and agentic_activation — but the
[backlog](../../cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md) frames that as a report the
default *should also* expose; of the four, only `agentic_activation` is a live schema
field today. The direction of travel and the four product lanes — dogfood, demo, product
surface, evidence — are set by the
[operating plan](../../CACHE-FRONTIER-OPERATING-PLAN.md).

## Where each external engine stands

The cross-engine story is deliberately conservative. Each engine gets exactly one
verdict from a **closed vocabulary** — `passive observe`, `active warm`, `exact evict`,
`prefix clone`, `paged KV`, or `unknown` — stating what fak can *prove it does with that
engine's cache today*, not the engine's internal mechanism
([external-engine-cache-capability-inventory.md](../../cache-frontier/external-engine-cache-capability-inventory.md)):

| Engine | Verdict today |
|---|---|
| **SGLang** | `passive observe` (RadixAttention reuse observed; only whole-radix reset) |
| **vLLM** | `passive observe` (paged-block residency observed; only whole-prefix reset) |
| **llama.cpp / llama-server** | `unknown` (no in-tree cache observation adapter) |
| **Ollama** | `unknown` (no in-tree cache observation adapter) |
| **LM Studio** | `unknown` (no in-tree adapter at all) |

No engine claims `active warm`, `exact evict`, or `prefix clone`. `unknown` here means
honest absence of evidence, not a guessed capability — and this whole lane is classified
`gen/second-next` in the program map, so expect these rows to move as observation
adapters land.

## What's default-on vs still planned

Honest status, because the gap is the point:

- **Witnessed / on:** `--managed-cache` defaults to `auto` and activates *only* when the
  session provably bills an API key (`--api-key-env` resolves a key on the Anthropic
  wire); a subscription-OAuth or passthrough session stays passive. When active it
  upgrades the stable-prefix `cache_control` breakpoint to Anthropic's 1h TTL tier, so a
  re-entry after a >5m idle gap costs a 0.1× cache **read** instead of re-writing the
  prefix (the 1h write is a one-time 2× premium), witnessed on `/metrics` as
  `fak_gateway_cache_ttl_upgrade_total`. `guard`/`serve` also persist a bounded,
  replayable provider-cache snapshot on session exit without extra flags, and the
  `fak.cache.default_usefulness.v1` scoring schema ships.
- **Planned / gated / open:** live false-warm/false-cold alarms on in-flight traffic; a
  first-class verb to query a *real* session image (not a demo binary); the four
  plane-separated headline numbers above; most external-engine observation adapters.
  Active warming is **not** live — `fak vcache status` reads: *"full vCache provider loop
  not yet executing warms."*

When in doubt about a number or a flag, read the source file, not this page. The backlog
rows carry per-item *landed vs open* annotations for exactly this reason.

## Try it

```bash
fak manage claude                                # run as usual; managed cache is auto
fak vcache score --json                          # per-plane score + the 7-facet usefulness fold
fak vcache status                                # what's live vs gated right now
fak cachevalue report --since 2026-06-22 --json  # OBSERVED/WITNESSED value, when available
```

## See also

- [Level 4 — The kernel-owned KV cache](level-4-kernel-kv-cache.md): the mechanism this
  frontier extends — addressable KV, exact-span eviction, WITNESSED accounting.
- [SOTA serving optimizations](../sota-optimizations.md) — what "tuned SOTA" means and
  which optimizations fak implements vs fronts.
- [Cache frontier operating plan](../../CACHE-FRONTIER-OPERATING-PLAN.md) · [vCache default enablement — next 50](../../cache-frontier/DEFAULT-ENABLEMENT-NEXT-50.md) · [external-engine cache capability inventory](../../cache-frontier/external-engine-cache-capability-inventory.md) — the program, the backlog, and the cross-engine verdicts.
- [Cache frontier review ledger](../../cache-frontier/README.md) — the recurring witnessed-value review.

---

*[← Level 4](level-4-kernel-kv-cache.md) · [full ladder](README.md)*
