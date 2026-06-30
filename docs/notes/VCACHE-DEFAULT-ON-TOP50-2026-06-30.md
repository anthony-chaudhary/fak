---
title: "Top-50: make fak's caching useful-by-default and honestly attributed"
description: "A ranked, actionable backlog (2026-06-30) to (a) decompose every cache saving by mechanism and owner so the provider's prompt-cache stops masquerading as fak's win, (b) make the vCache agentic loop (M2-M5) fire by default, and (c) push the vBlock/anchor abstraction into pure-fak, API, and sglang/vllm/llama paths. Each item has a default-on lever and a witness."
---

# Top-50: caching useful-by-default, honestly attributed

Companion to [`VCACHE-CONCEPT-REFRESH-2026-06-30.md`](VCACHE-CONCEPT-REFRESH-2026-06-30.md).
Ranking is by **leverage on the goal**: kill the "99% provider" conflation first
(items 1–12), then make the agentic loop actually fire (13–28), then push the
abstraction across every serving path (29–42), then remove legacy assumptions and
harden scoring (43–50).

Each item: **what · why · where (file) · default-on lever · witness**. "Witness"
is the artifact that proves it shipped — a test, a metric, a ledger row, a demo —
never a self-report. Every item is sized to one disjoint lane.

Acceptance gate for the whole list (the north-star): on a normal `fak guard --
claude` run with **no flags**, the operator sees one attribution line —
`avoided spend = provider P% + fak F% (compaction C, kv-reuse K, vdso V)` — and
`F` is non-zero on a multi-turn session because M2 fired by default.

---

## A. Granular attribution — kill the "99% provider" conflation (1–12)

1. **Per-mechanism saving record.** Add a `MechanismSavings` struct (provider-read,
   provider-write-premium, compaction-shed, kv-reuse, vdso-avoid) accumulated per
   session. *Why:* one frame, five owners. *Where:* `internal/gateway/metrics.go`
   (`AdjudicationSummary`). *Lever:* always on. *Witness:* unit test summing the
   five back to total avoided spend.

2. **One attribution headline.** Replace the standalone provider line at
   `cmd/fak/guard.go:1279` with `avoided spend = provider P% + fak F%` and a
   one-line fak breakdown. *Why:* the operator's only default cache number must
   name both owners. *Lever:* always on. *Witness:* `guard.go` exit-summary test
   asserting both P and F appear.

3. **`fak-turn` line shows fak's slice.** Extend `debug_stats.go:255` so the
   per-turn line carries `prov=… fak=…` not just `saved=` (provider-only today).
   *Where:* `internal/gateway/debug_stats.go`. *Witness:* `debug_stats_test.go`.

4. **Provider field on the ledger row.** Add `Provider string` to
   `cachevalueledger.Row` and `SavingsRow`. *Why:* Anthropic vs OpenAI vs SGLang
   reuse behave differently; today they're indistinguishable. *Where:*
   `internal/cachevalueledger/`, `cachevaluereport/track2.go`. *Witness:* fold test
   grouping by provider.

5. **Mechanism field on the ledger row.** Add `Mechanism` so Track-1/Track-2 folds
   can decompose. *Where:* same. *Witness:* `cachevaluereport` per-mechanism bucket
   test.

6. **Wire Track-2 per-session append (rung B, #1303).** `track2.go` defines
   `SavingsRow` but **nothing writes it live**; wire the guard/serve/run exit to
   append `docs/nightrun/cache-savings.jsonl`. *Why:* the OBSERVED-$ P&L is empty
   by construction today. *Where:* `cmd/fak/guard_child.go`, `serve.go`,
   `run_model.go`. *Lever:* always on. *Witness:* a real appended row after a
   guard session.

7. **"Why is fak's slice small?" diagnostic.** When `F` is near-zero, print the
   reason: `M2 off` / `no multi-turn reuse` / `anchor-starved` (the #1407
   pathology already detected at `guard.go:1303`). *Why:* the goal's exact
   complaint — distinguish "not firing" from "nothing to save." *Witness:*
   exit-summary test across the three causes.

8. **Provider-vs-fak split on `/metrics`.** Add `fak_cache_saved_by_owner{owner=…}`
   and `…_by_mechanism{mechanism=…}` gauges. *Where:* `internal/gateway/metrics.go`.
   *Witness:* metrics scrape test naming all five series.

9. **Conflation-scorecard rule for the cache headline.** Extend the conflation
   scorecard so an unlabeled cache number (owner not named) is a HARD defect.
   *Where:* `tools/conflation_scorecard.py`. *Witness:* scorecard row.

10. **`fak vcache observe` panel: owner split.** The observe panels already label
    OBSERVED vs DECISION; add an owner row (provider vs fak) to the family table.
    *Where:* `cmd/fak/vcache_observe.go`, `internal/vcacheobserve/panels.go`.
    *Witness:* observe golden test.

11. **Two-track card shows fak's authored slice.** `RenderTwoTrack` shows provider-$
    only; add the fak-authored token column (compaction + kv-reuse + vdso) beside it.
    *Where:* `internal/cachevaluereport/track2.go`. *Witness:* render test.

12. **Glossary + CLAIMS update.** Document the owner-split vocabulary and the new
    attribution line so the contract is stable. *Where:* `docs/glossary.md`,
    `CLAIMS.md`. *Witness:* doc-appeal + claim-repro scorecards green.

---

## B. Make the agentic loop fire by default — M2-M5 ON (13–28)

13. **Register M5 governor into the kernel.** It is "deliberately NOT registered"
    (`vcachegov/doc.go:25`). Wire a live loop that feeds it `PrefixStats` from
    `cacheobs`/`cachemeta.Lifecycle` and acts on its pin/lazy/evict verdict. *Why:*
    the headline feature exists only on paper. *Lever:* `--vcache-governor=on`
    defaulting **on** once witnessed. *Witness:* a governor decision recorded in a
    real session journal. **Progress 2026-06-30:** the live gateway now records
    DECISION verdicts from real provider-cache families into `/metrics` and a
    hash-chained `/debug/vars` journal; acting on pin/lazy/evict remains open.

14. **M2 star-anchor canonicalization as a default pre-flight gate.**
    `RecommendLayout` is advisory-only; make it *apply* the hoist (volatile content
    to the tail) before send, not just report it. *Where:* `internal/vcachestar/`,
    `cachemeta.RecommendLayout`. *Lever:* `--vcache-anchor=on` default on for the
    Anthropic path. *Witness:* before/after prefix-byte-stability test; a real turn
    whose anchor was rewritten. **Progress 2026-06-30:** the Anthropic raw
    preflight now applies the volatile-to-tail layout to top-level `system[]`
    blocks before inserting `cache_control`, and the gateway witness proves two
    different per-request UUIDs produce the same forwarded cache prefix. Full M2
    star manifests, sibling fan-out, and cross-surface canonicalization remain open.

15. **First-natural-request warming (M2, no dedicated warm).** Ensure the first
    request of a sibling burst warms the anchor and siblings read it — ordering
    only, free. *Where:* gateway scheduler. *Witness:* a two-request trace where
    request 2 reports `cache_read>0` on the anchor.

16. **Warmth-belief estimator on the live path (M1).** Fold `cached_tokens`
    feedback into `vcachecal` per family, decay on the clock, revive on a confirmed
    read. *Where:* `internal/vcachecal/estimator.go` ← `gateway`. *Lever:* on.
    *Witness:* prediction-error rate emitted to `/metrics` from a real session.

17. **Use warmth belief to steer, not just score.** On a believed-warm entry that
    reads 0, demote + byte-diff (localize the invalidator via
    `Diverge`/`FirstDivergeTokenOffset`) and log it. *Why:* the lethal
    "manifest says HIT, provider says MISS" (Law A1). *Witness:* a forced-mismatch
    test that fires the demote path.

18. **Per-provider calibration probe (M1).** A cadence job that probes TTL `T`,
    `Mₘᵢₙ`, `r` per (provider, model) and feeds the constants instead of
    hard-coding 0.1/1.25/2.0. *Where:* `internal/vcachecal/probe.go`,
    `tools/vcache_openai_probe.py`. *Witness:* a dated calibration row per provider.

19. **Dedicated warming under the break-even gate (M3).** `max_tokens:0` (explicit)
    / decode-1 (implicit), only when expected reuse ≥ break-even (≥2 Anthropic,
    ≥3 OpenAI). *Where:* new warmer in `vcachegov`/gateway. *Lever:*
    `--vcache-warm=auto` (off until a known burst). *Witness:* a warm that converts
    `k` misses to reads, proven from telemetry.

20. **Send-one-then-fan barrier (M3/M4).** Gate dependents on the first *streamed
    content delta*, not HTTP 200. *Why:* the warm-then-fan race (Law C2). *Where:*
    gateway scheduler. *Witness:* a race test that fails without the barrier.

21. **M4 recall cost-gate enforced live.** `ProveRecall` runs only from the CLI;
    make the gate actually *refuse* a net-negative single-unit chain rebuild at
    runtime and fall back to fresh prefill. *Where:* `internal/vcachechain/`.
    *Lever:* recall gated off by default (correct), but the *gate* must be live.
    *Witness:* a live refusal logged with the loss ratio.

22. **Affinity-key routing (M5 / §9).** Set `prompt_cache_key`/prefix-hash
    consistently across a chain so requests land on the same warm shard. *Where:*
    `vcachegov/affinity.go` ← gateway. *Witness:* correlated-hit test; a real
    chain that keeps its shard.

23. **Secret/retention gate live (Law D4).** `vcachegov/secret.go` classifies
    prefixes; enforce "never warm secrets" on the live warmer path. *Lever:* on.
    *Witness:* a regulated-content prefix that is refused warming.

24. **Rate-limit warm budget (M5 / §5.5).** Budget warms inside RPM/TPM headroom;
    degrade by warming fewer anchors, never by 429-ing. *Where:*
    `vcachegov/warmbudget.go`. *Witness:* a budget that caps warms under a synthetic
    tier.

25. **20-block intermediate breakpoints (M4 / Law C3).** Place a breakpoint every
    ~15 content blocks in long agentic turns to stay inside Anthropic's lookback,
    capped at 4. *Where:* `anthropic_cachebp.go`. *Witness:* a >20-block turn that
    keeps its rebate.

26. **Governor decisions on the live journal.** Every pin/lazy/evict/warm verdict
    writes a hash-chained journal row (same shape guard-RSI reads). *Why:* makes the
    loop auditable and feeds RSI. *Witness:* journal rows after a real session.
    **Progress 2026-06-30:** the gateway writes a bounded hash-chained
    `vcache_governor_journal` tail on `/debug/vars`; durable guard-RSI ingestion is
    still the follow-on.

27. **`fak vcache status` reflects live state.** Once registered, `status` must
    read the live governor/loop state, not the hard-coded "not wired" string.
    *Where:* `cmd/fak/vcache.go:494`. *Witness:* status flips to "live" in a run
    with the loop on.

28. **Default-on token-defaults lock for the new levers.** Add the M2/M5 default-on
    values to `token_defaults.go` with a paired lock test (the pattern
    `TestTokenDefault_CtxViewDefaultsOn` already uses). *Witness:* the lock test.

---

## C. Across every serving path — pure fak, API, sglang/vllm/llama (29–42)

29. **vBlock abstraction flows into pure-fak serve.** Bind `radixkv.Tree` nodes to
    `cachemeta.Entry` identities so the in-kernel prefix cache speaks the same
    vBlock vocabulary as the provider path. *Where:* `internal/radixkv/`,
    `internal/cacheobs/`. *Witness:* a served turn whose KV-reuse is attributed as a
    vBlock hit.

30. **Pure-fak KV-reuse on the attribution headline.** When `fak serve --engine
    inkernel`, mechanism #4 is the *dominant* fak saving — surface it in the same
    owner split as the provider path. *Witness:* serve exit shows `fak F%` driven by
    KV-reuse.

31. **Command SGLang radix, don't just observe it.** Today fak reads
    `cached_tokens` and polls `FAK_SGLANG_RADIX_URL`; add anchor-aware routing that
    *steers* requests to the worker with the hottest resident prefix. *Where:*
    `internal/engine/sglang.go`, fleet router. *Witness:* a routing decision that
    raised the radix hit rate in a bench.

32. **vLLM KV-event-driven warm-set.** Use the BlockStored/BlockRemoved stream to
    maintain a live warm-set and route on it. *Where:* `internal/engine/vllm.go`.
    *Witness:* residency-index routing test on real events.

33. **llama.cpp integration (none exists today).** Add a `llama-server` adapter
    that reads its prompt-cache / slot reuse signals so the `--n-cpu-moe` Ampere
    path (the GLM-5.2 wall) gets fak's attribution + anchor shaping. *Where:* new
    `internal/engine/llamacpp.go`. *Witness:* a llama.cpp serve whose reuse is
    attributed.

33b. **Map the llama.cpp slot/prompt-cache model first.** Before the adapter,
    document llama-server's `--slots`, `--prompt-cache`, and `cache_prompt` reuse
    semantics so the adapter reads real signals, not invented ones. *Witness:* a
    short capability note in `docs/serving/`.

34. **Unified cache-signal interface across engines.** One `CacheSignals` reader
    (provider-API, sglang, vllm, llama, pure-fak) so attribution code is
    engine-agnostic. *Where:* `internal/engine/`. *Witness:* interface test with all
    adapters.

35. **OpenAI chat-completions cache visibility.** The chat adapter reads no cache
    signal; switch the default OpenAI path to the Responses API (which reports
    `cached_tokens`) or surface the gap loudly. *Where:* `internal/agent/adapters.go`.
    *Witness:* an OpenAI turn with a non-null cached-token attribution.

36. **Gemini explicit `CachedContent` route (Law D5).** Route long-lived/sensitive
    vBlocks to Gemini's store-once handle (it has a deletion primitive, escaping
    shard roulette). *Where:* `adapters.go` Gemini path. *Witness:* a CachedContent
    create/use/delete round-trip.

37. **Anchor canonicalization shared by all paths.** The M2 canonicalizer (item 14)
    must run for provider AND engine paths, not Anthropic-only. *Witness:* a
    pure-fak and an SGLang turn both showing a hoisted anchor.

38. **Cross-worker shared-prefix reuse (the real multi-agent win).** The
    SESSION-VALUE-STACK 60.3x/4.1x result lives or dies on cross-worker prefix
    sharing; make the fleet router share warm anchors across workers by default.
    *Where:* fleet router + `PrefixResidencyIndex`. *Witness:* a multi-worker run
    where worker 2 reads worker 1's anchor.

39. **Gateway endpoint to inspect/steer cache.** Add `/v1/fak/cache` (read warm-set,
    request a warm, request an evict) so a caller can drive vCache through the API,
    not just observe it. *Where:* `internal/gateway/http.go`. *Witness:* an API
    round-trip that warms then reads.

40. **MCP `fak_cache_query` tool.** Expose the warm-set / attribution to an agent
    over MCP, beside `fak_memory_query`. *Where:* `internal/gateway/mcp.go`.
    *Witness:* tool-call test.

41. **Engine-cache eviction parity.** Where the engine supports it (pure-fak exact
    span), expose addressable eviction; where it doesn't (sglang/vllm whole-prefix),
    say so in the attribution. *Witness:* an eviction-scope row per engine.

42. **Integration-recipe docs per engine.** One copy-pasteable "turn on fak caching
    in front of {sglang,vllm,llama,Anthropic,OpenAI}" recipe. *Where:*
    `docs/serving/`. *Witness:* agent-readiness scorecard recipe rows.

---

## D. Remove legacy assumptions, harden scoring (43–50)

43. **Default score = realized, not forecast.** `fak vcache score` defaults to a
    synthetic `s=1.74` Zipf when no snapshot is present; make the realized snapshot
    the default and the forecast the loud fallback. *Where:* `cmd/fak/vcache.go:407`.
    *Witness:* score reads `active_source=telemetry` after a real session.

44. **Drop the hard-coded multiplier assumption per provider.** Replace the
    constants 0.1/1.25/2.0 with the M1-calibrated per-provider values where
    available. *Where:* `cache_pricing.go`, `vcachecal`. *Witness:* a priced turn
    using a calibrated `r`.

45. **Stop calling gated engines "up."** `vcache status` and docs say M4/M5 are
    "up"; until registered, say "implemented, off-path." (Resolved by item 27 once
    they're live; until then, relabel.) *Witness:* status string test.

46. **Measure `s` before trusting vCache (§5.2).** Compute workload concentration
    from real traffic and refuse the warm-the-tail strategy when `s≤1`; the right
    move is to *manufacture skew* via aggregation. *Where:* `vcachecal/concentration.go`.
    *Witness:* a measured-`s` row gating the strategy.

47. **Per-mechanism 2x gate in the scorecard.** `vcache score` grades one blended
    multiplier; add a per-owner gate so a provider-only win can't satisfy a fak 2x.
    *Where:* `internal/vcachescore/score.go`. *Witness:* a test where provider-2x but
    fak-1x fails the fak gate.

48. **Honesty fence: warmth never gates correctness (Law A2).** Add an architest /
    lint that fails if any live path elides resent context because "the provider has
    it." *Where:* `internal/architest/`. *Witness:* the lint catching a planted
    violation.

49. **Retire the offline-only framing of `fak vcache`.** Once M1-M5 are live, the
    CLI verbs become a *report on the live loop*, not a standalone proof; update the
    usage text and `docs/cli-reference.md`. *Witness:* cli-reference diff + doc
    scorecard.

50. **Weekly cache-frontier review starts from the owner split.** Update
    `CACHE-FRONTIER-OPERATING-PLAN.md`'s weekly loop to read the per-mechanism
    per-provider P&L first, so "is fak's cache paying off?" is answered by fak's
    slice, not the provider's. *Witness:* a dated review row using the new fields.

---

## Mapping to tickets

The keystone epic and child issues filed from this plan are tracked in
[`VCACHE-GATES-ON-TICKETS-2026-06-30.md`](VCACHE-GATES-ON-TICKETS-2026-06-30.md).
Every gate-enablement item (section B) carries a paired **QA** acceptance
(honesty test + witness) and **dogfood** acceptance (it runs on our own
`fak guard`/`fak serve` sessions and writes a ledger/journal row), so a gate can
only flip default-on once it is proven on our own traffic.
