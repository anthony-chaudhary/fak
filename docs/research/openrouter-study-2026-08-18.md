---
title: "OpenRouter study: transferable routing contracts and the gaps fak should close"
description: "A pinned, source-anchored study of OpenRouter's public API, SDKs, docs, releases, issues, and agent skills, reconciled against current fak routing evidence."
date: 2026-08-18
---

# OpenRouter study: transferable routing contracts and the gaps fak should close

**Verdict:** OpenRouter's most useful lesson for fak is not “add another upstream.” It is to make routing constraints a first-class, inspectable request contract. OpenRouter lets a caller say which providers are eligible, which privacy and quantization properties are mandatory, how eligible providers should be ordered, and whether fallback is allowed. Fak has the stronger local enforcement seam and richer task/aspect routing, but its provider choice is still mostly implicit in static model tiers. The highest-value missing spine is a fail-closed provider-constraint policy with an exclusion trace.

This is a dated field-borrow pass, not an endorsement or a parity claim. OpenRouter is a hosted model marketplace/router; fak is an agent kernel and local performance/security checkpoint. Their best defaults differ.

## Scope and evidence boundary

Observed on **2026-08-18**:

- Public documentation index: [OpenRouter `llms.txt`](https://openrouter.ai/docs/llms.txt), plus the linked pages for [principles](https://openrouter.ai/docs/guides/overview/principles), [provider selection](https://openrouter.ai/docs/guides/routing/provider-selection), [model fallbacks](https://openrouter.ai/docs/guides/routing/model-fallbacks), [Auto Exacto](https://openrouter.ai/docs/guides/routing/auto-exacto), [prompt caching](https://openrouter.ai/docs/guides/features/prompt-caching), and [presets](https://openrouter.ai/docs/guides/features/presets).
- Live unauthenticated [`GET /api/v1/models`](https://openrouter.ai/api/v1/models): **413 models**, **59 model namespaces**, **16 `:free` variants**; all 413 returned architecture, top-provider, and pricing objects in the captured observation. These are observed catalog values, not fak-controlled guarantees.
- [`OpenRouterTeam/typescript-sdk`](https://github.com/OpenRouterTeam/typescript-sdk) pinned at [`6d6e01a`](https://github.com/OpenRouterTeam/typescript-sdk/commit/6d6e01aae14eaa997a7aa66c2c433393d821ece4), release `v1.2.42`, Apache-2.0.
- [`OpenRouterTeam/go-sdk`](https://github.com/OpenRouterTeam/go-sdk) pinned at [`eb2fafb`](https://github.com/OpenRouterTeam/go-sdk/commit/eb2fafbf78d63bb0560d1e324071be2d495a9503), Apache-2.0. The generated [`ProviderPreferences`](https://github.com/OpenRouterTeam/go-sdk/blob/eb2fafbf78d63bb0560d1e324071be2d495a9503/models/components/providerpreferences.go) type is the most compact machine-readable statement of the routing contract.
- [`OpenRouterTeam/ai-sdk-provider`](https://github.com/OpenRouterTeam/ai-sdk-provider) pinned at [`b96b207`](https://github.com/OpenRouterTeam/ai-sdk-provider/commit/b96b20799eadeb72a180ef021b85254fc1500746), Apache-2.0.
- [`OpenRouterTeam/skills`](https://github.com/OpenRouterTeam/skills) pinned at [`f8fdfb7`](https://github.com/OpenRouterTeam/skills/commit/f8fdfb73b85a5e94109577a3aebc260f151a6dac). GitHub exposed no repository license on the observation date, so its prose is inspiration-only and must not be copied.
- Public TypeScript SDK releases, the latest 100 issues/PRs, repository history, and discussions were inspected. The release stream is generated from a changing API specification (multiple releases on 2026-08-17); SDK issue/PR history is useful for contract-integration failure modes, not for proving the hosted router's private implementation.

**Important limit:** the hosted routing engine is not in these repositories. Public docs and generated SDKs prove exposed behavior and schema. They do **not** prove the internal ranking algorithm, health sampling, commercial controls, or implementation quality. Claims below stay on the public side of that line.

## Feynman-simple value frame

- **For:** operators who need fak to choose among interchangeable model providers without silently relaxing privacy, cost, or capability requirements.
- **Problem:** fak can choose a model tier and enforce residency, but cannot yet express the full eligible-provider set and show why every rejected provider was rejected.
- **Today:** eligibility is spread across tier configuration, engine residency checks, static fallback, and provider-specific setup.
- **Better because:** one typed constraint contract can fail closed before dispatch and leave a deterministic decision receipt.
- **Witness:** table-driven tests where only/ignore, data policy, quantization, price, and fallback constraints each remove candidates with stable reason codes; an impossible policy returns no route rather than relaxing a requirement.

Problem centrality: **Enabling**. P1 managed context: the router receives a small typed policy rather than prose. P2 net-true efficiency: hard filters prevent paid or slow retries to ineligible endpoints; any ranking gain still needs live measurement. P3 bounded adaptation: constraints bound learned/telemetry routing. P4 integrated operations: the exclusion trace joins routing, audit, and incident diagnosis.

## What OpenRouter actually exposes

### 1. Eligibility is separate from ranking

`ProviderPreferences` separates hard filters from ordering:

- `only`, `ignore`, and explicit `order` constrain provider identity;
- `require_parameters` constrains request compatibility;
- `data_collection` and `zdr` constrain data handling;
- `quantizations` constrains serving format;
- `max_price` constrains prompt, completion, image, and request price;
- `sort` selects price, throughput, or latency ordering;
- `allow_fallbacks` controls whether another provider may be attempted.

This separation is the key transferable mechanism. A ranking score must never outrank a hard privacy or compatibility condition.

### 2. The default is operational, not merely cheapest

The provider-selection docs describe a default that first prefers providers with stable throughput and low latency, then load-balances among the preferred set. Price, throughput, and latency are explicit alternative sorts. A provider's recent errors can cause temporary de-prioritization. This is a useful frontier marker for fak's open telemetry work, but it is only documentation of service behavior; no public implementation was available to audit.

### 3. Fallback exists at two levels

OpenRouter exposes a model fallback list and provider fallback within a model. The docs enumerate triggers including context-length, moderation, rate-limit, and provider errors. The useful idea is typed fallback reason and ordered next action—not “retry everything.” Fak should preserve its stricter fail-closed constraints while borrowing the typed chain.

### 4. Routing can be task-specialized

Auto Exacto routes tool-calling requests using provider/model benchmark signals. The specific benchmark and hosted policy are not portable evidence, but the shape is: evaluate a capability on the workload that exercises it, then route only inside the hard-eligible cohort. Fak already goes further with aspect-level routing and ensembles; OpenRouter reinforces that generic model rank is not enough.

### 5. Catalog, configuration, and receipts are product surfaces

The model catalog returns normalized architecture, context, pricing, and endpoint facts. Presets give named reusable request/routing configuration. Generation and analytics APIs/skills treat per-request usage as queryable state. Together they make routing operable: discover choices, bind a policy, and inspect the outcome.

### 6. Prompt caching is provider-specific

OpenRouter documents different automatic and explicit cache behavior by provider. This validates fak's existing provider-cache normalization direction: do not pretend every provider has one cache contract. Requested hints, effective hints, cache reads/writes, TTL, and billed tokens need separate provenance.

## Current fak witness

The repository self-query was run with:

```text
fak-dev feature query "OpenRouter-inspired routing provider selection fallback presets usage accounting privacy prompt caching model discovery" --json
```

It surfaced current capability cards and source paths; every material result below was then checked against the current tree. Relevant module versions at the observation point were `internal/modelroute@r83+g093331453e`, `internal/engine@r67+gbe89afb6b9`, and `internal/gateway@r661+g7738a21047`.

| OpenRouter-shaped capability | Fak status | Current witness | Disposition |
|---|---|---|---|
| Per-task / per-aspect model selection | **PRESENT, stronger axis** | `internal/modelroute/modelroute.go`, `internal/modelroute/aspect_routing_test.go`, `docs/model-routing.md` | **DEFAULT** |
| Capability-aware tier selection | **PRESENT** | `modelroute.Requirements`, `TierConfig.RequiredCapabilities`, `internal/modelroute/capability_routing_test.go` | **DEFAULT** |
| Budget-aware model tiering | **PRESENT** | `BudgetMaxUSD`, `BudgetRemainingUSD`, conservative fallback tests in `internal/modelroute/modelroute_test.go` | **DEFAULT** |
| Region/residency enforcement | **PRESENT** | `internal/engine/engine.go` `residencyGate`; `internal/engine/residency_test.go` | **DEFAULT** |
| Ordered model/tier fallback with reason | **PRESENT / PARTIAL** | `modelroute.Decision` records requested/resolved tier and `FallbackReason`; execution fallback exists, but provider- and error-class policy is not one public typed contract | **DEFAULT**, extend |
| Ensemble / best-of / judge reduction | **PRESENT, broader than OpenRouter request contract** | `modelroute.EnsembleConfig`, reducers and judge tests | **OPTIONAL-MODULE** |
| Provider `only` / `ignore` eligibility | **ABSENT as a unified routing contract** | No corresponding fields in `modelroute.Requirements` or `TierConfig` | **DEFAULT** hard-filter spine |
| Data-collection / zero-data-retention requirement | **PARTIAL** | residency is enforced, but provider data-policy eligibility is not represented in `modelroute` | **DEFAULT** hard filter |
| Required provider parameter support | **PARTIAL** | coarse capabilities exist; request-parameter compatibility is not a provider eligibility contract | **DEFAULT** hard filter |
| Quantization eligibility | **PARTIAL / planned** | model capabilities can represent traits; open issue [#6225](https://github.com/anthony-chaudhary/fak/issues/6225) targets declared quantization routing | **OPTIONAL-MODULE** until endpoint facts are trustworthy |
| Max provider price and price sort | **PARTIAL** | model/tier budget exists; endpoint-specific prompt/completion/request ceilings and provider sorting do not | **DEFAULT** ceiling; **OPTIONAL-MODULE** sort |
| Live fastest/throughput provider sort | **ABSENT / planned** | open [#4588](https://github.com/anthony-chaudhary/fak/issues/4588) and telemetry epic [#600](https://github.com/anthony-chaudhary/fak/issues/600) | **OPTIONAL-MODULE** behind minimum-sample and stale-data gates |
| Pareto cost/latency/throughput policy | **ABSENT** | no provider Pareto frontier policy in `modelroute` | **WATCH** until the single-metric telemetry spine is measured |
| Live normalized model/endpoint catalog | **PARTIAL / planned** | static `internal/modelroute/registry.go`; gateway catalog integration is part of [#5632](https://github.com/anthony-chaudhary/fak/issues/5632) | **DEFAULT** normalized facts with provenance |
| Named reusable routing presets | **PARTIAL** | tiers/config files are reusable, but there is no portable preset identifier/version overlay contract | **RECIPE** first; promote only with demonstrated cross-client demand |
| Provider-specific cache semantics and receipts | **PRESENT / PARTIAL** | `docs/integrations/routers.md`, cache telemetry and open [#7308](https://github.com/anthony-chaudhary/fak/issues/7308), [#7309](https://github.com/anthony-chaudhary/fak/issues/7309), [#7310](https://github.com/anthony-chaudhary/fak/issues/7310) | **DEFAULT** normalization, provider adapters optional |
| Authoritative provider cost ledger | **PARTIAL / planned** | gateway usage ledger exists; session-keyed provider authority remains [#6651](https://github.com/anthony-chaudhary/fak/issues/6651) | **DEFAULT** |
| Hosted marketplace, credits, OAuth, rankings | **ABSENT by product choice** | fak supports upstream routers rather than becoming their billing marketplace | **EXCLUDE** |

## Best-default frontier

The combined best default is:

1. **Hard eligibility first:** policy floor, residency/data handling, required request features, quantization, and price ceilings.
2. **Task/aspect routing second:** choose the model class that can do the work.
3. **Operational ranking third:** rank only eligible endpoints using sufficiently fresh, sufficiently sampled latency/throughput/cost evidence.
4. **Typed fallback fourth:** retry only for declared error classes and never by relaxing a hard condition.
5. **Receipt always:** report selected route, excluded candidates with reason codes, metric provenance/freshness, and fallback steps.

OpenRouter contributes the explicit provider-preference contract. Fak contributes the non-bypassable local enforcement seam, aspect routing, conservative budget behavior, and receipt opportunity. Neither side alone is the desired end state.

## Bounded-superset opportunities

These alternatives should remain modular rather than displacing the default:

- **Price-first / latency-first / throughput-first:** useful for explicit cohorts, unsafe as universal defaults.
- **Pareto routing:** useful after trustworthy endpoint telemetry exists; premature before #4588/#600.
- **Free-model variants:** a recipe with explicit reliability/privacy warnings, not an implicit fallback.
- **Named presets:** useful at client boundaries, but the resolved policy and version must be recorded so a mutable preset cannot erase provenance.
- **External OpenRouter upstream:** keep as an OpenAI-compatible integration path; do not couple core routing policy to OpenRouter-specific billing or identity.

## Ranked borrow ledger

| Rank | Candidate | Classification | Why / next evidence |
|---:|---|---|---|
| 1 | Typed provider eligibility plus exclusion trace | **ABSENT → DEFAULT** | Smallest missing control-plane spine; filed as [#7354](https://github.com/anthony-chaudhary/fak/issues/7354); prove fail-closed behavior. |
| 2 | Live endpoint latency with freshness/sample gates | **ABSENT → OPTIONAL-MODULE** | Already [#4588](https://github.com/anthony-chaudhary/fak/issues/4588); requires measured runtime evidence. |
| 3 | Normalized live catalog with fact provenance | **PARTIAL → DEFAULT** | Already under [#5632](https://github.com/anthony-chaudhary/fak/issues/5632); must distinguish authored policy from observed provider facts. |
| 4 | Typed error-class fallback chain | **PARTIAL → DEFAULT** | Extend existing fallback without retrying policy denials or incompatible requests. |
| 5 | Endpoint-specific max-price contract | **PARTIAL → DEFAULT** | Extend budget controls; price facts need timestamp/currency/unit provenance. |
| 6 | Required-parameter compatibility filter | **PARTIAL → DEFAULT** | Prevent provider retries that could never honor the request. |
| 7 | Quantization constraint | **PARTIAL → OPTIONAL-MODULE** | Covered by [#6225](https://github.com/anthony-chaudhary/fak/issues/6225). |
| 8 | Requested-versus-effective cache receipt | **PARTIAL → DEFAULT** | Covered by [#7308](https://github.com/anthony-chaudhary/fak/issues/7308)-[#7310](https://github.com/anthony-chaudhary/fak/issues/7310). |
| 9 | Session-keyed authoritative provider cost | **PARTIAL → DEFAULT** | Covered by [#6651](https://github.com/anthony-chaudhary/fak/issues/6651). |
| 10 | Named/versioned presets | **PARTIAL → RECIPE** | First prove repeated cross-client configuration drift. |
| 11 | Pareto endpoint selection | **ABSENT → WATCH** | Depends on #4588/#600 and an explicit objective/tie-break contract. |
| 12 | Auto Exacto-style workload benchmark routing | **PARTIAL → OPTIONAL-MODULE** | Fak's aspect router is the seam; add tool-use telemetry only when benchmark transfer is validated. |
| 13 | OpenRouter SDK/client adapters | **PARTIAL → OPTIONAL-MODULE** | Generic adapter work is already [#6797](https://github.com/anthony-chaudhary/fak/issues/6797); avoid core vendor coupling. |
| 14 | Provider marketplace/billing/OAuth | **ABSENT → EXCLUDE** | Peripheral to fak's kernel value and would make the tail wag the dog. |
| 15 | Copy OpenRouter agent skills | **EXCLUDE** | No detected license; borrow concepts only. |

## Source-history findings that change the recommendation

- The generated SDK release cadence is high: `v1.2.18` through `v1.2.42` landed between 2026-08-09 and 2026-08-18. Fak should consume a versioned normalized adapter, not mirror the entire vendor schema in core policy.
- SDK issues and PRs repeatedly touch generated-type drift, streaming/error handling, and API compatibility. That supports a narrow boundary adapter and contract tests rather than broad direct dependency.
- OpenRouter's public skills expose analytics, generations, models, benchmarks, OAuth, media, and migration as separate skills. The transferable spirit is modular operator surfaces; the unlicensed prose and vendor-specific workflows are not importable.
- No public source was found for the hosted provider-ranking engine. Live-health and ranking claims therefore remain docs-observed and need independent fak witnesses before adoption.

## Decision and next work

Ship one minimal spine: add a provider-eligibility object at the `modelroute` seam with provider `only`/`ignore`, data-policy, required-parameter, and max-price constraints; evaluate it before ranking; return stable exclusion reasons; fail closed when no candidate remains. Keep quantization and live metric sorting in their existing dedicated issues so the first slice remains independently provable.

Issue [#7354](https://github.com/anthony-chaudhary/fak/issues/7354) carries the missing spine. Existing issues cover every other material follow-up found in this pass: #4588, #600, #5632, #6225, #6651, #6797, and #7308-#7310. Pareto routing remains WATCH and should not become backlog until endpoint telemetry is real enough to state a falsifiable done condition.

