# OpenAI prompt-caching watchlist — 2026-08-30

## Result

OpenAI's current prompt-caching contract is no longer only automatic caching plus
`prompt_cache_key` and `prompt_cache_retention`. GPT-5.6 and later add typed
`prompt_cache_options`, explicit content-block breakpoints, observable cache-write tokens,
and separate cache-write pricing. fak already handles the earlier contract well, but it
must now treat OpenAI caching as a model- and endpoint-versioned capability rather than one
uniform provider feature.

The highest-priority follow-up is [#10385](https://github.com/anthony-chaudhary/fak/issues/10385):
add current-schema parity without confusing the legacy maximum-retention policy with the new
minimum-TTL control. Existing issues retain ownership of lifecycle economics (#7310), exact
provider attribution (#9552), and per-turn cache bounds (#10188).

## Scope and method

- **Access date:** August 30, 2026.
- **Scope:** official OpenAI prompt-caching behavior, API schemas, economics, routing,
  retention, and data-policy implications; then a source-and-test inventory of fak.
- **Authority:** official OpenAI documentation and API reference listed under
  [Official sources](#official-sources). Repository claims are tied to named source or test
  paths. GitHub issues are used only for fak's planning history.
- **Interpretation rule:** generated API fields are not generalized across models or
  endpoints unless the guide documents that scope. An absent local match is reported as a
  non-finding, not proof that support is impossible.
- **Change scope:** this note and issue reconciliation only. The required implementation has
  a separate typed witness and should not be smuggled into a research commit.

## Official behavior inventory

### Automatic and explicit caching

- Prompt caching is enabled by default on supported models. A hit requires an exact rendered
  prefix match through a cache breakpoint. OpenAI describes the stored artifact as encrypted
  KV tensors rather than token text.
- **GPT-5.6 and later:** caching begins at 1,024 visible input tokens. The usage response can
  report the exact eligible boundary; hidden tokens are excluded. The preferred request
  surface is `prompt_cache_options`.
- **Earlier supported models:** the normal documented threshold is 2,048 visible tokens,
  although some models may cache shorter prefixes. Reported cached tokens are rounded down
  to a multiple of 128. Implicit breakpoint intervals are model-dependent; GPT-5.5 uses
  2,048-token intervals.
- GPT-5.6+ supports `prompt_cache_options.mode: "explicit"` and content-block
  `prompt_cache_breakpoint: {"mode":"explicit"}` markers. With explicit-only mode, a request
  without a marker neither reads nor writes prompt cache. A request may write at most four
  breakpoints. Top-level `instructions` cannot directly carry a marker.
- GPT-5.6 implicit mode places a breakpoint at the latest eligible message. Explicit markers
  may be combined with implicit mode, but the implicit breakpoint consumes one of the four
  write slots.

### Keys, routing, and rate behavior

- `prompt_cache_key` is a routing/grouping hint: related requests are more likely to reach a
  machine that has their cached prefix. It does not pin traffic or guarantee a hit.
- Caches are machine-local. The guide warns that traffic above roughly 15 requests per minute
  for one prefix/key combination can overflow to another machine and incur an initial miss.
  Deterministic sharding is therefore preferable to either one hot global key or unbounded
  per-request key cardinality.
- Cache entries are isolated by organization and processing region. Cached input still counts
  toward tokens-per-minute limits.

### Usage and economics

- Responses reports cache reads and writes under
  `usage.input_tokens_details.cached_tokens` and
  `usage.input_tokens_details.cache_write_tokens`.
- Chat Completions uses `usage.prompt_tokens_details.cached_tokens` and
  `usage.prompt_tokens_details.cache_write_tokens`. The API reference defines the latter as
  the unadjusted prompt-token count written to cache.
- GPT-5.6 cache reads cost 0.1 times ordinary input and cache writes cost 1.25 times ordinary
  input. Earlier models have model-specific cached-input prices and no documented separate
  write premium. Cost accounting must therefore be model-scoped, not merely
  `input - cached = uncached`.
- OpenAI publishes distinct Standard, Batch, Flex, and Fast-mode prices. Batch generally uses
  a separate rate-limit pool and a 50% discount; Flex uses Batch rates and can receive
  additional prompt-cache discounts. Service tier is part of the matched economic envelope.

## Endpoint, model, and retention matrix

| Surface | Model scope | Request controls | Read/write usage fields | Retention contract | Important limits |
|---|---|---|---|---|---|
| Responses | GPT-5.6+ | `prompt_cache_key`; preferred `prompt_cache_options.{mode,ttl}`; eligible content blocks may carry `prompt_cache_breakpoint` | `input_tokens_details.cached_tokens`; `input_tokens_details.cache_write_tokens` | `prompt_cache_options.ttl`; only documented value/default is `"30m"`, a minimum lifetime refreshed by reuse or write | 1,024 visible-token minimum; at most four writes per request |
| Chat Completions | GPT-5.6+ | Same top-level controls; breakpoint markers are content-block scoped | `prompt_tokens_details.cached_tokens`; `prompt_tokens_details.cache_write_tokens` | Same GPT-5.6+ minimum-TTL contract | Endpoint-specific usage nesting must be preserved |
| Responses and Chat Completions | earlier supported models | `prompt_cache_key`; deprecated `prompt_cache_retention` where supported | cached-token details; write-token availability must be read from the endpoint/model response, not assumed | `"in_memory"` or `"24h"` where supported; in-memory is typically 5–10 minutes idle and up to one hour; `24h` is typically available after about 30 minutes and may remain up to 24 hours | Normal threshold 2,048 visible tokens; cached count rounded down by 128 |
| Extended-retention subset | `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4`, `gpt-5.2`, named GPT-5.1/Codex variants, `gpt-5`, `gpt-5-codex`, `gpt-4.1` | Deprecated `prompt_cache_retention` | Model/endpoint response contract | Enumerated support only; GPT-5.5 and GPT-5.5 Pro accept only `"24h"` | Do not infer support for every earlier cached model |

`prompt_cache_retention` and `prompt_cache_options.ttl` are not aliases. OpenAI describes the
former as a maximum-retention selection and the latter as a minimum lifetime. In particular,
`"24h"` must not be mechanically migrated to `"30m"`.

## fak support and gap matrix

| Official behavior | Current fak handling/evidence | Risk if ignored | Action | Witness needed |
|---|---|---|---|---|
| Stable `prompt_cache_key` routing hint | Implemented in `internal/agent/adapters.go`; covered by `internal/agent/responses_prompt_cache_key_test.go` and closed #5186 | Key churn destroys affinity; one hot key can overflow provider-local routing | no change | Keep deterministic-key tests and add a high-volume sharding receipt when policy changes |
| Legacy `prompt_cache_retention` negotiation | `internal/agent/cachehint.go` models `in_memory` and `24h`; adapter emission and tests exist; closed #7309 records the shipped contract | Treating this as universal can send unsupported values or preserve a deprecated default indefinitely | docs | Model/endpoint capability table proving each emitted value |
| GPT-5.6+ `prompt_cache_options.{mode,ttl}` | No matching typed field was found in the inspected agent, gateway, adapter, or contract paths | fak cannot request current explicit mode or express the new minimum-TTL contract | code | Responses and Chat request-shape tests plus one live GPT-5.6 receipt |
| Explicit `prompt_cache_breakpoint` content markers | No local field or test match found | A shared prefix may never be written at the intended boundary; explicit-only requests may silently get no cache | code | Block-shape tests, no-marker negative test, four-write-limit test, live divergent-suffix hit |
| 1,024-token GPT-5.6 threshold and exact eligible boundaries | Existing notes and cache evidence predate the current contract; no model-scoped boundary witness was found | Threshold heuristics can label valid reuse as impossible or compare unlike envelopes | benchmark | Controlled 1,023/1,024-token and divergent-suffix ablation with usage receipt |
| Earlier-model 2,048 threshold, 128-token reporting granularity, model-dependent intervals | Existing managed-cache studies cover prefix behavior but do not constitute a current OpenAI model matrix | Apparent miss/hit deltas can be rounding or breakpoint-placement effects | docs | Versioned model matrix backed by official contract and targeted live samples |
| Responses `input_tokens_details.cached_tokens` | Parsed by `internal/canon/token_usage.go`; fixture `internal/canon/testdata/token_usage/openai.json`; gateway usage tests cover reporting | Losing the endpoint nesting makes provider-cache evidence unverifiable | no change | Preserve fixture and gateway normalization tests |
| Responses/Chat `cache_write_tokens` | Canonical `cache_write_tokens` exists and provider-cache tests consume it, but the inspected OpenAI fixture contains only `cached_tokens`; direct endpoint-specific parsing was not established | Generic write accounting may look supported while OpenAI writes remain zero or lose provenance | observability | Endpoint-specific fixtures for both nesting shapes and a live nonzero write receipt |
| GPT-5.6 read price 0.1x and write price 1.25x | `internal/gateway/cache_pricing.go` is peer-dirty during this audit; no clean committed-tip witness was accepted for the new model contract | Net savings can be overstated when write premiums, verification, or service tier are omitted | benchmark | Committed-tip pricing tests and net-true end-to-end receipt naming model and tier |
| Machine-local key routing and roughly 15 RPM overflow behavior | fak has affinity machinery in `internal/session/cache_affinity.go`, but provider overflow is not a guaranteed controllable boundary | A globally shared key can lower hit rate while dashboards blame content churn | observability | Per-key/prefix request rate, hit/write split, and deterministic shard identity |
| Service-tier interaction | Provider contracts and metrics retain service information, but cache evidence must join it for economic comparisons | Batch/Flex/Fast results can be compared as if they share price and limits | observability | Receipt joining model, endpoint, service tier, cached reads, writes, latency, and cost |
| Retention versus ZDR and residency | Current cache-hint policy includes privacy/retention concepts; no current GPT-5.6 policy witness was found | A performance hint can cross the operator's storage or regional-processing boundary | code | Policy tests for ZDR/residency capability, fail-closed unsupported combinations, scrubbed live receipt |

### Local evidence boundary

This audit inspected committed and working-tree paths directly, including
`internal/agent/cachehint.go`, `internal/agent/adapters.go`,
`internal/canon/token_usage.go`, `internal/openaiadapter/`, `internal/cachemeta/`,
`internal/modelroute/providercontract.go`, `internal/gateway/`, `internal/metrics/`, and
`internal/session/cache_affinity.go`. The shared checkout was peer-dirty. In particular,
`internal/gateway/cache_pricing.go` had an unrelated modification, so this note does not claim
that uncommitted work as shipped support.

The canonical `cache_write_tokens` field proves that fak has a provider-neutral place to store
write evidence. It does **not**, by itself, prove that both OpenAI endpoint wire shapes are
parsed, attributed, and priced correctly.

## Prioritized awareness and action list

1. **P0 awareness — version the contract.** Treat GPT-5.6+ prompt caching and earlier-model
   caching as separate capabilities. Retain legacy behavior for compatible models; do not
   replace every retention hint globally.
2. **P1 code — implement current request surfaces (#10385).** Add typed
   `prompt_cache_options` and eligible content-block breakpoints for Responses and Chat, with
   endpoint/model checks and write-slot validation.
3. **P1 observability — prove reads and writes (#9552, #10188).** Parse both endpoint nesting
   shapes, retain provider provenance, and expose cache writes separately from misses.
4. **P1 benchmark — update net-true economics.** Include the 1.25x write premium, 0.1x reads,
   service tier, setup/recovery, quality, and operating envelope. A high hit rate alone is not
   a savings claim.
5. **P2 policy — join retention to data controls.** Make ZDR, region, endpoint, model, and
   requested TTL visible to admission. Unsupported or ambiguous combinations should not be
   guessed.
6. **P2 routing — bound key temperature.** Monitor request rate per stable prefix/key and
   shard deterministically when a hot group loses locality. Avoid random keys, which erase
   reuse, and one universal key, which overloads routing.
7. **P2 benchmark — characterize breakpoints.** Test shared-prefix/divergent-suffix workloads,
   including the GPT-5.6 implicit-breakpoint trap and explicit-only no-marker behavior.

## Stable-prefix and operational hazards

- **Instructions and history:** developer messages, OpenAI-provided instructions, tool calls
  and results, and prior messages contribute to the rendered prefix. Timestamps, tenant data,
  or changing session facts near the front invalidate reuse from the first changed token.
- **Tool definitions:** names, descriptions, schemas, order, and tool instructions are prefix
  material. Keep the definitions stable and select availability with `tool_choice: "none"` or
  allowed-tool controls rather than deleting or reordering tools per turn.
- **Structured Outputs:** `text.format` and its JSON schema add rendered instructions. A schema
  edit can move or invalidate the cache boundary even when user text is unchanged.
- **Images, files, and audio:** multimodal content participates in the rendered context. Any
  changed content before a breakpoint can invalidate all later reuse.
- **History mutation:** summarization, truncation, message rewriting, and context compaction can
  reset provider reuse despite semantic equivalence. fak must distinguish provider-prefix
  preservation from its own logical-context correctness.
- **Breakpoint placement:** GPT-5.6's latest eligible implicit breakpoint may be after two
  requests diverge. A long common prefix alone does not prove a reusable entry was written at
  that boundary; use an explicit marker when the stable prefix ends before a dynamic suffix.
- **Write limits:** implicit mode can consume one of four write slots. A client-side planner
  must count effective writes, not only explicit markers.
- **Key cardinality and temperature:** stable keys improve affinity, but keys are neither cache
  identities nor locks. Observe prefix/key traffic and shard hot groups deterministically.

## Data-policy implications

- OpenAI states that API data is not used to train models unless the customer opts in.
  Standard abuse-monitoring logs may be retained for up to 30 days; application state has a
  separate contract.
- Responses and Chat Completions are generally eligible for Zero Data Retention, but prompt
  caching is documented as application state: encrypted KV tensors may be held in GPU-local
  storage, with a documented maximum expiration of 24 hours.
- For earlier models supporting both legacy modes, non-ZDR organizations default to `"24h"`
  while ZDR organizations default to `"in_memory"`. The guide also says supported models use
  extended prompt caching for organizations without ZDR.
- Caches do not cross regional-processing boundaries. In regions without Regional Processing
  support, extended caching may require temporary processing or storage outside the selected
  region. Regional processing may also carry a 10% pricing uplift for eligible models released
  on or after March 5, 2026.
- Admission therefore needs a joined decision across model, endpoint, organization data-control
  posture, processing region, retention request, and service tier. A latency-saving hint is not
  safe merely because no token text is stored.

## Explicit uncertainties and non-findings

- The prompt-caching guide says reads consider the latest **50** breakpoints, while generated
  API-reference material observed during this audit says **80**. This is an official-source
  inconsistency. fak should not enforce either as a durable client-side limit without a
  model/endpoint witness.
- `"30m"` is documented as GPT-5.6+'s minimum TTL and default; the exact effective expiration
  can be longer and is not observable or guaranteed. It is not a promise of expiration at
  exactly 30 minutes.
- The docs establish ZDR eligibility, prompt caching as application state, and a maximum
  retention boundary, but do not provide one exact GPT-5.6-under-ZDR expiration rule.
- “Supported models” is not one immutable list: automatic caching, explicit mode, and legacy
  extended retention have different scopes.
- No `prompt_cache_options` or `prompt_cache_breakpoint` implementation was found in the
  inspected local paths. That is a dated repository non-finding, not a claim that another
  branch, generated client, or future commit cannot contain it.
- The existence of canonical `cache_write_tokens` is not evidence that OpenAI previously lacked
  a write counter. As of this access date, official Responses and Chat schemas do expose one.

## Issue reconciliation

- [#10385](https://github.com/anthony-chaudhary/fak/issues/10385) — **open, new:** current
  OpenAI `prompt_cache_options`, explicit-breakpoint, and model/endpoint capability parity.
- [#7309](https://github.com/anthony-chaudhary/fak/issues/7309) — **closed:** prior provider
  cache-key and retention-hint negotiation; historical foundation, not current-schema parity.
- [#7310](https://github.com/anthony-chaudhary/fak/issues/7310) — **open:** provider TTL
  lifecycle governed by net reuse value.
- [#9552](https://github.com/anthony-chaudhary/fak/issues/9552) — **open:** exact provider
  token/cache attribution.
- [#10188](https://github.com/anthony-chaudhary/fak/issues/10188) — **open:** per-turn cache
  bounds and provider-path observability.
- [#5186](https://github.com/anthony-chaudhary/fak/issues/5186),
  [#5188](https://github.com/anthony-chaudhary/fak/issues/5188), and
  [#5190](https://github.com/anthony-chaudhary/fak/issues/5190) — **closed:** key churn,
  provider-blind reporting, and duplicate affinity-producer history.

## Official sources

All were accessed August 30, 2026. The older `platform.openai.com/docs/...` forms redirect to
these canonical developer pages.

- Prompt Caching guide: <https://developers.openai.com/api/docs/guides/prompt-caching>
- Responses create reference:
  <https://developers.openai.com/api/reference/resources/responses/methods/create/>
- Chat Completions create reference:
  <https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create/>
- API pricing: <https://developers.openai.com/api/docs/pricing>
- Data controls: <https://developers.openai.com/api/docs/guides/your-data>
- Batch API: <https://developers.openai.com/api/docs/guides/batch>
- Flex processing: <https://developers.openai.com/api/docs/guides/flex-processing>
