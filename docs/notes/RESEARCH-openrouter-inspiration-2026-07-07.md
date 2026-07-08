# Research: OpenRouter — what to borrow, what fak already has (2026-07-07)

Filed to explore integration/inspiration from OpenRouter for fak's routing spine. The
framing is deliberate and matches `docs/adoption/compare/vs-routers.md`: **OpenRouter and
fak are complements, not competitors.** A router answers *which model* serves a request;
fak governs *which effects* a tool call may have and routes *per-aspect* with ensembles.
So the useful question is not "how do we become OpenRouter" — it is "what does OpenRouter
express that fak's routing does not yet, and where does it attach?"

Confidence: OpenRouter mechanics below are from its public docs (provider routing, model
routing, `/models`); verify volatile field names against the live API before coding.

---

## 1. OpenRouter, distilled (the mechanisms worth studying)

- **Unified OpenAI-compatible endpoint.** One key, one wire, 300+ models across many
  upstream providers. Also normalizes response shape (choices always an array, `delta` vs
  `message`, unified `usage`/`finish_reason`, unsupported-param passthrough/ignore).
- **Automatic provider load-balancing + failover.** Under one model id, requests are
  balanced across the providers that serve it, weighted by ~inverse-square of price, and a
  provider that errored in the last ~30s is skipped (a live outage window). Balances on
  observed latency / error-rate / availability. This is TRANSPORT-level reliability (5xx,
  rate-limit, empty completion) — NOT correctness (a wrong-but-`200` answer is not caught).
- **Provider-preferences object** (per-request routing constraints):
  `sort` (price|throughput|latency), `order` (explicit provider priority), `only`/`ignore`
  (allow/deny providers), `allow_fallbacks` (bool), `require_parameters` (only providers
  supporting every requested param/tool), `data_collection: allow|deny`, `zdr`
  (zero-data-retention only), `quantizations` (filter by quant), **`max_price`** (ceiling on
  $/Mtok in+out — fail if nothing qualifies), `preferred_max_latency`,
  `preferred_min_throughput`.
- **Model-level fallback array.** `models: [a, b, c]` + `route: "fallback"` — try the next
  MODEL if one fails. Distinct from provider failover under a single model.
- **Slug variants.** `:floor` (cheapest), `:nitro` (fastest/highest-throughput), plus
  latest-aliases that auto-resolve to the newest snapshot.
- **Normalized model catalog** (`GET /api/v1/models`): per model `id`, `context_length`,
  `pricing{prompt,completion,image,request}`, `architecture{modality,tokenizer}`,
  `top_provider`, `supported_parameters`, `per_request_limits`.
- **Zero-completion insurance.** An empty/errored generation is not billed.

---

## 2. What fak ALREADY mirrors (do not reinvent)

| OpenRouter mechanism | fak equivalent | Where |
|---|---|---|
| OpenAI-compat + multi-wire | `TranscriptAdapter` (openai / openai-responses / anthropic / gemini / xai) | `internal/agent/adapters.go` |
| Provider selection + injection | `Config.Provider`; guard child-env (`ANTHROPIC_/OPENAI_BASE_URL`) | `internal/gateway/gateway.go`, `cmd/fak/guard_provider.go` |
| Cost/size/latency routing + fallback chain | `Router.Route`→`Decision{Tier,Fallbacks}`; `StrategyCostBased` (cheapest that fits); `ErrNoTier` structured refusal | `internal/gateway/routing.go` |
| Route-around-failed-model | `SetHealth` (manual) + `DemoteModel(model,cooldown)` (self-expiring, on entitlement-403) | `internal/gateway/routing.go` |
| Per-aspect + ensemble routing (the differentiator) | `Aspect`; `Plan`/`Member`/`Reduction` (first/vote/best-of/all-reduce/concat); `Route`/`Combine` | `internal/modelroute/modelroute.go`, `judge.go` |
| Price book / cost estimate | passive cost LENS: `PriceBook`, `DefaultPrices`, `EstimateSavings`, `--prices` | `internal/modelroute/cost.go` |
| Anthropic-wire third party (GLM/Qwen/DeepSeek) | route-profile template with closed refusal vocab | `internal/gateway/deepseek_anthropic.go` |
| Backend catalog (docs) | 47-target compatibility matrix incl. OpenRouter | `docs/integrations/compatibility-matrix.md` |
| Cache-aware placement | prefix-residency + power-of-two-choices (SGLang-Router-style) | `internal/gateway/residency_router.go` |

fak's tier router at `routing.go` is notably close to OpenRouter's provider routing already
— it has a cost strategy, an ordered fallback chain, health flags, and a fail-closed
`ErrNoTier`. The gaps below are the deltas on top of that, not greenfield.

---

## 3. Gaps — OpenRouter expresses it, fak's routing does not (yet)

1. **Price as an ACTIVE constraint.** fak's cost is a *passive, post-hoc estimate*
   (`cost.go` is explicit: "a cost LENS … never a bill"). OpenRouter's `max_price` is a
   *fail-closed gate*: no provider under the ceiling ⇒ refuse. fak has all the inputs
   (`Tier.CostPerMTok`, `PriceBook`) but never turns them into a ceiling.
2. **Retention / privacy as a caller-facing routing knob.** OpenRouter's
   `data_collection: deny` / `zdr` = "only route where data isn't retained." fak enforces
   this idea *more strongly* as a fail-closed residency floor — but there is no per-request
   `retention`/privacy preference that a caller sets and that the Decision surfaces. The
   floor is there; the *knob mapped onto it* is not.
3. **Live health-driven failover.** `DemoteModel` fires narrowly (entitlement-403) and
   `SetHealth` is manual. There is no live 5xx / rate-limit / latency scoreboard with an
   outage window feeding the router. (This IS the open learned-routing item, #600.)
4. **Normalized model catalog — record EXISTS, not surfaced.** `GET /v1/models`
   (`internal/gateway/http.go:1072`) advertises only the ONE served model (two in dual
   mode); its Codex-shaped metadata block (`http.go:1080`) is a STUB — every field is a
   hardcoded constant (`context_window: 272000` for *every* model), not a lookup. Yet fak
   ALREADY has the normalized record: `internal/modelscore.ModelEvidence` carries context
   window + cost{in,out} + benchmark scores WITH provenance (`modelscore.go:118`). It is
   SHIPPED in shape but populated only with `illustrative:true` placeholders
   (`builtin.json`) and NOT wired to `/v1/models`. So the gap is "populate + surface the
   record that exists," not "design a catalog." (`-latest`→dated alias resolution is also
   absent repo-wide — a separate, minor ergonomic gap.)
5. **Capability-aware routing (`require_parameters`).** fak carries per-provider quirk
   flags on `adapterRequest` (e.g. `OmitTemperature`, tool-as-text) but has no route-time
   filter that skips a target which cannot honor a requested param/tool.
6. **Request-wire model fallback array.** OpenRouter's `models:[...]`+`route:fallback`.
   fak's `ReduceFirst` is the decision-side analogue, but the live multi-backend DISPATCH is
   stub (topology #3 in `routers.md`).
7. **Slug ergonomics (`:floor`/`:nitro`).** fak uses routing *manifests/presets*
   (cost-saver / best-of-quality) — more principled; only a sugar gap.
8. **Zero-completion cost honesty.** `EstimateSavings` charges unpriced members at the
   frontier rate but does not zero-out a member that errored/returned empty.

---

## 4. Ranked opportunities (fit × cheap-to-ship)

**#1 — `max_price` price-ceiling constraint.** *Top pick.* Add an optional
`MaxCostPerMTok` to `routing.go`'s `RequestClass`: filter tier candidates to those at/under
the ceiling; if none qualify, return the EXISTING `ErrNoTier` (→ 503/replan). Mirror in
`modelroute` as a Subject/Plan budget checked against `PriceBook`. Pure, deterministic,
additive — matches `routing.go`'s stated ethos exactly ("data-in/decision-out, touches no
existing path"). Turns the passive cost lens into a load-bearing gate. Smallest, most
directly OpenRouter-inspired change. **Effort: low.**

**#2 — retention/privacy routing label bound to the residency floor.** *Highest-conviction
architectural borrow* — it plays to fak's unique strength (the floor), which OpenRouter
lacks. **Cheaper than first scored: the floor is already tag-driven, so this is an alias,
not new enforcement.** The rank-12 `residencyGate` (`internal/engine/engine.go:262`)
already DENYs a remote route when `sensitiveRoute` sees `Args.Scope==ScopeTenant` OR
`Meta["sensitivity"]`/`Meta["data_sensitivity"]` ∈ {sensitive,tenant,confidential,secret,
pii} (`engine.go:291`); `remoteRoute` is fail-closed and already treats `openrouter/`,
`litellm/`, `portkey`, `bedrock/`, `my-proxy` as remote (regression-tested in
`residency_test.go:76`). `modelroute/registry.go:46/187` mirrors it (`SensitiveLabel`,
`sensitive_remote`, fail-toward-sensitive). So the work is: accept a caller-facing
`retention: none` / `zdr: true` preference and translate it to the EXISTING sensitivity
tag (threaded Subject→`abi.ToolCall.Engine`→PDP via the ROUTE-BEFORE-ADJUDICATE contract),
then surface the resulting refuse-reason in the `Decision`. No new floor logic — a thin
adapter onto an enforced tag. Grep confirms `zdr`/`data_collection`/`retention` have zero
code symbols today. **Effort: low–medium.**

**#3 — surface the model catalog that already exists (`fak models` / richer `/v1/models`).**
The normalized record is already built (`modelscore.ModelEvidence`: context + cost{in,out} +
benchmarks + provenance) — it just carries placeholder data and isn't wired to the endpoint.
The win: populate it (via the `benchcatalog` ingest seam) and surface it as a list — `id`,
wire, `price{in,out}`, residency-class, context ceiling, source-of-truth pointer — replacing
the hardcoded-constant metadata stub in `http.go:1080`. `/api/v1/models` is the template;
this is the discovery surface fak lacks. Touches `cmd/fak` (build-fragile — gate via
git-archive+overlay per repo habit). **Effort: low–medium** (record exists; wiring + data).

**#4 — live health signal feeding `DemoteModel` (== #600).** Widen the demotion trigger
from entitlement-403 to a live 5xx / rate-limit / latency scoreboard with a ~30s outage
window (OpenRouter's exact mechanism), fed as a Subject SIGNAL so the pure decision stays
pure. Reinforces the existing learned-routing roadmap. **Effort: medium–high.**

**#5 — zero-completion cost honesty.** Exclude errored/empty members from the
`EstimateSavings` tally. A small honesty refinement consistent with the "not a bill" ethos.
**Effort: low, impact low.**

Lower: `:floor`/`:nitro` slug sugar (presets already cover the intent);
`require_parameters` capability filter (needs a capability table first).

---

## 5. Recommendation

Three clean, independently-shippable slices, each attaching to infra that ALREADY exists —
so all three are now low / low–medium effort:

- **#1 (price ceiling)** — smallest, purest; makes existing infra (`CostPerMTok` /
  `PriceBook`) load-bearing while staying inside `routing.go`'s pure/additive/`ErrNoTier`
  idiom. All fak pricing is static (consts/embedded JSON/flags), which is exactly what a
  ceiling check wants.
- **#2 (retention label → residency floor)** — the flagship *architectural* borrow: the one
  OpenRouter idea fak can express BETTER than OpenRouter, because fak owns the enforcement
  floor OpenRouter only advises on. And it is a thin adapter onto the already-enforced
  sensitivity tag, not new floor code.
- **#3 (catalog)** — best standalone discovery win; the normalized record
  (`modelscore.ModelEvidence`) already exists and just needs population + wiring to
  `/v1/models`. Natural home for the price data #1 introduces.

Start with **#1** (fastest to green) or **#3** (most visible), then **#2** as the flagship.
**#4** (live health→failover) is larger and belongs under open issue #600.

Cross-refs: `docs/adoption/compare/vs-routers.md`, `docs/integrations/routers.md`,
`docs/model-routing.md`, `docs/integrations/compatibility-matrix.md`; open issue #600
(learned routing), topology #3 stub (dispatch-through-router).
