# Portable fast mode: provider, model, and agent latency controls

Status: research packet and issue map under [#8959](https://github.com/anthony-chaudhary/fak/issues/8959)  
Observed at: `2026-08-25`  
Evidence boundary: official provider documentation, one research paper, FAK self-query, direct source inspection, and GitHub issue dedupe. No live paid-provider latency run was made.

## Result

`fast` is not one switch. It can mean faster token emission, lower time to first token, a smaller or lower-effort model, fewer output tokens, a premium capacity class, preserved prompt-cache reuse, fewer serial calls, or a shorter agent critical path. These mechanisms have different quality, cost, reliability, and cache effects.

The portable contract should therefore be a typed latency intent with explicit quality, cost, residency, cache, deadline, and fallback fences. FAK resolves that intent into supported provider, model, and orchestration mechanisms, then records requested, resolved, launched, and realized behavior separately. A requested fast tier is not evidence that the provider served it.

The issue portfolio is:

1. [#8960](https://github.com/anthony-chaudhary/fak/issues/8960) — minimal portable fast-profile resolver and receipt spine;
2. [#8962](https://github.com/anthony-chaudhary/fak/issues/8962) — provider service/speed-tier capability facts and requested-versus-realized readback;
3. [#8961](https://github.com/anthony-chaudhary/fak/issues/8961) — worker width selected from witnessed coordination break-even rather than maximum fan-out;
4. [#8963](https://github.com/anthony-chaudhary/fak/issues/8963) — paired time-to-accepted-outcome benchmark across tier, model, cache, and worker controls;
5. [#8975](https://github.com/anthony-chaudhary/fak/issues/8975) — selective delayed hedging for eligible buffered model calls, with cancellation, drain, and unknown provider-work accounting.

Ticket-scope readback is green for all five leaves: strict project-work, witness, scale, and born-routed review reports `dispatchable=5`, and the smallness lint reports one deliverable and one witness for each. Cohort planning produces two collision-safe path waves because #8960 and #8961 both own the orchestration planner seam. The epic itself remains a non-dispatch parent by design.

## FAK witness

The following dogfood queries were run from the repository:

```text
fak capabilities "fast mode" --json --limit 6
fak capabilities "reasoning effort speed quality" --json --limit 6
fak capabilities "microagents concurrency orchestration" --json --limit 6
fak capabilities "parallel subagents coordination overhead" --json --limit 6
fak capabilities "time to first token streaming" --json --limit 6
fak capabilities "speculative execution cancellation stragglers" --json --limit 6
```

The first and fifth queries surfaced adjacent turn-avoidance, routing, context, and streaming capabilities. The reasoning-effort and agent-coordination phrasings returned no matching outcome cards. Direct source inspection and `go run ./cmd/fak-dev index docs|leaves|verbs|claims` supplied the lexical and ownership cross-check.

| Capability | State | Direct FAK evidence | Disposition |
|---|---|---|---|
| Claude launch speed posture | **PRESENT**, harness-specific | `cmd/fak/dispatch_model_policy.go:134-165` resolves `auto|fast|standard`; `cmd/fak/dispatch_tick.go:568,695,888` carries the selection into launch and a sidecar. Closed #5017 owns the shipped binding. | Consume as the first binding; do not re-file generic Claude fast mode. |
| Portable conversation intent | **PRESENT** as a narrow spine | `pkg/conversationprofile/conversationprofile.go:14-59` carries portable intent, binding, and receipt shapes from closed #6878. | Extend through #6877; do not create a second generic intent plane. |
| Model latency routing | **PARTIAL** | `internal/modelroute/modelroute.go:123-179` has only `any|interactive|batch`; `:355-383` applies ordered static rules. #4588 owns live per-model latency and `Fastest`. | Feed constraints/evidence into the owner. |
| Provider service/speed tiers | **SHIPPED** | `internal/modelroute/providercontract.go` records provenance-bearing OpenAI and Anthropic tier contracts; `internal/agent/adapters.go` preserves standard-mode bytes, binds fast requests, and reads realized tiers; `/v1/models` publishes only contract-supported modes. Requested, wire, realized, downgrade, cache-rewarm, and unknown premium-price state remain distinct in `ServiceTierReceipt`. | #8962. |
| Portable orchestration controls | **PARTIAL** | `internal/orchestration/orchestration.go:14-50,314-411` owns `off|auto|ultracode`, caps, capabilities, roles, leases, and reconciliation, but no end-to-end latency objective or evidence-fed width. | #8960 and #8961 consume #5964. |
| Paired agent accounting | **PARTIAL** | `internal/ultracodebench/ultracodebench.go` already gates `GAIN|NO_GAIN|ABSTAIN` on accepted effects, wall time, authoritative cost, retries, activation, and independent witness. It does not join provider tier, TTFT/TPOT, fallback, and cache transition into one fast-profile run. | #8963 extends the existing evaluator. |
| Provider Predicted Outputs | **ABSENT** | No OpenAI `prediction` request carrier or accepted/rejected prediction-token receipt was found. Native speculative decoding is separately owned by #23. | Optional module only after the core receipt and benchmark exist. |

Overall verdict: **PARTIAL**. FAK ships most of the individual controls, but not the cross-layer contract or realized receipt.

## External source ledger

Provider behavior is **INSPIRE-ONLY**. No provider prose, example, or implementation is copied. These pages are mutable; refresh them when provider APIs, tier names, supported models, pricing, or fallback rules change.

| Source | State observed | Fact used | Constraint / refresh trigger |
|---|---|---|---|
| [OpenAI Fast mode](https://developers.openai.com/api/docs/guides/fast-mode) | Shipped; observed `2026-08-25` | `service_tier: "fast"` can provide up to 2.5x faster processing; responses report the realized service tier; rapid ramp can downgrade to `default`. | Premium pricing, model/endpoint limits, and ramp policy are mutable. The vendor multiplier is not FAK evidence. |
| [Anthropic Fast mode](https://platform.claude.com/docs/en/build-with-claude/fast-mode) | Research preview; observed `2026-08-25` | Same model weights, higher output tokens per second rather than lower TTFT; changing fast/standard posture invalidates prompt-cache reuse; explicit fallback can cause another cache miss. | Beta header, model/platform eligibility, pricing, and cache semantics can change. |
| [Anthropic service tiers](https://platform.claude.com/docs/en/api/service-tiers) | Shipped; observed `2026-08-25` | `auto` may use priority capacity and fall back to standard; response usage reports the realized tier. | Priority is a committed-capacity contract, not merely an inference-speed toggle. |
| [Google Gemini optimization](https://ai.google.dev/gemini-api/docs/optimization) | Shipped; observed `2026-08-25` | Standard, Flex, Priority, Batch, and caching are different latency/cost/reliability contracts; Priority can gracefully downgrade. | Availability, prices, and model support are mutable. Google prose is CC BY 4.0 and code samples Apache 2.0; this packet uses behavioral facts only. |
| [OpenAI Predicted Outputs](https://developers.openai.com/api/docs/guides/predicted-outputs) | Shipped on a narrow model/API envelope; observed `2026-08-25` | Responses report accepted and rejected prediction tokens; rejected prediction tokens are billed. | Model and feature exclusions make this a narrow patch/regeneration accelerator, not a portable default. |
| [OpenAI latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization) | Current guidance; observed `2026-08-25` | Faster processing, fewer output tokens, fewer calls, parallelism/speculation, streaming, and avoiding model calls are distinct levers. | Guidance is not a performance claim for FAK. |
| [Towards a Science of Scaling Agent Systems, arXiv:2512.08296v3](https://arxiv.org/html/2512.08296v3) | Research paper v3, `2026-04-08` | Across the paper's 260 benchmark configurations, coordination topology and task structure materially changed scaling; tool-heavy work and already-strong single-agent baselines often showed negative returns, and independent agents amplified errors more than centralized coordination. | Benchmark-specific evidence, not a universal law. Re-run on FAK workloads before choosing thresholds or defaults. |

Implementation repositories and framework docs were also surveyed for worker limits, cancellation, and drain behavior. Exact commits from OpenAI Codex, Google ADK, AutoGen, OpenAI Agents Python, Pydantic AI, and LangGraph remain useful future study targets, but their source anchors were not independently re-opened in this pass; no code-level borrow or issue depends on them.

## Facts and inferences

Facts established by current sources and FAK inspection:

- provider speed and capacity controls have different request, fallback, cache, and receipt semantics;
- a provider may realize a different service tier from the one requested;
- higher output-token rate does not imply lower TTFT or faster accepted completion;
- streaming changes perceived latency without proving lower completion wall time;
- increasing worker width can add model turns, context, cancellation, lease, and reconciliation work;
- FAK currently exposes fixed orchestration widths and separate performance controls, not one fast intent.

Inferences that require paired FAK evidence:

- a portable intent can preserve provider-specific semantics without flattening them into a boolean;
- cache-coherent session posture may beat per-call fastest selection on repeated-prefix agent traces;
- task-aware model, effort, output, and worker-width selection may beat a provider fast tier alone;
- adaptive worker width can improve the critical path when it consumes a comparable frontier and falls back to one worker when evidence is absent;
- delayed model-call hedging can help a narrow, idempotent read-only tail, but only after loser generation, cancellation, billing, and drain work are counted;
- Predicted Outputs can help high-similarity whole-file edits when accepted-token share and net wall time cross a frozen threshold.

## Candidate disposition

| Mechanism | Disposition | Why / falsifier |
|---|---|---|
| Requested/resolved/realized fast receipt | **DEFAULT** contract | Necessary for every claim. Falsified as net-new only if all supported paths already preserve distinct states and downgrade reasons. |
| Quality-constrained model, effort, output, cache, and fallback fences | **DEFAULT** contract | Prevents “fast” from silently buying lower quality, wider authority, higher cost, or a cold cache. |
| Task-shape admission before multiagent fan-out | **DEFAULT** selector | Choose the smallest qualifying width; hold at one for missing, stale, incomparable, or non-gain evidence. Falsified if the FAK frontier shows unconditional positive returns across the target envelope. |
| Provider inference-speed tier | **OPTIONAL MODULE** | Opt in until paired live evidence shows deadline gain net of premium, cache invalidation, fallback, and throttling. |
| Predicted Outputs for patch-shaped work | **OPTIONAL MODULE** | Admit only supported, tool-free, high-similarity regeneration. Retire if accepted-token share and wall time do not beat ordinary generation. |
| Delayed read-only model hedge with cancellation/drain | **OPTIONAL MODULE / WATCH**, tracked by #8975 | Reopen after the core receipt can charge loser work; never apply to effectful calls. |
| Flex/Batch placement for noncritical leaves | **RECIPE** | It can shorten the interactive critical path while increasing leaf latency; promote only after repeated workload evidence. |
| Cache warm-leader barrier before a shared-prefix burst | **RECIPE / EXPERIMENT** | Test whether avoided duplicate cache writes beat the serialized warmup cost. |
| Blanket microagents or maximum concurrency | **EXCLUDE** | Coordination and error amplification can dominate. A typed profile selects by evidence; it does not promise more workers are faster. |
| Silent fallback to another provider, model, tier, or engine | **EXCLUDE** | It destroys quality, residency, and performance attribution. Explicit operator-selected migration/interoperability remains allowed with a truthful receipt. |

## Dedupe and ownership map

| Owner | Relationship |
|---|---|
| #6877 / closed #6878 | Portable intent envelope and first conversation binding. #8959 is the latency family below it. |
| #5964 / #5970 / #5971 / #5973 | Harness-neutral orchestration plan, capability degradation, execution, and receipts. #8960 and #8961 consume these seams. |
| #4588 / #600 / #595 | Live latency/model routing and the broader model-route program. #8962 owns only the selected provider request's service tier and readback. |
| #7309 | Provider cache-hint and retention negotiation. #8962 records tier/cache interaction but does not reopen generic cache negotiation. |
| #8337 / #8679 / #6058 | Scout-versus-worker and coordination-overhead evidence. #8961 consumes their comparable frontier instead of duplicating a microagent benchmark. |
| #5796 / #5798 | Governed tool-call execution width. Provider emission width and local tool execution width remain separate. |
| #4053 | Error-class retry/fallthrough. Provider-tier fallback must compose with it. |
| #4102 / #4104 / #4106 | Tool speculation family. Fast intent does not reopen speculative tool execution. |
| closed #6160 / #6167 / #6185 | Straggler cancellation, selective scheduling, and negative live-matrix evidence. #8975 consumes them as predecessors rather than duplicating an experiment. |
| #23 / #8759 / #8403 | Native inference, regression gates, and engine attribution. Provider fast tiers never substitute another engine silently. |
| #2022 / #8170 / #2028 | Small-model escalation, witnessed specialist-effect reuse, and microagent quality ablation. No generic microagent issue was filed. |

## Coverage frontier

The first supported envelope is two heterogeneous provider/harness bindings, interactive and autonomous tasks, explicit standard/fast/degraded/unsupported states, and worker widths `1,2,4,8` within existing hard caps. The benchmark must join TTFT, TPOT/OTPS, critical-path and end-to-end wall time, prompt/cache tokens, provider-authoritative cost, retries, fallback, discarded calls, worker time, lease wait, invalidation, reconciliation, and independent outcome acceptance.

Outside the first envelope:

- provider-specific Flex/Batch/committed-capacity recipes;
- residency-bounded regional capacity widening;
- provider Predicted Outputs;
- live learned width selection or online bandits;
- effectful speculative workers;
- native kernel optimization and generic model leaderboards.

Stop expanding when a cohort lacks a repeatable accepted-outcome witness, when source churn exceeds observed value, or when the mechanism belongs to an existing owner above.

## Refresh and unavailable surfaces

Refresh this packet when a provider changes tier names, model eligibility, price, cache semantics, fallback, or response fields; when FAK closes #8959, #8960, #8961, #8962, #8963, or #8975 or one of the consuming owners; or before promoting any optional module to a default.

Provider scheduler implementations, production queue/load distributions, contract-specific service credits, account eligibility, and the methodology behind vendor “up to” multipliers are unavailable. No inference about them is made. Live provider availability, realized billing class, cache behavior, TTFT, token rate, and downgrade behavior remain unwitnessed until #8963 captures a scrubbed paid-endpoint bundle.
