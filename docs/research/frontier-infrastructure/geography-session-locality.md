# Geography and session-locality evidence ledger

**As of:** 2026-08-27. **Tracker:** #9323. This ledger separates billing geography,
request origin, serving region, local time, WAN routing, sovereignty, and KV/session
affinity. Those variables are related, but no source supports treating them as one
universal regional-demand distribution.

## Public-chat geography and session sources (#9370)

| Source | Geography/session fields | Safe use | Hard limit |
|---|---|---|---|
| LMSYS-Chat-1M | Timestamp, language, anonymized user ID, turns, model | Conversation/turn geometry and temporal sampling within the public platform | No verified country or provider-wide population. |
| WildChat | Timestamp, inferred country/language, anonymized IP-derived user, turns | Bounded country-local diurnal and multilingual analysis | Self-selected public traffic; IP identity/country errors and no billing denominator. |
| Chatbot Arena | Pairwise votes, users, prompts/conversations, model pairs | Preference and repeated-pair sampling | Votes are not request share or model demand. |
| OpenAssistant | Languages, branching trees, messages, ratings, volunteers | Multilingual branching-state and annotation workload | Crowdsourced collection, not organic sessions or geography. |

## Evidence matrix

| Source | Population and window | Geography / time variable | Measured parameters | Session / locality mechanism | Limits and missing fields |
|---|---|---|---|---|---|
| OpenRouter, *State of AI* | More than 100T production tokens across tasks, geographies, and time | Weekly share of global **spend** by continent | North America below 50% for most observed weeks; Europe roughly mid-teens to low twenties; Asia rises from about 13% to 31%; English exceeds 80% of tokens and Simplified Chinese is nearly 5% | None disclosed for geography | Spend is not request, token, user, tenant, or physical serving-region share; observation dates, comparable geographic denominators, timezones, CIs, and serving locations are not reported |
| SkyLB / SkyWalker | WildChat-derived requests mapped across six countries, then replayed in a cross-region evaluation | Country-local diurnal curves and complementary regional peaks | Evaluated system reports 1.12-2.06x throughput, 1.74-6.30x lower latency, and 25% lower serving cost; round-robin reaches 2.64x peak KV-memory imbalance | Consistent hashing and a dynamic prefix trie preserve user/prefix affinity across load balancing | WildChat is opt-in public chatbot traffic, not a provider billing/API census; system gains are not country-demand fit quality or CIs |
| SageServe | More than 8M production-trace requests, four open models, and three regions, with empirical and simulated evaluation components | Region, SLO class, capacity type, and multiscale placement | Up to 25% GPU-hour savings and up to 80% scaling-overhead reduction in the evaluated envelope | Model placement and traffic management, not a disclosed session distribution | No universal geography, timezone, tenant, or session parameters; results depend on forecast, spot supply, placement time, and SLO mix |
| Chutes, *A Year in LLM Serving* | 6.122B requests, 314,970 users, 9,174 models over 365 days | User cadence and daily periodicity; geography omitted | About 22% of users show daily periodicity; same `(user, model)` requests have strong sub-second-to-minute locality; prefix reuse persists from seconds to hours | User-model affinity and token-prefix reuse distance | No timezone, country, region, residency, or WAN route; daily periodicity cannot be assigned to geography |
| ServeGen | One-day production text, multimodal, and reasoning traces | One-day temporal curves and client-composed rates; geography omitted | Top 29 of 2,412 text clients account for 90% of requests; clients differ materially in rate and burstiness | Multi-turn reasoning requests are nearly 10%; reconstruction is client-conditioned | One-day window; no tenant organization, timezone, origin region, serving region, WAN, retry, or CIs |

## Required benchmark dimensions

Record these independently instead of deriving one from another:

1. request-origin region and local timestamp;
2. billing/account geography and tenant or organization identifier;
3. serving region, model, hardware, capacity class, and SLO;
4. request count, input tokens, output tokens, spend, and accepted useful work;
5. session/user/model identity, turn index, prefix identity, and reuse distance;
6. WAN latency, residency/sovereignty eligibility, failover, and retry cause;
7. observation window, estimator, fit family, parameters, fit quality, CI, and drift.

At minimum, test three separate scenarios: independent regional peaks, complementary
regional peaks with WAN routing, and session-affine demand where moving traffic destroys
KV locality. A fourth sovereignty-constrained scenario should forbid otherwise cheap
routes. Report local-only and cross-region results under the same model, hardware, SLO,
and quality envelope.

## What is still missing

- Direct provider request, token, user, and tenant shares by origin geography and timezone.
- A mapping from request origin to physical serving region, WAN path, residency decision,
  retry/failover, and useful completion.
- Production session-length and tool-call-chain distributions outside coding-agent traces.
- Confidence intervals, fit diagnostics, and drift for geographic demand distributions.
- Microsoft and other first-party provider traces with comparable denominators.

The bounded conclusion is therefore narrower than “global traffic follows a known law”:
regional demand changes over time, complementary peaks can create routing opportunity,
and session locality can make that opportunity expensive. The current evidence does not
justify one permanent continent mix, one timezone model, or converting spend share into
compute demand.
