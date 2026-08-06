# Knowing the denominator for an agentic trajectory

**Research note — 2026-08-06**

## Decision

A useful trajectory denominator is **not repo size, token count, or the shortest historical run**.
It is the estimated cost of the cheapest *currently feasible* trajectory that reaches an explicit
quality bar, conditioned on the capability surface visible at dispatch time:

\[
D(x, e, q, t) = \inf_{\pi \in \Pi(e,t)} E[C(\pi, x) \mid Q(\pi,x) \ge q]
\]

where `x` is the request, `e` is a versioned environment/capability snapshot, `q` is the
required answer/effect quality, `t` is the information available at prediction time, and `C`
is a cost vector collapsed by an operator-declared utility function. `D` is latent: the system
must publish a calibrated interval or quantiles, not pretend to know an oracle scalar.

The operational contract should therefore be:

> Before starting, quote a quality-conditioned cost range and the capability/coverage assumptions
> behind it. During execution, update the quote from observed progress and newly discovered missing
> capabilities. At completion, report realized cost, the original quote, and regret against the best
> *replayable feasible* route known after the fact.

This gives a user the confidence they actually want: not “the agent searched 30% of the repo,” but
“under this tool/index state, 80% of comparable requests that met this evidence bar completed within
this envelope; this request is still inside/outside that envelope.”

## Why the denominator is hard

There are three different unknowns that are easy to collapse into one misleading number.

1. **Task uncertainty** — what work is required is not fully known from the prompt.
2. **Environment uncertainty** — an MCP server may be configured but unreachable, a tool may be
   discoverable but unauthorized, or an index may be stale or omit a relevant corpus.
3. **Policy/quality uncertainty** — “answered” ranges from plausible prose to a cited answer, a green
   patch, or an externally witnessed effect. A cheap trajectory that misses the required evidence is
   not in the denominator's feasible set.

The hindsight optimum is also unobservable. We see routes that were attempted, not all routes that
could have been attempted. Historical minimum cost is optimistically biased by easy instances;
historical mean is pessimistically polluted by bad routing and missing capabilities. An unqualified
“optimal trajectory” denominator therefore rewards shortcutting quality or compares the run with an
impossible oracle.

## Four denominators, only one suitable for the user promise

Keep these names separate in data and UI:

| Name | Definition | Use |
|---|---|---|
| `declared_budget` | User/operator limit | Hard stop or approval boundary; not an efficiency baseline. |
| `predicted_feasible` | Pre-dispatch distribution of minimum cost among known feasible routes at quality `q` | **Primary user quote.** |
| `best_replayable` | Cheapest post-run route that can be replayed against the recorded environment and pass the same witness | Retrospective regret and training label. |
| `oracle_lower_bound` | Optimistic bound with perfect routing/information | Research only; never a user-facing efficiency claim. |

A run's normalized efficiency can be reported after completion as

\[
R = C_{actual} / D_{best\_replayable}
\]

with `R >= 1` only when the replay set is credible. Before execution, report predicted cost quantiles
(`p50`, `p80`, `p95`) rather than a ratio whose denominator has not yet been observed. If no comparable
support exists, say `cold-start`, emit structural bounds, and abstain from a calibrated probability.

## Cost and quality are vectors

### Cost

Record at least:

- wall-clock elapsed time and critical-path time;
- model input/output/cached tokens and provider cost;
- tool calls, retries, failed calls, and bytes read;
- human interventions/approvals;
- external compute time;
- irreversible or security-sensitive operations.

Do not silently turn this into one universal score. A quote selects a declared utility profile, for
example `interactive_latency`, `dollar_min`, or `high_assurance`, with explicit weights and hard
constraints. Preserve the raw vector so a later profile can recompute it.

### Quality / done condition

The quality threshold must be fixed before pricing and matched to the request class:

- answer: grounded citations and an answer rubric;
- repository question: provenance plus relevant-region coverage;
- code change: failing-before/passing-after witness and repository gates;
- external effect: independent read-back of the effect;
- exploration: stated recall target or saturation rule.

This composes with `trajctl`: its objective, plan, budgets, rubric score, progress curve, and scorer
calibration provide the trajectory-side observations. The missing layer is the pre-dispatch capability
snapshot and the counterfactual route/cost model.

## Capability-conditioned prediction

A server being “enabled” is not a binary capability. Build a signed, versioned **capability snapshot**
for every quote:

```json
{
  "snapshot_id": "sha256:...",
  "captured_at": "...",
  "request_class": "repo_question",
  "quality_contract": "cited-answer/v1",
  "capabilities": [
    {
      "id": "mcp:github/search_code",
      "discovered": true,
      "reachable": true,
      "authorized_scope": "repo:R",
      "freshness": "2026-08-06T...Z",
      "corpus_revision": "...",
      "latency_p50_ms": 180,
      "failure_rate_7d": 0.01,
      "price_model": "..."
    }
  ],
  "indexes": [
    {
      "id": "repo-symbol-index",
      "scope_manifest": "sha256:...",
      "source_revision": "...",
      "completeness_lower_bound": 0.94,
      "freshness_lag_s": 14
    }
  ],
  "missing_or_unknown": ["private issue tracker authorization"]
}
```

MCP discovery (`tools/list` plus list-change notifications) supplies an inventory, not readiness or
coverage. Probe only cheap, safe health/read operations; policy must gate probes as it gates normal
calls. Cache snapshots by digest, but invalidate on tool-list changes, auth/scope changes, index
revision changes, policy changes, or material health drift.

Represent a route as required capability classes and alternatives, not hard-coded tool names. Example:
`repo_search AND (symbol_index OR bounded_text_scan) AND citation_readback`. Route feasibility is then
computed against the snapshot. Missing optional capabilities increase predicted cost; missing required
capabilities causes a typed abstention or a request to enable/authorize them.

This should compose with existing fak surfaces:

- `capindex` supplies declared capability identity and overlap/conflict information;
- `toolcoverage` distinguishes covered, uncovered, and stale tool-policy rows;
- `sotacoverage` demonstrates a useful denominator discipline: enumerate an eligible universe, classify
  each member, and refuse silent omission;
- `trajctl` supplies objective progress, cost observations, and scorer qualification.

None alone answers the question. The quote joins their snapshots under one request/quality contract.

## Predicting the initial denominator

Use a staged model so cold starts remain honest.

### 1. Classify and decompose

Map the request to a multi-label task class and a small requirement graph: likely corpora, evidence
class, effect class, and uncertainty. Keep ambiguity as competing hypotheses instead of prematurely
choosing one. Features should include repository topology/revision, request semantics, capability
snapshot, policy, index freshness, and chosen quality contract.

### 2. Enumerate route templates

Generate feasible route templates from capability classes. Include direct lookup, indexed retrieval,
bounded scan, inspect-then-act, and external read-back variants. Reject routes that cannot satisfy the
quality contract. The estimated denominator is the lower envelope of the route distributions, after a
multiple-comparisons optimism correction or held-out calibration; simply taking the minimum of noisy
predictions is biased low.

### 3. Predict a distribution

For each route, predict joint success and cost:

\[
P(Q \ge q \mid x,e,\pi), \quad P(C \le c \mid Q \ge q,x,e,\pi)
\]

Select the cheapest route whose lower confidence bound on success clears the risk target. Quantile
regression or survival models naturally handle timeouts and censored runs. Use hierarchical backoff:
exact request class + environment, class + capability signature, broad class, then structural cold-start
bounds. A point estimate without support count, interval, and calibration cohort is invalid.

### 4. Quote the envelope

A useful quote is compact:

```text
quality: cited-answer/v1
route: indexed repo search -> targeted reads -> citation check
cost: p50 3.2k tokens / 38s; p80 5.1k / 71s; p95 9.4k / 160s
confidence: calibrated on 184 comparable runs; 30-day p80 hit rate 0.78
assumptions: symbol index at HEAD; GitHub MCP read scope healthy
coverage floor: 0.91 relevant-region probability
fallback: bounded scan (+p80 84s) if index probe fails
```

This is a service-level estimate, not a guarantee. The user can choose “fast,” “balanced,” or
“high-assurance” by changing quality/risk/utility, while the underlying raw prediction remains auditable.

## Correcting the denominator online

The initial quote will be wrong. Correction is a feature, not an embarrassment.

At each meaningful observation (tool result, failed probe, phase completion, evidence gain), update a
posterior over remaining work:

\[
\hat C_{total,t} = C_{spent,t} + \hat C_{remaining}(state_t, e_t, q)
\]

Use `trajctl`'s scored progress observations as one state signal, but do not extrapolate from progress
alone. Add discovered subproblems, capability failures, retrieval yield, contradiction count, and
witness status. Publish changes only when they cross a material threshold to avoid noisy UI churn.

A rational continuation rule compares expected value of the next action with its cost:

\[
E[\Delta Q \mid a,state] / E[\Delta C \mid a,state]
\]

Continue while the best feasible action can plausibly close the quality gap within the revised risk
budget. Otherwise switch routes, ask for a capability/clarification, lower the quality target only with
user consent, or abstain. The first quote stays immutable in the ledger; revisions are appended, so
bad initial calibration cannot be hidden by moving the denominator after seeing the run.

### Counterfactual correction after the run

Create training labels from independently replayable evidence, not the agent's narrative:

1. Record the actual event trace and capability snapshot.
2. Identify plausible alternative route templates available in that snapshot.
3. Replay or shadow-evaluate alternatives on sampled tasks under the same quality witness.
4. Set `best_replayable` to the cheapest passing route, retaining uncertainty where replay is incomplete.
5. Attribute excess cost to typed causes: routing error, capability miss, stale index, tool failure,
   quality rework, user scope change, or irreducible discovery.

Randomized exploration is needed on a small safe cohort; otherwise the system only learns about routes
its current policy already chooses. Use inverse-propensity or doubly robust evaluation when comparing
logged policies, and never infer that an untried route would have passed merely because it looked short.

## Coverage: use a query-relative denominator

“Files read / files in repo” is usually the wrong denominator. A one-line ownership question may need
one manifest in a million-line monorepo; a license audit may require nearly every file. Repository size
is a predictor of cost and uncertainty, not coverage itself.

Define coverage against a **query-specific eligible evidence universe** `U(x,q,e)` and report multiple
layers:

1. **Source availability coverage** — relevant corpora reachable and authorized / corpora believed
   necessary. This can be bounded from manifests and capability snapshots.
2. **Index coverage** — indexed eligible artifacts / eligible artifacts, revision- and freshness-aware.
3. **Retrieval coverage** — probability mass of relevant evidence represented in retrieved candidates.
4. **Inspection coverage** — weighted relevant regions actually examined / estimated relevant regions.
5. **Claim coverage** — answer claims with supporting evidence / evidence-requiring claims.
6. **Witness coverage** — required done-condition checks actually executed / checks declared up front.

Only (1), (2), (5), and (6) often have enumerable denominators. Relevant-region recall (3/4) generally
does not. Estimate it with bounds rather than fake precision:

- independent retrieval channels and overlap (capture-recapture) to estimate unseen relevant regions;
- Good–Turing-style singleton/doubleton mass or diminishing novel-evidence yield;
- mutation/seeded-evidence tests on benchmark queries;
- stratified sampling over repository modules, languages, generated/vendored status, and ownership;
- disagreement between lexical, symbol, semantic, history, and dependency retrieval;
- explicit exclusions with their estimated mass.

Capture-recapture assumptions are fragile because retrieval channels are correlated. Report the
estimator, channel dependence warning, and interval. Saturation (“three searches found nothing new”)
is evidence only when the searches are meaningfully independent.

A proposed coverage object:

```json
{
  "universe_manifest": "sha256:...",
  "query_scope": "behavior of retry routing at HEAD",
  "source_availability": {"lower": 1.0, "method": "enumerated"},
  "index_coverage": {"lower": 0.97, "revision": "..."},
  "relevant_region_recall": {"lower": 0.82, "upper": 0.96, "method": "overlap+sampling"},
  "claim_coverage": {"value": 1.0, "supported": 7, "eligible": 7},
  "exclusions": [{"class": "vendored", "reason": "outside query contract"}]
}
```

Coverage should influence the quality-success probability and widen the cost interval. It should not be
multiplied into a single “efficiency score”; that hides whether a run was cheap because it was focused
or because it skipped evidence.

## Calibration and governance

Evaluate the quote as a prediction system, not by showcasing successful runs.

- **Interval/quantile calibration:** p80 cost envelopes should contain approximately 80% of passing
  comparable runs, including censored runs.
- **Selective risk/coverage:** plot error rate against the fraction of requests the system is willing
  to quote rather than mark cold-start/abstain.
- **Quality calibration:** predicted probability of satisfying `q` versus witnessed outcomes.
- **Route regret:** actual cost versus `best_replayable`, stratified by request and capability signature.
- **Drift:** capability, model, policy, repository, and index revisions each split calibration cohorts.
- **Fairness of comparison:** compare routes under identical quality contracts and utility profiles.
- **Observer cost:** include probing, indexing, tracing, and replays in net cost.

Use conformalized residual intervals or another finite-sample calibration wrapper where exchangeability
is plausible; detect and disclose when drift breaks that premise. Reliability diagrams, Brier/log loss,
coverage error, interval width, and abstention rate belong beside median error. `trajctl` already makes
the important architectural choice that scorer calibration is separate from objective scoring; retain
that separation for denominator predictors.

## Minimal end-to-end spine

Do not start with a universal learned oracle. The smallest useful implementation is a replayable quote
ledger for one request class, likely `repo_question`:

1. Snapshot `capindex`, tool-policy coverage, index manifests/revisions, and cheap health probes.
2. Declare `quality_contract`, utility profile, and a hand-authored route-template set.
3. Produce structural `p50/p80/p95` bounds from historical matched runs, with hierarchical fallback and
   `cold-start` when sample support is insufficient.
4. Append immutable quote revisions at trajectory phase boundaries.
5. On completion, bind witnessed quality and raw cost to the original quote.
6. Backtest quantile coverage and route regret on held-out chronological cohorts.

The spine's self-check should contain synthetic tasks where: an index is present and fresh; present but
stale; an MCP tool disappears after discovery; the cheapest route fails quality; a run is censored; and
a cold-start request correctly abstains. The proof is not “the command emits JSON”; it is held-out
coverage near the advertised quantiles and a captured example showing a denominator revision after a
capability failure.

## Failure modes to guard explicitly

- **Goodharting quality:** denominator falls because the witness weakened.
- **Capability theater:** configured/discovered is treated as reachable, authorized, complete, and fresh.
- **Hindsight leakage:** post-run discoveries enter the pre-run feature vector.
- **Minimum-of-noise bias:** selecting the smallest route estimate creates systematic underquotes.
- **Moving baseline:** revised denominator overwrites the original quote.
- **Coverage theater:** bytes/files read masquerade as relevant-evidence recall.
- **Survivorship bias:** only completed trajectories train the model; timeouts disappear.
- **Policy confounding:** a route looks slow because policy required approval, but the snapshot omitted it.
- **Exploration starvation:** current routing never gathers outcomes for plausible alternatives.
- **Unpriced measurement:** index builds, probes, shadow runs, and human labels are excluded from cost.
- **False universality:** one model pools unrelated request classes and repository regimes.

## Relationship to adjacent research

This design combines established ideas rather than claiming an exact denominator is directly known:

- Rice's **algorithm selection problem** frames choosing a route from instance features.
- Query performance prediction in information retrieval motivates estimating difficulty before retrieval,
  while retrieval-channel disagreement and judged pools expose the unknown-recall problem.
- Selective classification and **risk–coverage** curves justify abstaining on unsupported quotes.
- Conformal risk control motivates calibrated bounds rather than raw model confidence.
- Survival analysis handles timeout/censoring; contextual bandits and off-policy evaluation handle route
  learning from logged choices.
- Rational metareasoning/value of computation supplies the online continue/switch/stop criterion.
- Good–Turing/unseen-species and capture-recapture estimators offer imperfect but explicit ways to bound
  unseen relevant evidence.
- FrugalGPT and model-routing work show cost/quality route selection, but this proposal extends the
  environment to tools, indexes, policy, evidence, and external effects.

## Sources

External sources were checked on 2026-08-06.

- J. R. Rice, “The Algorithm Selection Problem” (1976), DOI
  <https://doi.org/10.1016/S0065-2458(08)60520-3>.
- Y. Geifman and R. El-Yaniv, “Selective Classification for Deep Neural Networks” (2017),
  <https://arxiv.org/abs/1705.08500>.
- A. Angelopoulos et al., “Conformal Risk Control” (2022),
  <https://arxiv.org/abs/2208.02814>.
- D. Golovin and A. Krause, “Adaptive Submodularity” (2011), JAIR
  <https://www.jair.org/index.php/jair/article/view/10640>.
- C. Hauff, L. Azzopardi, D. Hiemstra, and F. de Jong, “Query Performance Prediction: Evaluation Contrasted with Effectiveness”
  (2010), DOI <https://doi.org/10.1007/978-3-642-12275-0_20>.
- O. Vinyals et al., “Estimating the Unseen” (2011), DOI
  <https://doi.org/10.1145/1993636.1993727>.
- L. Chen, M. Zaharia, and J. Zou, “FrugalGPT” (2023),
  <https://arxiv.org/abs/2305.05176>.
- Model Context Protocol, “Tools” specification (2025-06-18),
  <https://modelcontextprotocol.io/specification/2025-06-18/server/tools>.
- fak trajectory-control foundation:
  [`TRAJECTORY-CONTROL-SCORE-ANYTHING-2026-07-03.md`](TRAJECTORY-CONTROL-SCORE-ANYTHING-2026-07-03.md).
- fak prediction calibration standard:
  [`../standards/prediction-calibration.md`](../standards/prediction-calibration.md).

## Bottom line

The denominator is a **calibrated, quality-constrained, capability-conditioned counterfactual cost
distribution**. It is quoted before the run, revised without erasing history, and grounded after the run
against replayable alternatives. Coverage is a separate query-relative evidence claim that informs the
quality probability and interval width. Repo size, enabled MCP count, and files read are useful features;
none is the denominator.



