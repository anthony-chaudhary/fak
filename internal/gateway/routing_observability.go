package gateway

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// Routing observability — the gateway-side LIVE emit of modelroute's per-aspect
// decision surface (#603, epic #595). modelroute/observe.go ships the PURE half: a
// DecisionRecord, an append-only DecisionJournal, and a Counts() rollup over the
// per-rule / per-strategy / per-aspect distribution. That file's lane note DEFERS
// the live /metrics emit (extending the gateway + the fak_gateway_* family) to this
// issue, because the gateway is the in-lane owner of the metric NAMES. This file is
// that deferred half: it folds every routing Decision the gateway takes on the served
// hot path (routeDecision, the single classification seam buildCall routes through)
// into a modelroute.DecisionJournal AND renders the journal's Counts() into the
// fak_gateway_routing_* family. The journal IS the per-route audit trail #603 names
// (Records() replays every route); the metrics are a pure projection of its Counts(),
// so the scrape and the audit can never disagree — the same accumulator-on-
// gatewayMetrics discipline harness_coherence.go keeps for its family.
//
// "Once routing is live" (the acceptance wording): routing IS live in the gateway —
// buildCall calls routeEngine -> routeDecision -> route.Route on every served tool
// call when a RouteManifest is configured. With no manifest the seam is never
// reached, so the family renders at 0 (the panels exist; nothing has routed yet).
// The fold here makes the live decision OBSERVABLE without changing the route itself.

// routingMetrics accumulates the per-decision journal and renders the
// fak_gateway_routing_* family. It is the SINGLE source of truth: the /metrics
// scrape (writeRoutingMetrics) and the operator-line roll-up (summary) both fold its
// Counts(), so the two surfaces can never disagree. Its own lock guards the journal —
// kept off the other gatewayMetrics locks, a distinct (and rare, one-fold-per-routed-
// call) hot path.
type routingMetrics struct {
	mu      sync.Mutex
	journal modelroute.DecisionJournal
}

func newRoutingMetrics() *routingMetrics { return &routingMetrics{} }

// observe folds one routing Decision into the journal, under the manifest version
// that produced it (used only to content-address the decision digest) and the
// routing overhead the caller measured. It returns the appended DecisionRecord so a
// caller (a debug line / a test) can read the per-aspect decision directly. nil-safe.
func (r *routingMetrics) observe(version string, d modelroute.Decision, overhead time.Duration) modelroute.DecisionRecord {
	if r == nil {
		return modelroute.DecisionRecord{}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.journal.Record(version, d, overhead)
}

// counts is a locked copy of the journal's per-dimension rollup for rendering / the
// operator line. Both surfaces fold THIS, so a scrape and the exit line agree.
func (r *routingMetrics) counts() modelroute.Counts {
	if r == nil {
		return modelroute.Counts{
			ByRule:     map[string]int{},
			ByStrategy: map[string]int{},
			ByAspect:   map[string]int{},
		}
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.journal.Counts()
}

// records returns a copy of the journal's decision records — the after-the-fact audit
// read #603 names ("the journal records the decision"). nil-safe.
func (r *routingMetrics) records() []modelroute.DecisionRecord {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.journal.Records()
}

// allStrategies is the closed Strategy set, emitted in a stable order at 0 so every
// strategy panel exists before the first routed call (the same emit-at-0 discipline
// the harness-coherence and compaction families keep). Per-rule / per-aspect series
// are inherently open (a manifest names its own rules, a deployment its own aspects),
// so those are emitted only for keys the journal actually saw.
var allStrategies = []modelroute.Strategy{
	modelroute.StrategyDefault,
	modelroute.StrategySingle,
	modelroute.StrategyEnsemble,
}

// observeRouteDecision folds one routing Decision into the routing journal, nil-safe
// at the gatewayMetrics layer so the call site (routeDecision) need not guard a Server
// built without metrics or without a routing manifest. It is the single entry point
// the served route path uses; the accumulator it updates is the shared source both
// /metrics and the operator roll-up read. Returns the appended record.
func (m *gatewayMetrics) observeRouteDecision(version string, d modelroute.Decision, overhead time.Duration) modelroute.DecisionRecord {
	if m == nil || m.routing == nil {
		return modelroute.DecisionRecord{}
	}
	return m.routing.observe(version, d, overhead)
}

// routingSummary is the gatewayMetrics-level accessor for the operator-line roll-up,
// nil-safe so a bare Server still renders a zero summary.
func (m *gatewayMetrics) routingSummary() RoutingSummary {
	if m == nil || m.routing == nil {
		return RoutingSummary{}
	}
	return m.routing.summary()
}

// writeRoutingMetrics renders the fak_gateway_routing_* family from the journal's
// Counts(). It is the gateway-visible form of modelroute's decision surface: the
// per-strategy / per-rule / per-aspect decision distribution, the ensemble and
// scout-call totals, and the summed routing overhead — every series a pure projection
// of the SAME Counts() the operator roll-up reads, so the two views agree by
// construction. The family renders at 0 when nothing has routed (no manifest, or no
// served tool call yet), so the panels exist before the first route.
func (m *gatewayMetrics) writeRoutingMetrics(b *strings.Builder) {
	if m == nil || m.routing == nil {
		return
	}
	c := m.routing.counts()

	writeCounter(b, "fak_gateway_routing_decisions_total",
		"WITNESSED (fak authored): per-aspect model-routing decisions the gateway took on the served path. The denominator for the by-strategy/rule/aspect families below; 0 until a RouteManifest is configured AND a tool call has routed.", int64(c.Total))

	// By strategy (closed set): which routing strategy served each decision. Emit the
	// whole closed set at 0 so every panel exists pre-first-route.
	writeHelpType(b, "fak_gateway_routing_decisions_by_strategy_total",
		"WITNESSED (fak authored): routing decisions by selected strategy (default|single|ensemble). default = no rule matched (the fail-closed default plan); single = a one-model pick; ensemble = a multi-model plan folded by a reduction.", "counter")
	for _, st := range allStrategies {
		fmt.Fprintf(b, "fak_gateway_routing_decisions_by_strategy_total{strategy=%q} %d\n", string(st), c.ByStrategy[string(st)])
	}

	// By matched rule (open set): which manifest rule fired. "(default)" is the
	// fail-closed no-match bucket. Emitted in a deterministic order (descending count,
	// ties by name) so two scrapes over the same journal print the same table.
	writeHelpType(b, "fak_gateway_routing_decisions_by_rule_total",
		"WITNESSED (fak authored): routing decisions by the matched manifest rule name. \"(default)\" is the fail-closed bucket where no rule matched. One series per rule the gateway has actually routed through.", "counter")
	for _, rule := range c.SortedRules() {
		fmt.Fprintf(b, "fak_gateway_routing_decisions_by_rule_total{rule=%q} %d\n", promQuote(rule), c.ByRule[rule])
	}

	// By routed aspect (open set): the granularity of the routed subject (tool_call,
	// query, state, …). "(none)" is an unset aspect. Sorted for a stable scrape.
	writeHelpType(b, "fak_gateway_routing_decisions_by_aspect_total",
		"WITNESSED (fak authored): routing decisions by the routed subject's aspect (the granularity fak routes at: request|tool_call|query|state|step|…). \"(none)\" is an unset aspect.", "counter")
	for _, asp := range sortedKeys(c.ByAspect) {
		fmt.Fprintf(b, "fak_gateway_routing_decisions_by_aspect_total{aspect=%q} %d\n", promQuote(asp), c.ByAspect[asp])
	}

	writeCounter(b, "fak_gateway_routing_ensemble_decisions_total",
		"WITNESSED (fak authored): routing decisions whose selected plan is a multi-model ensemble (len(members) > 1, folded by a reduction). A subset of the decisions total.", int64(c.EnsembleHits))

	writeCounter(b, "fak_gateway_routing_scout_calls_total",
		"WITNESSED (fak authored): scout classify-first calls the routed plans named (1 per decision whose plan set a scout model, summed). The cheap classify-then-route probe count.", int64(c.ScoutCalls))

	writeHelpType(b, "fak_gateway_routing_overhead_seconds_total",
		"WITNESSED (fak authored): cumulative wall-clock the routing decision itself cost (summed Decision overhead), the verifiable routing-overhead axis that sets fak apart from a learned-predictor router. Pure-function routing, so this stays tiny.", "counter")
	fmt.Fprintf(b, "fak_gateway_routing_overhead_seconds_total %s\n", promFloat(c.TotalOverhead.Seconds()))
}

// sortedKeys returns the keys of a string->int map in ascending order, for a stable
// scrape over an open label set (rules use SortedRules' count order instead).
func sortedKeys(m map[string]int) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// RoutingSummary is the operator-line roll-up of the routing family — the SAME numbers
// the fak_gateway_routing_* scrape reports (both fold Counts()), so the exit line and
// /metrics agree by construction.
type RoutingSummary struct {
	Total         int            `json:"total"`
	ByStrategy    map[string]int `json:"by_strategy,omitempty"`
	ByRule        map[string]int `json:"by_rule,omitempty"`
	EnsembleHits  int            `json:"ensemble_hits"`
	ScoutCalls    int            `json:"scout_calls"`
	OverheadNanos int64          `json:"overhead_ns"`
}

// summary folds the live journal into the operator-line roll-up. Every count is the
// SAME number the fak_gateway_routing_* scrape reports (both fold Counts()), so the
// exit line can never disagree with the metrics.
func (r *routingMetrics) summary() RoutingSummary {
	c := r.counts()
	sum := RoutingSummary{
		Total:         c.Total,
		EnsembleHits:  c.EnsembleHits,
		ScoutCalls:    c.ScoutCalls,
		OverheadNanos: c.TotalOverhead.Nanoseconds(),
	}
	if c.Total > 0 {
		sum.ByStrategy = nonZero(c.ByStrategy)
		sum.ByRule = nonZero(c.ByRule)
	}
	return sum
}

// nonZero copies the entries of m with a positive count (a compact roll-up that omits
// the emit-at-0 placeholder strategies).
func nonZero(m map[string]int) map[string]int {
	out := make(map[string]int, len(m))
	for k, v := range m {
		if v > 0 {
			out[k] = v
		}
	}
	if len(out) == 0 {
		return nil
	}
	return out
}
