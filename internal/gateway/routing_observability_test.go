package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestRoutingMetricsEmitsPerAspectDecisionFamily is the #603 acceptance witness: a
// REAL routed request (driven through buildCall, the production served-call seam that
// routes every tool call when a RouteManifest is configured) emits the per-aspect
// modelroute.Decision to BOTH /metrics (the fak_gateway_routing_* family, with the
// rule/strategy/aspect labels present) AND the decision journal (the audit trail). It
// also proves the operator roll-up folds the SAME Counts(), so the two surfaces can
// never disagree.
//
// It exercises routing LIVE — not a hand-folded accumulator — so it witnesses that the
// route the gateway actually took is the one that reaches the metric.
func TestRoutingMetricsEmitsPerAspectDecisionFamily(t *testing.T) {
	// A manifest with a single-model PICK rule (fetch -> remote:openai), an ensemble rule
	// (risky_write -> two guard models, vote), and a fail-closed default. Routing every
	// kind of decision through one server proves the by-strategy family is populated by the
	// LIVE route, not a stub.
	m := &modelroute.Manifest{
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "test"}}},
		Rules: []modelroute.Rule{
			{
				Name:  "pick-fetch",
				Match: modelroute.Match{Tool: "fetch"},
				Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: "remote:openai"}}},
			},
			{
				Name:  "guard-write",
				Match: modelroute.Match{Tool: "risky_write"},
				Plan: modelroute.Plan{
					Members: []modelroute.Member{{Model: "guard-a"}, {Model: "guard-b"}},
					Reduce:  modelroute.ReduceVote,
				},
			},
		},
	}
	if err := m.Validate(); err != nil {
		t.Fatalf("manifest should validate: %v", err)
	}
	s := routeServer(t, m)
	ctx := context.Background()

	// Drive three LIVE routing decisions through the production seam:
	//  - "fetch"        -> the single-model PICK rule (strategy=single, rule=pick-fetch)
	//  - "risky_write"  -> the ensemble rule          (strategy=ensemble, rule=guard-write)
	//  - "unmatched"    -> no rule matches            (strategy=default,  rule=(default))
	for _, tc := range []struct {
		tool     string
		readOnly bool
	}{
		{"fetch", true},
		{"risky_write", false},
		{"unmatched_tool", true},
	} {
		if _, err := s.buildCall(ctx, tc.tool, `{}`, tc.readOnly, "", ""); err != nil {
			t.Fatalf("buildCall(%q): %v", tc.tool, err)
		}
	}

	// (1) /metrics carries the routing family with the per-aspect decision distribution —
	// the rule and strategy labels the acceptance criterion names are present and correct.
	text := s.renderMetrics()
	for _, want := range []string{
		"fak_gateway_routing_decisions_total 3",
		`fak_gateway_routing_decisions_by_strategy_total{strategy="single"} 1`,
		`fak_gateway_routing_decisions_by_strategy_total{strategy="ensemble"} 1`,
		`fak_gateway_routing_decisions_by_strategy_total{strategy="default"} 1`,
		`fak_gateway_routing_decisions_by_rule_total{rule="pick-fetch"} 1`,
		`fak_gateway_routing_decisions_by_rule_total{rule="guard-write"} 1`,
		`fak_gateway_routing_decisions_by_rule_total{rule="(default)"} 1`,
		// Every routed subject here is a tool call -> the aspect bucket is tool_call.
		`fak_gateway_routing_decisions_by_aspect_total{aspect="tool_call"} 3`,
		"fak_gateway_routing_ensemble_decisions_total 1",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("/metrics missing %q\n--- metrics ---\n%s", want, routingSlice(text))
		}
	}
	// The overhead family is rendered (emit-at-0 discipline); its value is non-negative.
	if !strings.Contains(text, "fak_gateway_routing_overhead_seconds_total ") {
		t.Fatalf("/metrics missing the routing overhead series\n%s", routingSlice(text))
	}

	// (2) The decision JOURNAL recorded each route (the after-the-fact audit trail). Every
	// record carries the matched rule, strategy, aspect, and a content-address digest.
	records := s.metrics.routing.records()
	if len(records) != 3 {
		t.Fatalf("journal recorded %d decisions, want 3", len(records))
	}
	gotRules := map[string]modelroute.DecisionRecord{}
	for _, r := range records {
		if r.Aspect != modelroute.AspectToolCall {
			t.Fatalf("record aspect = %q, want tool_call", r.Aspect)
		}
		if r.Digest == "" {
			t.Fatalf("record %+v has no digest (the #615 content-address binding the audit to evidence)", r)
		}
		gotRules[r.RuleName] = r
	}
	if got := gotRules["pick-fetch"].Strategy; got != modelroute.StrategySingle {
		t.Fatalf("pick-fetch strategy = %q, want single", got)
	}
	if got := gotRules["guard-write"]; got.Strategy != modelroute.StrategyEnsemble || got.Members != 2 {
		t.Fatalf("guard-write record = %+v, want ensemble with 2 members", got)
	}
	if got := gotRules[""].Strategy; got != modelroute.StrategyDefault {
		t.Fatalf("unmatched decision strategy = %q, want default", got)
	}

	// (3) The operator-line roll-up folds the SAME Counts() the scrape rendered -> the two
	// surfaces agree by construction (the #603 single-source-of-truth requirement).
	sum := s.metrics.routingSummary()
	if sum.Total != 3 || sum.EnsembleHits != 1 {
		t.Fatalf("summary disagrees with metrics: %+v", sum)
	}
	if sum.ByStrategy["single"] != 1 || sum.ByStrategy["ensemble"] != 1 || sum.ByStrategy["default"] != 1 {
		t.Fatalf("summary by-strategy disagrees: %+v", sum.ByStrategy)
	}
	if sum.ByRule["pick-fetch"] != 1 || sum.ByRule["guard-write"] != 1 || sum.ByRule["(default)"] != 1 {
		t.Fatalf("summary by-rule disagrees: %+v", sum.ByRule)
	}
}

// TestRoutingMetricsDormantWithoutManifest proves the honest baseline: with NO routing
// manifest the seam is never reached, so the family renders the emit-at-0 panels and
// records nothing — wired-but-quiet, never a phantom count.
func TestRoutingMetricsDormantWithoutManifest(t *testing.T) {
	s := routeServer(t, nil)
	if _, err := s.buildCall(context.Background(), "anything", `{}`, false, "", ""); err != nil {
		t.Fatalf("buildCall: %v", err)
	}
	if n := len(s.metrics.routing.records()); n != 0 {
		t.Fatalf("no manifest must record 0 decisions, got %d", n)
	}
	text := s.renderMetrics()
	for _, want := range []string{
		"fak_gateway_routing_decisions_total 0",
		// The closed strategy set is still emitted at 0 so the panels exist.
		`fak_gateway_routing_decisions_by_strategy_total{strategy="single"} 0`,
		`fak_gateway_routing_decisions_by_strategy_total{strategy="ensemble"} 0`,
		`fak_gateway_routing_decisions_by_strategy_total{strategy="default"} 0`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("/metrics missing %q\n%s", want, routingSlice(text))
		}
	}
	if sum := s.metrics.routingSummary(); sum.Total != 0 {
		t.Fatalf("dormant summary total = %d, want 0", sum.Total)
	}
}

// routingSlice trims a rendered /metrics blob to the fak_gateway_routing_* lines for a
// readable failure dump (the whole scrape is large).
func routingSlice(text string) string {
	var b strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if strings.Contains(line, "fak_gateway_routing_") {
			b.WriteString(line)
			b.WriteByte('\n')
		}
	}
	return b.String()
}
