package gateway

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func dualServer(t *testing.T, localID string) *Server {
	t.Helper()
	dp, err := NewDualPlanner(agent.NewMockPlanner("upstream"), agent.NewMockPlanner(localID), localID)
	if err != nil {
		t.Fatalf("NewDualPlanner: %v", err)
	}
	// Mirror the shape New builds: a dual deployment is always pointed at an upstream,
	// and the classifier has to know WHICH one before it can call the proxied leg
	// bought. This one is a vendor API, so those turns genuinely were.
	return &Server{planner: dp, upstream: upstreamAttribution([]string{"https://api.anthropic.com"})}
}

// Under a dual planner the side is a property of the REQUEST, and the classifier must
// get it from the planner's own routing predicate rather than a second copy of the
// rules — otherwise the metric drifts from what actually served the turn.
func TestServedLocalityFollowsTheDualPlannerRouting(t *testing.T) {
	s := dualServer(t, "qwen-local")
	for _, tc := range []struct {
		name  string
		model string
		want  servingLocality
	}{
		{"the local id decodes in-kernel", "qwen-local", localitySelfHosted},
		{"the reserved alias too", "local", localitySelfHosted},
		{"case and padding do not change the side", "  QWEN-LOCAL ", localitySelfHosted},
		{"any other id proxies upstream", "claude-opus-5", localityVendor},
		// An omitted model routes to the proxy (the planner's documented default
		// side), so it is vendor — not unknown, and certainly not local.
		{"an omitted model is what the proxy served", "", localityVendor},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := s.servedLocality(tc.model); got != tc.want {
				t.Errorf("servedLocality(%q) = %v, want %v", tc.model, got, tc.want)
			}
			// The classifier and the planner must never disagree about the same id.
			if routesLocal := s.planner.(*DualPlanner).RoutesLocal(tc.model); routesLocal != (tc.want == localitySelfHosted) {
				t.Errorf("RoutesLocal(%q) = %v but locality says %v", tc.model, routesLocal, tc.want)
			}
		})
	}
}

// Every other deployment has a constant side, and an unwired one must stay UNKNOWN
// rather than defaulting into a group.
func TestServedLocalityUsesTheDeploymentSideWhenNotDual(t *testing.T) {
	for _, tc := range []struct {
		name string
		srv  *Server
		want servingLocality
	}{
		{"proxy-only pointed at a vendor API serves bought tokens",
			&Server{planner: agent.NewMockPlanner("x"), upstream: upstreamAttribution([]string{"https://api.openai.com/v1"})}, localityVendor},
		{"in-kernel-only serves our own", &Server{planner: agent.NewMockPlanner("x"), servedSide: localitySelfHosted}, localitySelfHosted},
		// The zero value. A planner wired in directly (a test, or a host that
		// bypassed the selector) must not be attributed to either side.
		{"an unwired side stays unclassified", &Server{planner: agent.NewMockPlanner("x")}, localityUnknown},
		{"a nil server classifies nothing", nil, localityUnknown},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := tc.srv.servedLocality("anything"); got != tc.want {
				t.Errorf("servedLocality = %v, want %v", got, tc.want)
			}
		})
	}
}

// The buffered path holds sampling opts rather than a bare id; it must fold them the
// same way the planner does, or it can classify a turn differently from the side that
// served it.
func TestServedLocalityOfOptsMatchesTheRoutedSide(t *testing.T) {
	s := dualServer(t, "qwen-local")
	if got := s.servedLocalityOf([]agent.SampleOpt{agent.WithModel("qwen-local")}); got != localitySelfHosted {
		t.Errorf("opts naming the local id = %v, want local", got)
	}
	if got := s.servedLocalityOf([]agent.SampleOpt{agent.WithModel("claude-opus-5")}); got != localityVendor {
		t.Errorf("opts naming an upstream id = %v, want vendor", got)
	}
	// A nil opt in the slice must not panic the accounting path.
	if got := s.servedLocalityOf([]agent.SampleOpt{nil, agent.WithModel("qwen-local"), nil}); got != localitySelfHosted {
		t.Errorf("opts with nil entries = %v, want local", got)
	}
	if got := s.servedLocalityOf(nil); got != localityVendor {
		t.Errorf("no opts at all = %v, want vendor (the planner's default side)", got)
	}
}

// The load-bearing property: the split is a pair of SUBSETS of the unsplit totals.
// An unclassified turn is counted in the totals and in neither group, so the
// shortfall measures how much volume went unattributed instead of being absorbed
// into whichever side happens to be convenient.
func TestObserveInferenceServedSplitsWithoutDisturbingTheTotals(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInferenceServed(localitySelfHosted, 100, 10, 0, 0, "stop", time.Second)
	m.observeInferenceServed(localitySelfHosted, 200, 20, 0, 0, "stop", time.Second)
	m.observeInferenceServed(localityVendor, 300, 30, 0, 0, "stop", time.Second)
	// Unclassified: a path not yet taught to classify, or a mock.
	m.observeInferenceServed(localityUnknown, 400, 40, 0, 0, "stop", time.Second)
	// And a legacy caller that never passes a side at all.
	m.observeInference(500, 50, 0, 0, "stop", time.Second)

	s := m.adjudicationSummary()
	if s.InputTokens != 1500 || s.OutputTokens != 150 {
		t.Fatalf("unsplit totals = %d in / %d out, want 1500/150 — the split must not disturb them",
			s.InputTokens, s.OutputTokens)
	}
	if s.SelfHostedTurns != 2 || s.SelfHostedInputTokens != 300 || s.SelfHostedOutputTokens != 30 {
		t.Errorf("self-hosted = %d turns %d in %d out, want 2/300/30",
			s.SelfHostedTurns, s.SelfHostedInputTokens, s.SelfHostedOutputTokens)
	}
	if s.VendorTurns != 1 || s.VendorInputTokens != 300 || s.VendorOutputTokens != 30 {
		t.Errorf("vendor = %d turns %d in %d out, want 1/300/30",
			s.VendorTurns, s.VendorInputTokens, s.VendorOutputTokens)
	}
	// 90 of 150 output tokens classified; the other 60 belong to neither side.
	if got := s.SelfHostedOutputTokens + s.VendorOutputTokens; got != 60 {
		t.Errorf("classified output = %d, want 60", got)
	}
	if s.SelfHostedOutputTokens+s.VendorOutputTokens > s.OutputTokens {
		t.Errorf("the split exceeded the total it is a subset of")
	}
}

// A build or deployment that classified nothing must serialize NO split at all. Six
// zeros on the wire would read as "we self-host nothing" — a claim about the fleet
// that no turn ever supported.
func TestAnUnclassifiedSummaryKeepsTheSplitOffTheWire(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInference(500, 50, 0, 0, "stop", time.Second)
	b, err := json.Marshal(m.adjudicationSummary())
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{"self_hosted_turns", "self_hosted_input_tokens", "self_hosted_output_tokens",
		"vendor_turns", "vendor_input_tokens", "vendor_output_tokens"} {
		if strings.Contains(string(b), f) {
			t.Errorf("unclassified summary serialized %q; absence is what carries 'not instrumented'", f)
		}
	}
	if !strings.Contains(string(b), "output_tokens") {
		t.Fatalf("expected the unsplit total to still serialize: %s", b)
	}

	// One classified turn and the split appears — an EARNED zero on the other side.
	m.observeInferenceServed(localityVendor, 10, 1, 0, 0, "stop", time.Second)
	b2, err := json.Marshal(m.adjudicationSummary())
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b2), "vendor_turns") {
		t.Errorf("a classified turn must serialize its side: %s", b2)
	}
	if strings.Contains(string(b2), "self_hosted_turns") {
		t.Errorf("nothing was served locally, so the local group must stay absent: %s", b2)
	}
}

// A negative token count from an upstream must not let a subset exceed its total.
func TestAttributeServedTurnClampsNegatives(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInferenceServed(localitySelfHosted, -5, -7, 0, 0, "stop", time.Second)
	s := m.adjudicationSummary()
	if s.SelfHostedTurns != 1 {
		t.Errorf("SelfHostedTurns = %d, want 1 (the turn still happened)", s.SelfHostedTurns)
	}
	if s.SelfHostedInputTokens != 0 || s.SelfHostedOutputTokens != 0 {
		t.Errorf("negative usage leaked into the split: %d in / %d out",
			s.SelfHostedInputTokens, s.SelfHostedOutputTokens)
	}
}
