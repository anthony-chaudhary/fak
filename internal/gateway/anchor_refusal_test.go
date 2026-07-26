package gateway

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

// The gateway half of #3622. internal/metrics/anchor_refusal_test.go already pins the pure
// fold; what is untested without this file is the WIRING — that observePlacement actually
// feeds the monitor, that adjudicationSummary carries the report out, and that a session
// which never placed reports nothing rather than a fabricated clean verdict. Those are the
// links that break silently: the fold keeps passing its own tests while the operator
// surface goes quiet, which is the exact failure mode the watchdog exists to prevent.

// observe drives n placement attempts with one outcome through the real producer path.
func observe(m *gatewayMetrics, outcome string, n int) {
	for i := 0; i < n; i++ {
		m.observePlacement(agent.BreakpointOutcome{Reason: outcome})
	}
}

// TestAnchorPlacementIsNilUntilAPlacementIsAttempted pins the absence. A cold session, a
// non-Anthropic wire, or a body the anchor never applied to must leave the summary field
// unset — and the JSON byte-identical to before this field existed — because a 0% refused
// report the gateway never measured reads as evidence the anchor is healthy.
func TestAnchorPlacementIsNilUntilAPlacementIsAttempted(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	if got := m.adjudicationSummary().AnchorPlacement; got != nil {
		t.Fatalf("AnchorPlacement = %+v on a session with no placements, want nil", got)
	}
	sum := m.adjudicationSummary()
	if sum.AnchorRefusedRising() || sum.AnchorRefusalAlarmed() {
		t.Error("a session that never placed must not raise or alarm")
	}
	if sum.AnchorRefusalBanner() != "" {
		t.Errorf("banner = %q, want empty on a session that never placed", sum.AnchorRefusalBanner())
	}
	if sum.AnchorRefusalOutcomes() != nil {
		t.Error("outcomes must be nil on a session that never placed")
	}
	b, err := json.Marshal(sum)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "anchor_placement") {
		t.Errorf("the summary wire must omit anchor_placement entirely, got: %s", b)
	}
}

// TestObservePlacementFeedsTheLiveMonitor is the wiring assertion: outcomes recorded on the
// cumulative counter reach the rolling fold and come back out through the summary. If the
// observePlacement call site is ever dropped the counters keep incrementing and this reds,
// which is the only way that regression is visible.
func TestObservePlacementFeedsTheLiveMonitor(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	observe(m, metrics.AnchorOutcomePlaced, 3)
	observe(m, "volatile_head", 2)

	r := m.adjudicationSummary().AnchorPlacement
	if r == nil {
		t.Fatal("AnchorPlacement is nil after 5 observed placements: the monitor is not wired")
	}
	if r.Turns != 5 {
		t.Errorf("Turns = %d, want 5", r.Turns)
	}
	if r.Earned != 3 || r.Refused != 2 {
		t.Errorf("(earned, refused) = (%d, %d), want (3, 2)", r.Earned, r.Refused)
	}
	// The monitor mirrors the counter it rides on; a divergence means one of the two writes
	// was lost under the shared lock.
	m.compactMu.Lock()
	placed, volatile := m.placementAttempts[metrics.AnchorOutcomePlaced], m.placementAttempts["volatile_head"]
	m.compactMu.Unlock()
	if int(placed) != r.Earned || int(volatile) != r.Refused {
		t.Errorf("counter (%d placed, %d volatile) disagrees with monitor (%d earned, %d refused)",
			placed, volatile, r.Earned, r.Refused)
	}
}

// TestAlreadySetNeverRaisesAtTheGateway carries the fold's false-positive guard up to the
// surface an operator actually reads. A Claude-Code-shaped session is ~100% already_set by
// construction: the client authored its own cache_control and every turn IS cached. Pricing
// that as a refusal would alarm every healthy client on earth, which is how a monitor
// becomes a thing operators mute.
func TestAlreadySetNeverRaisesAtTheGateway(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	observe(m, "already_set", 40)

	sum := m.adjudicationSummary()
	r := sum.AnchorPlacement
	if r == nil {
		t.Fatal("a session that attempted 40 placements must report, even when all deferred")
	}
	if r.Deferred != 40 || r.Refused != 0 {
		t.Errorf("(deferred, refused) = (%d, %d), want (40, 0)", r.Deferred, r.Refused)
	}
	if sum.AnchorRefusedRising() || sum.AnchorRefusalAlarmed() {
		t.Error("an all-already_set session must never raise ANCHOR_REFUSED_RISING")
	}
	if got := sum.AnchorRefusalOutcomes(); len(got) != 0 {
		t.Errorf("refused outcomes = %v, want none: already_set is deferred, not refused", got)
	}
}

// TestRisingKeysOnCrossingNotEndState is the reason there are two predicates rather than
// one. A session that turned volatile and then recovered ends un-alarmed, but it still spent
// turns paying uncached — an exit artifact reporting only the final instant would hide
// exactly the mid-session degradation this watchdog is for.
func TestRisingKeysOnCrossingNotEndState(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	observe(m, "volatile_head", 8)             // go volatile: cross the threshold
	observe(m, metrics.AnchorOutcomePlaced, 8) // then fully recover

	sum := m.adjudicationSummary()
	if sum.AnchorPlacement == nil {
		t.Fatal("AnchorPlacement is nil after 16 placements")
	}
	if !sum.AnchorRefusedRising() {
		t.Errorf("AnchorRefusedRising = false after a volatile stretch (findings=%d): a recovered "+
			"session must still report that it degraded", sum.AnchorPlacement.Findings)
	}
	if sum.AnchorRefusalAlarmed() {
		t.Error("AnchorRefusalAlarmed = true after full recovery: the end-state twin must clear")
	}
	if sum.AnchorRefusalBanner() == "" {
		t.Error("a session with a finding must render a banner row")
	}
}

// TestAnchorRefusalOutcomesReturnsOnlyRefusals pins the drilldown the banner prints under the
// finding. The operator's first question after "the anchor stopped earning" is WHICH bail it
// was — volatile_head (the caller's head carries a per-request token) and no_stable_head (no
// system[]/tools[] block at all) point at different fixes — so the earned and deferred
// buckets must not be mixed in.
func TestAnchorRefusalOutcomesReturnsOnlyRefusals(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	observe(m, metrics.AnchorOutcomePlaced, 2)
	observe(m, "already_set", 2)
	observe(m, "volatile_head", 5)
	observe(m, "no_stable_head", 1)

	got := m.adjudicationSummary().AnchorRefusalOutcomes()
	if len(got) != 2 {
		t.Fatalf("refused outcomes = %v, want exactly the two refusal buckets", got)
	}
	for _, tal := range got {
		if tal.Class != metrics.AnchorRefused {
			t.Errorf("outcome %q has class %q in the refusal drilldown", tal.Outcome, tal.Class)
		}
	}
	// Deterministic order (turns desc), so the dominant bail is what an operator reads first.
	if got[0].Outcome != "volatile_head" || got[0].Turns != 5 {
		t.Errorf("first refusal = %+v, want volatile_head with 5 turns", got[0])
	}
}

// TestMonitorIsBuiltLazilyForADirectlyConstructedMetrics covers the posture observePlacement
// documents: a Server assembled without newGatewayMetrics still monitors. Without the lazy
// build the fold would nil-pointer or silently no-op on that path, and it is not the path
// any other test in this package exercises.
func TestMonitorIsBuiltLazilyForADirectlyConstructedMetrics(t *testing.T) {
	m := &gatewayMetrics{}
	if m.anchorRefusalReport() != nil {
		t.Fatal("a metrics with no monitor and no observations must report nil")
	}
	observe(m, "volatile_head", 1)
	r := m.anchorRefusalReport()
	if r == nil || r.Turns != 1 || r.Refused != 1 {
		t.Fatalf("report = %+v, want 1 refused turn from a lazily built monitor", r)
	}
}
