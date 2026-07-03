package gateway

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// The offensive cache-breakpoint placement family must witness every attempt outcome (placed +
// each bail reason) and must emit the "placed" row even at zero, so a passthrough that never
// placed reads as visible-zero rather than an absent panel. "placed" is the fak-authored slice;
// "already_set" is the Claude-Code shape fak leaves to the client's own cache.
func TestCacheBreakpointPlacementMetrics(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observePlacement(agent.BreakpointOutcome{Reason: agent.BreakpointReasonNone})         // a real placement
	m.observePlacement(agent.BreakpointOutcome{Reason: agent.BreakpointReasonNone})         // second turn places again
	m.observePlacement(agent.BreakpointOutcome{Reason: agent.BreakpointReasonAlreadySet})   // client had its own bp
	m.observePlacement(agent.BreakpointOutcome{Reason: agent.BreakpointReasonNoStableHead}) // no head to anchor

	var b strings.Builder
	m.writeCompactionMetrics(&b)
	got := b.String()
	for _, want := range []string{
		`fak_gateway_cache_breakpoint_placement_total{outcome="placed"} 2`,
		`fak_gateway_cache_breakpoint_placement_total{outcome="already_set"} 1`,
		`fak_gateway_cache_breakpoint_placement_total{outcome="no_stable_head"} 1`,
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("placement metrics missing %q:\n%s", want, got)
		}
	}
}

// Zero attempts: the "placed" row still renders at 0 (panel exists pre-first-attempt), and no
// phantom reason rows appear — the always-on witness reads visible-zero, not absent.
func TestCacheBreakpointPlacementMetricsZero(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	var b strings.Builder
	m.writeCompactionMetrics(&b)
	got := b.String()
	if !strings.Contains(got, `fak_gateway_cache_breakpoint_placement_total{outcome="placed"} 0`) {
		t.Fatalf("zero-state placed row missing:\n%s", got)
	}
	if strings.Count(got, "fak_gateway_cache_breakpoint_placement_total{") != 1 {
		t.Fatalf("unexpected extra placement rows in zero state:\n%s", got)
	}
}
