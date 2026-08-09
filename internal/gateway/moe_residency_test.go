package gateway

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// moe_residency_test.go — witnesses for the OPERATOR half of R6 (#5617, epic #5606): the gateway
// surfaces that turn the planner's activated-expert ledger into something a human or a scrape can
// read.
//
// What only this layer can get wrong is what it says when there is nothing to say. A proxy planner
// has no ring; a local planner whose operator declared no expert budget never builds one. Both are
// ordinary configurations, and both must render as ABSENT rather than as a ring reporting zero
// hits — the two are indistinguishable on a dashboard, and the ladder's entire premise is that
// knowing whether experts are resident is the thing you could not previously find out.

type moeResidencyPlanner struct{ ledger agent.MoEResidencyLedger }

func (p moeResidencyPlanner) Complete(context.Context, []agent.Message, []agent.ToolDef, ...agent.SampleOpt) (*agent.Completion, error) {
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "ok"}}, nil
}

func (p moeResidencyPlanner) Model() string { return "moe-residency-test" }

func (p moeResidencyPlanner) MoEResidencyStats() agent.MoEResidencyLedger { return p.ledger }

// moeEngagedLedger is a serve that ran a real ring: 800 stagings, 600 of them served from resident
// experts, no refusal, under a 1 MiB budget it never quite filled.
func moeEngagedLedger() agent.MoEResidencyLedger {
	return agent.MoEResidencyLedger{
		Requests:    4,
		Tokens:      1000,
		Lookups:     800,
		Hits:        600,
		PageIns:     200,
		Evictions:   150,
		Refusals:    0,
		PageInBytes: 200 * 4096,
		BudgetBytes: 1 << 20,
		PeakBytes:   900 << 10,
		Last: model.MoEResidencyReport{
			Shape: model.MoEShape{Experts: 256, ExpertsPerToken: 8, Layers: 61, ActivatedFraction: 8.0 / 256.0},
			Ring:  model.ExpertRingStats{Enabled: true},
			Placement: model.ExpertPlacementReport{
				Basis: "pin-set", BasisWidth: 8, Coverage: 0.75, Drift: 0.25, ObservedTouches: 200,
			},
			// Under a shared ring (R7) the coalescing ledger travels with the report. AgentsPerPageIn
			// is meaningless without the B it was measured at, so the fixture carries both — and the
			// render is gated on the Shared block for exactly that reason.
			Shared: &model.SharedExpertRingStats{
				Enabled: true, Agents: 3, PeakAgents: 3,
				Demands: 600, DistinctServes: 500, CrossAgentHits: 300,
			},
			Rates:          model.MoEResidencyRates{AgentsPerPageIn: 2.5, CrossAgentHitRate: 0.5},
			Reconciliation: model.MoEResidencyReconciliation{OK: true},
		},
	}
}

// TestMoEResidencySurfacesAreAbsentUntilARingIsEngaged is the rung's most important negative. Both
// silences are load-bearing: a gateway in front of an upstream provider has no local ring to report,
// and a local serve whose operator declared no expert budget has one that was never built. Emitting
// zeros for either would put a permanent "0% expert hit rate" on every dashboard that scrapes fak.
func TestMoEResidencySurfacesAreAbsentUntilARingIsEngaged(t *testing.T) {
	srv := newTestServer(t)
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_moe_") {
		t.Fatalf("a planner that does not report MoE residency emitted the family:\n%s", text)
	}
	if vars := srv.debugVars(time.Now()); vars.MoEResidency != nil {
		t.Fatalf("debug moe_residency present for a non-reporting planner: %+v", vars.MoEResidency)
	}

	// The other silence: a reporter whose ledger never moved, which is every local serve today.
	srv.planner = moeResidencyPlanner{}
	if text := srv.renderMetrics(); strings.Contains(text, "fak_gateway_moe_") {
		t.Fatalf("a serve that declared no expert budget emitted the family; a scrape of "+
			"`staging_total 0` is indistinguishable from a ring that missed everything:\n%s", text)
	}
	if vars := srv.debugVars(time.Now()); vars.MoEResidency != nil {
		t.Fatalf("debug moe_residency present for an unengaged ring: %+v", vars.MoEResidency)
	}
}

// TestMoEResidencyMetricsPartitionStagingsAndOmitDerivableRates pins both halves of the family's
// design: the three staging outcomes PARTITION the ring's lookups (so a scrape can recover the hit
// rate by summing them), and the ratios themselves are deliberately not emitted, because PromQL
// derives them over the window the operator asks about rather than over this process's lifetime.
func TestMoEResidencyMetricsPartitionStagingsAndOmitDerivableRates(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = moeResidencyPlanner{ledger: moeEngagedLedger()}
	text := srv.renderMetrics()

	for _, want := range []string{
		"fak_gateway_moe_expert_staging_total{outcome=\"hit\"} 600",
		"fak_gateway_moe_expert_staging_total{outcome=\"page_in\"} 200",
		"fak_gateway_moe_expert_staging_total{outcome=\"refused\"} 0",
		"fak_gateway_moe_expert_evictions_total 150",
		"fak_gateway_moe_expert_page_in_bytes_total 819200",
		"fak_gateway_moe_residency_requests_total 4",
		"fak_gateway_moe_residency_tokens_total 1000",
		"fak_gateway_moe_residency_reconciliation_failures_total 0",
		"fak_gateway_moe_expert_budget_bytes 1048576",
		"fak_gateway_moe_expert_peak_resident_bytes 921600",
		"fak_gateway_moe_placement_drift{basis=\"pin-set\"}",
		"fak_gateway_moe_placement_served_share{basis=\"pin-set\"}",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("metrics missing %q:\n%s", want, text)
		}
	}
	// The refusal series must be emitted at zero rather than omitted: a refused staging is the
	// signal that a declared budget stopped being enforced, and an absent series cannot be alerted
	// on for "became non-zero".
	if !strings.Contains(text, "fak_gateway_moe_expert_staging_total{outcome=\"refused\"} 0") {
		t.Fatal("refusal series omitted at zero; it is the one series whose zero is the good news " +
			"and whose absence would silence the alert")
	}
	// 600 + 200 + 0 == 800 lookups: the outcomes partition the lookups, which is why no separate
	// lookups series exists to drift out of agreement with them.
	l := moeEngagedLedger()
	if l.Hits+l.PageIns+l.Refusals != l.Lookups {
		t.Fatalf("fixture is not self-consistent: %d+%d+%d != %d lookups",
			l.Hits, l.PageIns, l.Refusals, l.Lookups)
	}
	for _, unwanted := range []string{"moe_expert_hit_rate", "moe_expert_hit_ratio", "moe_expert_bytes_per_token", "moe_expert_refusal_rate"} {
		if strings.Contains(text, unwanted) {
			t.Fatalf("metrics emit %q, a ratio of two counters already in this family; PromQL must "+
				"derive it so the window is the operator's choice, not this process's uptime", unwanted)
		}
	}
	// The shape gauge is what makes every byte number above readable: 8 of 256 experts.
	if !strings.Contains(text, "fak_gateway_moe_activated_fraction") {
		t.Fatalf("no activated-fraction gauge; the byte rates are unreadable without k/E:\n%s", text)
	}
	if !strings.Contains(text, "fak_gateway_moe_agents_per_page_in") {
		t.Fatalf("shared-ring report carried %.2f agents per page-in and the gauge is missing:\n%s",
			moeEngagedLedger().Last.Rates.AgentsPerPageIn, text)
	}
}

// TestMoEResidencyDebugVarsCarryTheRatesAndNameWhatBroke covers the human surface. It divides once,
// here, because /debug/vars is read by eye; and when the ring's own accounting disagreed with
// itself it must name the failing check rather than leave an operator to guess which of the numbers
// beside it went wrong.
func TestMoEResidencyDebugVarsCarryTheRatesAndNameWhatBroke(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = moeResidencyPlanner{ledger: moeEngagedLedger()}

	v := srv.debugVars(time.Now()).MoEResidency
	if v == nil {
		t.Fatal("debug moe_residency absent for a serve that engaged a ring")
	}
	if v.HitRate != 600.0/800.0 {
		t.Fatalf("hit rate %.6f, want %.6f (hits over hits+page-ins)", v.HitRate, 600.0/800.0)
	}
	if v.ExpertBytesPerToken != float64(200*4096)/1000 {
		t.Fatalf("expert bytes/token %.3f, want %.3f", v.ExpertBytesPerToken, float64(200*4096)/1000)
	}
	if v.PeakBudgetUsed != float64(900<<10)/float64(1<<20) {
		t.Fatalf("peak budget used %.4f", v.PeakBudgetUsed)
	}
	if v.RefusalRate != 0 {
		t.Fatalf("refusal rate %.4f on a ledger with no refusal", v.RefusalRate)
	}
	if v.Experts != 256 || v.ExpertsPerToken != 8 {
		t.Fatalf("shape %d/%d did not reach the block that makes its byte rates readable",
			v.ExpertsPerToken, v.Experts)
	}
	if v.PlacementBasis != "pin-set" || v.PlacementDrift != 0.25 {
		t.Fatalf("placement basis=%q drift=%.3f", v.PlacementBasis, v.PlacementDrift)
	}
	if v.PlacementServedShare != 0.75 {
		t.Fatalf("placement served share %.3f, want 0.75 — the complement of the drift beside it",
			v.PlacementServedShare)
	}
	if v.SharedRingAgents != 3 || v.AgentsPerPageIn != 2.5 {
		t.Fatalf("coalescing carried %d agents at %.2f per page-in; the ratio is uninterpretable "+
			"without the B it was measured at, so both must survive", v.SharedRingAgents, v.AgentsPerPageIn)
	}
	if len(v.FailedChecks) != 0 {
		t.Fatalf("healthy serve named failing checks: %v", v.FailedChecks)
	}

	// Now the alarm. A ring whose accounting disagrees with itself makes every field above wrong,
	// so the block must both count it and say which identity broke.
	broken := moeEngagedLedger()
	broken.ReconciliationFailures = 2
	broken.Last.Reconciliation = model.MoEResidencyReconciliation{
		OK: false,
		Checks: []model.MoEResidencyCheck{
			{Name: "lookups-identity", OK: false, Detail: "800 != 799"},
			{Name: "resident-within-budget", OK: true},
		},
	}
	srv.planner = moeResidencyPlanner{ledger: broken}

	v = srv.debugVars(time.Now()).MoEResidency
	if v.ReconciliationFailures != 2 {
		t.Fatalf("reconciliation failures = %d, want 2", v.ReconciliationFailures)
	}
	if len(v.FailedChecks) != 1 || v.FailedChecks[0] != "lookups-identity" {
		t.Fatalf("failed checks %v, want exactly the one that failed", v.FailedChecks)
	}
	if text := srv.renderMetrics(); !strings.Contains(text, "fak_gateway_moe_residency_reconciliation_failures_total 2") {
		t.Fatalf("the scrape did not carry the reconciliation alarm:\n%s", text)
	}
	// And it still renders the rest: a broken ring is worth an alarm beside the numbers, not a
	// blank block that hides which serve is affected.
	if v.Requests != 4 {
		t.Fatalf("an unreconciled serve blanked its own block (%d requests)", v.Requests)
	}
}
