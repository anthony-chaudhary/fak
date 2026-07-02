package vcacheobserve

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

func TestPlanProviderActionsRidesNaturalWithoutSpending(t *testing.T) {
	plan := PlanProviderActions(twoFamilies(), true)
	if plan.Schema != ProviderActionSchema || plan.Turns != 6 || plan.FamilyCount != 2 || !plan.WindowCapped {
		t.Fatalf("plan header = %+v", plan)
	}
	if plan.Counts.Noop != 2 || plan.Counts.Gated != 0 || plan.Counts.Ready != 0 {
		t.Fatalf("counts = %+v, want two no-op ride-natural rows", plan.Counts)
	}
	if plan.Transport.Ready || plan.Transport.Mode != "decision_only" {
		t.Fatalf("transport = %+v, want decision-only non-spend surface", plan.Transport)
	}
	for _, row := range plan.Actions {
		if row.Decision != vcachegov.DecisionRideNatural || row.Action != "ride_natural" || row.State != ActionNoop {
			t.Fatalf("row = %+v, want ride-natural no-op", row)
		}
		if row.CacheReadTokens <= 0 {
			t.Fatalf("row should carry observed provider cache-read counters: %+v", row)
		}
	}
}

func TestPlanProviderActionsGatesHeartbeatPinsUntilTransportExists(t *testing.T) {
	const sec = int64(1000)
	turns := []Turn{
		{Family: "bursty", UnixMillis: 0, InputTokens: 100, CacheCreation: 40000},
		{Family: "bursty", UnixMillis: 700 * sec, InputTokens: 50, CacheRead: 40000},
	}
	plan := PlanProviderActions(turns, false)
	if len(plan.Actions) != 1 {
		t.Fatalf("actions = %+v, want one family", plan.Actions)
	}
	row := plan.Actions[0]
	if row.Decision != vcachegov.DecisionHeartbeatPin || row.Action != "heartbeat_pin" || row.State != ActionGated {
		t.Fatalf("row = %+v, want gated heartbeat pin", row)
	}
	if plan.Counts.Gated != 1 || plan.Counts.Noop != 0 || plan.Counts.Ready != 0 {
		t.Fatalf("counts = %+v, want one gated row", plan.Counts)
	}
}

func TestPlanProviderActionsEmptyWindowIsExplicit(t *testing.T) {
	plan := PlanProviderActions(nil, false)
	if plan.Schema != ProviderActionSchema || len(plan.Actions) != 0 || plan.FamilyCount != 0 || plan.Turns != 0 {
		t.Fatalf("empty plan = %+v", plan)
	}
	if plan.Transport.Reason == "" || plan.CorrectnessLaw == "" {
		t.Fatalf("empty plan should still explain its evidence boundary: %+v", plan)
	}
}

func TestPlanProviderActionsIgnoresContextOnlyRows(t *testing.T) {
	plan := PlanProviderActions([]Turn{{
		Family:              "context",
		ContextEvents:       1,
		ContextShedTokens:   900,
		ContextDroppedTurns: 1,
	}}, false)
	if plan.Turns != 0 || plan.FamilyCount != 0 || len(plan.Actions) != 0 {
		t.Fatalf("context-only rows must not invent provider actions: %+v", plan)
	}
}
