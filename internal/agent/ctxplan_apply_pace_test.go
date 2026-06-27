package agent

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
)

// TestApplyPaceComposesBudget proves the genuine wire (#628): a throttled session.Pace drives
// the planner's resident-context Budget DOWN, an un-paced one leaves it untouched, and the
// scale is taken from the construction baseline so the call is idempotent and restorable.
func TestApplyPaceComposesBudget(t *testing.T) {
	const base = 4096
	const baseline = 2048

	sp := NewSessionPlanner(base)
	if sp.Budget != base {
		t.Fatalf("fresh planner Budget = %d, want %d", sp.Budget, base)
	}

	// Half the per-turn output -> half the window.
	if got := sp.ApplyPace(session.Pace{MaxTokensPerTurn: 1024}, baseline); got != 2048 || sp.Budget != 2048 {
		t.Fatalf("ApplyPace(half) -> Budget %d (returned %d), want 2048", sp.Budget, got)
	}

	// Idempotent: re-applying the SAME pace lands on the same Budget (scales the baseline, not
	// the already-throttled 2048 -> never compounds to 1024).
	if got := sp.ApplyPace(session.Pace{MaxTokensPerTurn: 1024}, baseline); got != 2048 {
		t.Fatalf("ApplyPace(half) re-applied -> %d, want 2048 (must not compound)", got)
	}

	// A harder throttle scales from the baseline, not from the prior throttle.
	if got := sp.ApplyPace(session.Pace{MaxTokensPerTurn: 512}, baseline); got != 1024 {
		t.Fatalf("ApplyPace(quarter) -> %d, want 1024 (scale the baseline)", got)
	}

	// Clearing the pace restores the full baseline window.
	if got := sp.ApplyPace(session.Pace{}, baseline); got != base || sp.Budget != base {
		t.Fatalf("ApplyPace(cleared) -> Budget %d (returned %d), want the restored base %d", sp.Budget, got, base)
	}
}

// TestApplyPaceChangesPlanning proves ApplyPace is a LIVE wire, not a dead field: the lowered
// Budget actually reaches PlanTurn and elides spans a full window would have kept resident. A
// vacuity guard fails the test if the full-budget plan didn't fit everything (then the
// comparison would prove nothing) — the same load-bearing discipline the goal-pin test uses.
func TestApplyPaceChangesPlanning(t *testing.T) {
	const base = 4096
	const baseline = 2048

	// A session whose content clearly exceeds the throttled window but fits the full one: 16
	// distinct user spans of ~120 tokens each (~480 bytes) ≈ 1900 tokens of body.
	body := strings.Repeat("alpha beta gamma delta ", 20) // ~460 bytes -> ~115 tokens
	msgs := []Message{{Role: RoleSystem, Content: "system preamble"}}
	for i := 0; i < 16; i++ {
		msgs = append(msgs, Message{Role: RoleUser, Content: body + "marker" + string(rune('a'+i))})
	}

	full := NewSessionPlanner(base)
	fullPlan := full.PlanTurn(msgs)

	throttled := NewSessionPlanner(base)
	// Throttle to the floor (base/8 = 512 tokens) — far below the ~1900 tokens of body.
	if got := throttled.ApplyPace(session.Pace{MaxTokensPerTurn: 1}, baseline); got != base/session.MinPlannerBudgetDivisor {
		t.Fatalf("deep throttle Budget = %d, want the floor %d", got, base/session.MinPlannerBudgetDivisor)
	}
	throttledPlan := throttled.PlanTurn(msgs)

	// Vacuity guard: the full window must actually fit MORE than the throttled floor can, or the
	// comparison is meaningless.
	if len(fullPlan.Selected) <= len(throttledPlan.Selected) {
		t.Fatalf("premise broken: full plan kept %d spans, throttled kept %d — the throttle didn't bind, tighten the session/budget so the assertion isn't vacuous",
			len(fullPlan.Selected), len(throttledPlan.Selected))
	}
}
