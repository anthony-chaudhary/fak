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

// TestApplyThroughputComposesBudget proves the #1585 wire: a session measurably falling
// behind its EXPECTED throughput — no configured MaxTokensPerTurn cap at all — still drives
// the planner's resident-context Budget DOWN, a session keeping pace leaves it untouched, and
// the scale is taken from the construction baseline so the call is idempotent and restorable,
// exactly like ApplyPace's contract.
func TestApplyThroughputComposesBudget(t *testing.T) {
	const base = 4096

	sp := NewSessionPlanner(base)

	// Observed at half the expected rate -> half the window.
	slow := session.Throughput{ObservedTokensPerSec: 50, ExpectedTokensPerSec: 100}
	if got := sp.ApplyThroughput(slow); got != 2048 || sp.Budget != 2048 {
		t.Fatalf("ApplyThroughput(half pace) -> Budget %d (returned %d), want 2048", sp.Budget, got)
	}

	// Idempotent: re-applying the SAME observation lands on the same Budget (scales the
	// baseline, never compounds).
	if got := sp.ApplyThroughput(slow); got != 2048 {
		t.Fatalf("ApplyThroughput(half pace) re-applied -> %d, want 2048 (must not compound)", got)
	}

	// A deeper observed slowdown scales from the baseline, not from the prior throttle.
	quarter := session.Throughput{ObservedTokensPerSec: 25, ExpectedTokensPerSec: 100}
	if got := sp.ApplyThroughput(quarter); got != 1024 {
		t.Fatalf("ApplyThroughput(quarter pace) -> %d, want 1024 (scale the baseline)", got)
	}

	// Catching back up to the expected rate restores the full baseline window.
	if got := sp.ApplyThroughput(session.Throughput{ObservedTokensPerSec: 100, ExpectedTokensPerSec: 100}); got != base || sp.Budget != base {
		t.Fatalf("ApplyThroughput(on pace) -> Budget %d (returned %d), want the restored base %d", sp.Budget, got, base)
	}

	// No observation yet is a no-op.
	if got := sp.ApplyThroughput(session.Throughput{}); got != base {
		t.Fatalf("ApplyThroughput(no signal) -> %d, want the base %d unchanged", got, base)
	}
}

// TestApplyThroughputChangesPlanning is the throughput analogue of TestApplyPaceChangesPlanning:
// a session OBSERVED to be running well behind its expected throughput (never mind any
// configured cap) actually elides spans a full window would have kept resident. The vacuity
// guard fails the test if the full-budget plan did not already keep strictly more than the
// throttled-floor plan can, so the comparison cannot pass for a vacuous reason.
func TestApplyThroughputChangesPlanning(t *testing.T) {
	const base = 4096

	body := strings.Repeat("alpha beta gamma delta ", 20) // ~460 bytes -> ~115 tokens
	msgs := []Message{{Role: RoleSystem, Content: "system preamble"}}
	for i := 0; i < 16; i++ {
		msgs = append(msgs, Message{Role: RoleUser, Content: body + "marker" + string(rune('a'+i))})
	}

	full := NewSessionPlanner(base)
	fullPlan := full.PlanTurn(msgs)

	throttled := NewSessionPlanner(base)
	// Observed at a near-stalled rate against the expected one -> the floor (base/8=512).
	crawling := session.Throughput{ObservedTokensPerSec: 0.001, ExpectedTokensPerSec: 1000}
	if got := throttled.ApplyThroughput(crawling); got != base/session.MinPlannerBudgetDivisor {
		t.Fatalf("near-stalled throughput Budget = %d, want the floor %d", got, base/session.MinPlannerBudgetDivisor)
	}
	throttledPlan := throttled.PlanTurn(msgs)

	if len(fullPlan.Selected) <= len(throttledPlan.Selected) {
		t.Fatalf("premise broken: full plan kept %d spans, throughput-throttled kept %d — the observed signal didn't bind",
			len(fullPlan.Selected), len(throttledPlan.Selected))
	}
}

// TestApplyPaceAndThroughputTakesTighterConstraint proves the folded entry point reconciles
// a configured cap with an observed throughput signal by taking whichever is more
// constraining — a session that is both configured-throttled AND running behind its expected
// rate must not have one signal silently override the other.
func TestApplyPaceAndThroughputTakesTighterConstraint(t *testing.T) {
	const base = 4096
	const baselineOutput = 2048

	sp := NewSessionPlanner(base)
	pace := session.Pace{MaxTokensPerTurn: 512}                                   // quarter of baselineOutput -> composed 1024
	tp := session.Throughput{ObservedTokensPerSec: 50, ExpectedTokensPerSec: 100} // half of expected -> composed 2048
	if got := sp.ApplyPaceAndThroughput(pace, tp, baselineOutput); got != 1024 {
		t.Fatalf("ApplyPaceAndThroughput = %d, want the tighter (configured) budget 1024", got)
	}
	if sp.Budget != 1024 {
		t.Fatalf("sp.Budget = %d, want 1024", sp.Budget)
	}

	// Flip which axis is tighter: the observed signal now dominates.
	pace2 := session.Pace{MaxTokensPerTurn: 1536}                                  // mild throttle, close to base
	tp2 := session.Throughput{ObservedTokensPerSec: 10, ExpectedTokensPerSec: 100} // deep observed slowdown
	if got := sp.ApplyPaceAndThroughput(pace2, tp2, baselineOutput); got != base/session.MinPlannerBudgetDivisor {
		t.Fatalf("ApplyPaceAndThroughput (observed-dominant) = %d, want the floor %d", got, base/session.MinPlannerBudgetDivisor)
	}

	// Clearing both restores the full baseline window.
	if got := sp.ApplyPaceAndThroughput(session.Pace{}, session.Throughput{}, baselineOutput); got != base {
		t.Fatalf("ApplyPaceAndThroughput (cleared) = %d, want the restored base %d", got, base)
	}
}
