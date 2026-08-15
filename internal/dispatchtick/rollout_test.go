package dispatchtick

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// rolloutAcct builds a routable, deterministic account row at an explicit ModelTier
// so a test never depends on the tier-inference heuristics. It is named distinctly
// from the C5 tierroute_test.go acct helper (a different signature) to avoid a
// same-package redeclaration in the shared dispatchtick test binary.
func rolloutAcct(tag string, tier int, avail bool) AccountRow {
	return AccountRow{
		Account:   ".claude-" + tag,
		Tag:       tag,
		Product:   "claude",
		Model:     "opus",
		ModelTier: tier,
		Available: avail,
	}
}

// poolAll is a full pool: a frontier (T1), a mid (T2) and a routine (T3) account,
// all available. For routine work the tier route picks the cheapest (T3).
func poolAll() []AccountRow {
	return []AccountRow{rolloutAcct("frontier", 1, true), rolloutAcct("mid", 2, true), rolloutAcct("local", 3, true)}
}

// routineIssue is the tier metadata for low-risk routine work: required and
// optimal both T2, which floors the route at the cheapest account tier (3).
func routineIssue() IssueTier {
	return IssueTier{Required: modelroute.TierT2, Optimal: modelroute.TierT2, HasTier: true}
}

func TestShadowModeObservesWouldChooseWithoutApplying(t *testing.T) {
	// Live path currently selects a frontier (tier-1) account; the routine issue
	// would choose the cheapest (tier-3). Shadow must REPORT the delta and change
	// nothing.
	d := EvaluateRollout(RolloutInput{
		Mode:        RolloutShadow,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        poolAll(),
		Product:     "claude",
		CurrentTier: 1,
	})

	if !d.ModeValid {
		t.Fatalf("shadow mode should be valid")
	}
	if d.WouldChooseTier != 3 {
		t.Fatalf("would_choose tier: want 3 (cheapest for routine), got %d", d.WouldChooseTier)
	}
	if d.Delta != DeltaCheaper {
		t.Fatalf("delta: want %q, got %q", DeltaCheaper, d.Delta)
	}
	if !d.Differs {
		t.Fatalf("differs should be true (1 vs 3)")
	}
	// The load-bearing shadow invariant: it NEVER switches the live selection.
	if d.Applied {
		t.Fatalf("shadow must never apply a route")
	}
	if d.AppliedTier != 1 {
		t.Fatalf("shadow must leave the live selection (tier 1) in effect, got %d", d.AppliedTier)
	}
	if d.Reason != RolloutReasonShadowObserveOnly {
		t.Fatalf("reason: want %q, got %q", RolloutReasonShadowObserveOnly, d.Reason)
	}
}

func TestCanaryAppliesOnlyToRoutineScope(t *testing.T) {
	// In-scope: routine work, a cheaper route exists, no regression -> APPLIED.
	in := RolloutInput{
		Mode:        RolloutCanary,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        poolAll(),
		Product:     "claude",
		CurrentTier: 1,
	}
	d := EvaluateRollout(in)
	if !d.InCanaryScope {
		t.Fatalf("routine work must be in canary scope")
	}
	if !d.Applied || d.AppliedTier != 3 {
		t.Fatalf("canary should apply the cheaper tier-3 route, got applied=%v tier=%d", d.Applied, d.AppliedTier)
	}
	if d.Reason != RolloutReasonCanaryApplied {
		t.Fatalf("reason: want %q, got %q", RolloutReasonCanaryApplied, d.Reason)
	}

	// OUT of scope: the SAME cheaper route exists, but the work is normal-impl.
	// Canary must NOT apply — default behavior is unchanged outside canary scope.
	for _, class := range []modelroute.WorkClass{
		modelroute.ClassNormalImpl,
		modelroute.ClassUltraHard,
		modelroute.ClassSecurityRelease,
	} {
		out := in
		out.Class = class
		d := EvaluateRollout(out)
		if d.InCanaryScope {
			t.Fatalf("class %q must NOT be in canary scope", class)
		}
		if d.Applied {
			t.Fatalf("class %q must not be canaried (cheaper route ignored outside scope)", class)
		}
		if d.AppliedTier != 1 {
			t.Fatalf("class %q must keep the live selection (tier 1), got %d", class, d.AppliedTier)
		}
		if d.Reason != RolloutReasonCanaryOutOfScope {
			t.Fatalf("class %q reason: want %q, got %q", class, RolloutReasonCanaryOutOfScope, d.Reason)
		}
	}
}

func TestCanaryRollsBackOnRegression(t *testing.T) {
	base := RolloutInput{
		Mode:        RolloutCanary,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        poolAll(),
		Product:     "claude",
		CurrentTier: 1,
	}
	// A quality (outcome-parity) regression rolls the canary back to the live tier.
	quality := base
	quality.Signal = QualitySignal{QualityRegression: true, Note: "cheaper launch missed outcome parity"}
	d := EvaluateRollout(quality)
	if !d.RolledBack || d.Applied {
		t.Fatalf("quality regression must roll back and not apply, got rolledBack=%v applied=%v", d.RolledBack, d.Applied)
	}
	if d.AppliedTier != 1 {
		t.Fatalf("rollback must restore the live selection (tier 1), got %d", d.AppliedTier)
	}
	if d.Reason != RolloutReasonCanaryRolledBack {
		t.Fatalf("reason: want %q, got %q", RolloutReasonCanaryRolledBack, d.Reason)
	}

	// A refusal regression (a dropped DENY reason) rolls back too — the C6 honesty
	// trap wired to the rollback.
	refusal := base
	refusal.Signal = QualitySignal{RefusalRegression: true}
	if d := EvaluateRollout(refusal); !d.RolledBack || d.Applied {
		t.Fatalf("refusal regression must roll back and not apply, got rolledBack=%v applied=%v", d.RolledBack, d.Applied)
	}
}

func TestCanaryRouteRefusedAndNotCheaperKeepLiveSelection(t *testing.T) {
	// Route refused: no account is available at all -> keep the live selection.
	none := []AccountRow{rolloutAcct("frontier", 1, false), rolloutAcct("local", 3, false)}
	d := EvaluateRollout(RolloutInput{
		Mode:        RolloutCanary,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        none,
		Product:     "claude",
		CurrentTier: 2,
	})
	if d.Applied || d.AppliedTier != 2 {
		t.Fatalf("refused route must keep the live selection (tier 2), got applied=%v tier=%d", d.Applied, d.AppliedTier)
	}
	if d.Reason != RolloutReasonCanaryRouteRefused {
		t.Fatalf("reason: want %q, got %q", RolloutReasonCanaryRouteRefused, d.Reason)
	}

	// Would-choose is MORE capable than current (only a frontier account is up, but
	// current is already the cheapest). Canary never upgrades routine work.
	onlyFrontier := []AccountRow{rolloutAcct("frontier", 1, true), rolloutAcct("local", 3, false)}
	up := EvaluateRollout(RolloutInput{
		Mode:        RolloutCanary,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        onlyFrontier,
		Product:     "claude",
		CurrentTier: 3,
	})
	if up.Delta != DeltaMoreCapable {
		t.Fatalf("delta: want %q, got %q", DeltaMoreCapable, up.Delta)
	}
	if up.Applied || up.AppliedTier != 3 {
		t.Fatalf("canary must not upgrade routine work, got applied=%v tier=%d", up.Applied, up.AppliedTier)
	}
	if up.Reason != RolloutReasonCanaryNotCheaper {
		t.Fatalf("reason: want %q, got %q", RolloutReasonCanaryNotCheaper, up.Reason)
	}

	// Would-choose MATCHES current -> no change needed.
	same := EvaluateRollout(RolloutInput{
		Mode:        RolloutCanary,
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        poolAll(),
		Product:     "claude",
		CurrentTier: 3,
	})
	if same.Applied || same.Reason != RolloutReasonCanaryNoChange {
		t.Fatalf("matching tier should report no-change, got applied=%v reason=%q", same.Applied, same.Reason)
	}
}

func TestRolloutModeGuardCanaryOffDefaultUnknown(t *testing.T) {
	base := RolloutInput{
		Class:       modelroute.ClassRoutine,
		Issue:       routineIssue(),
		Rows:        poolAll(),
		Product:     "claude",
		CurrentTier: 1,
	}

	// OFF is the default: nothing computed, nothing applied, live selection stands.
	off := base
	off.Mode = RolloutOff
	if d := EvaluateRollout(off); d.Applied || d.AppliedTier != 1 || d.WouldChooseTier != 0 || d.Reason != RolloutReasonOffCurrentSelection {
		t.Fatalf("off mode must leave live selection untouched and compute nothing, got %+v", d)
	}

	// ON (broad default-on) is OUT OF SCOPE — the guard refuses it. It is a KNOWN
	// rung of the shared ladder and still unreachable here: naming it the shared
	// way must not make it look implemented (#6090).
	def := base
	def.Mode = RolloutOn
	d := EvaluateRollout(def)
	if !d.ModeValid {
		t.Fatalf("on is a known rung of the shared ladder")
	}
	if d.Applied || d.AppliedTier != 1 || d.Reason != RolloutReasonOnOutOfScope {
		t.Fatalf("default-on routing must be refused, got %+v", d)
	}
	// The retired private spelling is NOT a rung any more: it fails closed like any
	// other unknown string, rather than quietly reaching the refusal arm.
	retired := base
	retired.Mode = RolloutMode("default")
	if r := EvaluateRollout(retired); r.ModeValid || r.Applied || r.Reason != RolloutReasonUnknownMode {
		t.Fatalf("the retired %q spelling must fail closed, got %+v", "default", r)
	}

	// An unknown mode fails closed.
	unknown := base
	unknown.Mode = RolloutMode("frobnicate")
	u := EvaluateRollout(unknown)
	if u.ModeValid || u.Applied || u.Reason != RolloutReasonUnknownMode {
		t.Fatalf("unknown mode must fail closed, got %+v", u)
	}
}

func TestShadowReportDryRunLaunchesNothing(t *testing.T) {
	items := []ShadowItem{
		{ID: "watchdog-1", Class: modelroute.ClassRoutine, Issue: routineIssue(), Rows: poolAll(), Product: "claude", CurrentTier: 1},
		{ID: "status-2", Class: modelroute.ClassRoutine, Issue: routineIssue(), Rows: poolAll(), Product: "claude", CurrentTier: 3},
		{ID: "impl-3", Class: modelroute.ClassNormalImpl, Issue: routineIssue(), Rows: poolAll(), Product: "claude", CurrentTier: 1},
	}
	rep := FoldShadowReport(items)

	if rep.Schema != ShadowReportSchema || rep.Mode != RolloutShadow {
		t.Fatalf("report header wrong: schema=%q mode=%q", rep.Schema, rep.Mode)
	}
	if rep.Items != 3 || len(rep.Rows) != 3 {
		t.Fatalf("want 3 rows, got items=%d rows=%d", rep.Items, len(rep.Rows))
	}
	// The whole point of a dry-run: NOTHING is applied, ever.
	if rep.AnyApplied {
		t.Fatalf("a shadow dry-run must launch/apply nothing")
	}
	for _, row := range rep.Rows {
		if row.Applied {
			t.Fatalf("row %q applied in a shadow readout", row.ID)
		}
	}
	// watchdog-1 (1->3) and impl-3 (1->3) are cheaper; status-2 (3->3) is same.
	if rep.Cheaper != 2 || rep.Same != 1 {
		t.Fatalf("delta tally: want cheaper=2 same=1, got cheaper=%d same=%d", rep.Cheaper, rep.Same)
	}
	// Only the routine cheaper item is a canary candidate; impl-3 is out of scope.
	if rep.CanaryEligible != 1 {
		t.Fatalf("canary-eligible: want 1 (routine cheaper only), got %d", rep.CanaryEligible)
	}
}
