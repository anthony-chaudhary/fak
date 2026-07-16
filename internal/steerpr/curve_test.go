package steerpr

import (
	"reflect"
	"strings"
	"testing"
)

// allSignals and allRungs are the closed vocabularies restated INDEPENDENTLY of
// the production consts, so a table below cannot silently agree with a broken
// enum. If production drops or renames a value, these stay put and the coverage
// counts pinned in the tests fail.
var allSignals = []CurveSignal{CurveHealthy, CurveStall, CurveDrift, CurveDetourOverrun}
var allRungs = []CurveRung{RungW3, RungW2, RungW1, RungW0}

// signalNeedsAttentionOracle is a hand-written truth table for NeedsAttention,
// derived from the issue text ("surface a DRIFT or DETOUR_OVERRUN curve ... even
// when the band is CLEARED") and NOT from the code under test.
var signalNeedsAttentionOracle = map[CurveSignal]bool{
	CurveHealthy:       false,
	CurveStall:         false,
	CurveDrift:         true,
	CurveDetourOverrun: true,
}

// rungActionableOracle is a hand-written truth table for Actionable, derived from
// trajctl's rung doctrine ("never act on a bare W0 self-report"); the empty rung
// stands in for "unset", which must also fail closed.
var rungActionableOracle = map[CurveRung]bool{
	RungW3: true, RungW2: true, RungW1: true, RungW0: false, CurveRung(""): false,
}

// TestNeedsAttentionOverFullVocabulary pins NeedsAttention against the independent
// oracle across every signal, and proves a nil curve (no bound objective) needs
// no attention — the degrade-cleanly case.
func TestNeedsAttentionOverFullVocabulary(t *testing.T) {
	for _, sig := range allSignals {
		want := signalNeedsAttentionOracle[sig]
		got := (&Curve{ObjectiveID: "o", Signal: sig, Rung: RungW3}).NeedsAttention()
		if got != want {
			t.Errorf("NeedsAttention(%q) = %v, want %v", sig, got, want)
		}
	}
	if len(signalNeedsAttentionOracle) != len(allSignals) {
		t.Fatalf("oracle covers %d signals, vocabulary has %d", len(signalNeedsAttentionOracle), len(allSignals))
	}
	var nilCurve *Curve
	if nilCurve.NeedsAttention() {
		t.Error("nil curve (no bound objective) must not need attention")
	}
}

// TestActionableHonorsRungDoctrine pins Actionable against the independent oracle:
// only a bare W0 (or unset) rung fails to clear the bar. This is the "never act on
// a bare self-report" doctrine made mechanical.
func TestActionableHonorsRungDoctrine(t *testing.T) {
	for rung, want := range rungActionableOracle {
		got := (&Curve{ObjectiveID: "o", Signal: CurveDrift, Rung: rung}).Actionable()
		if got != want {
			t.Errorf("Actionable(rung=%q) = %v, want %v", rung, got, want)
		}
	}
	var nilCurve *Curve
	if nilCurve.Actionable() {
		t.Error("nil curve must not be actionable")
	}
}

// TestAnnotateRendersSignalAndRung is done-condition #1: a unit bound to an
// objective renders its curve signal AND its witness rung. The render captures
// the bytes the operator surface emits — the display witness for a display change.
func TestAnnotateRendersSignalAndRung(t *testing.T) {
	c := &Curve{ObjectiveID: "#5038", Signal: CurveHealthy, Rung: RungW3, Latest: 0.7, Delta: 0.1, Detail: "progress 0.70 (delta +0.10)"}
	got := c.Annotate()
	if !strings.Contains(got, string(CurveHealthy)) {
		t.Errorf("annotation %q must name the signal %q", got, CurveHealthy)
	}
	if !strings.Contains(got, string(RungW3)) {
		t.Errorf("annotation %q must name the witness rung %q", got, RungW3)
	}
	if !strings.Contains(got, "progress 0.70") {
		t.Errorf("annotation %q must carry the curve detail", got)
	}
	// A HEALTHY, W3 curve is on course and fully evidenced: no attention marker,
	// no not-actionable tag.
	if strings.Contains(got, "⚠") {
		t.Errorf("a HEALTHY curve must not carry the attention marker: %q", got)
	}
	if strings.Contains(got, "not actionable") {
		t.Errorf("a W3 curve must not read as not-actionable: %q", got)
	}
}

// TestDriftVisibleOnClearedUnit is done-condition #2 AND the acceptance gate: a
// DRIFT curve must be visible on a unit whose band is CLEARED — the "individually
// correct, collectively wrong" case. If a drifting objective can hide behind a
// clean band, the ticket is not done.
func TestDriftVisibleOnClearedUnit(t *testing.T) {
	cleared := Unit{Leaf: "steerpr", Band: BandCleared}
	drift := Curve{ObjectiveID: "#5038", Signal: CurveDrift, Rung: RungW3, Detail: "progress declined 0.60 -> 0.40 (delta -0.20)"}
	unit := cleared.WithCurve(drift)

	// The band is untouched — the curve is a SEPARATE axis, never folded in.
	if unit.Band != BandCleared {
		t.Fatalf("binding a curve must not move the band: got %q, want CLEARED", unit.Band)
	}
	// The drift is surfaced despite the clean band.
	if !unit.Curve.NeedsAttention() {
		t.Fatal("a DRIFT curve on a CLEARED unit must need attention")
	}
	ann := unit.Curve.Annotate()
	if !strings.HasPrefix(ann, "⚠ curve") {
		t.Errorf("a DRIFT annotation must lead with the attention marker: %q", ann)
	}
	if !strings.Contains(ann, string(CurveDrift)) {
		t.Errorf("annotation %q must name DRIFT", ann)
	}

	// DriftHiddenByBand is the acceptance witness: the CLEARED+DRIFT unit is in the
	// set, and a plain CLEARED unit with no curve is not.
	clean := Unit{Leaf: "other", Band: BandCleared}
	hidden := DriftHiddenByBand([]Unit{unit, clean})
	if len(hidden) != 1 || hidden[0].Leaf != "steerpr" {
		t.Fatalf("DriftHiddenByBand must return exactly the CLEARED+DRIFT unit, got %+v", hidden)
	}
}

// TestNoObjectiveRendersCleanly is done-condition #3: a unit with no bound
// objective renders cleanly — no curve, no annotation, no warning, and never in
// the hidden-drift set. This is the common case today and must not read as a gap.
func TestNoObjectiveRendersCleanly(t *testing.T) {
	unit := Unit{Leaf: "steerpr", Band: BandCleared}
	if unit.Curve != nil {
		t.Fatal("a fresh unit must have no curve")
	}
	if unit.Curve.Annotate() != "" {
		t.Errorf("a curveless unit must render an empty annotation, got %q", unit.Curve.Annotate())
	}
	if unit.Curve.NeedsAttention() {
		t.Error("a curveless unit must not need attention")
	}
	if got := DriftHiddenByBand([]Unit{unit}); len(got) != 0 {
		t.Errorf("a curveless CLEARED unit is not hidden drift, got %+v", got)
	}
	// WithCurve on a zero ObjectiveID is a no-op bind: still curveless.
	if u := unit.WithCurve(Curve{Signal: CurveDrift}); u.Curve != nil {
		t.Error("WithCurve on an empty ObjectiveID must leave the unit curveless")
	}
}

// TestW0CurveRenderedButNotActionable is done-condition #4: a W0-only curve is
// rendered (the operator sees it) but explicitly marked not actionable, so it can
// never be mistaken for evidence that may gate an action.
func TestW0CurveRenderedButNotActionable(t *testing.T) {
	c := &Curve{ObjectiveID: "#5038", Signal: CurveDrift, Rung: RungW0, Detail: "self-reported decline"}
	if c.Actionable() {
		t.Fatal("a bare W0 curve must not be actionable")
	}
	ann := c.Annotate()
	if !strings.Contains(ann, string(RungW0)) {
		t.Errorf("annotation %q must name the W0 rung", ann)
	}
	if !strings.Contains(ann, "not actionable") {
		t.Errorf("a W0 curve must be tagged not-actionable, got %q", ann)
	}
	// It is still RENDERED, not suppressed: the drift signal is present.
	if !strings.Contains(ann, string(CurveDrift)) {
		t.Errorf("a W0 DRIFT curve must still render its signal, got %q", ann)
	}
}

// TestAttachCurvesJoin proves the caller-supplied join binds by whatever key the
// caller owns and degrades cleanly for units with no objective — steerpr never
// touches the join key, keeping the leaf trajctl-free.
func TestAttachCurvesJoin(t *testing.T) {
	units := []Unit{
		{Leaf: "steerpr", Resolves: []string{"#5038"}, Band: BandCleared},
		{Leaf: "gateway", Band: BandResidual}, // no objective
	}
	byIssue := map[string]Curve{
		"#5038": {ObjectiveID: "#5038", Signal: CurveDrift, Rung: RungW3},
	}
	AttachCurves(units, func(u Unit) (Curve, bool) {
		for _, ref := range u.Resolves {
			if c, ok := byIssue[ref]; ok {
				return c, true
			}
		}
		return Curve{}, false
	})
	if units[0].Curve == nil || units[0].Curve.Signal != CurveDrift {
		t.Errorf("unit bound to #5038 must carry the DRIFT curve, got %+v", units[0].Curve)
	}
	if units[1].Curve != nil {
		t.Errorf("unit with no objective must stay curveless, got %+v", units[1].Curve)
	}
	// Re-attaching with an empty lookup clears a previously bound curve — the join
	// is authoritative, not additive.
	AttachCurves(units, func(Unit) (Curve, bool) { return Curve{}, false })
	if units[0].Curve != nil {
		t.Error("a lookup that finds nothing must clear a stale curve")
	}
}

// TestCurveCannotLaunderABand is the anti-gaming proof the pin in
// antigaming_test.go demands for the new Unit.Curve field: a curve is the
// objective-progress axis and must NEVER improve the attention band. Binding even
// a HEALTHY curve onto a RESIDUAL unit leaves the band RESIDUAL — the two axes are
// orthogonal, exactly the "do not fold the curve into the band" doctrine.
func TestCurveCannotLaunderABand(t *testing.T) {
	// Structural: the band's fold reads []Commit, and Commit carries no curve
	// field, so a curve has nowhere to enter the band by construction.
	if hasField(reflect.TypeOf(Commit{}), "Curve") {
		t.Fatal("Commit must not carry a Curve field — the curve must never reach FoldBand")
	}
	// Behavioural: the healthiest possible curve cannot rescue a RESIDUAL unit.
	residual := Unit{Leaf: "x", Band: BandResidual}
	healed := residual.WithCurve(Curve{ObjectiveID: "#5038", Signal: CurveHealthy, Rung: RungW3})
	if healed.Band != BandResidual {
		t.Errorf("band = %q after binding a HEALTHY curve, want RESIDUAL: a curve must not improve a band", healed.Band)
	}
	// AttachCurves over a folded unit set never rewrites a band.
	units, _ := FoldUnits([]Commit{
		{SHA: "u1", Subject: "fix(x): unproven (fak x)", Leaf: "x", Type: "fix", Verdict: VerdictUnwitnessed},
	})
	if len(units) != 1 || units[0].Band != BandResidual {
		t.Fatalf("precondition: want one RESIDUAL unit, got %+v", units)
	}
	AttachCurves(units, func(Unit) (Curve, bool) {
		return Curve{ObjectiveID: "#5038", Signal: CurveHealthy, Rung: RungW3}, true
	})
	if units[0].Band != BandResidual {
		t.Errorf("band = %q after AttachCurves, want RESIDUAL: attaching a curve must not touch the band", units[0].Band)
	}
	if units[0].Curve == nil {
		t.Error("the curve should still be carried for rendering — orthogonal, not dropped")
	}
}

func hasField(t reflect.Type, name string) bool {
	_, ok := t.FieldByName(name)
	return ok
}

// TestAnnotateAttentionMarkerMatchesNeedsAttention proves the rendered marker and
// the NeedsAttention predicate can never disagree across the full vocabulary — the
// ⚠ an operator sees is exactly the set of curves the code flags.
func TestAnnotateAttentionMarkerMatchesNeedsAttention(t *testing.T) {
	for _, sig := range allSignals {
		for _, rung := range allRungs {
			c := &Curve{ObjectiveID: "o", Signal: sig, Rung: rung}
			marked := strings.HasPrefix(c.Annotate(), "⚠")
			if marked != c.NeedsAttention() {
				t.Errorf("signal=%q rung=%q: marker=%v but NeedsAttention=%v", sig, rung, marked, c.NeedsAttention())
			}
		}
	}
}
