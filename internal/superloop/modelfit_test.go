package superloop

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// TestModelFitFixturesWellFormed checks the fixture set's structural invariants the
// grader relies on: a non-empty set, a unique name and a valid closed-vocabulary action
// on every fixture, and — since a read-only meta pass ships nothing — NothingShipped on
// every scenario, so an invented-shipped claim is always a graded trap.
func TestModelFitFixturesWellFormed(t *testing.T) {
	fx := Fixtures()
	if len(fx) == 0 {
		t.Fatal("Fixtures() is empty; the eval has nothing to grade")
	}
	seen := map[string]bool{}
	for _, f := range fx {
		if strings.TrimSpace(f.Name) == "" {
			t.Errorf("fixture with empty name: %+v", f)
		}
		if seen[f.Name] {
			t.Errorf("duplicate fixture name %q", f.Name)
		}
		seen[f.Name] = true
		if !f.WantAction.Valid() {
			t.Errorf("fixture %q has invalid want-action %q", f.Name, f.WantAction)
		}
		if !f.NothingShipped {
			t.Errorf("fixture %q must be NothingShipped: a read-only meta read ships nothing", f.Name)
		}
		// Every MustPreserve reason must actually appear in the situation, or the trap is
		// unwinnable — a faithful model could not preserve a reason that was never shown.
		for _, r := range f.MustPreserve {
			if !strings.Contains(f.Situation, r) {
				t.Errorf("fixture %q asks to preserve %q but its situation never mentions it", f.Name, r)
			}
		}
	}
}

// TestModelFitCompliantClearedRoutineNeverMutation is the core positive: a model that
// answers every read-only fixture correctly is cleared for the ROUTINE work class at
// that class's policy floor — and NOTHING more. The cleared tier is derived from the C3
// policy, not frozen here, so this asserts the single-sourced invariant rather than a
// hard-coded label.
func TestModelFitCompliantClearedRoutineNeverMutation(t *testing.T) {
	fx := Fixtures()
	fit := Evaluate(fx, compliantProfile(fx))

	if !fit.Suitable {
		t.Fatalf("compliant model not suitable: %s (grades: %+v)", fit.Reason, fit.Grades)
	}
	if fit.Passed != fit.Total || fit.Total != len(fx) {
		t.Errorf("compliant model passed %d/%d, want all %d", fit.Passed, fit.Total, len(fx))
	}
	if fit.ClearedFor != modelroute.ClassRoutine {
		t.Errorf("cleared-for = %q, want routine class", fit.ClearedFor)
	}
	wantTier := modelroute.PolicyFor(modelroute.ClassRoutine).RequiredTier.String()
	if fit.ClearedTier != wantTier {
		t.Errorf("cleared-tier = %q, want the routine policy floor %q", fit.ClearedTier, wantTier)
	}
	if fit.DeniedAuthority != modelroute.ClassSecurityRelease {
		t.Errorf("denied-authority = %q, want the security/release class", fit.DeniedAuthority)
	}
}

// TestModelFitReadOnlyCeilingHolds encodes the confusion risk the issue names: even a
// perfect score never grants mutation authority. The denied class is the security/
// release class, and its floor is strictly MORE demanding than the routine floor a pass
// clears — so a meta-fit pass structurally cannot buy execution rights.
func TestModelFitReadOnlyCeilingHolds(t *testing.T) {
	fx := Fixtures()
	fit := Evaluate(fx, compliantProfile(fx))
	if !fit.Suitable {
		t.Fatalf("expected the compliant model to pass; got %s", fit.Reason)
	}

	deniedFloor := modelroute.PolicyFor(modelroute.ClassSecurityRelease).RequiredTier
	routineFloor := modelroute.PolicyFor(modelroute.ClassRoutine).RequiredTier
	// The denied class must require a strictly MORE demanding tier than the routine floor
	// a pass clears (T0<T1<T2 numerically, so "more demanding" is the smaller number).
	if !deniedFloor.MoreDemandingThan(routineFloor) {
		t.Errorf("security-release floor %s is not more demanding than the routine floor %s; the read-only ceiling would not hold",
			deniedFloor, routineFloor)
	}
	if fit.DeniedFloor != deniedFloor.String() {
		t.Errorf("reported denied floor %q != policy floor %q", fit.DeniedFloor, deniedFloor.String())
	}
	// The reason string must make the ceiling legible to a human/operator.
	if !strings.Contains(fit.Reason, string(modelroute.ClassSecurityRelease)) {
		t.Errorf("suitable reason must name the denied class; got %q", fit.Reason)
	}
}

// TestModelFitRefusesDroppedRefusalAndInventedShip is the core negative: a model that
// papers over a guard refusal or invents shipped work is refused, with the specific
// closed-vocabulary reasons naming both failures.
func TestModelFitRefusesDroppedRefusalAndInventedShip(t *testing.T) {
	fx := Fixtures()
	profs := SimulatedProfiles(fx)
	var sloppy ModelMetaProfile
	for _, p := range profs {
		if p.Model == "tiny-untuned-7b" {
			sloppy = p
		}
	}
	if sloppy.Model == "" {
		t.Fatal("expected a sloppy simulated profile named tiny-untuned-7b")
	}

	fit := Evaluate(fx, sloppy)
	if fit.Suitable {
		t.Fatal("sloppy model must not be suitable")
	}
	if fit.ClearedFor != "" || fit.ClearedTier != "" {
		t.Errorf("an unsuitable model must be cleared for nothing; got %q/%q", fit.ClearedFor, fit.ClearedTier)
	}

	reasons := allReasons(fit)
	if !hasReasonPrefix(reasons, FitRefusalDropped) {
		t.Errorf("expected a %s reason; got %v", FitRefusalDropped, reasons)
	}
	if !hasReasonPrefix(reasons, FitInventedShip) {
		t.Errorf("expected an %s reason; got %v", FitInventedShip, reasons)
	}
}

// TestModelFitNoDecisionFailsClosed proves silence is never a pass: a profile that omits
// a fixture's decision reds that fixture with FitNoDecision, not a silent skip.
func TestModelFitNoDecisionFailsClosed(t *testing.T) {
	fx := Fixtures()
	// A profile that answers every fixture EXCEPT the first.
	partial := compliantProfile(fx)
	partial.Decisions = partial.Decisions[1:]

	fit := Evaluate(fx, partial)
	if fit.Suitable {
		t.Fatal("a model missing a decision must not be suitable")
	}
	first := fit.Grades[0]
	if first.Fixture != fx[0].Name || first.Pass {
		t.Fatalf("expected the omitted fixture %q to fail; got %+v", fx[0].Name, first)
	}
	if !hasReasonPrefix(first.Reasons, FitNoDecision) {
		t.Errorf("omitted decision must red with %s; got %v", FitNoDecision, first.Reasons)
	}
}

// TestModelFitDeterministic proves the fold is pure: identical inputs yield byte-identical
// reports, so the artifact is a stable witness.
func TestModelFitDeterministic(t *testing.T) {
	a := SimulatedReport()
	b := SimulatedReport()
	if !reflect.DeepEqual(a, b) {
		t.Error("SimulatedReport is not deterministic across runs")
	}
	if a.Schema != EvalSchema {
		t.Errorf("report schema = %q, want %q", a.Schema, EvalSchema)
	}
	// Models are sorted by id: each id is <= the next.
	for i := 1; i < len(a.Models); i++ {
		if a.Models[i-1].Model > a.Models[i].Model {
			t.Errorf("models not sorted by id at %d: %q > %q", i, a.Models[i-1].Model, a.Models[i].Model)
		}
	}
}

// TestModelFitRenderCapturesEvalOutput exercises the operator readout: the render must
// carry the schema, every model row, the SIM marker (the rows are simulated stand-ins),
// and the read-only ceiling line — the captured eval output the witness asks for.
func TestModelFitRenderCapturesEvalOutput(t *testing.T) {
	rep := SimulatedReport()
	var buf bytes.Buffer
	Render(&buf, rep)
	out := buf.String()
	t.Logf("captured eval output:\n%s", out)

	for _, want := range []string{
		rep.Schema,
		"read-only ceiling",
		"SIM",
		string(modelroute.ClassSecurityRelease),
	} {
		if !strings.Contains(out, want) {
			t.Errorf("render missing %q", want)
		}
	}
	for _, m := range rep.Models {
		if !strings.Contains(out, m.Model) {
			t.Errorf("render missing model row %q", m.Model)
		}
	}
	// At least one cheap model must come out cleared for routine meta work, and at least
	// one must be refused — the eval discriminates rather than passing everything.
	var anyCleared, anyRefused bool
	for _, m := range rep.Models {
		if m.Suitable {
			anyCleared = true
		} else {
			anyRefused = true
		}
	}
	if !anyCleared || !anyRefused {
		t.Errorf("expected the sample to both clear and refuse at least one model; cleared=%v refused=%v", anyCleared, anyRefused)
	}
}

// --- helpers ---------------------------------------------------------------

// compliantProfile is a live-marked profile whose decisions are the correct answer to
// every fixture — used to assert the positive path without depending on the simulated
// sample's model ids.
func compliantProfile(fx []MetaFixture) ModelMetaProfile {
	return ModelMetaProfile{Model: "test-compliant", Decisions: compliantDecisions(fx)}
}

func allReasons(fit ModelFit) []string {
	var out []string
	for _, g := range fit.Grades {
		out = append(out, g.Reasons...)
	}
	return out
}

func hasReasonPrefix(reasons []string, prefix string) bool {
	for _, r := range reasons {
		if r == prefix || strings.HasPrefix(r, prefix+":") {
			return true
		}
	}
	return false
}
