package dispatchtick

import "testing"

func TestWorkShapeForIssue_LabelDerivation(t *testing.T) {
	cases := []struct {
		name   string
		labels []string
		want   WorkShape
	}{
		{"untagged is unknown", nil, ShapeUnknown},
		{"unrelated labels are unknown", []string{"bug", "tier/T0"}, ShapeUnknown},
		{"surgical label", []string{"shape/surgical"}, ShapeSurgical},
		{"bounded synonym", []string{"shape/bounded"}, ShapeSurgical},
		{"churning label", []string{"shape/churning"}, ShapeChurning},
		{"exploratory synonym", []string{"shape/exploratory"}, ShapeChurning},
		{"case-insensitive and trimmed", []string{"  Shape/CHURNING "}, ShapeChurning},
		{"shape label alongside tier labels", []string{"tier/T0", "tier/ultra", "shape/churning"}, ShapeChurning},
		// A contradictory signal is not a reason to guess.
		{"both families is unknown", []string{"shape/surgical", "shape/churning"}, ShapeUnknown},
		{"both families via synonyms is unknown", []string{"shape/bounded", "shape/exploratory"}, ShapeUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := WorkShapeForIssue(c.labels); got != c.want {
				t.Errorf("WorkShapeForIssue(%v) = %q, want %q", c.labels, got, c.want)
			}
		})
	}
}

func TestModelHoldsChurningSlot_WitnessedProfile(t *testing.T) {
	if ModelHoldsChurningSlot(WorkerModelFable) {
		t.Errorf("%s must NOT hold a churning slot (witnessed restart-amnesia starvation)", WorkerModelFable)
	}
	if !ModelHoldsChurningSlot(WorkerModelOpus) {
		t.Errorf("%s must hold a churning slot", WorkerModelOpus)
	}
	// Fail OPEN: an unprofiled model is never blocked by a gate that has no witness for it.
	if !ModelHoldsChurningSlot("some-unprofiled-model") {
		t.Error("an unknown model must fail open (assumed capable)")
	}
	if !ModelHoldsChurningSlot("") {
		t.Error("a blank model must fail open")
	}
}

// TestAssessPlacement_GoldenWitnessedFableOnChurning reproduces the placement failure
// §1.3 witnessed: the cheap model dropped into a hard, churning slot. The gate must
// refuse it BEFORE launch and name the safe re-route target.
func TestAssessPlacement_GoldenWitnessedFableOnChurning(t *testing.T) {
	v := AssessPlacement(ShapeChurning, WorkerModelFable)
	if v.OK {
		t.Fatal("fable on a churning slot must be refused before launch")
	}
	if v.Reason != PlacementShapeMismatch {
		t.Errorf("reason = %q, want %q", v.Reason, PlacementShapeMismatch)
	}
	if v.SafeModel != ChurningSafeModel {
		t.Errorf("SafeModel = %q, want %q", v.SafeModel, ChurningSafeModel)
	}
	if ChurningSafeModel != WorkerModelOpus {
		t.Errorf("ChurningSafeModel = %q, want the opus id %q", ChurningSafeModel, WorkerModelOpus)
	}
	if v.Detail == "" {
		t.Error("a refused placement must carry a legible detail")
	}
}

func TestAssessPlacement_InertCases(t *testing.T) {
	cases := []struct {
		name  string
		shape WorkShape
		model string
	}{
		{"surgical shape never gates", ShapeSurgical, WorkerModelFable},
		{"unknown shape never gates (conservative degrade)", ShapeUnknown, WorkerModelFable},
		{"churning + capable model is fine", ShapeChurning, WorkerModelOpus},
		{"unpinned seat default is not gated", ShapeChurning, ""},
		{"churning + unprofiled model fails open", ShapeChurning, "some-unprofiled-model"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			v := AssessPlacement(c.shape, c.model)
			if !v.OK {
				t.Errorf("AssessPlacement(%q, %q) must be OK, got refusal %q", c.shape, c.model, v.Reason)
			}
			if v.SafeModel != "" || v.Reason != "" {
				t.Errorf("an OK verdict must carry no reason/safe-model, got %+v", v)
			}
		})
	}
}

// TestAssessPlacement_ReconcilesUltraBucket pins the reconciliation the issue asks for:
// DefaultTierLaunchTable routes BucketUltra to fable+ultracode. A churning ultra-hard
// issue must therefore NOT be silently left on fable — the gate catches exactly that
// table's output.
func TestAssessPlacement_ReconcilesUltraBucket(t *testing.T) {
	prof := DefaultTierLaunchTable()[BucketUltra]
	if prof.Model != WorkerModelFable || !prof.Ultracode {
		t.Fatalf("precondition: ultra bucket = %+v, want fable+ultracode", prof)
	}
	// Untagged-for-shape ultra work is unchanged (conservative degrade)...
	if v := AssessPlacement(WorkShapeForIssue([]string{"tier/ultra"}), prof.Model); !v.OK {
		t.Error("an ultra issue with no shape label must not be gated")
	}
	// ...but an ultra issue explicitly labelled churning is re-routed off fable.
	shape := WorkShapeForIssue([]string{"tier/ultra", "shape/churning"})
	v := AssessPlacement(shape, prof.Model)
	if v.OK || v.SafeModel != WorkerModelOpus {
		t.Errorf("churning ultra work must re-route fable->opus, got %+v", v)
	}
}

func TestWorkShapeClosedSet(t *testing.T) {
	for _, s := range []WorkShape{ShapeSurgical, ShapeChurning, ShapeUnknown} {
		if !s.Valid() {
			t.Errorf("%q should be a known WorkShape", s)
		}
	}
	if WorkShape("shape/bogus").Valid() {
		t.Error("an unknown token must not validate as a WorkShape")
	}
}
