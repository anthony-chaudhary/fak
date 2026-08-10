package issuepolicy

import "testing"

func TestScaleFitDerivesFromStepsWhenUndeclared(t *testing.T) {
	c := completeCandidate() // WorkUnit "leaf", ExpectedSteps 3, no Scale declared.
	fit := scaleFit(c)
	if fit.Effective != ScaleLeaf {
		t.Fatalf("effective = %q, want S1 (leaf, 3 steps)", fit.Effective)
	}
	if fit.Source != "steps" {
		t.Fatalf("source = %q, want steps", fit.Source)
	}
	if !fit.Dispatchable {
		t.Fatalf("S1 leaf must be dispatchable: %+v", fit)
	}
	if len(fit.Flags) != 0 {
		t.Fatalf("well-formed leaf should carry no scale flags: %v", fit.Flags)
	}
}

func TestScaleFitDeriveBands(t *testing.T) {
	cases := []struct {
		steps int
		want  Scale
	}{
		{1, ScaleStep},
		{8, ScaleLeaf},
		{9, ScaleFeature},
		{100, ScaleFeature},
		{101, ScaleEpic},
		{1000, ScaleEpic},
		{5000, ScaleProgram},
	}
	for _, tc := range cases {
		if got, ok := scaleFromSteps(tc.steps); !ok || got != tc.want {
			t.Errorf("scaleFromSteps(%d) = %q ok=%v, want %q", tc.steps, got, ok, tc.want)
		}
	}
	if _, ok := scaleFromSteps(0); ok {
		t.Errorf("scaleFromSteps(0) should carry no information")
	}
}

func TestScaleFitFlagsWitnessUnderScale(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = "feature"
	c.ExpectedSteps = 40 // S2 feature by both shape and steps.
	// Witness is only a commit/test — a scale-1 witness for scale-2 work.
	c.Witness = "go test ./internal/foo"
	c.AcceptanceGate = "the resolving commit lands with a green build"
	c.DoneCondition = "The refactor is finished."
	fit := scaleFit(c)
	if fit.Effective != ScaleFeature {
		t.Fatalf("effective = %q, want S2", fit.Effective)
	}
	if !has(fit.Flags, flagWitnessUnderScale) {
		t.Fatalf("S2 work with a commit/test witness must flag witness_under_scale: %+v", fit)
	}
	if fit.NeedsWitness != "integration/live witness" {
		t.Fatalf("needs_witness = %q, want integration/live witness", fit.NeedsWitness)
	}
}

func TestScaleFitIntegrationWitnessSatisfiesFeature(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = "feature"
	c.ExpectedSteps = 40
	c.Witness = "end-to-end probe against the live served turn passes"
	fit := scaleFit(c)
	if has(fit.Flags, flagWitnessUnderScale) {
		t.Fatalf("an integration witness must satisfy S2: %+v", fit)
	}
	if fit.WitnessScale != ScaleFeature {
		t.Fatalf("witness_scale = %q, want S2", fit.WitnessScale)
	}
}

func TestScaleFitFlagsDeclaredContradictsSteps(t *testing.T) {
	c := completeCandidate()
	c.Scale = "S1"
	c.ExpectedSteps = 50 // A 50-step change is S2, not the declared S1.
	fit := scaleFit(c)
	if fit.Effective != ScaleLeaf {
		t.Fatalf("effective honors the declaration: %q, want S1", fit.Effective)
	}
	if !has(fit.Flags, flagScaleContradictsSteps) {
		t.Fatalf("declared S1 vs 50-step budget must contradict: %+v", fit)
	}
}

func TestScaleFitFlagsDeclaredInvalid(t *testing.T) {
	c := completeCandidate()
	c.Scale = "medium"
	fit := scaleFit(c)
	if !has(fit.Flags, flagScaleDeclaredInvalid) {
		t.Fatalf("a scale that names no tier must flag declared_invalid: %+v", fit)
	}
}

func TestScaleFitFlagsUndeclaredWhenNothingToDerive(t *testing.T) {
	c := completeCandidate()
	c.Scale = ""
	c.WorkUnit = ""
	c.ExpectedSteps = 0
	fit := scaleFit(c)
	if fit.Effective != ScaleUnknown {
		t.Fatalf("effective = %q, want unknown", fit.Effective)
	}
	if !has(fit.Flags, flagScaleUndeclared) {
		t.Fatalf("no scale, no steps, no shape must flag undeclared: %+v", fit)
	}
}

func TestParseScaleAcceptsTierNames(t *testing.T) {
	cases := map[string]Scale{
		"S0":           ScaleStep,
		"step":         ScaleStep,
		"leaf":         ScaleLeaf,
		"S2 (feature)": ScaleFeature,
		"feature":      ScaleFeature,
		"epic":         ScaleEpic,
		"program":      ScaleProgram,
		"`S3`":         ScaleEpic,
	}
	for in, want := range cases {
		if got, ok := parseScale(in); !ok || got != want {
			t.Errorf("parseScale(%q) = %q ok=%v, want %q", in, got, ok, want)
		}
	}
	if _, ok := parseScale("medium"); ok {
		t.Errorf("parseScale(medium) should not resolve to a tier")
	}
}

func TestReviewCandidateStrictScaleHoldsUndeclared(t *testing.T) {
	c := completeCandidate()
	c.Scale = ""
	c.WorkUnit = ""
	c.ExpectedSteps = 0
	// Advisory by default: the readout flags it, but it stays dispatchable.
	advisory := ReviewCandidate(c, Options{})
	if has(advisory.Reasons, ReasonScaleUndeclared) {
		t.Fatalf("undeclared scale must be advisory by default: %+v", advisory.Reasons)
	}
	if !has(advisory.Scale.Flags, flagScaleUndeclared) {
		t.Fatalf("readout should still flag undeclared: %+v", advisory.Scale)
	}
	// Strict: the same flag now holds dispatch triage-only.
	strict := ReviewCandidate(c, Options{StrictScale: true})
	if !has(strict.Reasons, ReasonScaleUndeclared) {
		t.Fatalf("StrictScale must hold an undeclared scale: %+v", strict.Reasons)
	}
	if strict.Dispatchability != TriageOnly {
		t.Fatalf("dispatchability = %q, want triage_only", strict.Dispatchability)
	}
}

func TestReviewCandidateFeatureScaleIsNotADispatchLeaf(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = "feature" // S2 by shape...
	c.ExpectedSteps = 4    // ...even though the step budget is leaf-sized.
	// No StrictScale: the S2+ leaf gate is always-on, like the oversized-steps gate.
	review := ReviewCandidate(c, Options{})
	if review.Dispatchability == Dispatchable {
		t.Fatalf("an S2 feature must not be dispatchable: %+v", review)
	}
	if !has(review.Reasons, ReasonNotDispatchLeaf) {
		t.Fatalf("effective scale S2+ must trip ISSUE_NOT_DISPATCH_LEAF: %+v", review.Reasons)
	}
}

func TestReviewCandidateStrictScaleHoldsWitnessMismatch(t *testing.T) {
	c := completeCandidate()
	c.WorkUnit = "feature"
	c.ExpectedSteps = 40
	c.Witness = "go test ./internal/foo"
	c.AcceptanceGate = "the resolving commit lands"
	c.DoneCondition = "The capability is complete."
	strict := ReviewCandidate(c, Options{StrictScale: true})
	if !has(strict.Reasons, ReasonWitnessScaleMismatch) {
		t.Fatalf("StrictScale must hold a feature witnessed only by a commit/test: %+v", strict.Reasons)
	}
	if strict.Scale.Effective != ScaleFeature {
		t.Fatalf("effective = %q, want S2", strict.Scale.Effective)
	}
}
