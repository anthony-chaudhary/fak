package ctxplan

import "testing"

func TestPlanPinRoundTrip(t *testing.T) {
	steps := []StepPin{{StepID: "discover", Text: "inspect current state"}, {StepID: "ship", Text: "land the witnessed fix"}}
	got := NewPlanPin(steps)
	want := NewPlanPin(steps)
	if got.Digest != want.Digest {
		t.Fatalf("digest is not deterministic: %q != %q", got.Digest, want.Digest)
	}
	if !got.Verify() {
		t.Fatal("fresh plan pin did not verify")
	}

	steps[0].Text = "mutated by caller"
	if got.Steps[0].Text != "inspect current state" {
		t.Fatal("NewPlanPin retained the caller's mutable slice")
	}
}

func TestPlanPinTamper(t *testing.T) {
	pin := NewPlanPin([]StepPin{{StepID: "one", Text: "original"}})
	pin.Steps[0].Text = "tampered"
	if pin.Verify() {
		t.Fatal("tampered plan verified")
	}
}

func TestPlanPinStepReorder(t *testing.T) {
	one := StepPin{StepID: "one", Text: "first"}
	two := StepPin{StepID: "two", Text: "second"}
	forward := NewPlanPin([]StepPin{one, two})
	reverse := NewPlanPin([]StepPin{two, one})
	if forward.Digest == reverse.Digest {
		t.Fatalf("step reorder preserved digest %q", forward.Digest)
	}
}

func TestPlanPinFieldBoundariesCannotCollide(t *testing.T) {
	left := NewPlanPin([]StepPin{{StepID: "a", Text: "bc"}})
	right := NewPlanPin([]StepPin{{StepID: "ab", Text: "c"}})
	if left.Digest == right.Digest {
		t.Fatalf("field-boundary collision: %q", left.Digest)
	}
}

func TestPlanPinIsZero(t *testing.T) {
	if !(PlanPin{}).IsZero() {
		t.Fatal("zero PlanPin was not zero")
	}
	if NewPlanPin(nil).IsZero() {
		t.Fatal("content-addressed empty plan was reported as zero")
	}
}
