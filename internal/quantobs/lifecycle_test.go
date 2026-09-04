package quantobs

import (
	"testing"
)

// Invariant: Quantization observability payloads must record typed lifecycle events without data corruption.
// Guard: Build verifies valid schema versions and maps inputs to complete event streams.

func TestQuantObsLifecycle(t *testing.T) {
	t.Parallel()

	in := Input{
		SchemaVersion:      SchemaVersion,
		ArtifactFormat:     CodeGGUF,
		EffectivePrecision: CodeINT4,
		Recipe:             CodeWeightOnly,
		RuntimeDelegation:  CodeRuntimeLocal,
		Conversion:         CodeConversionNone,
		MemoryResidency:    CodeResidencyAccelerator,
		ResidencyMeasured:  true,
	}

	res := Build(in)
	if res.Outcome != OutcomeObserved {
		t.Fatalf("expected OutcomeObserved, got %v: %v", res.Outcome, res.Reason)
	}
	if len(res.Events) == 0 {
		t.Fatal("expected non-empty events")
	}
}
