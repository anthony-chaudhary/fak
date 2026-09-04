package quantobs

import (
	"testing"
)

// BenchmarkQuantObsObservation measures the performance of constructing and validating categorical telemetry events.
func BenchmarkQuantObsObservation(b *testing.B) {
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

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Build(in)
		if res.Outcome != OutcomeObserved {
			b.Fatalf("unexpected outcome: %v", res.Outcome)
		}
	}
}
