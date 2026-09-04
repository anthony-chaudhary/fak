package humanctl

import (
	"testing"
	"time"
)

// BenchmarkHumanCtl benchmarks control envelope composition, index lookup, and validation.
func BenchmarkHumanCtl(b *testing.B) {
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Lookup("redirect")
		_, _ = Lookup("reinforce")

		instructions, err := Compose(
			Instruction{Verb: Reinforce, Reason: "stable progress"},
			Instruction{Verb: Verify, Target: "evidence"},
		)
		if err != nil {
			b.Fatalf("unexpected compose failure: %v", err)
		}

		env := Envelope{
			Instruction: instructions[1],
			Delivery:    DeliveryNextSafePoint,
			Addressee:   Addressee{Kind: AddresseeSubagent, Cardinality: CardinalityOne, IDs: []string{"worker-1"}},
			Lifetime:    Lifetime{Duration: DurationUntilExpiry, ExpiresAt: now.Add(time.Hour)},
			Outcome:     Outcome{Receipt: ReceiptAcknowledged, Admission: AdmissionAccepted, Effect: EffectPending},
		}
		if err := env.ValidateAt(now); err != nil {
			b.Fatalf("unexpected validate failure: %v", err)
		}
	}
}
