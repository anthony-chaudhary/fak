package incidentrsi

import (
	"errors"
	"fmt"
	"testing"
	"time"
)

func BenchmarkIncidentRSI(b *testing.B) {
	b.Run("Fingerprint", func(b *testing.B) {
		input := Input{
			Source:        SourceUnexpectedHook,
			Operation:     "git_commit",
			ErrorClass:    "ValidationError",
			CauseIdentity: "schema_mismatch",
			OccurredAt:    time.Now().UTC(),
			Developer:     true,
			Expected:      false,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = Fingerprint(input)
		}
	})

	b.Run("DebounceObserve", func(b *testing.B) {
		d := NewDebouncer(DefaultDebounceConfig(), &MemoryBurstStore{}, nil)
		errTest := errors.New("simulated product error")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			obs := DebounceObservation{
				Fingerprint:   "irsi-benchmark-fingerprint-stable",
				ProducerMajor: 1,
				ObservationID: fmt.Sprintf("obs-%d", i%256),
			}
			_ = d.Observe(obs, errTest)
		}
	})
}
