package computetune

import (
	"context"
	"testing"
	"time"
)

func BenchmarkComputeTune(b *testing.B) {
	profile := DefaultStrixHaloProfile()
	arbitrator := NewStorageComputeArbitrator(profile)

	b.Run("ArbitrationDecide", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			dec := arbitrator.Decide(4000, 2.5)
			if dec.Verdict == "" {
				b.Fatal("empty verdict")
			}
		}
	})

	b.Run("TuneProfiles", func(b *testing.B) {
		p := Profile{
			Operation: OpMatMul,
			M:         2,
			N:         2,
			K:         2,
			DType:     "f32",
			Compat: Compatibility{
				Backend:          "cpu",
				Device:           "bench-device",
				SoftwareRevision: "v1",
				KernelRevision:   "v1",
			},
			Frequency: 10,
			Weight:    1.0,
		}
		ref := fakeCandidate{
			id: "ref",
			result: func(Profile) ([]float32, error) {
				return []float32{1, 2, 3, 4}, nil
			},
		}
		candidate := fakeCandidate{
			id: "fast",
			result: func(Profile) ([]float32, error) {
				return []float32{1, 2, 3, 4}, nil
			},
		}
		candidates := []Candidate{ref, candidate}
		measure := func(_ context.Context, c Candidate, _ Profile) (time.Duration, error) {
			return 5 * time.Microsecond, nil
		}
		policy := Policy{
			Warmup:            0,
			Repeats:           1,
			Statistic:         "median",
			TimerDomain:       "monotonic",
			FallbackCandidate: "ref",
		}
		ctx := context.Background()

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			manifest, _, err := Tune(ctx, []Profile{p}, candidates, ref, exact, measure, policy)
			if err != nil {
				b.Fatalf("Tune failed: %v", err)
			}
			if _, ok := manifest.Lookup(p); !ok {
				b.Fatal("profile lookup failed")
			}
		}
	})

	for i := 0; i < b.N; i++ {
		_ = arbitrator.Decide(2048, 1.0)
	}
}
