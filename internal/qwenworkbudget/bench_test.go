package qwenworkbudget

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func BenchmarkQwenWorkBudget(b *testing.B) {
	audit := auditWith(
		transcript("claude", "q1", "Qwen3", 40, 10, 50, 2),
		transcript("codex", "q2", "qwen-2.5", 20, 5, 25, 1),
	)
	policy := Policy{
		QwenAmplificationPolicy: trajectory.QwenAmplificationPolicy{Enforce: true},
		MaxInputPerOutput:       2.0,
	}
	packet := Packet{
		Boundary:        BoundaryLaunch,
		Engine:          "fak-native/qwen3",
		Audit:           &audit,
		UsefulWitnesses: 1,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		receipt := policy.Evaluate(packet)
		if !receipt.Eligible {
			b.Fatalf("expected eligible receipt during benchmark, got: %+v", receipt)
		}
	}
}

func TestBenchmarkQwenWorkBudget(t *testing.T) {
	res := testing.Benchmark(BenchmarkQwenWorkBudget)
	if res.N <= 0 {
		t.Fatalf("benchmark did not run: %+v", res)
	}
}
