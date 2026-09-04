package epochbridge

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// BenchmarkEpochBridge benchmarks lineage mapping across both generation-0 roots
// and multi-generation continuation chains.
func BenchmarkEpochBridge(b *testing.B) {
	root := session.State{TraceID: "root-trace"}
	c1 := session.ContinuationID("root-trace", 1)
	gen1 := session.State{
		TraceID:     c1,
		ParentTrace: "root-trace",
		Generation:  1,
	}

	b.Run("RootGen0", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sc := SpecContextFor(root)
			if sc.Epoch != 0 || sc.ParentEpoch != 0 {
				b.Fatal("invalid gen0 epoch")
			}
		}
	})

	b.Run("ContinuationGen1", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			sc := SpecContextFor(gen1)
			if sc.Epoch == 0 {
				b.Fatal("unexpected zero epoch for continuation")
			}
		}
	})

	b.Run("Outcome", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if GenerationOutcome() != abi.OutcomeCommitted {
				b.Fatal("unexpected outcome")
			}
		}
	})
}
