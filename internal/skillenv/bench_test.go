package skillenv

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kvmmu"
	"github.com/anthony-chaudhary/fak/internal/model"
)

func TestBenchmarkSuite_Sanity(t *testing.T) {
	table := New(nil, nil, nil)
	if _, _, err := table.Pin("test-skill", "1.0.0"); err != nil {
		t.Fatalf("sanity pin failed: %v", err)
	}
	if v, ok := table.ActiveVersion("test-skill"); !ok || v != "1.0.0" {
		t.Fatalf("sanity active version = %q, ok=%v", v, ok)
	}
}

func BenchmarkActiveVersion_Pinned(b *testing.B) {
	table := New(nil, nil, nil)
	table.Pin("code-review", "2.0.0")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v, ok := table.ActiveVersion("code-review")
		if !ok || v == "" {
			b.Fatal("unexpected missing version")
		}
	}
}

func BenchmarkActiveVersion_Unpinned(b *testing.B) {
	table := New(nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = table.ActiveVersion("unpinned-skill")
	}
}

func BenchmarkActiveVersion_Parallel(b *testing.B) {
	table := New(nil, nil, nil)
	table.Pin("code-review", "2.0.0")

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			v, ok := table.ActiveVersion("code-review")
			if !ok || v == "" {
				b.Fatal("unexpected missing version")
			}
		}
	})
}

func BenchmarkPin(b *testing.B) {
	table := New(nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = table.Pin("test-skill", "1.0.0")
	}
}

func BenchmarkSwap(b *testing.B) {
	table := New(nil, nil, nil)
	table.Pin("test-skill", "1.0.0")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if i%2 == 0 {
			_, _, _ = table.Swap("test-skill", "1.0.0", "2.0.0")
		} else {
			_, _, _ = table.Swap("test-skill", "2.0.0", "1.0.0")
		}
	}
}

func BenchmarkUnpin(b *testing.B) {
	table := New(nil, nil, nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		table.Pin("test-skill", "1.0.0")
		_, _, _ = table.Unpin("test-skill")
	}
}

func BenchmarkList(b *testing.B) {
	table := New(nil, nil, nil)
	for i := 0; i < 20; i++ {
		table.Pin(fmt.Sprintf("skill-%02d", i), "1.0.0")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m := table.List()
		if len(m) != 20 {
			b.Fatal("unexpected list length")
		}
	}
}

func BenchmarkBlastRadius_Residency(b *testing.B) {
	mmu := ctxmmu.New()
	kvctx := kvmmu.NewWithGate(model.NewSynthetic(model.Config{
		HiddenSize: 32, NumLayers: 2, NumHeads: 4, NumKVHeads: 2, HeadDim: 8,
		IntermediateSize: 64, VocabSize: 48, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1, ModelType: "llama",
	}).NewSession(), mmu)
	table := New(nil, mmu, kvctx)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _, _ = table.Pin("monitored-skill", "1.0.0")
	}
}
