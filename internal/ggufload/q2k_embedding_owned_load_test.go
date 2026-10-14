package ggufload

import (
	"bytes"
	"fmt"
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

func q2OwnedRowsEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

// The paired collector arms differ only in adopting or copying the fresh
// TensorBytes payload. This is a host-copy witness, not startup timing.
// fak-test:runtime fast est=100ms lane=default
func TestQ2KEmbeddingOwnedLoad(t *testing.T) {
	t.Run("collector avoids one payload copy", func(t *testing.T) {
		path := writeV41EmbeddingFixture(t, TensorQ2_K)
		ws, err := OpenWeights(path)
		if err != nil {
			t.Fatal(err)
		}
		defer ws.Close()
		cfg, err := ws.File.Config()
		if err != nil {
			t.Fatal(err)
		}
		info, ok := ws.Tensor("token_embd.weight")
		if !ok {
			t.Fatal("fixture has no embedding")
		}
		var sink *model.Model
		run := func(copyPayload bool) {
			tw := ws.computeQ4KTensorWork(info, cfg, false, nil, q4kLoadOptions{residentQ2KEmbedding: true}, 1)
			if tw.err != nil || len(tw.pending) != 1 || !tw.pending[0].q2kEmbed {
				panic("fixture missed the actual packed embedding loader branch")
			}
			builder := model.NewQuantBuilder(cfg, cfg.TieWordEmbeddings)
			if copyPayload {
				pt := tw.pending[0]
				embed, err := model.NewQ2KEmbedding(pt.raw, pt.shape[0], pt.shape[1])
				if err != nil {
					panic(err)
				}
				if err := builder.SetQ2KEmbedding(embed); err != nil {
					panic(err)
				}
			} else if err := applyQ4KTensorWork(tw, nil, cfg, builder, nil, false); err != nil {
				panic(err)
			}
			var err error
			sink, err = builder.Build()
			if err != nil {
				panic(err)
			}
		}
		copied := testing.AllocsPerRun(5, func() { run(true) })
		owned := testing.AllocsPerRun(5, func() { run(false) })
		if copied-owned != 1 || sink == nil || !sink.HasQ2KEmbedding() {
			t.Fatalf("removed payload allocations=%g, want1 (copied=%g owned=%g)", copied-owned, copied, owned)
		}
		t.Logf("one packed payload copy removed: %d bytes; copied=%g owned=%g allocations", sink.Q2KEmbedding.Bytes(), copied, owned)
	})

	t.Run("public load owns bytes after source close", func(t *testing.T) {
		path := writeV41EmbeddingFixture(t, TensorQ2_K)
		m, err := LoadModelQ4KProfileOptions(path, nil, WithQ2KEmbeddingResident(true))
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = m.CloseWeights() })
		_, want := q2KFixtureBlock()
		got := make([]float32, 256)
		if err := m.Q2KEmbedding.GatherRow(2, got, 1); err != nil || !q2OwnedRowsEqual(got, want) {
			t.Fatalf("loaded embedding row differs: %v", err)
		}
		// The public load has closed its WeightSource. Its table must remain
		// independent of later changes to the checkpoint file.
		if err := os.WriteFile(path, []byte("replaced checkpoint"), 0600); err != nil {
			t.Fatal(err)
		}
		if err := m.Q2KEmbedding.GatherRow(2, got, 1); err != nil || !q2OwnedRowsEqual(got, want) {
			t.Fatalf("embedding retained source-file storage: %v", err)
		}
	})

	t.Run("copy constructor and failed transfer keep ownership", func(t *testing.T) {
		raw, want := q2KFixtureBlock()
		copied, err := model.NewQ2KEmbedding(raw, 1, 256)
		if err != nil {
			t.Fatal(err)
		}
		clear(raw)
		got := make([]float32, 256)
		if err := copied.GatherRow(0, got, 1); err != nil || !q2OwnedRowsEqual(got, want) {
			t.Fatalf("public copy constructor retained mutable caller storage: %v", err)
		}
		for _, dims := range [][2]int{{0, 256}, {1, 255}, {2, 256}} {
			data, _ := q2KFixtureBlock()
			before := append([]byte(nil), data...)
			copyOut, copyErr := model.NewQ2KEmbedding(data, dims[0], dims[1])
			ownedOut, ownedErr := model.NewQ2KEmbeddingOwned(data, dims[0], dims[1])
			if copyOut != nil || ownedOut != nil || copyErr == nil || fmt.Sprint(copyErr) != fmt.Sprint(ownedErr) || !bytes.Equal(data, before) {
				t.Fatalf("failed ownership transfer changed validation or source: %v / %v", copyErr, ownedErr)
			}
		}
	})
}
