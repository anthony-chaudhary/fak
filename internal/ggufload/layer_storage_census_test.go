package ggufload

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=1s
func TestCensusQ4KLayerStorage(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ4_K, TensorQ4_K, false, true)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	c, err := ws.CensusQ4KLayerStorage()
	if err != nil {
		t.Fatal(err)
	}
	const layerBytes = 3 * dim * dim / 256 * 144
	const embeddingBytes = vocab * dim * 4
	const headBytes = vocab * dim / 32 * 36
	if len(c.Layers) != 2 || c.Layers[0] != layerBytes || c.Layers[1] != 0 || c.Embedding != embeddingBytes || c.Head != headBytes {
		t.Fatalf("census = %+v", c)
	}
	m, err := LoadModelQ4KProfileOptions(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = m.CloseWeights() })
	if got := m.ResidentReport().TotalResidentBytes; got != c.Layers[0]+c.Embedding+c.Head {
		t.Fatalf("loader storage = %d", got)
	}
	_, err = ws.CensusQ4KLayerStorage(WithStreamedDenseQ4KWorkingSet(1))
	if !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
		t.Fatalf("streamed census error = %v", err)
	}
	for i := range ws.File.Tensors {
		if ws.File.Tensors[i].Name == "blk.0.attn_v.weight" {
			ws.File.Tensors[i].Name = "blk.2.attn_v.weight"
			break
		}
	}
	if _, err := ws.CensusQ4KLayerStorage(); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
		t.Fatalf("out-of-range ownership error = %v", err)
	}
}

// fak-test:runtime fast est=1s
func TestCensusQ4KLayerStorageMixedRetention(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	const dim, vocab = 256, 4
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ6_K, false, false)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	for _, tc := range []struct {
		name            string
		opts            []Q4KLoadOption
		embedding, head int64
	}{
		{"packed", nil, vocab * dim * 4, vocab * 210},
		{"q8", []Q4KLoadOption{WithDenseKQuantResident(false)}, vocab * dim * 4, vocab * dim / 32 * 36},
		{"packed_embedding", []Q4KLoadOption{WithQ2KEmbeddingResident(true)}, vocab * 84, vocab * 210},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c, err := ws.CensusQ4KLayerStorage(tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			const layer = 3 * dim * dim / 256 * 144
			if len(c.Layers) != 2 || c.Layers[0] != layer || c.Layers[1] != 0 || c.Embedding != tc.embedding || c.Head != tc.head {
				t.Fatalf("census = %+v", c)
			}
			m, err := LoadModelQ4KProfileOptions(path, nil, tc.opts...)
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { _ = m.CloseWeights() })
			if got := m.ResidentReport().TotalResidentBytes; got != layer+tc.embedding+tc.head {
				t.Fatalf("loaded bytes = %d", got)
			}
		})
	}
}

// fak-test:runtime fast est=1s
func TestLayerStorageOverflow(t *testing.T) {
	n := int64(math.MaxInt64)
	if err := addLayerStorageBytes(&n, 1); err == nil || n != math.MaxInt64 {
		t.Fatal("sum overflow accepted")
	}
	n = 0
	if err := addLayerStorageBytes(&n, uint64(math.MaxInt64)+1); err == nil || n != 0 {
		t.Fatal("conversion overflow accepted")
	}
}

// fak-test:runtime fast est=1s
func TestCensusQ4KLayerStorageRejectsTiedPackedTable(t *testing.T) {
	t.Setenv("FAK_W3_MLP", "")
	path := buildQwen35GGUFFixture(t, "qwen35", 256, 4, TensorQ6_K, TensorQ4_K, false, true)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	if _, err := ws.CensusQ4KLayerStorage(); !errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
		t.Fatalf("tied packed table error = %v", err)
	}
}
