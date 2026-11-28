package memoryindex

import (
	"fmt"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestANNRecall(t *testing.T) {
	const dim = 16
	idxF32, err := NewANNIndex(dim, QuantF32)
	if err != nil {
		t.Fatal(err)
	}
	idxI8, err := NewANNIndex(dim, QuantI8)
	if err != nil {
		t.Fatal(err)
	}

	// Insert 16 orthogonal basis clusters + perturbed siblings.
	for c := 0; c < dim; c++ {
		v := make([]float32, dim)
		v[c] = 1.0
		v[(c+1)%dim] = 0.05
		id := fmt.Sprintf("mem-%02d", c)
		if err := idxF32.Add(id, v); err != nil {
			t.Fatal(err)
		}
		if err := idxI8.Add(id, v); err != nil {
			t.Fatal(err)
		}
	}

	if idxI8.StorageBytes() >= idxF32.StorageBytes()/2 {
		t.Fatalf("i8 storage=%d not significantly smaller than f32=%d", idxI8.StorageBytes(), idxF32.StorageBytes())
	}

	// Recall@1 across all 16 clusters must be 1.0 for both F32 and I8.
	for c := 0; c < dim; c++ {
		q := make([]float32, dim)
		q[c] = 1.0
		wantID := fmt.Sprintf("mem-%02d", c)
		for _, idx := range []*ANNIndex{idxF32, idxI8} {
			got := idx.Search(q, 3)
			if len(got) != 3 || got[0].ID != wantID {
				t.Fatalf("quant=%d cluster=%d top=%+v, want %s", idx.Quantization(), c, got, wantID)
			}
		}
	}
}
