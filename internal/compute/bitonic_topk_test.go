package compute

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func referenceSortTopK(values []float32, k int) []BitonicTopKEntry {
	entries := make([]BitonicTopKEntry, len(values))
	for i, v := range values {
		entries[i] = BitonicTopKEntry{
			Value: v,
			Index: int32(i),
		}
	}

	sort.Slice(entries, func(i, j int) bool {
		return bitonicCompare(entries[i], entries[j])
	})

	return entries[:k]
}

func TestPersistentBitonicTopKWitness(t *testing.T) {
	// First witness requirements (#9942):
	// 1. One-block persistent selection up to N=2048 where k approaches N
	// 2. Stable tie/NaN/index ordering against CPU reference sort for N/K sweep
	// 3. Exact deterministic index and value parity

	rng := rand.New(rand.NewSource(777))

	testCases := []struct {
		n int
		k int
	}{
		{n: 16, k: 14},   // k approaches N
		{n: 16, k: 16},   // k == N
		{n: 64, k: 60},   // k approaches N
		{n: 128, k: 120}, // k approaches N
		{n: 256, k: 250}, // k approaches N
		{n: 512, k: 500}, // k approaches N
		{n: 1024, k: 1000},
		{n: 1024, k: 1024},
		{n: 2048, k: 2000}, // N=2048 limit
		{n: 2048, k: 2047}, // k = N - 1
		{n: 2048, k: 2048}, // k = N
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("N=%d_K=%d", tc.n, tc.k), func(t *testing.T) {
			values := make([]float32, tc.n)
			for i := range values {
				values[i] = rng.Float32()*100 - 50
			}

			// Deliberately introduce ties:
			values[1] = 42.0
			values[tc.n/2] = 42.0
			values[tc.n-2] = 42.0

			// Deliberately introduce NaNs:
			values[0] = float32(math.NaN())
			values[tc.n/3] = float32(math.NaN())

			want := referenceSortTopK(values, tc.k)

			got, receipt, err := PersistentBitonicTopK(values, tc.k)
			if err != nil {
				t.Fatalf("PersistentBitonicTopK failed: %v", err)
			}

			if receipt.N != tc.n || receipt.K != tc.k {
				t.Fatalf("receipt mismatch: N=%d, K=%d", receipt.N, receipt.K)
			}
			if len(got) != tc.k {
				t.Fatalf("got length %d != k %d", len(got), tc.k)
			}

			// Verify exact element-by-element match (including index tie-break)
			for i := 0; i < tc.k; i++ {
				wantNaN := math.IsNaN(float64(want[i].Value))
				gotNaN := math.IsNaN(float64(got[i].Value))

				if wantNaN != gotNaN {
					t.Fatalf("NaN mismatch at rank %d: got=%v, want=%v", i, got[i].Value, want[i].Value)
				}
				if !wantNaN && math.Abs(float64(got[i].Value-want[i].Value)) > 1e-6 {
					t.Fatalf("value mismatch at rank %d: got=%v, want=%v", i, got[i].Value, want[i].Value)
				}
				if got[i].Index != want[i].Index {
					t.Fatalf("index tie-break mismatch at rank %d: got index %d, want index %d (value=%v)",
						i, got[i].Index, want[i].Index, got[i].Value)
				}
			}
		})
	}
}

func TestPersistentBitonicTopKFailClosed(t *testing.T) {
	// Empty slice
	if _, _, err := PersistentBitonicTopK(nil, 1); err == nil {
		t.Fatal("expected error on empty values")
	}

	// k <= 0
	if _, _, err := PersistentBitonicTopK([]float32{1, 2, 3}, 0); err == nil {
		t.Fatal("expected error on k = 0")
	}

	// k > n
	if _, _, err := PersistentBitonicTopK([]float32{1, 2, 3}, 5); err == nil {
		t.Fatal("expected error on k > n")
	}
}
