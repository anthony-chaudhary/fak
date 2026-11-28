package compute

import (
	"math"
	"testing"
)

func TestInt8GEMMBlocking(t *testing.T) {
	t.Parallel()
	const m, n, k = 5, 7, 9
	a := make([]int8, m*k)
	b := make([]int8, k*n)
	for i := range a {
		a[i] = int8((i % 11) - 5)
	}
	for i := range b {
		b[i] = int8((i % 7) - 3)
	}
	rs := []float32{0.5, 0.25, 1.0, 0.125, 2.0}
	cs := []float32{1.0, 0.5, 0.25, 1.5, 0.75, 2.0, 0.5}

	cfg := Int8GEMMBlockConfig{MC: 2, NC: 3, KC: 4}
	packed, err := PackInt8B(b, k, n, cfg)
	if err != nil {
		t.Fatal(err)
	}
	got, err := BlockedInt8GEMM(a, packed, m, n, k, rs, cs, cfg)
	if err != nil {
		t.Fatal(err)
	}
	// Naive unblocked reference.
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var acc int32
			for p := 0; p < k; p++ {
				acc += int32(a[i*k+p]) * int32(b[p*n+j])
			}
			want := float32(acc) * rs[i] * cs[j]
			if math.Abs(float64(got[i*n+j]-want)) > 1e-6 {
				t.Fatalf("C[%d,%d] = %v, want %v", i, j, got[i*n+j], want)
			}
		}
	}
	if _, err := PackInt8B(b[:len(b)-1], k, n, cfg); err == nil {
		t.Fatal("expected error for mismatched B length")
	}
}
