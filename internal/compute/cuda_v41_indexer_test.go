//go:build cuda && cgo && fak_cuda_v41_indexer

package compute

import (
	"math"
	"testing"
)

// This is an opt-in physical witness, not production qualification. Compile the
// library with FAK_CUDA_V41_INDEXER and --ftz=false, and require the real backend.
// fak-test:runtime integration est=2s lane=optin
func TestCUDAV41IndexerSIMTExact(t *testing.T) {
	be := cudaGDNBackend(t)
	if available, err := be.V41IndexerScoreAdmission(); err != nil || available {
		t.Fatalf("candidate admission = %v, %v; want healthy but unqualified", available, err)
	}
	for _, rows := range []int{0, 1, 17, 8192} {
		const heads, dim = 3, 7
		qv, kv, wv := make([]float32, heads*dim), make([]float32, rows*dim), []float32{0.75, -0.5, 0.25}
		for i := range qv {
			qv[i] = float32(i%13-6) / 7
		}
		for i := range kv {
			kv[i] = float32(i%19-9) / 11
		}
		cudaV41IndexerWitnessCase(t, be, qv, kv, wv, rows, heads, dim)
	}
	// Exercise both sides of dispatch, partial 64-row tiles and the bounded
	// shared-key dimension. These remain requested physical witnesses, not
	// source-only proof that the tiled kernel was dispatched or is faster.
	for _, shape := range []struct{ rows, heads, dim int }{
		{63, 3, 128}, {64, 3, 128}, {65, 3, 128}, {127, 3, 7},
		{129, 3, 1}, {65, 3, 129}, {65, 1, 128}, {8192, 32, 128},
	} {
		qv := make([]float32, shape.heads*shape.dim)
		kv := make([]float32, shape.rows*shape.dim)
		wv := make([]float32, shape.heads)
		for i := range qv {
			qv[i] = float32(i%13-6) / 7
		}
		for i := range kv {
			kv[i] = float32(i%19-9) / 11
		}
		for i := range wv {
			wv[i] = float32(i%7-3) / 5
		}
		cudaV41IndexerWitnessCase(t, be, qv, kv, wv, shape.rows, shape.heads, shape.dim)
	}
	// Finite input can produce subnormal values, infinity or NaN. Preserve the
	// reference's arithmetic classification; NaN payload identity is unspecified.
	for _, fixture := range []struct{ q, keys, weights []float32 }{
		{[]float32{math.Float32frombits(0x00800000), 0}, []float32{0.5, 0}, []float32{1}},
		{[]float32{math.Float32frombits(0x80000000), 0}, []float32{1, 0}, []float32{-1}},
		{[]float32{math.MaxFloat32, math.MaxFloat32}, []float32{2, -2}, []float32{1}},
		{[]float32{math.MaxFloat32, 1}, []float32{2, 0}, []float32{-1}},
	} {
		cudaV41IndexerWitnessCase(t, be, fixture.q, fixture.keys, fixture.weights, 1, 1, 2)
	}
	if be.SupportsV41IndexerScore() {
		t.Fatal("running a package test must not enable production capability")
	}
}

func cudaV41IndexerWitnessCase(t *testing.T, be *cudaBackend, qv, kv, wv []float32, rows, heads, dim int) {
	t.Helper()
	q := mkResident(be, []int{heads, dim}, qv)
	defer be.Free(q)
	w := mkResident(be, []int{heads}, wv)
	defer be.Free(w)
	keys := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{rows, dim}}
	if rows != 0 {
		keys = mkResident(be, []int{rows, dim}, kv)
		defer be.Free(keys)
	}
	before := cudaV41IndexerCompleted()
	out, err := be.cudaV41IndexerScore(q, keys, w, rows, heads, dim, true)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(out)
	if rows == 0 {
		if out.Numel() != 0 || cudaV41IndexerCompleted() != before {
			t.Fatal("empty score dispatched or allocated logical output")
		}
		return
	}
	if cudaV41IndexerCompleted() != before+1 {
		t.Fatal("no completed native scorer witness")
	}
	got, err := be.cudaV41IndexerReadback(out, rows)
	if err != nil {
		t.Fatal(err)
	}
	for row := 0; row < rows; row++ {
		var want float32
		for h := 0; h < heads; h++ {
			var dot float32
			for d := 0; d < dim; d++ {
				dot = float32(dot + float32(qv[h*dim+d]*kv[row*dim+d]))
			}
			if dot < 0 {
				dot = 0
			}
			want = float32(want + float32(dot*wv[h]))
		}
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got[row])) {
				t.Fatalf("row%d: want NaN got%g", row, got[row])
			}
		} else if math.Float32bits(want) != math.Float32bits(got[row]) {
			t.Fatalf("row%d: bits %08x want %08x", row, math.Float32bits(got[row]), math.Float32bits(want))
		}
	}
	// No unqualified invocation may become selected merely because the witness
	// arm ran once. It must return an error without another native launch.
	if _, err := be.V41IndexerScore(q, keys, w, rows, heads, dim); err == nil || cudaV41IndexerCompleted() != before+1 {
		t.Fatal("ordinary unqualified scorer dispatched or failed open")
	}
}
