//go:build cuda && cgo && fak_cuda_v41_shared_attention

package compute

import (
	"errors"
	"fmt"
	"math"
	"testing"
	"time"
)

// Opt-in physical witness for the CUDA FlashMLA-class candidate (fak#13993).
// Build the library with -DFAK_CUDA_V41_SHARED_ATTENTION and run with
// -tags 'cuda fak_cuda_v41_shared_attention'. Each dispatch is pinned to the
// host FlashMLA-class reference at the tile/split partition it actually ran,
// and to the binary64 oracle.
// fak-test:runtime integration est=20s lane=optin
func TestCUDAV41SharedAttentionMatchesFlashMLAReference(t *testing.T) {
	be := cudaGDNBackend(t)
	if !be.SupportsV41SharedAttention() {
		t.Fatal("candidate built but native admission refused")
	}
	const bound = 4e-6 // outputs are bounded by max|kv| = 1; see v41_sparse_mla_decode_test.go
	for _, tc := range []struct{ rows, heads, dim int }{
		{1, 4, 512}, {15, 4, 512}, {16, 4, 512}, {17, 4, 512}, {63, 4, 512}, {64, 4, 512},
		{65, 4, 512}, {130, 4, 512}, {640, 64, 512}, {1152, 128, 512}, {2049, 1, 512},
		{33, 3, 7}, {100, 9, 128}, {40, 5, 1500},
	} {
		for _, hasSink := range []bool{false, true} {
			for _, splits := range []int{0, 1, 2, 7} {
				name := fmt.Sprintf("r%d_h%d_d%d_sink%t_split%d", tc.rows, tc.heads, tc.dim, hasSink, splits)
				t.Run(name, func(t *testing.T) {
					qv, kvv, sinkv := v41SparseMLAInputs(int64(tc.rows*131+tc.heads*7+splits), tc.rows, tc.heads, tc.dim)
					got, run := cudaV41SharedAttentionCase(t, be, qv, kvv, sinkv, tc.rows, tc.heads, tc.dim, hasSink, splits)
					var useSink []float32
					if hasSink {
						useSink = sinkv
					}
					scale := float32(1 / math.Sqrt(float64(tc.dim)))
					ref, err := v41SparseMLADecodeReference(qv, kvv, useSink, tc.rows, tc.heads, tc.dim, scale, hasSink,
						v41SparseMLADecodeOptions{TileRows: run.Tile, Splits: run.Splits})
					if err != nil {
						t.Fatal(err)
					}
					if diff := v41SparseMLAMaxDiff(got, ref.Out); diff > bound {
						t.Fatalf("tile=%d splits=%d max |cuda-reference| = %g, bound %g", run.Tile, run.Splits, diff, bound)
					}
					oracle := v41SharedAttentionFloat64Oracle(qv, kvv, useSink, tc.heads, tc.dim, scale)
					if diff := v41SparseMLAMaxDiff(got, oracle); diff > bound {
						t.Fatalf("tile=%d splits=%d max |cuda-binary64| = %g, bound %g", run.Tile, run.Splits, diff, bound)
					}
				})
			}
		}
	}
}

// Prefill runs one shared-attention call per query position, so the prefill
// shapes are long selected-row counts (R up to 8192) and a run of consecutive
// positions whose row count grows by one, each pinned like the decode table.
// fak-test:runtime integration est=30s lane=optin
func TestCUDAV41SharedAttentionPrefillShapes(t *testing.T) {
	be := cudaGDNBackend(t)
	const bound = 4e-6
	check := func(t *testing.T, rows, heads, dim int, seed int64) {
		t.Helper()
		qv, kvv, sinkv := v41SparseMLAInputs(seed, rows, heads, dim)
		got, run := cudaV41SharedAttentionCase(t, be, qv, kvv, sinkv, rows, heads, dim, true, 0)
		scale := float32(1 / math.Sqrt(float64(dim)))
		ref, err := v41SparseMLADecodeReference(qv, kvv, sinkv, rows, heads, dim, scale, true,
			v41SparseMLADecodeOptions{TileRows: run.Tile, Splits: run.Splits})
		if err != nil {
			t.Fatal(err)
		}
		if diff := v41SparseMLAMaxDiff(got, ref.Out); diff > bound {
			t.Fatalf("rows=%d tile=%d splits=%d max |cuda-reference| = %g, bound %g", rows, run.Tile, run.Splits, diff, bound)
		}
		if diff := v41SparseMLAMaxDiff(got, v41SharedAttentionFloat64Oracle(qv, kvv, sinkv, heads, dim, scale)); diff > bound {
			t.Fatalf("rows=%d tile=%d splits=%d max |cuda-binary64| = %g, bound %g", rows, run.Tile, run.Splits, diff, bound)
		}
	}
	for _, rows := range []int{2048, 4096, 8192} {
		t.Run(fmt.Sprintf("r%d_h64_d512", rows), func(t *testing.T) { check(t, rows, 64, 512, int64(rows)) })
	}
	t.Run("positions_1_to_48_h8_d512", func(t *testing.T) {
		for rows := 1; rows <= 48; rows++ {
			check(t, rows, 8, 512, int64(9000+rows))
		}
	})
}

// A finite input whose scaled score overflows must surface the typed score
// producer with its gathered slot, and publish no output.
// fak-test:runtime integration est=1s lane=optin
func TestCUDAV41SharedAttentionScoreFaultIsTyped(t *testing.T) {
	be := cudaGDNBackend(t)
	const rows, heads, dim = 5, 2, 4
	qv := make([]float32, heads*dim)
	kvv := make([]float32, rows*dim)
	for i := range kvv {
		kvv[i] = 0.25
	}
	qv[dim] = math.MaxFloat32 // head 1 only
	kvv[3*dim] = 4            // slot 3 overflows for head 1
	q := mkResident(be, []int{heads, dim}, qv)
	defer be.Free(q)
	kv := mkResident(be, []int{rows, dim}, kvv)
	defer be.Free(kv)
	sink := mkResident(be, []int{1}, []float32{0})
	defer be.Free(sink)
	_, _, err := be.cudaV41SharedAttention(q, kv, sink, rows, heads, dim, 1, V41SharedAttentionPlain, false, 0)
	var arithmetic *V41SharedAttentionArithmeticError
	if !errors.As(err, &arithmetic) {
		t.Fatalf("error = %v, want *V41SharedAttentionArithmeticError", err)
	}
	if arithmetic.Stage != V41SharedAttentionStageScore || arithmetic.Head != 1 || arithmetic.SelectedSlot != 3 {
		t.Fatalf("fault = %+v, want score head 1 slot 3", *arithmetic)
	}
	if !be.SupportsV41SharedAttention() {
		t.Fatal("an arithmetic fault must not close admission")
	}
}

// Per-op timing at V4.1 decode geometry: wall ms includes launch, status
// readback and synchronization; kernel ms is the native event pair.
// fak-test:runtime integration est=10s lane=optin
func TestCUDAV41SharedAttentionTiming(t *testing.T) {
	be := cudaGDNBackend(t)
	for _, tc := range []struct{ rows, heads int }{{640, 64}, {1152, 64}, {1152, 128}, {4224, 128}} {
		const dim, iters = 512, 50
		qv, kvv, sinkv := v41SparseMLAInputs(7, tc.rows, tc.heads, dim)
		q := mkResident(be, []int{tc.heads, dim}, qv)
		kv := mkResident(be, []int{tc.rows, dim}, kvv)
		sink := mkResident(be, []int{tc.heads}, sinkv)
		scale := float32(1 / math.Sqrt(dim))
		var kernel float64
		var run cudaV41SharedAttentionRun
		var wall time.Duration
		for i := 0; i < iters+5; i++ {
			start := time.Now()
			out, r, err := be.cudaV41SharedAttention(q, kv, sink, tc.rows, tc.heads, dim, scale, V41SharedAttentionPlain, true, 0)
			if err != nil {
				t.Fatal(err)
			}
			elapsed := time.Since(start)
			be.Free(out)
			if i >= 5 {
				wall += elapsed
				kernel += float64(r.KernelMS)
			}
			run = r
		}
		hostStart := time.Now()
		if _, err := v41SparseMLADecodeReference(qv, kvv, sinkv, tc.rows, tc.heads, dim, scale, true,
			v41SparseMLADecodeOptions{TileRows: run.Tile, Splits: run.Splits}); err != nil {
			t.Fatal(err)
		}
		host := time.Since(hostStart)
		t.Logf("V41SA_RECEIPT rows=%d heads=%d dim=%d tile=%d splits=%d wall_ms=%.4f kernel_ms=%.4f host_ref_ms=%.3f",
			tc.rows, tc.heads, dim, run.Tile, run.Splits, float64(wall.Microseconds())/1000/iters, kernel/iters,
			float64(host.Microseconds())/1000)
		be.Free(q)
		be.Free(kv)
		be.Free(sink)
	}
}

func cudaV41SharedAttentionCase(t *testing.T, be *cudaBackend, qv, kvv, sinkv []float32, rows, heads, dim int, hasSink bool, splits int) ([]float32, cudaV41SharedAttentionRun) {
	t.Helper()
	q := mkResident(be, []int{heads, dim}, qv)
	defer be.Free(q)
	kv := mkResident(be, []int{rows, dim}, kvv)
	defer be.Free(kv)
	sinkHost := []float32{0}
	if hasSink {
		sinkHost = sinkv
	}
	sink := mkResident(be, []int{len(sinkHost)}, sinkHost)
	defer be.Free(sink)
	before := cudaV41SharedAttentionCompleted()
	scale := float32(1 / math.Sqrt(float64(dim)))
	out, run, err := be.cudaV41SharedAttention(q, kv, sink, rows, heads, dim, scale, V41SharedAttentionPlain, hasSink, splits)
	if err != nil {
		t.Fatal(err)
	}
	defer be.Free(out)
	if cudaV41SharedAttentionCompleted() != before+1 {
		t.Fatal("no completed native shared-attention witness")
	}
	if splits > 0 && run.Splits != min(splits, (rows+run.Tile-1)/run.Tile) {
		t.Fatalf("forced splits %d ran %d (tile %d)", splits, run.Splits, run.Tile)
	}
	return be.Read(out), run
}
