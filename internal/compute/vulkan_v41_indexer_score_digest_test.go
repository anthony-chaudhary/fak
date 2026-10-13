//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

// Indexer score digests are compared across shader builds: a relayout that only
// moves where each serial dot and the ordered head sum run must leave every bit
// equal. Shapes cover partial and multiple 64-head chunks, decode row counts up
// to long-context compressed rows, and an overflow shape whose +/-Inf and NaN
// bits are part of the digest. FAK_V41_INDEXER_SCORE_BENCH=<n> repeats each
// shape n times for timestamps.
// fak-test:runtime fast est=4s lane=default
func TestVulkanV41IndexerScoreDecodeShapeDigest(t *testing.T) {
	v := vk(t)
	if !v.SupportsV41IndexerScore() {
		t.Skip("V4.1 indexer score pipeline unavailable")
	}
	repeats := 1
	if n, err := strconv.Atoi(os.Getenv("FAK_V41_INDEXER_SCORE_BENCH")); err == nil && n > 1 {
		repeats = n
	}
	type shape struct {
		rows, heads, dim int
		scale            float32
	}
	shapes := []shape{{1, 3, 7, 1}, {65, 65, 33, 1}, {130, 130, 128, 1}, {7, 64, 128, 3e19}}
	for _, rows := range []int{1, 2, 63, 64, 65, 130, 640, 2049, 4096, 16384} {
		shapes = append(shapes, shape{rows, 64, 128, 1})
	}
	for _, s := range shapes {
		q := make([]float32, s.heads*s.dim)
		for i := range q {
			q[i] = decodeSplitValue(i, s.rows) * s.scale
		}
		keys := make([]float32, s.rows*s.dim)
		for i := range keys {
			keys[i] = decodeSplitValue(i, s.rows+7) * s.scale
		}
		weights := make([]float32, s.heads)
		for i := range weights {
			weights[i] = decodeSplitValue(i, s.rows+13) * 0.125
		}
		got, err := v41IndexerScoreRun(v, q, keys, weights, s.rows, s.heads, s.dim, repeats)
		if err != nil {
			t.Fatalf("rows=%d heads=%d dim=%d: %v", s.rows, s.heads, s.dim, err)
		}
		want := v41IndexerScoreSerialOracle(q, keys, weights, s.rows, s.heads, s.dim)
		for i := range got {
			g, w := got[i], want[i]
			if math.Float32bits(g) != math.Float32bits(w) && !(g != g && w != w) {
				t.Fatalf("rows=%d heads=%d dim=%d score[%d] bits %08x want %08x", s.rows, s.heads, s.dim, i, math.Float32bits(g), math.Float32bits(w))
			}
		}
		t.Logf("INDEXER_SCORE_DIGEST rows%d_heads%d_dim%d_scale%g %s", s.rows, s.heads, s.dim, s.scale, decodeSplitDigest(got))
	}
}

// v41IndexerScoreSerialOracle is the one-invocation-per-row order: each head dot
// in dimension order, ReLU, weight, then heads summed in order. The float32
// conversions keep every product and sum separately rounded (no fused form).
func v41IndexerScoreSerialOracle(q, keys, weights []float32, rows, heads, dim int) []float32 {
	out := make([]float32, rows)
	for r := 0; r < rows; r++ {
		var score float32
		for h := 0; h < heads; h++ {
			var dot float32
			for d := 0; d < dim; d++ {
				dot = float32(dot + float32(q[h*dim+d]*keys[r*dim+d]))
			}
			if dot < 0 {
				dot = 0
			}
			score = float32(score + float32(dot*weights[h]))
		}
		out[r] = score
	}
	return out
}

func v41IndexerScoreRun(v *vulkanBackend, q, keys, weights []float32, rows, heads, dim, repeats int) ([]float32, error) {
	upload := func(shape []int, values []float32) Tensor {
		return v.UploadClass(NewF32(Default(), shape, values), F32, MemoryActivation, "indexer score digest input")
	}
	qt, kt, wt := upload([]int{heads, dim}, q), upload([]int{rows, dim}, keys), upload([]int{heads}, weights)
	defer v.Free(qt)
	defer v.Free(kt)
	defer v.Free(wt)
	var got []float32
	for i := 0; i < repeats; i++ {
		v.BeginBatch()
		out, err := v.V41IndexerScore(qt, kt, wt, rows, heads, dim)
		if err != nil {
			v.FlushBatch()
			return nil, err
		}
		values := v.Read(out)
		v.Free(out)
		v.FlushBatch()
		if got != nil && decodeSplitDigest(got) != decodeSplitDigest(values) {
			return nil, fmt.Errorf("repeat %d changed output bits", i)
		}
		got = values
	}
	return got, nil
}
