package model

import (
	"math"
	"sync/atomic"
	"testing"
)

func qwen35TiledAttentionClose(t *testing.T, name string, got, want []float32, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s lengths %d != %d", name, len(got), len(want))
	}
	for i, x := range got {
		d := math.Abs(float64(x) - float64(want[i]))
		if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) || d > tolerance*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("%s[%d] got=%g want=%g delta=%g", name, i, x, want[i], d)
		}
	}
}

// fak-test:runtime fast est=100ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_tiled_attention*
func TestQwen35TiledAttentionTiles(t *testing.T) {
	for _, shape := range [][4]int{{19, 6, 2, 9}, {35, 24, 4, 256}, {65, 4, 2, 64}} {
		rows, heads, kvHeads, dim := shape[0], shape[1], shape[2], shape[3]
		for _, base := range []int{0, 17} {
			for _, window := range []int{-1, 1, 11} {
				for _, cap := range []float32{0, 2.5} {
					width, kw := heads*dim, kvHeads*dim
					q, k, v := make([]float32, rows*width), make([]float32, (base+rows)*kw), make([]float32, (base+rows)*kw)
					for i := range q {
						q[i] = float32(i%29-14) / 31
					}
					for i := range k {
						k[i], v[i] = float32(i%31-15)/37, float32(i%23-11)/17
					}
					beforeQ, beforeK, beforeV := append([]float32(nil), q...), append([]float32(nil), k...), append([]float32(nil), v...)
					want, got := make([]float32, len(q)), make([]float32, len(q))
					scale := float32(1 / math.Sqrt(float64(dim)))
					attnPrefillInto(want, q, k, v, rows, base, heads, dim, kw, heads/kvHeads, window, 0, scale, cap, fdot, nil)
					scratch := make([]float32, 2*((2*qwen35AttentionKVTile+qwen35AttentionQueryTile)*dim+qwen35AttentionQueryTile*qwen35AttentionKVTile+2*qwen35AttentionQueryTile))
					for i := range scratch {
						scratch[i] = float32(math.NaN())
					}
					if !qwen35PrefillTiledAttentionRun(got, q, k, v, scratch, rows, base, heads, dim, kw, heads/kvHeads, window, scale, cap, nil, 2) {
						t.Fatal("supported tile declined")
					}
					qwen35TiledAttentionClose(t, "online versus score-row", got, want, 2e-5)
					assertFloat32BitsEqual(t, "query read-only", beforeQ, q)
					assertFloat32BitsEqual(t, "key read-only", beforeK, k)
					assertFloat32BitsEqual(t, "value read-only", beforeV, v)
				}
			}
		}
	}
}

// fak-test:runtime fast est=20ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_tiled_attention*
func TestQwen35TiledAttentionRefusesBeforeWrites(t *testing.T) {
	const rows, heads, dim, group = 17, 4, 8, 2
	const width, kw = heads * dim, heads / group * dim
	for _, mode := range []string{"short-loan", "loan-capacity-only", "short-output", "short-query", "short-key", "short-value", "geometry", "zero-window", "position-overflow", "worker-budget", "observer", "nan-query", "inf-key", "inf-value", "nan-scale", "inf-cap", "dot-overflow", "value-overflow", "out-query-alias", "scratch-query-alias", "scratch-output-alias"} {
		t.Run(mode, func(t *testing.T) {
			q, k, v := make([]float32, rows*width), make([]float32, rows*kw), make([]float32, rows*kw)
			out, scratch := make([]float32, rows*width), make([]float32, 8192)
			for i := range q {
				q[i] = .25
			}
			for i := range k {
				k[i], v[i] = .5, .75
			}
			for i := range out {
				out[i] = 7
			}
			for i := range scratch {
				scratch[i] = 11
			}
			base, nheads, window, workers := 0, heads, -1, 2
			scale, cap := float32(.5), float32(0)
			var obs AttnObserver
			switch mode {
			case "short-loan":
				scratch = scratch[:1:1]
			case "loan-capacity-only":
				scratch = scratch[:1]
			case "short-output":
				out = out[:len(out)-1]
			case "short-query":
				q = q[:len(q)-1]
			case "short-key":
				k = k[:len(k)-1]
			case "short-value":
				v = v[:len(v)-1]
			case "geometry":
				nheads = 3
			case "zero-window":
				window = 0
			case "position-overflow":
				base = int(^uint(0) >> 1)
			case "worker-budget":
				workers = 67
			case "observer":
				obs = func(int, int, int, []int, []float32) {}
			case "nan-query":
				q[0] = float32(math.NaN())
			case "inf-key":
				k[0] = float32(math.Inf(1))
			case "inf-value":
				v[0] = float32(math.Inf(-1))
			case "nan-scale":
				scale = float32(math.NaN())
			case "inf-cap":
				cap = float32(math.Inf(1))
			case "dot-overflow":
				q[0], k[0] = math.MaxFloat32, math.MaxFloat32
			case "value-overflow":
				v[0] = math.MaxFloat32
			case "out-query-alias":
				out = q
			case "scratch-query-alias":
				scratch = make([]float32, 8192)
				q = scratch[:rows*width]
			case "scratch-output-alias":
				scratch = make([]float32, 8192)
				out = scratch[:rows*width]
			}
			before := make([][]float32, 0, 5)
			for _, x := range [][]float32{out, q, k, v, scratch} {
				before = append(before, append([]float32(nil), x...))
			}
			if qwen35PrefillTiledAttentionRun(out, q, k, v, scratch, rows, base, nheads, dim, kw, group, window, scale, cap, obs, workers) {
				t.Fatal("unsupported candidate admitted")
			}
			for i, x := range [][]float32{out, q, k, v, scratch} {
				assertFloat32BitsEqual(t, "refusal writes", before[i], x)
			}
		})
	}
}

// fak-test:runtime fast est=2000ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill*
// Uses the real Q4_K full-attention producer with Q8-minority Q/K and Q4_K V/O.
// The two-head GQA group cannot take the old three-query loan route, so mutation
// of the captured private qf proves the new selected consumer actually ran.
func TestQwen35TiledAttentionFullConsumer(t *testing.T) {
	savedWorkers, savedSource := numWorkers, workerBudgetSource
	t.Cleanup(func() { numWorkers, workerBudgetSource = savedWorkers, savedSource })
	if err := SetWorkers(2); err != nil {
		t.Fatal(err)
	}
	cfg := qwen35HybridQ4KTestCfg()
	m := NewSynthetic(cfg)
	m.Quantize()
	fillQ4KMajority(t, m, cfg)
	left, right := m.NewSession(), m.NewSession()
	const rows = 65
	x := make([]float32, rows*cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%29-14) / 19
	}
	producer := func(s *Session, capture *([]float32)) (hybridQ4KProj, hybridQ4KGroup, hybridQ4KQuantize) {
		proj := func(name string, x []float32, xq *q8Panel) []float32 {
			if qt := m.q4kw[name]; qt != nil {
				return s.q4kGemmDispatch(name, qt, x, rows)
			}
			return s.q8GemmDispatch(name, m.q8(name), xq)
		}
		group := func(names []string, x []float32, xq *q8Panel) [][]float32 {
			out := make([][]float32, len(names))
			for i, name := range names {
				out[i] = proj(name, x, xq)
			}
			*capture = out[0]
			return out
		}
		qz := func(x []float32, rows, width int, _ ...string) *q8Panel { return quantizeBatchPanel(x, rows, width) }
		return proj, group, qz
	}
	var lqf, rqf []float32
	lp, lg, lq := producer(left, &lqf)
	rp, rg, rq := producer(right, &rqf)
	// Caller ownership false is the retained custom-callback oracle.
	want := left.prefillQwen35FullAttnLayerQ4KOwnedQF(3, x, rows, 0, lp, lg, lq, false)
	got := right.prefillQwen35FullAttnLayerQ4KOwnedQF(3, x, rows, 0, rp, rg, rq, true)
	changed := false
	for i := range lqf {
		if math.Float32bits(lqf[i]) != math.Float32bits(rqf[i]) {
			changed = true
			break
		}
	}
	if changed != qwen35PrefillTiledAttentionQualification {
		t.Fatal("default-off/selected consumer reach changed")
	}
	// Online rounding can be amplified by the unchanged downstream Q8 input
	// quantizer. This component bound is NOT a model/logit acceptance claim.
	qwen35TiledAttentionClose(t, "full quantized consumer", got, want, 2e-3)
	assertKVCacheUnchanged(t, "attention KV unchanged", left.Cache, right.Cache)
	before := append([]float32(nil), got...)
	for i := range rqf {
		rqf[i] = float32(math.NaN())
	}
	assertFloat32BitsEqual(t, "returned output owns values", before, got)

	// An observer must still see every old score row and cannot lose semantics
	// merely because the candidate tag is selected.
	observed := m.NewSession()
	var count atomic.Int64
	m.SetAttnObserver(func(int, int, int, []int, []float32) { count.Add(1) })
	t.Cleanup(func() { m.SetAttnObserver(nil) })
	var oqf []float32
	op, og, oq := producer(observed, &oqf)
	seen := observed.prefillQwen35FullAttnLayerQ4KOwnedQF(3, x, rows, 0, op, og, oq, true)
	if count.Load() != int64(rows*cfg.NumHeads) {
		t.Fatal("observer rows lost")
	}
	assertFloat32BitsEqual(t, "observer legacy output", want, seen)
	assertFloat32BitsEqual(t, "observer did not loan qf", lqf, oqf)
}

// fak-test:runtime fast est=500ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill*
func TestQwen35TiledAttentionQ8Consumer(t *testing.T) {
	savedWorkers, savedSource := numWorkers, workerBudgetSource
	t.Cleanup(func() { numWorkers, workerBudgetSource = savedWorkers, savedSource })
	if err := SetWorkers(2); err != nil {
		t.Fatal(err)
	}
	cfg := qwen35HybridQ4KTestCfg() // legal shared geometry, but all projections Q8
	m := NewSynthetic(cfg)
	m.Quantize()
	left, right := m.NewSession(), m.NewSession()
	const rows = 65
	x := make([]float32, rows*cfg.HiddenSize)
	for i := range x {
		x[i] = float32(i%29-14) / 19
	}
	qz := func(x []float32, rows, width int) *q8Panel { return quantizeBatchPanel(x, rows, width) }
	var seen atomic.Int64
	m.SetAttnObserver(func(int, int, int, []int, []float32) { seen.Add(1) })
	want := left.prefillQwen35FullAttnLayerQ(3, x, rows, 0, qz)
	m.SetAttnObserver(nil)
	if seen.Load() != int64(rows*cfg.NumHeads) {
		t.Fatal("legacy oracle not observed")
	}
	got := right.prefillQwen35FullAttnLayerQ(3, x, rows, 0, qz)
	qwen35TiledAttentionClose(t, "Q8 full attention consumer", got, want, 2e-3)
	assertKVCacheUnchanged(t, "Q8 KV unchanged", left.Cache, right.Cache)
}
