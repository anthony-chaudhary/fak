package model

import (
	"math"
	"testing"
)

func qwen35RopeRowsTestConfig(q4k bool) Config {
	cfg := qwen35HybridTestCfg()
	if q4k {
		cfg = qwen35HybridQ4KTestCfg()
	}
	cfg.LayerTypes = []string{"linear_attention", "full_attention", "linear_attention", "full_attention"}
	cfg.FullAttentionInterval = 2
	cfg.PartialRotaryFactor = 0.5
	return cfg
}

// fak-test:runtime fast est=10ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_rope_rows*
func TestQwen35PrefillRopeRowsAdmission(t *testing.T) {
	cfg := qwen35RopeRowsTestConfig(false)
	s := &Session{M: &Model{Cfg: cfg}}
	if got := s.newQwen35PrefillRopeRows(7, 16); (got != nil) != qwen35PrefillRopeRowsQualification {
		t.Fatal("default/qualification selection changed")
	}
	for _, edit := range []func(*Session){
		func(s *Session) { s.Metal = true },
		func(s *Session) { s.MetalQ4K = true },
		func(s *Session) { s.M.Cfg = qwen35HybridTestCfg() }, // no cross-layer reuse
		func(s *Session) { s.M.Cfg.RopeThetaPerLayer = []float64{0, 20000} },
	} {
		x := &Session{M: &Model{Cfg: cfg}}
		edit(x)
		if x.newQwen35PrefillRopeRows(0, 16) != nil {
			t.Fatal("unsupported producer/configuration admitted")
		}
	}
	for _, shape := range [][2]int{{-1, 16}, {0, 0}, {int(^uint(0) >> 1), 2}} {
		if s.newQwen35PrefillRopeRows(shape[0], shape[1]) != nil {
			t.Fatal("invalid position extent admitted")
		}
	}
}

// fak-test:runtime fast est=20ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill_rope_rows*
func TestQwen35PrefillRopeRowsExact(t *testing.T) {
	cfg := qwen35RopeRowsTestConfig(false)
	configs := []Config{cfg}
	for _, mode := range []string{"linear", "llama3", "yarn", "dynamic"} {
		c := cfg
		c.RopeScaling, c.RopeFactor, c.RopeOrigContext = mode, 8, 2048
		c.RopeLowFreqFactor, c.RopeHighFreqFactor = 1, 4
		configs = append(configs, c)
	}
	long := cfg
	long.MaxPositionEmbeddings = 4096
	long.LongRope = &RopeScaling{Type: "longrope", OriginalMaxPositionEmbeddings: 2048, ShortFactor: []float64{1, 2}, LongFactor: []float64{2, 4}}
	configs = append(configs, long)
	for _, c := range configs {
		for _, base := range []int{0, 7, 1 << 20} {
			r := &qwen35PrefillRopeRows{base: base, rows: 5}
			if !r.prepare(c, 1, base, 5) {
				t.Fatal("canonical first layer declined")
			}
			parFor(5, 2, func(lo, hi int) {
				for i := lo; i < hi; i++ {
					r.row(i)
				}
			})
			firstCos, firstSin := r.table.cos[0], r.table.sin[0]
			for _, layer := range []int{1, 3, 1} {
				if !r.prepare(c, layer, base, 5) {
					t.Fatal("identical canonical layer declined")
				}
				for i := 0; i < 5; i++ {
					gotC, gotS := r.row(i)
					wantC, wantS := ropeRowForLayer(c, layer, base+i)
					assertFloat32BitsEqual(t, "cos", wantC, gotC)
					assertFloat32BitsEqual(t, "sin", wantS, gotS)
				}
				if &r.table.cos[0][0] != &firstCos[0] || &r.table.sin[0][0] != &firstSin[0] {
					t.Fatal("a later layer rebuilt the row")
				}
			}
			if len(r.table.cos) != 5 || len(r.table.sin) != 5 {
				t.Fatal("table included the cached prefix")
			}
			other := &qwen35PrefillRopeRows{base: base, rows: 5}
			if !other.prepare(c, 1, base, 5) {
				t.Fatal("independent request declined")
			}
			oc, _ := other.row(0)
			if &oc[0] == &firstCos[0] {
				t.Fatal("requests share mutable row storage")
			}
			before := append([]float32(nil), firstCos...)
			changed := c
			changed.RopeTheta = c.RopeTheta * 3
			changed.PartialRotaryFactor = 1
			if r.prepare(changed, 3, base, 5) || r.prepare(c, 3, base+1, 5) || r.prepare(c, 3, base, 4) {
				t.Fatal("incompatible row identity reused")
			}
			assertFloat32BitsEqual(t, "refusal unchanged", before, r.table.cos[0])
			if !r.prepare(c, 3, base, 5) {
				t.Fatal("refusal corrupted the prior owner")
			}
			if a := testing.AllocsPerRun(4, func() { r.row(0) }); a != 0 {
				t.Fatalf("read-only reused row allocated: %g", a)
			}
		}
	}
}

// fak-test:runtime fast est=2000ms
// fak-test:justify why=contract when=changed:internal/model/qwen35_prefill*
// Exercises the real full-attention consumers, not a trigonometry-only kernel.
// The legacy nil-owner entry is the bit-exact control; Q4_K uses both resident
// Q4_K and Q8-minority projections, with the same synchronous local callbacks.
func TestQwen35PrefillRopeRowsFullAttention(t *testing.T) {
	for _, q4k := range []bool{false, true} {
		cfg := qwen35RopeRowsTestConfig(q4k)
		m := NewSynthetic(cfg)
		m.Quantize()
		if q4k {
			fillQ4KMajority(t, m, cfg)
		}
		const rows = 4
		left, right := m.NewSession(), m.NewSession()
		x := make([]float32, rows*cfg.HiddenSize)
		for i := range x {
			x[i] = float32(i%29-14) / 19
		}
		r := &qwen35PrefillRopeRows{rows: rows}
		for _, layer := range []int{1, 3} {
			var want, got []float32
			if !q4k {
				qz := func(x []float32, rows, width int) *q8Panel {
					p := &q8Panel{}
					quantizeBatchPanelInto(p, x, rows, width)
					return p
				}
				want = left.prefillQwen35FullAttnLayerQ(layer, x, rows, 0, qz)
				got = right.prefillQwen35FullAttnLayerQWithRopeRows(layer, x, rows, 0, qz, r)
			} else {
				producer := func(s *Session) (hybridQ4KProj, hybridQ4KGroup, hybridQ4KQuantize) {
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
						return out
					}
					quantize := func(x []float32, rows, width int, _ ...string) *q8Panel {
						p := &q8Panel{}
						quantizeBatchPanelInto(p, x, rows, width)
						return p
					}
					return proj, group, quantize
				}
				lp, lg, lq := producer(left)
				rp, rg, rq := producer(right)
				want = left.prefillQwen35FullAttnLayerQ4KOwnedQF(layer, x, rows, 0, lp, lg, lq, true)
				got = right.prefillQwen35FullAttnLayerQ4KWithRopeRows(layer, x, rows, 0, rp, rg, rq, true, r)
			}
			assertFloat32BitsEqual(t, "full attention output", want, got)
			assertKVCacheUnchanged(t, "full attention KV", left.Cache, right.Cache)
			for _, v := range got {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					t.Fatal("nonfinite fixture output")
				}
			}
		}
		if len(r.table.cos) != rows || r.table.cos[0] == nil {
			t.Fatal("real attention consumer never reached the table")
		}
	}
}
