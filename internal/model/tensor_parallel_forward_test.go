package model

import (
	"encoding/binary"
	"math"
	"sort"
	"testing"
)

// tensor_parallel_forward_test.go — the gates for WIRING the tensor-parallel decomposition
// into the LIVE forward path (tensor_parallel_forward.go). The standalone primitives
// (tensor_parallel_test.go / tensor_parallel_attn_test.go) proved sharding-invariance for a
// MINIMAL attention/FFN kernel that omits RoPE, qk-norm, softcap, ALiBi, the output gate,
// biases, and the sliding window. These gates prove the SAME decomposition over the REAL
// model layer, with every one of those features RE-ENTERED, equals the live (HF-oracle-gated)
// attnSeq / mlpSeq / Forward:
//
//   - ranks=1  ==(max|Δ|=0)  the live sub-layer / Forward — the wired path is arithmetically
//     identical to the HF-gated path, so it inherits the HF oracle (oracle_test.go).
//   - per-rank pre-o_proj bands concat ==(max|Δ|=0) the 1-rank band — column-parallel
//     projection+attention shards bit-exactly EVEN WITH RoPE/qk-norm/gate/softcap re-entered
//     (they are head-local).
//   - the row-parallel o_proj/down AllReduce ==(max|Δ|=0) a rank-order reference (reduction
//     order pinned) and within documented round-off of the live monolith.
//
// Coverage is a feature MATRIX (basic, softcap, gate, ALiBi, sliding window, GELU FFN,
// qk-norm, biases) so each omitted feature is exercised re-entering the sharded path.

// --- test model construction ------------------------------------------------------------

// tpFwdBaseCfg is a dense Llama-style config with nKV=4 (attention shardable across {1,2,4})
// and I=96 (FFN shardable across {1,2,3,4,6,8}). HeadDim*NumHeads == HiddenSize == 64.
func tpFwdBaseCfg() Config {
	return Config{
		HiddenSize: 64, NumLayers: 3, NumHeads: 8, NumKVHeads: 4, HeadDim: 8,
		IntermediateSize: 96, VocabSize: 40, RMSNormEps: 1e-5, RopeTheta: 10000,
		ModelType: "llama", EOSTokenID: -1,
	}
}

// tpTensor is an injected f32 tensor (shape + values) for features NewSynthetic does not emit
// (qk-norm weights, projection/MLP biases).
type tpTensor struct {
	shape []int
	vals  []float32
}

// tpInjectTensors appends f32 tensors to a synthetic model's manifest+raw. m.tensor() derives
// its zero-copy view from m.raw per call, and append preserves the existing prefix bytes, so
// previously-manifested tensors keep their offsets while the new ones land in the appended
// region. Inject BEFORE any forward pass (no live views are held across the append).
func tpInjectTensors(m *Model, extra map[string]tpTensor) {
	names := make([]string, 0, len(extra))
	for n := range extra {
		names = append(names, n)
	}
	sort.Strings(names) // deterministic offsets
	base := len(m.raw)
	var add []byte
	for _, n := range names {
		t := extra[n]
		m.manifest[n] = tensorMeta{Dtype: "F32", Shape: t.shape, Offset: base + len(add), Nbytes: len(t.vals) * 4}
		for _, f := range t.vals {
			var b [4]byte
			binary.LittleEndian.PutUint32(b[:], math.Float32bits(f))
			add = append(add, b[:]...)
		}
	}
	m.raw = append(m.raw, add...)
}

// tpInjectQKNorm adds per-layer q_norm/k_norm.weight (length HeadDim) with values near 1.0.
func tpInjectQKNorm(m *Model) {
	cfg := m.Cfg
	extra := map[string]tpTensor{}
	for l := 0; l < cfg.NumLayers; l++ {
		for _, suffix := range []string{"q_norm", "k_norm"} {
			vals := make([]float32, cfg.HeadDim)
			for i := range vals {
				vals[i] = 0.9 + 0.2*float32(i%5)/4 // [0.9,1.1]
			}
			extra[layerName(l, "self_attn."+suffix+".weight")] = tpTensor{shape: []int{cfg.HeadDim}, vals: vals}
		}
	}
	tpInjectTensors(m, extra)
}

// tpInjectBiases adds q/k/v/o projection biases and gate/up/down MLP biases (small random).
func tpInjectBiases(m *Model) {
	cfg := m.Cfg
	H, I, hd := cfg.HiddenSize, cfg.IntermediateSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	rnd := func(n int, seed uint64) []float32 { return tpRand(n, seed) }
	extra := map[string]tpTensor{}
	for l := 0; l < cfg.NumLayers; l++ {
		ul := uint64(l + 1)
		extra[layerName(l, "self_attn.q_proj.bias")] = tpTensor{[]int{nH * hd}, rnd(nH*hd, 100*ul+1)}
		extra[layerName(l, "self_attn.k_proj.bias")] = tpTensor{[]int{nKV * hd}, rnd(nKV*hd, 100*ul+2)}
		extra[layerName(l, "self_attn.v_proj.bias")] = tpTensor{[]int{nKV * hd}, rnd(nKV*hd, 100*ul+3)}
		extra[layerName(l, "self_attn.o_proj.bias")] = tpTensor{[]int{H}, rnd(H, 100*ul+4)}
		extra[layerName(l, "mlp.gate_proj.bias")] = tpTensor{[]int{I}, rnd(I, 100*ul+5)}
		extra[layerName(l, "mlp.up_proj.bias")] = tpTensor{[]int{I}, rnd(I, 100*ul+6)}
		extra[layerName(l, "mlp.down_proj.bias")] = tpTensor{[]int{H}, rnd(H, 100*ul+7)}
	}
	tpInjectTensors(m, extra)
}

// tpFeatureVariant is one row of the feature matrix: a config mutation + optional tensor
// injection, so each omitted-feature is exercised re-entering the sharded path.
type tpFeatureVariant struct {
	name   string
	cfg    func() Config
	inject func(*Model)
}

func tpFeatureVariants() []tpFeatureVariant {
	return []tpFeatureVariant{
		{"basic", tpFwdBaseCfg, nil},
		{"softcap", func() Config { c := tpFwdBaseCfg(); c.AttnSoftcap = 30; return c }, nil},
		{"gate", func() Config { c := tpFwdBaseCfg(); c.AttnOutputGate = true; return c }, nil},
		{"alibi", func() Config { c := tpFwdBaseCfg(); c.Alibi = true; return c }, nil},
		{"window", func() Config { c := tpFwdBaseCfg(); c.Window = []int{2, 2, 2}; return c }, nil},
		{"gelu", func() Config { c := tpFwdBaseCfg(); c.ActGeluTanh = true; return c }, nil},
		{"qknorm", func() Config { c := tpFwdBaseCfg(); c.QKNorm = true; return c }, tpInjectQKNorm},
		{"bias", tpFwdBaseCfg, tpInjectBiases},
	}
}

func tpBuildVariant(v tpFeatureVariant) *Model {
	m := NewSynthetic(v.cfg())
	if v.inject != nil {
		v.inject(m)
	}
	return m
}

// --- comparison helpers -----------------------------------------------------------------

func tpRows(seq, hidden int, seed uint64) [][]float32 {
	x := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		x[t] = tpRand(hidden, seed+uint64(t)*7+1)
	}
	return x
}

// bitExactRows reports whether every element is Float32bits-identical, and the max abs diff.
func bitExactRows(a, b [][]float32) (bool, float64) {
	exact := true
	mx := 0.0
	for t := range a {
		for i := range a[t] {
			if math.Float32bits(a[t][i]) != math.Float32bits(b[t][i]) {
				exact = false
			}
			if d := math.Abs(float64(a[t][i] - b[t][i])); d > mx {
				mx = d
			}
		}
	}
	return exact, mx
}

// --- gates ------------------------------------------------------------------------------

// TestTPForwardRanks1MatchesLive is the transitive-HF gate: at ranks=1 the wired sub-layers
// are BIT-IDENTICAL (max|Δ|=0) to the live attnSeq / mlpSeq across the whole feature matrix.
// A 1-shard plan makes the AllGather a no-op concat and the AllReduce a single matRows over
// the full width, so the only non-bit-exact step (the row-parallel reassociation) vanishes;
// the per-band projection over the full band slices the whole weight, i.e. it IS matRows.
func TestTPForwardRanks1MatchesLive(t *testing.T) {
	const seq = 6
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			cfg := m.Cfg
			xn := tpRows(seq, cfg.HiddenSize, uint64(len(v.name))*13+1)
			rp := newRopeForLayer(cfg, 0, seq)

			attnPlan, err := NewTPPlan(cfg.NumKVHeads, 1)
			if err != nil {
				t.Fatalf("attn plan: %v", err)
			}
			gotAttn, err := m.tpAttnLayer(0, xn, rp, attnPlan, LocalCollective{})
			if err != nil {
				t.Fatalf("tpAttnLayer: %v", err)
			}
			refAttn := m.attnSeq(0, xn, rp)
			if exact, mx := bitExactRows(gotAttn, refAttn); !exact {
				t.Fatalf("[%s] tpAttnLayer(ranks=1) != attnSeq: max|Δ|=%.3e (want bit-exact 0)", v.name, mx)
			}

			ffnPlan, err := NewTPPlan(cfg.IntermediateSize, 1)
			if err != nil {
				t.Fatalf("ffn plan: %v", err)
			}
			gotFFN, err := m.tpFFNLayer(0, xn, ffnPlan, LocalCollective{})
			if err != nil {
				t.Fatalf("tpFFNLayer: %v", err)
			}
			refFFN := m.mlpSeq(0, xn)
			if exact, mx := bitExactRows(gotFFN, refFFN); !exact {
				t.Fatalf("[%s] tpFFNLayer(ranks=1) != mlpSeq: max|Δ|=%.3e (want bit-exact 0)", v.name, mx)
			}
		})
	}
}

// TestTPForwardColumnParallelBandDisjoint proves the head-parallel projection+attention shards
// BIT-EXACTLY with every feature re-entered: the per-rank pre-o_proj attention bands,
// concatenated in rank order along the head dimension, equal the single-rank full-band output
// at max|Δ|=0. This is the column-parallel disjointness witness (heads are independent;
// RoPE/qk-norm/the gate/softcap are head-local), now over the LIVE feature set rather than the
// minimal standalone kernel.
func TestTPForwardColumnParallelBandDisjoint(t *testing.T) {
	const seq = 6
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			cfg := m.Cfg
			hd, nH := cfg.HeadDim, cfg.NumHeads
			grp := cfg.GroupSize()
			xn := tpRows(seq, cfg.HiddenSize, uint64(len(v.name))*29+3)
			rp := newRopeForLayer(cfg, 0, seq)

			full, err := m.tpAttnBandPreProj(0, xn, rp, attnHeadBand{KVLo: 0, KVHi: cfg.NumKVHeads, QLo: 0, QHi: nH})
			if err != nil {
				t.Fatalf("full band: %v", err)
			}
			for _, ranks := range []int{1, 2, 4} {
				if ranks > cfg.NumKVHeads {
					continue
				}
				plan, err := NewTPPlan(cfg.NumKVHeads, ranks)
				if err != nil {
					t.Fatalf("plan ranks=%d: %v", ranks, err)
				}
				// Concatenate each rank's band along the head dim, in rank order.
				concat := make([][]float32, seq)
				for t2 := 0; t2 < seq; t2++ {
					concat[t2] = make([]float32, 0, nH*hd)
				}
				for _, s := range plan.Shards {
					band := attnBandForShard(s, grp)
					part, err := m.tpAttnBandPreProj(0, xn, rp, band)
					if err != nil {
						t.Fatalf("band %v: %v", band, err)
					}
					for t2 := 0; t2 < seq; t2++ {
						concat[t2] = append(concat[t2], part[t2]...)
					}
				}
				if exact, mx := bitExactRows(concat, full); !exact {
					t.Fatalf("[%s] ranks=%d concat band != full band: max|Δ|=%.3e (want bit-exact 0)", v.name, ranks, mx)
				}
			}
		})
	}
}

// TestTPForwardReductionRankOrderPinned pins the row-parallel o_proj / down AllReduce to a
// rank-order reference at max|Δ|=0 across rank counts — a reordered or dropped rank in the
// reduction (which the loose vs-monolith round-off bound cannot see) is caught here.
func TestTPForwardReductionRankOrderPinned(t *testing.T) {
	const seq = 5
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			cfg := m.Cfg
			xn := tpRows(seq, cfg.HiddenSize, uint64(len(v.name))*37+5)
			rp := newRopeForLayer(cfg, 0, seq)
			for _, ranks := range []int{1, 2, 4} {
				if ranks <= cfg.NumKVHeads {
					plan, _ := NewTPPlan(cfg.NumKVHeads, ranks)
					got, err := m.tpAttnLayer(0, xn, rp, plan, LocalCollective{})
					if err != nil {
						t.Fatalf("tpAttnLayer ranks=%d: %v", ranks, err)
					}
					ref, err := m.tpAttnLayerReference(0, xn, rp, plan)
					if err != nil {
						t.Fatalf("tpAttnLayerReference ranks=%d: %v", ranks, err)
					}
					if exact, mx := bitExactRows(got, ref); !exact {
						t.Fatalf("[%s] attn ranks=%d != rank-order ref: max|Δ|=%.3e", v.name, ranks, mx)
					}
				}
			}
			for _, ranks := range []int{1, 2, 3, 4, 8} {
				if ranks > cfg.IntermediateSize {
					continue
				}
				plan, _ := NewTPPlan(cfg.IntermediateSize, ranks)
				got, err := m.tpFFNLayer(0, xn, plan, LocalCollective{})
				if err != nil {
					t.Fatalf("tpFFNLayer ranks=%d: %v", ranks, err)
				}
				ref, err := m.tpFFNLayerReference(0, xn, plan)
				if err != nil {
					t.Fatalf("tpFFNLayerReference ranks=%d: %v", ranks, err)
				}
				if exact, mx := bitExactRows(got, ref); !exact {
					t.Fatalf("[%s] ffn ranks=%d != rank-order ref: max|Δ|=%.3e", v.name, ranks, mx)
				}
			}
		})
	}
}

// TestTPForwardWithinRoundOff confirms the multi-rank sharded sub-layers match the live
// attnSeq / mlpSeq within the documented AllReduce reassociation round-off (NOT bit-exact at
// ranks>1, by design — the o_proj/down contraction reassociates). The relative bound governs
// large coordinates; an absolute bound backstops near-zero ones. The 1e-3 bound matches
// cmd/tpcheck's "HIGH" threshold; this test only bounds the reassociation MAGNITUDE — the
// bit-exact rank-order guard is TestTPForwardReductionRankOrderPinned, so a reduction LOGIC
// error is caught there at max|Δ|=0, not hidden inside this magnitude bound.
func TestTPForwardWithinRoundOff(t *testing.T) {
	const seq = 6
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			cfg := m.Cfg
			xn := tpRows(seq, cfg.HiddenSize, uint64(len(v.name))*41+7)
			rp := newRopeForLayer(cfg, 0, seq)

			refAttn := m.attnSeq(0, xn, rp)
			for _, ranks := range []int{2, 4} {
				if ranks > cfg.NumKVHeads {
					continue
				}
				plan, _ := NewTPPlan(cfg.NumKVHeads, ranks)
				got, err := m.tpAttnLayer(0, xn, rp, plan, LocalCollective{})
				if err != nil {
					t.Fatalf("tpAttnLayer ranks=%d: %v", ranks, err)
				}
				rel, absSmall := maxRelAbsRows(got, refAttn)
				if rel > 1e-3 || absSmall > 1e-3 {
					t.Fatalf("[%s] attn ranks=%d vs attnSeq: rel=%.3e absSmall=%.3e exceeds round-off", v.name, ranks, rel, absSmall)
				}
			}

			refFFN := m.mlpSeq(0, xn)
			for _, ranks := range []int{2, 4, 8} {
				if ranks > cfg.IntermediateSize {
					continue
				}
				plan, _ := NewTPPlan(cfg.IntermediateSize, ranks)
				got, err := m.tpFFNLayer(0, xn, plan, LocalCollective{})
				if err != nil {
					t.Fatalf("tpFFNLayer ranks=%d: %v", ranks, err)
				}
				rel, absSmall := maxRelAbsRows(got, refFFN)
				if rel > 1e-3 || absSmall > 1e-3 {
					t.Fatalf("[%s] ffn ranks=%d vs mlpSeq: rel=%.3e absSmall=%.3e exceeds round-off", v.name, ranks, rel, absSmall)
				}
			}
		})
	}
}

// TestForwardTPMatchesForward is the end-to-end gate: a full ForwardTP over a multi-layer
// model reproduces Forward — bit-identical logits at ranks=1 (so it inherits the HF oracle),
// and across rank counts the same argmax at EVERY position with last-position logits within
// the accumulated AllReduce round-off. Run over the feature matrix.
func TestForwardTPMatchesForward(t *testing.T) {
	ids := []int{1, 2, 3, 4, 5, 6}
	combos := []struct{ attn, ffn int }{{1, 1}, {2, 2}, {4, 4}, {2, 8}, {4, 3}}
	for _, v := range tpFeatureVariants() {
		t.Run(v.name, func(t *testing.T) {
			m := tpBuildVariant(v)
			full := m.Forward(ids)
			for _, c := range combos {
				act, err := m.ForwardTP(ids, TPConfig{AttnRanks: c.attn, FFNRanks: c.ffn, Coll: LocalCollective{}})
				if err != nil {
					t.Fatalf("[%s] ForwardTP(a=%d,f=%d): %v", v.name, c.attn, c.ffn, err)
				}
				// argmax-identical at every position.
				for pos := range full.Logits {
					if argmax(act.Logits[pos]) != argmax(full.Logits[pos]) {
						t.Fatalf("[%s] a=%d f=%d pos %d argmax tp=%d forward=%d",
							v.name, c.attn, c.ffn, pos, argmax(act.Logits[pos]), argmax(full.Logits[pos]))
					}
				}
				exact, mxLogits := bitExactRows(act.Logits, full.Logits)
				if c.attn == 1 && c.ffn == 1 {
					if !exact {
						t.Fatalf("[%s] ForwardTP(ranks=1) logits != Forward: max|Δ|=%.3e (want bit-exact 0)", v.name, mxLogits)
					}
				} else if mxLogits > 5e-3 {
					t.Fatalf("[%s] a=%d f=%d logits max|Δ|=%.3e exceeds round-off bound", v.name, c.attn, c.ffn, mxLogits)
				}
			}
		})
	}
}

// TestForwardTPFailsClosed proves ForwardTP refuses every architecture whose TP decomposition
// it does not yet implement (rather than silently mis-serving it), and the f32-residency
// guard. Each case names the later sub-lever it belongs to.
func TestForwardTPFailsClosed(t *testing.T) {
	ids := []int{1, 2, 3}

	// MoE FFN.
	moe := tpFwdBaseCfg()
	moe.NumExperts = 4
	moe.NumExpertsPerTok = 2
	if _, err := (&Model{Cfg: moe}).ForwardTP(ids, TPConfig{}); err == nil {
		t.Fatalf("ForwardTP should reject MoE")
	}

	// DenseMLP activation MLP.
	dm := tpFwdBaseCfg()
	dm.DenseMLP = true
	if _, err := (&Model{Cfg: dm}).ForwardTP(ids, TPConfig{}); err == nil {
		t.Fatalf("ForwardTP should reject DenseMLP")
	}

	// ParallelResidual topology.
	pr := tpFwdBaseCfg()
	pr.BlockTopology = ParallelResidual
	if _, err := (&Model{Cfg: pr}).ForwardTP(ids, TPConfig{}); err == nil {
		t.Fatalf("ForwardTP should reject ParallelResidual")
	}

	// Linear-attention (GDN) layer.
	la := tpFwdBaseCfg()
	la.LayerTypes = []string{"linear_attention", "full_attention", "full_attention"}
	if _, err := (&Model{Cfg: la}).ForwardTP(ids, TPConfig{}); err == nil {
		t.Fatalf("ForwardTP should reject linear-attention layers")
	}

	// glm_moe_dsa (MLA+DSA).
	glm := tpFwdBaseCfg()
	glm.ModelType = "glm_moe_dsa"
	if _, err := (&Model{Cfg: glm}).ForwardTP(ids, TPConfig{}); err == nil {
		t.Fatalf("ForwardTP should reject glm_moe_dsa")
	}

	// f32-residency guard: a model whose q_proj tensor is absent must error (quant-aware TP
	// sharding is a later lever).
	missing := NewSynthetic(tpFwdBaseCfg())
	delete(missing.manifest, layerName(0, "self_attn.q_proj.weight"))
	if _, err := missing.ForwardTP(ids, TPConfig{AttnRanks: 2, FFNRanks: 2, Coll: LocalCollective{}}); err == nil {
		t.Fatalf("ForwardTP should reject a model missing an f32-resident projection")
	}
}

// TestForwardTPRankCountClamped confirms a requested rank count above the shardable dimension
// is clamped (not an error): asking for more attention ranks than nKV uses nKV.
func TestForwardTPRankCountClamped(t *testing.T) {
	m := NewSynthetic(tpFwdBaseCfg())
	ids := []int{1, 2, 3, 4}
	full := m.Forward(ids)
	act, err := m.ForwardTP(ids, TPConfig{AttnRanks: 64, FFNRanks: 1024, Coll: LocalCollective{}})
	if err != nil {
		t.Fatalf("ForwardTP with over-large ranks should clamp, got: %v", err)
	}
	for pos := range full.Logits {
		if argmax(act.Logits[pos]) != argmax(full.Logits[pos]) {
			t.Fatalf("clamped ForwardTP pos %d argmax mismatch", pos)
		}
	}
}
