package model

// family_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU numeric
// oracle for non-Llama families (#1271 Lane 1, support-maturity epic #1243).
//
// The gap it closes: family_conformance_test.go proves STRUCTURE (the forward runs,
// shapes are right) and oracle_test.go proves NUMERICS against a real HF export —
// but the export lives in .cache/ and SKIPs in CI, so a non-Llama family's numeric
// claim is asserted, not proven, in CI (covmatrix PROOF-PATH-ONLY, rung M3). This
// file is the M4 witness style the honesty fence names for the cpu cell: a numeric
// reference the production forward did NOT define, runnable on a bare CI box.
//
// Independence discipline: the reference below is a plain scalar transcription of
// the FAMILY's semantics as HuggingFace defines them (for OLMo2:
// transformers/models/olmo2/modeling_olmo2.py — q/k RMSNorm over the FULL projection
// width after projection and before RoPE, rotate_half RoPE, causal GQA softmax
// attention, SwiGLU MLP, and the Post-Norm placement x += norm(body(x)) using
// post_attention_layernorm / post_feedforward_layernorm). It deliberately reuses
// NONE of the production machinery: tensors are decoded from the manifest bytes
// directly, matmuls/norms/softmax are naive in-order scalar loops, and the block
// dataflow is hardcoded to the family's published topology rather than routed
// through cfg.BlockTopology. A wiring bug in derivation, norm routing, qk-norm
// width, RoPE, GQA grouping, or residual placement therefore diverges the two
// sides by O(0.01..1) — far above the reduction-order tolerance.
//
// The fixture is built with synthBuildRaw on the family's REAL tensor names (OLMo2
// has no input_layernorm; it carries post_attention + post_feedforward norms) and,
// unlike NewSynthetic, gives every norm weight a distinct NON-UNIT gain so weight
// application and norm-tensor routing are numerically live, not masked by 1.0.
//
// Adding a family = a fixture builder + a reference forward transcribed from that
// family's HF modeling file, then flip its covmatrix OracleInCI bit in the same
// commit. Do NOT flip the bit without the reference — that is exactly the
// self-reported promotion docs/standards/support-maturity-honesty-fence.md forbids.

import (
	"encoding/binary"
	"math"
	"testing"
)

// cpuOracleTol is the numeric gate between the production forward (fdot 8-accumulator
// reductions) and the in-order scalar reference. Reduction-order drift on these tiny
// dims is ~1e-6; any semantic divergence lands orders of magnitude above this.
const cpuOracleTol = 1e-4

// cpuOracleTensor decodes one named f32 tensor straight from the manifest bytes —
// independent of the production tensor() view path.
func cpuOracleTensor(t *testing.T, m *Model, name string) []float32 {
	t.Helper()
	meta, ok := m.manifest[name]
	if !ok {
		t.Fatalf("fixture tensor %q missing from manifest", name)
	}
	n := meta.Nbytes / 4
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(m.raw[meta.Offset+i*4:]))
	}
	return out
}

// cpuOracleMatVec is the naive HF Linear: y[o] = sum_i W[o][i] * x[i], W row-major [out,in].
func cpuOracleMatVec(w, x []float32, out, in int) []float32 {
	y := make([]float32, out)
	for o := 0; o < out; o++ {
		var s float32
		row := w[o*in : (o+1)*in]
		for i := 0; i < in; i++ {
			s += row[i] * x[i]
		}
		y[o] = s
	}
	return y
}

// cpuOracleRMSNorm is the plain HF RMSNorm: x / sqrt(mean(x^2)+eps) * w.
func cpuOracleRMSNorm(x, w []float32, eps float32) []float32 {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * inv * w[i]
	}
	return out
}

// cpuOracleRope rotates one head vector in place at position pos with HF's
// non-interleaved rotate_half convention: inv_freq[j] = theta^(-2j/hd).
func cpuOracleRope(hv []float32, pos, hd int, theta float64) {
	half := hd / 2
	for j := 0; j < half; j++ {
		angle := float64(pos) / math.Pow(theta, float64(2*j)/float64(hd))
		c, s := float32(math.Cos(angle)), float32(math.Sin(angle))
		a, b := hv[j], hv[j+half]
		hv[j] = a*c - b*s
		hv[j+half] = b*c + a*s
	}
}

func cpuOracleSoftmax(s []float32) {
	mx := s[0]
	for _, v := range s {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range s {
		e := float32(math.Exp(float64(v - mx)))
		s[i] = e
		sum += e
	}
	for i := range s {
		s[i] /= sum
	}
}

func cpuOracleSilu(z float32) float32 { return z / (1 + float32(math.Exp(float64(-z)))) }

// olmo2OracleCfg is the tiny OLMo2 fixture config. nH*hd (32) deliberately differs
// from HiddenSize (24) so a projection-width/hidden-width conflation cannot cancel.
func olmo2OracleCfg() Config {
	return Config{
		HiddenSize:        24,
		NumLayers:         3,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  40,
		VocabSize:         53,
		ModelType:         "olmo2",
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
}

// newOlmo2OracleModel builds the fixture on OLMo2's REAL tensor roster: q/k/v/o
// projections, FULL-projection-width q_norm/k_norm, post_attention_layernorm +
// post_feedforward_layernorm (and no input_layernorm — OLMo2 has none). Norm
// weights get distinct non-unit gains in (0.75, 1.25) so gain application is
// numerically observable; matmul weights use the shared deterministic LCG fill.
func newOlmo2OracleModel() *Model {
	cfg := olmo2OracleCfg()
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"model.embed_tokens.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		tensors = append(tensors,
			ts{p + "self_attn.q_proj.weight", []int{nH * hd, H}},
			ts{p + "self_attn.k_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.v_proj.weight", []int{nKV * hd, H}},
			ts{p + "self_attn.o_proj.weight", []int{H, nH * hd}},
			ts{p + "self_attn.q_norm.weight", []int{nH * hd}},
			ts{p + "self_attn.k_norm.weight", []int{nKV * hd}},
			ts{p + "post_attention_layernorm.weight", []int{H}},
			ts{p + "post_feedforward_layernorm.weight", []int{H}},
			ts{p + "mlp.gate_proj.weight", []int{I, H}},
			ts{p + "mlp.up_proj.weight", []int{I, H}},
			ts{p + "mlp.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors, ts{"model.norm.weight", []int{H}})

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		if isCPUOracleNormWeight(name) {
			return 1 + 0.25*next() // distinct non-unit gains, well-conditioned
		}
		return synthMatmulFill(name, next)
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func isCPUOracleNormWeight(name string) bool {
	for _, suffix := range []string{"layernorm.weight", "q_norm.weight", "k_norm.weight"} {
		if len(name) >= len(suffix) && name[len(name)-len(suffix):] == suffix {
			return true
		}
	}
	return name == "model.norm.weight"
}

// olmo2Reference runs the independent OLMo2 forward: per-position logits for ids.
// Every step below is the HF Olmo2 dataflow, hardcoded — NOT routed through
// cfg.BlockTopology or the production sublayer composition.
func olmo2Reference(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH, nKV, hd := cfg.NumHeads, cfg.NumKVHeads, cfg.HeadDim
	grp := nH / nKV
	eps := float32(cfg.RMSNormEps)
	theta := cfg.RopeTheta
	seq := len(ids)

	embed := cpuOracleTensor(t, m, "model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := layerPrefix(l)
		wq := cpuOracleTensor(t, m, p+"self_attn.q_proj.weight")
		wk := cpuOracleTensor(t, m, p+"self_attn.k_proj.weight")
		wv := cpuOracleTensor(t, m, p+"self_attn.v_proj.weight")
		wo := cpuOracleTensor(t, m, p+"self_attn.o_proj.weight")
		qnw := cpuOracleTensor(t, m, p+"self_attn.q_norm.weight")
		knw := cpuOracleTensor(t, m, p+"self_attn.k_norm.weight")
		postAttn := cpuOracleTensor(t, m, p+"post_attention_layernorm.weight")
		postFFN := cpuOracleTensor(t, m, p+"post_feedforward_layernorm.weight")
		wg := cpuOracleTensor(t, m, p+"mlp.gate_proj.weight")
		wu := cpuOracleTensor(t, m, p+"mlp.up_proj.weight")
		wd := cpuOracleTensor(t, m, p+"mlp.down_proj.weight")

		// --- attention sub-layer, Post-Norm: x += postAttnNorm(attn(x)) ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			// HF Olmo2Attention: q_norm/k_norm are RMSNorms over the FULL projection
			// width (num_heads*head_dim), applied after projection, before RoPE.
			q[tt] = cpuOracleRMSNorm(cpuOracleMatVec(wq, x[tt], nH*hd, H), qnw, eps)
			k[tt] = cpuOracleRMSNorm(cpuOracleMatVec(wk, x[tt], nKV*hd, H), knw, eps)
			v[tt] = cpuOracleMatVec(wv, x[tt], nKV*hd, H)
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
			for h := 0; h < nKV; h++ {
				cpuOracleRope(k[tt][h*hd:(h+1)*hd], tt, hd, theta)
			}
		}
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := k[j][kvh*hd : (kvh+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s * scale
				}
				cpuOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut := cpuOracleMatVec(wo, concat, H, nH*hd)
			nout := cpuOracleRMSNorm(attnOut, postAttn, eps)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}

		// --- MLP sub-layer, Post-Norm: x += postFFNNorm(SwiGLU(x)) ---
		for tt := 0; tt < seq; tt++ {
			gate := cpuOracleMatVec(wg, x[tt], I, H)
			up := cpuOracleMatVec(wu, x[tt], I, H)
			for i := 0; i < I; i++ {
				gate[i] = cpuOracleSilu(gate[i]) * up[i]
			}
			mlpOut := cpuOracleMatVec(wd, gate, H, I)
			nout := cpuOracleRMSNorm(mlpOut, postFFN, eps)
			for i := 0; i < H; i++ {
				x[tt][i] += nout[i]
			}
		}
	}

	norm := cpuOracleTensor(t, m, "model.norm.weight")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := cpuOracleRMSNorm(x[tt], norm, eps)
		logits[tt] = cpuOracleMatVec(embed, xf, V, H) // tied head: logits = E @ xf
	}
	return logits
}

func cpuOracleMaxAbsDiff(a, b []float32) float64 {
	var mx float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > mx {
			mx = d
		}
	}
	return mx
}

// TestOlmo2CPUNumericOracle is the OLMo2×cpu M4 witness: the production forward
// (Forward, and the cached Prefill/Step decode path) must reproduce the independent
// HF-semantics reference on every position within cpuOracleTol. This is the CI
// numeric oracle covmatrix.OracleInCI cites for OLMo2 — if this test reds, the
// honesty fence demotes the cell back to M3 (drop the covmatrix bit with it).
func TestOlmo2CPUNumericOracle(t *testing.T) {
	m := newOlmo2OracleModel()
	if err := m.Cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	// The derivation must land the published OLMo2 axes — the reference hardcodes them.
	if m.Cfg.BlockTopology != PostNorm {
		t.Fatalf("olmo2 derived topology = %v, want PostNorm", m.Cfg.BlockTopology)
	}
	if !m.Cfg.QKNorm {
		t.Fatal("olmo2 derived QKNorm = false, want true (OLMo2 always carries qk-norms)")
	}

	ids := []int{3, 17, 5, 23, 41, 2, 19}
	ref := olmo2Reference(t, m, ids)

	// Full-prefill Forward: every position must match the reference.
	act := m.Forward(ids)
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
			t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
		}
	}

	// Cached decode path: Prefill then Step must match the reference at the same
	// positions (the reference is cacheless, so Step(id) at position len(ids) is
	// compared against a reference run over the extended prompt).
	s := m.NewSession()
	pf := s.Prefill(ids)
	if d := cpuOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
		t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
	next := 11
	st := s.Step(next)
	extRef := olmo2Reference(t, m, append(append([]int(nil), ids...), next))
	if d := cpuOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
		t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
	}
}

// TestOlmo2CPUNumericOracleIsSensitive proves the oracle is non-vacuous: perturbing
// ONE weight element must move the compared logits by far more than the tolerance.
// A comparison that stays green under a perturbed fixture would be a fake witness.
func TestOlmo2CPUNumericOracleIsSensitive(t *testing.T) {
	m := newOlmo2OracleModel()
	if err := m.Cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	ref := olmo2Reference(t, m, ids)

	// Perturb one q_proj element of layer 0 in the RAW fixture bytes, then re-run
	// the production forward against the UNPERTURBED reference.
	meta := m.manifest["model.layers.0.self_attn.q_proj.weight"]
	orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[meta.Offset:]))
	binary.LittleEndian.PutUint32(m.raw[meta.Offset:], math.Float32bits(orig+0.5))

	act := m.Forward(ids)
	var worst float64
	for tt := range ids {
		if d := cpuOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous", worst)
	}
}
