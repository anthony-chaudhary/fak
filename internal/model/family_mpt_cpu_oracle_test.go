package model

// family_mpt_cpu_oracle_test.go — the weight-free, checkpoint-independent CPU numeric
// oracle for the MPT family (#1271 Lane 1, support-maturity epic #1243). It is the
// per-family sibling of family_cpu_oracle_test.go, whose header states the doctrine:
// a numeric reference the production forward did NOT define, runnable on a bare CI
// box, so the MPT×cpu covmatrix cell can rise from PROOF-PATH-ONLY (M3) to matured
// (M4). Flipping covmatrix's OracleInCI bit without this reference is exactly the
// self-promotion docs/standards/support-maturity-honesty-fence.md forbids.
//
// Independence discipline (family_cpu_oracle_test.go:13-24): the reference below
// reuses NONE of the production machinery. Tensors are decoded from the manifest
// bytes; matmul / LayerNorm / GELU / softmax / the ALiBi slope table are naive
// in-order scalar loops written here; and the block dataflow is hardcoded to MPT's
// published topology rather than routed through cfg.BlockTopology, normCfg, act(),
// alibiScoreBias, or the FFN dispatch. Not one production helper is called from
// mptReference.
//
// MPT semantics transcribed from HuggingFace transformers/models/mpt/modeling_mpt.py:
//
//   - MptBlock (PreNorm, sequential residual):
//     h = h + attn(norm_1(h));  h = h + ffn(norm_2(h))
//     — norm_1/norm_2 are nn.LayerNorm with `.bias = None` (MPT is a no-bias model:
//     MptConfig.no_bias defaults True), so the norm is MEAN-SUBTRACTING with a gain
//     and no shift. This is the axis that separates MPT from every RMSNorm family.
//   - MptAttention: ONE fused `Wqkv` nn.Linear(hidden, 3*hidden, bias=False), then
//     `mixed_qkv.chunk(3, dim=2)` → q = rows [0,H), k = rows [H,2H), v = rows [2H,3H);
//     each reshaped to (n_heads, head_dim). Default MPT is multi-head (n_kv_heads ==
//     n_heads, head_dim = hidden / n_heads). softmax_scale defaults to
//     1/sqrt(hidden/n_heads) = 1/sqrt(head_dim). out_proj is bias-free.
//   - Positional encoding is ALiBi, NOT RoPE: build_mpt_alibi_tensor gives
//     slope_h = 2^-((idx+1) * alibi_bias_max / pow2), pow2 = 2^ceil(log2(n_heads)),
//     alibi_bias_max default 8, and — when n_heads is not a power of two — the slope
//     vector is reordered `concat(slopes[1::2], slopes[::2])[:n_heads]`. The additive
//     score bias is slope_h * (key - key_length + 1). No rotation is applied to q/k.
//   - MptMLP is a DENSE activation MLP, not SwiGLU: down_proj(GELU(up_proj(x))) with
//     nn.GELU(approximate="none"), i.e. the exact erf GELU. There is no gate branch.
//   - Head: transformer.norm_f LayerNorm, then lm_head TIED to transformer.wte.
//
// Fixture: built with synthBuildRaw on MPT's REAL checkpoint vocabulary
// (transformer.blocks.N.{norm_1,norm_2,attn.Wqkv,attn.out_proj,ffn.up_proj,
// ffn.down_proj} + transformer.wte / transformer.norm_f), then handed to the LOADER's
// rename+split (materializeMPTTensors, splitFusedProjections) so the production
// forward sees the canonical names it reads. The reference deliberately reads the
// ORIGINAL transformer.* rows and does its own Wqkv chunk, which puts MPT's fused-QKV
// split and its whole rename map inside the compared surface. Every norm weight gets
// a distinct NON-UNIT gain so norm routing is numerically live, not masked by 1.0.
//
// Two lineage fixtures run through the same reference: n_heads=4 (a power of two, the
// mpt-7b/30b shape) and n_heads=6 (not a power of two, which is the only way to make
// build_mpt_alibi_tensor's slope REORDER branch numerically live).

import (
	"encoding/binary"
	"math"
	"strings"
	"testing"
)

// mptOracleAlibiBiasMax is HF MptConfig.attn_config.alibi_bias_max's default. It is
// hardcoded here (not read from cfg) so the production default-when-zero is compared,
// not shared.
const mptOracleAlibiBiasMax = 8.0

// mptOracleLayerNorm is the plain torch nn.LayerNorm MPT uses, with bias=None:
// (x - mean) / sqrt(var + eps) * w, var the BIASED (1/n) variance. Naive in-order.
func mptOracleLayerNorm(x, w []float32, eps float32) []float32 {
	var mean float32
	for _, v := range x {
		mean += v
	}
	mean /= float32(len(x))
	var ss float32
	for _, v := range x {
		d := v - mean
		ss += d * d
	}
	inv := float32(1.0 / math.Sqrt(float64(ss/float32(len(x))+eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = (v - mean) * inv * w[i]
	}
	return out
}

// mptOracleGelu is nn.GELU(approximate="none"): 0.5*x*(1+erf(x/sqrt(2))).
func mptOracleGelu(z float32) float32 {
	z64 := float64(z)
	return float32(0.5 * z64 * (1 + math.Erf(z64/math.Sqrt2)))
}

// mptOracleAlibiSlopes transcribes build_mpt_alibi_tensor's per-head slope vector:
//
//	pow2   = 2^ceil(log2(n_heads))
//	base   = [1..pow2] * (alibi_bias_max / pow2)
//	slopes = 1 / 2^base
//	if pow2 != n_heads: slopes = concat(slopes[1::2], slopes[::2])[:n_heads]
func mptOracleAlibiSlopes(nHeads int, biasMax float64) []float64 {
	pow2 := 1
	for pow2 < nHeads {
		pow2 *= 2
	}
	full := make([]float64, pow2)
	for i := 0; i < pow2; i++ {
		full[i] = 1.0 / math.Pow(2, float64(i+1)*(biasMax/float64(pow2)))
	}
	if pow2 == nHeads {
		return full
	}
	reordered := make([]float64, 0, pow2)
	for i := 1; i < pow2; i += 2 {
		reordered = append(reordered, full[i])
	}
	for i := 0; i < pow2; i += 2 {
		reordered = append(reordered, full[i])
	}
	return reordered[:nHeads]
}

// mptOracleMatVec is the naive HF Linear: y[o] = sum_i W[o][i]*x[i], W row-major [out,in].
// (Deliberately a local copy, not the shared cpuOracleMatVec, so this file's reference is
// self-contained and cannot be perturbed from another family's lane.)
func mptOracleMatVec(w, x []float32, out, in int) []float32 {
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

func mptOracleSoftmax(s []float32) {
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

// mptOracleCfg is a tiny MPT config in MPT's own terms: d_model = n_heads*head_dim
// (MPT derives head_dim, so the fused Wqkv is exactly [3*d_model, d_model]) and
// ffn hidden = expansion_ratio(4) * d_model. Only the keys a real MPT config.json
// carries are set — LayerNorm / DenseMLP / ActGeluErf / Alibi / head_dim / n_kv_heads
// are left for deriveConfigAxes, so the derivation itself is under test.
func mptOracleCfg(nHeads, headDim int) Config {
	hidden := nHeads * headDim
	return Config{
		HiddenSize:        hidden,
		NumLayers:         3,
		NumHeads:          nHeads,
		IntermediateSize:  4 * hidden,
		VocabSize:         53,
		ModelType:         "mpt",
		RMSNormEps:        1e-5, // MPT layer_norm_epsilon
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	}
}

func mptOracleIsNormWeight(name string) bool {
	return strings.HasSuffix(name, "norm_1.weight") ||
		strings.HasSuffix(name, "norm_2.weight") ||
		name == "transformer.norm_f.weight"
}

func mptOracleBlockPrefix(l int) string { return "transformer.blocks." + itoa(l) + "." }

// newMPTOracleModel builds the fixture on MPT's REAL checkpoint tensor names and then
// runs the loader's MPT rename + fused-QKV split so the manifest also carries the
// canonical rows the production forward reads. Both name sets alias the SAME raw bytes,
// so the reference (which reads the transformer.* rows) and production (which reads the
// canonical rows) are looking at identical numbers through different names — any drift
// in the rename map or the Wqkv cut shows up as a numeric divergence.
func newMPTOracleModel(t *testing.T, nHeads, headDim int) *Model {
	t.Helper()
	cfg := mptOracleCfg(nHeads, headDim)
	if err := cfg.deriveConfigAxes(configJSONHints{}); err != nil {
		t.Fatalf("deriveConfigAxes: %v", err)
	}
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize

	type ts = synthTensor
	var tensors []ts
	tensors = append(tensors, ts{"transformer.wte.weight", []int{V, H}})
	for l := 0; l < cfg.NumLayers; l++ {
		p := mptOracleBlockPrefix(l)
		tensors = append(tensors,
			ts{p + "norm_1.weight", []int{H}},
			ts{p + "attn.Wqkv.weight", []int{3 * H, H}},
			ts{p + "attn.out_proj.weight", []int{H, H}},
			ts{p + "norm_2.weight", []int{H}},
			ts{p + "ffn.up_proj.weight", []int{I, H}},
			ts{p + "ffn.down_proj.weight", []int{H, I}},
		)
	}
	tensors = append(tensors, ts{"transformer.norm_f.weight", []int{H}})

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case mptOracleIsNormWeight(name):
			return 1 + 0.25*next() // distinct non-unit gains, well-conditioned
		case name == "transformer.wte.weight":
			return next() * 0.2 // wider so distinct ids separate cleanly
		}
		return next() * 0.1
	})

	// The loader's MPT path: transformer.* -> canonical names, then the fused
	// Wqkv row-range cut into q/k/v. Pure manifest surgery; raw is untouched.
	if err := materializeMPTTensors(cfg, man); err != nil {
		t.Fatalf("materializeMPTTensors: %v", err)
	}
	if err := splitFusedProjections(cfg, man); err != nil {
		t.Fatalf("splitFusedProjections: %v", err)
	}
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

// mptReference runs the independent MPT forward and returns per-position logits.
// Every step is the HF MptModel dataflow written out by hand: no production norm,
// activation, attention, FFN-dispatch or topology helper is touched.
func mptReference(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, I, V := cfg.HiddenSize, cfg.IntermediateSize, cfg.VocabSize
	nH := cfg.NumHeads
	hd := H / nH // MPT: head_dim = d_model / n_heads
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)

	slopes := mptOracleAlibiSlopes(nH, mptOracleAlibiBiasMax)
	// HF applies the alibi row built for the CURRENT key length to every query
	// (the tensor's query axis is broadcast), i.e. bias = slope*(key - keyLen + 1).
	keyLen := seq

	embed := cpuOracleTensorMPT(t, m, "transformer.wte.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		p := mptOracleBlockPrefix(l)
		norm1 := cpuOracleTensorMPT(t, m, p+"norm_1.weight")
		wqkv := cpuOracleTensorMPT(t, m, p+"attn.Wqkv.weight")
		wo := cpuOracleTensorMPT(t, m, p+"attn.out_proj.weight")
		norm2 := cpuOracleTensorMPT(t, m, p+"norm_2.weight")
		wup := cpuOracleTensorMPT(t, m, p+"ffn.up_proj.weight")
		wdown := cpuOracleTensorMPT(t, m, p+"ffn.down_proj.weight")

		// --- attention sub-layer, PreNorm: h += attn(norm_1(h)) ---
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			xn := mptOracleLayerNorm(x[tt], norm1, eps)
			// ONE fused projection, then chunk(3): q | k | v, each d_model wide.
			mixed := mptOracleMatVec(wqkv, xn, 3*H, H)
			q[tt] = append([]float32(nil), mixed[0:H]...)
			k[tt] = append([]float32(nil), mixed[H:2*H]...)
			v[tt] = append([]float32(nil), mixed[2*H:3*H]...)
			// No RoPE: MPT is ALiBi-positional, q/k leave the projection unrotated.
		}
		scale := float32(1.0 / math.Sqrt(float64(hd)))
		for tt := 0; tt < seq; tt++ {
			concat := make([]float32, H)
			for h := 0; h < nH; h++ {
				qh := q[tt][h*hd : (h+1)*hd]
				scores := make([]float32, tt+1)
				for j := 0; j <= tt; j++ {
					kh := k[j][h*hd : (h+1)*hd]
					var s float32
					for d := 0; d < hd; d++ {
						s += qh[d] * kh[d]
					}
					scores[j] = s*scale + float32(slopes[h]*float64(j-keyLen+1))
				}
				mptOracleSoftmax(scores)
				o := concat[h*hd : (h+1)*hd]
				for j := 0; j <= tt; j++ {
					vh := v[j][h*hd : (h+1)*hd]
					for d := 0; d < hd; d++ {
						o[d] += scores[j] * vh[d]
					}
				}
			}
			attnOut := mptOracleMatVec(wo, concat, H, H)
			for i := 0; i < H; i++ {
				x[tt][i] += attnOut[i]
			}
		}

		// --- MLP sub-layer, PreNorm: h += down(gelu(up(norm_2(h)))) (no gate) ---
		for tt := 0; tt < seq; tt++ {
			xn := mptOracleLayerNorm(x[tt], norm2, eps)
			up := mptOracleMatVec(wup, xn, I, H)
			for i := 0; i < I; i++ {
				up[i] = mptOracleGelu(up[i])
			}
			mlpOut := mptOracleMatVec(wdown, up, H, I)
			for i := 0; i < H; i++ {
				x[tt][i] += mlpOut[i]
			}
		}
	}

	normF := cpuOracleTensorMPT(t, m, "transformer.norm_f.weight")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := mptOracleLayerNorm(x[tt], normF, eps)
		logits[tt] = mptOracleMatVec(embed, xf, V, H) // lm_head tied to wte
	}
	return logits
}

// cpuOracleTensorMPT decodes one named f32 tensor straight from the manifest bytes,
// independent of the production tensor() view path. (Local copy of the shared decoder
// so this file stands alone.)
func cpuOracleTensorMPT(t *testing.T, m *Model, name string) []float32 {
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

func mptOracleMaxAbsDiff(a, b []float32) float64 {
	var mx float64
	for i := range a {
		d := math.Abs(float64(a[i] - b[i]))
		if d > mx {
			mx = d
		}
	}
	return mx
}

// TestMPTCPUNumericOracle is the MPT×cpu M4 witness: the production forward (Forward,
// and the cached Prefill/Step decode path) must reproduce the independent HF-semantics
// reference at every position within cpuOracleTol. This is the CI numeric oracle
// covmatrix.OracleInCI cites for MPT — if it reds, the honesty fence demotes the cell
// back to M3 (drop the covmatrix bit with it).
func TestMPTCPUNumericOracle(t *testing.T) {
	for _, lineage := range []struct {
		name    string
		nHeads  int
		headDim int
	}{
		// n_heads a power of two: the mpt-7b (32) / mpt-30b (64) shape.
		{"heads_pow2", 4, 8},
		// n_heads NOT a power of two: the only shape that makes
		// build_mpt_alibi_tensor's slope-reorder branch numerically live.
		{"heads_nonpow2", 6, 4},
	} {
		t.Run(lineage.name, func(t *testing.T) {
			m := newMPTOracleModel(t, lineage.nHeads, lineage.headDim)
			// The derivation must land the published MPT axes — the reference hardcodes them.
			if m.Cfg.BlockTopology != PreNorm {
				t.Fatalf("mpt derived topology = %v, want PreNorm", m.Cfg.BlockTopology)
			}
			if !m.Cfg.LayerNorm {
				t.Fatal("mpt derived LayerNorm = false, want true (norm_1/norm_2 are nn.LayerNorm)")
			}
			if !m.Cfg.DenseMLP {
				t.Fatal("mpt derived DenseMLP = false, want true (MptMLP has no gate branch)")
			}
			if !m.Cfg.ActGeluErf {
				t.Fatal("mpt derived ActGeluErf = false, want true (nn.GELU(approximate=\"none\"))")
			}
			if !m.Cfg.Alibi {
				t.Fatal("mpt derived Alibi = false, want true (MPT is ALiBi, not RoPE)")
			}
			if m.Cfg.NumKVHeads != lineage.nHeads {
				t.Fatalf("mpt derived NumKVHeads = %d, want %d (default MPT is multi-head)", m.Cfg.NumKVHeads, lineage.nHeads)
			}
			if m.Cfg.HeadDim != lineage.headDim {
				t.Fatalf("mpt derived HeadDim = %d, want %d (d_model/n_heads)", m.Cfg.HeadDim, lineage.headDim)
			}

			ids := []int{3, 17, 5, 23, 41, 2, 19}
			ref := mptReference(t, m, ids)

			// Full-prefill Forward: every position must match the reference.
			act := m.Forward(ids)
			for tt := range ids {
				if d := mptOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > cpuOracleTol {
					t.Errorf("Forward logits pos %d: max|delta| vs reference = %.3e (tol %.0e)", tt, d, cpuOracleTol)
				}
			}

			// Cached decode path: Prefill then Step must match the reference at the same
			// positions (the reference is cacheless, so Step(id) at position len(ids) is
			// compared against a reference run over the extended prompt).
			s := m.NewSession()
			pf := s.Prefill(ids)
			if d := mptOracleMaxAbsDiff(pf, ref[len(ids)-1]); d > cpuOracleTol {
				t.Errorf("Prefill last logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
			}
			next := 11
			st := s.Step(next)
			extRef := mptReference(t, m, append(append([]int(nil), ids...), next))
			if d := mptOracleMaxAbsDiff(st, extRef[len(ids)]); d > cpuOracleTol {
				t.Errorf("Step logits: max|delta| vs reference = %.3e (tol %.0e)", d, cpuOracleTol)
			}
		})
	}
}

// TestMPTCPUNumericOracleIsSensitive proves the MPT reference comparison is
// non-vacuous: perturbing ONE norm_1 LayerNorm gain in the raw fixture bytes, then
// re-running the production forward against the UNPERTURBED reference, must move the
// compared logits far above the tolerance. A comparison that stayed green under a
// perturbed fixture would be a fake witness.
func TestMPTCPUNumericOracleIsSensitive(t *testing.T) {
	m := newMPTOracleModel(t, 4, 8)
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	ref := mptReference(t, m, ids)

	meta := m.manifest["transformer.blocks.0.norm_1.weight"]
	orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[meta.Offset:]))
	binary.LittleEndian.PutUint32(m.raw[meta.Offset:], math.Float32bits(orig+0.5))

	act := m.Forward(ids)
	var worst float64
	for tt := range ids {
		if d := mptOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("perturbed fixture still within tolerance (max|delta|=%.3e) — the oracle is vacuous", worst)
	}
}

// TestMPTCPUNumericOracleWqkvSplitIsLive proves the fused-QKV surface is actually
// inside the compared region: the reference chunks transformer.*.attn.Wqkv itself,
// production reads the loader's split q/k/v views, so perturbing ONE byte inside the
// k-chunk of the fused tensor (which the reference reads under a different name) must
// diverge. This is the MPT-specific rename+split axis no other family exercises.
func TestMPTCPUNumericOracleWqkvSplitIsLive(t *testing.T) {
	m := newMPTOracleModel(t, 4, 8)
	H := m.Cfg.HiddenSize
	ids := []int{3, 17, 5, 23, 41, 2, 19}
	ref := mptReference(t, m, ids)

	// Row H is the first row of the k chunk (rows [0,H) q, [H,2H) k, [2H,3H) v).
	meta := m.manifest["transformer.blocks.0.attn.Wqkv.weight"]
	off := meta.Offset + H*H*4
	orig := math.Float32frombits(binary.LittleEndian.Uint32(m.raw[off:]))
	binary.LittleEndian.PutUint32(m.raw[off:], math.Float32bits(orig+0.5))

	act := m.Forward(ids)
	var worst float64
	for tt := range ids {
		if d := mptOracleMaxAbsDiff(act.Logits[tt], ref[tt]); d > worst {
			worst = d
		}
	}
	if worst <= cpuOracleTol {
		t.Fatalf("perturbing the fused Wqkv k-chunk left the comparison green (max|delta|=%.3e) — the split is not in the compared surface", worst)
	}
}
