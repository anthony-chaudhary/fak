package model

// v41_latent_norm_parity_test.go — the independent numeric oracle for the V4.1
// Q/KV latent norms through the FULL forward (fak#13322).
//
// WHY THIS EXISTS. The #13290 witness (v41_attn_latent_norm_test.go) proved the
// norms are APPLIED on the full path, but its scalar oracle calls the production
// helpers rmsnormCfg/matRows, so it is a production-vs-production tautology, and
// its full-geometry fixture carries UNIT latent-norm gains (v41RawFullFlattenedMHC
// fills attn.wq_a_norm.weight / attn.kv_norm.weight with 1.0). Unit gains cannot
// distinguish a correct order from a wrong one, nor a real norm from a no-op.
//
// This leaf closes that gap with a genuinely independent scalar full-forward
// oracle that
//
//  1. reads the fixture's own f32 bytes (cpuOracleTensor),
//  2. transcribes the FULL-path dataflow — flattened four-stream mHC projection,
//     the persistent residual streams, the causal sink contraction, the router
//     and the shared expert — using only the weight-free scalar primitives
//     cpuOracleMatVec / cpuOracleRMSNorm / cpuOracleRopeTailInterleaved and the
//     oracleV41* scalar kernels; never rmsnormCfg, matRows, v41ProjMatRows,
//     v41MHCProjectFull or V41SparseAttentionSink (the production machinery
//     under test), and
//  3. applies the pinned reference order
//         qr = q_norm(wq_a(x)) ; q = wq_b(qr)
//         kv = kv_norm(wkv(x)) ; rope on the tail
//     over an opt-in fixture whose latent-norm gains are nonuniform and non-unit,
//     so a missing norm, a mis-ordered norm or substituted unit gains each move
//     the expected logits beyond tolerance.
//
// The default reduced fixture (v41ReducedModel) and the unit-gain full fixture
// (v41RawFullFlattenedMHC) are left byte-for-byte unchanged; this file only adds
// an opt-in patched fixture and an independent oracle. Test-only, no production
// edit.

import (
	"encoding/binary"
	"math"
	"testing"
)

// v41LatentNormPatchedMHC returns v41RawFullFlattenedMHC with nonuniform,
// non-unit Q-latent and KV-latent gain vectors written into the two norm tensors.
// The gains are deterministic and strictly increasing, so a scalar oracle that
// drops or reorders the norm cannot coincidentally agree.
func v41LatentNormPatchedMHC(t *testing.T) *Model {
	t.Helper()
	m := v41RawFullFlattenedMHC(t)
	cfg := m.Cfg

	// Non-unit, nonuniform gains: a smooth ramp strictly inside (0.5, 1.5).
	qGain := make([]float32, cfg.QLoraRank)
	for i := range qGain {
		qGain[i] = float32(0.5 + float64(i)/float64(cfg.QLoraRank))
	}
	kvRank := v41KVLoraRank
	kvGain := make([]float32, kvRank)
	for i := range kvGain {
		kvGain[i] = float32(0.5 + float64(i)/float64(kvRank))
	}
	v41WriteTensorF32(t, m, layerName(0, "attn.wq_a_norm.weight"), qGain)
	v41WriteTensorF32(t, m, layerName(0, "attn.kv_norm.weight"), kvGain)

	// Guard the fixture itself: if either vector were all-ones the test would be
	// vacuous (it could not tell the norm from its absence).
	if v41AllOnes(qGain) || v41AllOnes(kvGain) {
		t.Fatalf("latent-norm gains are unit; the fixture is not discriminating")
	}
	return m
}

// v41WriteTensorF32 overwrites a manifest tensor's raw f32 elements with vals.
// It fails loudly on a width mismatch so a drifted fixture cannot silently
// partial-write. Test-only; changes no production layout.
func v41WriteTensorF32(t *testing.T, m *Model, name string, vals []float32) {
	t.Helper()
	meta, ok := m.manifest[name]
	if !ok {
		t.Fatalf("v41WriteTensorF32: missing tensor %s", name)
	}
	if meta.Nbytes != len(vals)*4 {
		t.Fatalf("v41WriteTensorF32: %s has %d bytes, want %d for %d values", name, meta.Nbytes, len(vals)*4, len(vals))
	}
	for i, v := range vals {
		binary.LittleEndian.PutUint32(m.raw[meta.Offset+i*4:], math.Float32bits(v))
	}
}

func v41AllOnes(v []float32) bool {
	for _, x := range v {
		if x != 1 {
			return false
		}
	}
	return true
}

// v41LatentNormOpts selects the negative control applied by the independent
// oracle; the zero value is the correct pinned reference.
type v41LatentNormOpts struct {
	omitQ     bool // skip the Q latent norm (the pre-#13290 producer)
	omitKV    bool // skip the KV latent norm
	qAfterB   bool // apply the Q norm AFTER wq_b instead of before (wrong order)
	unitGains bool // substitute all-ones gains (the norm as a no-op)
}

// v41OracleMHCProjectFull transcribes the flattened four-stream mHC mix
// projection: the four width-H streams laid end to end, ONE shared RMS over the
// whole 4H flattened residual, and the 24 mix coefficients as the projection
// scaled by that rsqrt. transposed selects the stored [4H, 24] orientation. It is
// a naive scalar transcription, not v41MHCProjectFull.
func v41OracleMHCProjectFull(wMix []float32, streams [][]float32, H int, eps float32, transposed bool) []float64 {
	flatWidth := 4 * H
	xflat := make([]float64, 0, flatWidth)
	for _, s := range streams {
		for _, v := range s {
			xflat = append(xflat, float64(v))
		}
	}
	var ss float64
	for _, v := range xflat {
		ss += v * v
	}
	rsqrt := 1 / math.Sqrt(ss/float64(flatWidth)+float64(eps))
	mixes := make([]float64, v41MHCMixWidth)
	for m := 0; m < v41MHCMixWidth; m++ {
		var s float64
		if transposed {
			for i := 0; i < flatWidth; i++ {
				s += float64(wMix[i*v41MHCMixWidth+m]) * xflat[i]
			}
		} else {
			row := wMix[m*flatWidth : (m+1)*flatWidth]
			for i := 0; i < flatWidth; i++ {
				s += float64(row[i]) * xflat[i]
			}
		}
		mixes[m] = s * rsqrt
	}
	return mixes
}

// v41OracleForwardLatentNorm is the independent scalar reference for the FULL
// V4.1 forward. See v41OracleForwardLatentNormHidden for the dataflow.
func v41OracleForwardLatentNorm(t *testing.T, m *Model, ids []int, opts v41LatentNormOpts) [][]float32 {
	t.Helper()
	logits, _ := v41OracleForwardLatentNormHidden(t, m, ids, opts)
	return logits
}

// v41OracleForwardLatentNormHidden is the independent scalar reference for the
// FULL V4.1 forward. It reproduces v41Layer's full path (flattened mHC
// projection, four persistent streams, causal sink contraction, router +
// routed/shared experts, post-mix written back into all four streams) and
// interposes the Q/KV latent norms at the pinned reference order. opts
// perturbations are for negative controls. It also returns the post-stack hidden
// state so a probe can localize a divergence to the layer body vs the head.
func v41OracleForwardLatentNormHidden(t *testing.T, m *Model, ids []int, opts v41LatentNormOpts) ([][]float32, [][]float32) {
	t.Helper()
	cfg := m.Cfg
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)

	tensor := func(name string) []float32 { return cpuOracleTensor(t, m, name) }

	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	// Persistent four-stream set per position: stream 0 carries the live hidden,
	// streams 1..3 the persistent residual (zero-initialized).
	streams := make([][][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
		set := make([][]float32, 4)
		set[0] = x[tt]
		for h := 1; h < 4; h++ {
			set[h] = make([]float32, H)
		}
		streams[tt] = set
	}

	qGain := tensor(layerName(0, "attn.wq_a_norm.weight"))
	kvGain := tensor(layerName(0, "attn.kv_norm.weight"))
	if len(qGain) != cfg.QLoraRank {
		t.Fatalf("fixture q latent norm width %d, want %d", len(qGain), cfg.QLoraRank)
	}
	if len(kvGain) != v41KVLoraRank {
		t.Fatalf("fixture kv latent norm width %d, want %d", len(kvGain), v41KVLoraRank)
	}
	gainOrUnit := func(w []float32) []float32 {
		if !opts.unitGains {
			return w
		}
		ones := make([]float32, len(w))
		for i := range ones {
			ones[i] = 1
		}
		return ones
	}

	scale := cfg.attnScale()
	hcIters := 1
	hcEps := float64(1e-6)
	if cfg.DeepSeekV41 != nil {
		if cfg.DeepSeekV41.HCSinkhornIters > 0 {
			hcIters = cfg.DeepSeekV41.HCSinkhornIters
		}
		if cfg.DeepSeekV41.HCEps > 0 {
			hcEps = cfg.DeepSeekV41.HCEps
		}
	}
	routeScale := cfg.RoutedScalingFactor
	if routeScale == 0 {
		routeScale = 1.5
	}

	for l := 0; l < cfg.NumLayers; l++ {
		ffnNorm := tensor(layerName(l, "ffn_norm.weight"))
		wMix := tensor(layerName(l, "mhc.mixes.weight"))
		mixBase := tensor(layerName(l, "mhc.base"))
		mixScale := tensor(layerName(l, "mhc.scale"))
		wQA := tensor(layerName(l, "attn.wq_a.weight"))
		wQB := tensor(layerName(l, "attn.wq_b.weight"))
		wKV := tensor(layerName(l, "attn.wkv.weight"))
		woA := tensor(layerName(l, "attn.wo_a.weight"))
		woB := tensor(layerName(l, "attn.wo_b.weight"))
		sink := tensor(layerName(l, "attn.sink"))
		wGate := tensor(layerName(l, "ffn.gate.weight"))
		gateBias := tensor(layerName(l, "ffn.gate.e_score_correction_bias"))
		shW1 := tensor(layerName(l, "ffn.shared_experts.w1.weight"))
		shW3 := tensor(layerName(l, "ffn.shared_experts.w3.weight"))
		shW2 := tensor(layerName(l, "ffn.shared_experts.w2.weight"))

		// ---- mHC split over the persistent flattened four-stream residual ----
		hcPre := make([][]float64, seq)
		hcPost := make([][]float64, seq)
		hcComb := make([][]float64, seq)
		preByPos := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			mixes := v41OracleMHCProjectFull(wMix, streams[tt], H, eps, true)
			p, po, c := oracleV41MHCKernel(mixes, toF64(mixScale), toF64(mixBase), 4, hcIters, hcEps)
			hcPre[tt], hcPost[tt], hcComb[tt] = p, po, c
			streams4 := [][]float64{toF64(streams[tt][0]), toF64(streams[tt][1]), toF64(streams[tt][2]), toF64(streams[tt][3])}
			preByPos[tt] = toF32(oracleV41MHCPre(streams4, p))
		}

		// ---- attention projections with the latent norms ----
		qHeads := make([][]float32, seq)
		kvRows := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			c := preByPos[tt]
			qLat := cpuOracleMatVec(wQA, c, cfg.QLoraRank, H)
			var q []float32
			switch {
			case opts.qAfterB:
				// Wrong order: project first, then norm the up-projected query.
				q = cpuOracleMatVec(wQB, qLat, nH*hd, cfg.QLoraRank)
				q = cpuOracleRMSNorm(q, gainOrUnit(v41GainPad(qGain, nH*hd)), eps)
			default:
				if !opts.omitQ {
					qLat = cpuOracleRMSNorm(qLat, gainOrUnit(qGain), eps)
				}
				q = cpuOracleMatVec(wQB, qLat, nH*hd, cfg.QLoraRank)
			}
			// kv = kv_norm(wkv(x)) at the published latent rank, sliced to hd.
			kv := cpuOracleMatVec(wKV, c, v41KVLoraRank, H)
			if !opts.omitKV {
				kv = cpuOracleRMSNorm(kv, gainOrUnit(kvGain), eps)
			}
			kv = kv[:hd]
			// Mirror v41RopeTableForLayer: the table is sized to QKRopeHeadDim, uses
			// the PER-LAYER theta, and folds in the YaRN attention-factor rescale.
			// (The reduced fixture pins these off; the published full config does
			// not, so a bare theta table would rotate a different vector.)
			cos, sin := v41OracleRopeTable(t, cfg, l, tt)
			for h := 0; h < nH; h++ {
				v41OracleRopeTailInterleaved(q[h*hd:(h+1)*hd], cos, sin, cfg.QKRopeHeadDim)
			}
			v41OracleRopeTailInterleaved(kv, cos, sin, cfg.QKRopeHeadDim)
			qHeads[tt] = q
			kvRows[tt] = kv
		}

		// ---- causal sink contraction + grouped output ----
		// Mirror V41SparseAttentionSink's PRECISION: the score dot and the
		// weighted value accumulation are float32 (the production kernel keeps
		// f32), so the independent oracle must not silently compute in float64
		// and drift. Only the exp is promoted through float64, exactly as exp32.
		attnOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			rows := tt + 1
			o := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				qh := qHeads[tt][h*hd : (h+1)*hd]
				maxScore := sink[h]
				dots := make([]float32, rows)
				for i := 0; i < rows; i++ {
					var d float32
					for j := 0; j < hd; j++ {
						d += qh[j] * kvRows[i][j]
					}
					d *= scale
					dots[i] = d
					if d > maxScore {
						maxScore = d
					}
				}
				sum := float32(math.Exp(float64(sink[h] - maxScore)))
				for i := 0; i < rows; i++ {
					sum += float32(math.Exp(float64(dots[i] - maxScore)))
				}
				if sum == 0 {
					continue
				}
				for i := 0; i < rows; i++ {
					w := float32(math.Exp(float64(dots[i]-maxScore))) / sum
					for j := 0; j < hd; j++ {
						o[h*hd+j] += w * kvRows[i][j]
					}
				}
			}
			attnOut[tt] = v41OracleGroupedOutput(o, woA, woB, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
		}

		// ---- router + routed/shared experts, then the mHC post-mix ----
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], ffnNorm, eps)
			router := cpuOracleMatVec(wGate, xn, cfg.NumExperts, H)
			picks, weights := oracleV41Route(toF64(router), toF64(gateBias), cfg.NumExpertsPerTok, routeScale)
			routed := make([]float64, H)
			for pi, e := range picks {
				stem := "ffn.experts." + itoa(e)
				w1 := tensor(layerName(l, stem+".w1.weight"))
				w3 := tensor(layerName(l, stem+".w3.weight"))
				w2 := tensor(layerName(l, stem+".w2.weight"))
				y := oracleSwiGLU(w1, w3, w2, xn, cfg.MoEIntermediateSize, H)
				for d := range routed {
					routed[d] += weights[pi] * y[d]
				}
			}
			shared := oracleSwiGLU(shW1, shW3, shW2, xn, cfg.MoEIntermediateSize, H)
			moe := oracleV41SharedExpertAdd(routed, shared)

			delta := make([]float64, H)
			for i := 0; i < H; i++ {
				delta[i] = float64(attnOut[tt][i]) + moe[i]
			}
			residual := [][]float64{
				toF64(streams[tt][0]), toF64(streams[tt][1]),
				toF64(streams[tt][2]), toF64(streams[tt][3]),
			}
			next := oracleV41MHCPost(delta, residual, hcPost[tt], hcComb[tt], 4, false)
			// Full path writes ALL FOUR post-mix streams back; stream 0 is the
			// live hidden. (oracleV41MHCPost returns []float64; the production
			// path stores f32, so round through f32 at the write-back boundary.)
			for h := 0; h < 4; h++ {
				copy(streams[tt][h], toF32(next[h]))
			}
			copy(x[tt], toF32(next[0]))
		}
	}

	norm := tensor("model.norm.weight")
	head := tensor("lm_head.weight")
	logits := make([][]float32, seq)
	for tt := 0; tt < seq; tt++ {
		xf := cpuOracleRMSNorm(x[tt], norm, eps)
		logits[tt] = cpuOracleMatVec(head, xf, cfg.VocabSize, H)
	}
	return logits, x
}

// v41OracleHiddenAfterLayers returns the post-stack hidden state the oracle would
// feed into the final norm, for a probe that localizes a divergence.
func v41OracleHiddenAfterLayers(t *testing.T, m *Model, ids []int) []float32 {
	t.Helper()
	_, x := v41OracleForwardLatentNormHidden(t, m, ids, v41LatentNormOpts{})
	return x[0]
}

// v41OracleRopeTable is the independent transcription of the V4.1 rotary table:
// QKRopeHeadDim/2 frequencies over the rope width, the per-layer theta, and the
// YaRN attention-factor rescale. It is deliberately NOT a call to
// v41RopeTableForLayer.
func v41OracleRopeTable(t *testing.T, cfg Config, layer, p int) (cos, sin []float32) {
	t.Helper()
	theta := cfg.RopeTheta
	if layer >= 0 && layer < len(cfg.RopeThetaPerLayer) && cfg.RopeThetaPerLayer[layer] != 0 {
		theta = cfg.RopeThetaPerLayer[layer]
	}
	scale := cfg.ropeAttentionFactor()
	n := cfg.QKRopeHeadDim / 2
	cos = make([]float32, n)
	sin = make([]float32, n)
	for j := 0; j < n; j++ {
		a := float64(p) / math.Pow(theta, float64(2*j)/float64(cfg.QKRopeHeadDim))
		cv := float32(math.Cos(a))
		sv := float32(math.Sin(a))
		if scale != 0 && scale != 1 {
			s := float32(scale)
			cv *= s
			sv *= s
		}
		cos[j] = cv
		sin[j] = sv
	}
	return cos, sin
}

// v41OracleRopeTailInterleaved rotates only the last ropeDim components of hv in
// the adjacent-pair (complex) convention, consuming the cos/sin table. It is the
// independent transcription of the reference rotate, not applyRopeTailInterleaved.
func v41OracleRopeTailInterleaved(hv, cos, sin []float32, ropeDim int) {
	tail := len(hv) - ropeDim
	for j := 0; j < ropeDim/2; j++ {
		i := tail + 2*j
		a, b := hv[i], hv[i+1]
		// Pin each product to f32 (no FMA fusion), matching the production
		// kernel's bit-determinism guarantee.
		hv[i] = float32(a*cos[j]) - float32(b*sin[j])
		hv[i+1] = float32(b*cos[j]) + float32(a*sin[j])
	}
}

// v41GainPad extends a q-lora-width gain vector to width by cycling, for the
// qAfterB negative control (a wrong-order oracle still needs a well-formed gain
// of the up-projected query width).
func v41GainPad(g []float32, width int) []float32 {
	out := make([]float32, width)
	for i := range out {
		out[i] = g[i%len(g)]
	}
	return out
}

// TestV41ForwardParityLatentNorm is the #13322 acceptance witness. It drives the
// REAL Model.Forward over an opt-in full-geometry fixture whose Q/KV latent-norm
// gains are nonuniform and non-unit, and requires the logits to agree with an
// independent scalar oracle that reuses none of the production forward machinery.
// Negative controls inside the oracle (omit Q norm, omit KV norm, wrong order,
// unit gains) must each move the expected logits beyond tolerance, proving the
// agreement is not vacuous.
func TestV41ForwardParityLatentNorm(t *testing.T) {
	m := v41LatentNormPatchedMHC(t)
	prompts := [][]int{{0, 1}, {2, 3, 5}}

	t.Run("forward matches independent oracle on nonuniform gains", func(t *testing.T) {
		for pi, ids := range prompts {
			act := m.Forward(ids)
			if act == nil || len(act.Logits) != len(ids) {
				t.Fatalf("prompt %d: Forward returned %v positions, want %d", pi, act, len(ids))
			}
			want := v41OracleForwardLatentNorm(t, m, ids, v41LatentNormOpts{})
			for tPos := range ids {
				v41LogitsClose(t, "latent-norm-parity", act.Logits[tPos], want[tPos])
			}
		}
	})

	t.Run("negative controls diverge beyond tolerance", func(t *testing.T) {
		ids := prompts[0]
		good := sysFlatten(v41OracleForwardLatentNorm(t, m, ids, v41LatentNormOpts{}))
		controls := []struct {
			name string
			opts v41LatentNormOpts
		}{
			{"omit Q latent norm", v41LatentNormOpts{omitQ: true}},
			{"omit KV latent norm", v41LatentNormOpts{omitKV: true}},
			{"apply Q norm after wq_b", v41LatentNormOpts{qAfterB: true}},
			{"substitute unit gains", v41LatentNormOpts{unitGains: true}},
		}
		for _, ctl := range controls {
			bad := sysFlatten(v41OracleForwardLatentNorm(t, m, ids, ctl.opts))
			if !v41LogitsDiverge(good, bad, cpuOracleTol) {
				t.Fatalf("negative control %q did not move the expected logits beyond tol %.0e; the parity check is not discriminating", ctl.name, cpuOracleTol)
			}
		}
	})
}

// sysFlatten flattens a per-position logit matrix for the divergence check.
func sysFlatten(rows [][]float32) []float32 {
	var out []float32
	for _, r := range rows {
		out = append(out, r...)
	}
	return out
}

// v41LogitsDiverge reports whether any element of a differs from b by more than
// tol (the inverse of v41LogitsClose, used to require a negative control to move).
func v41LogitsDiverge(a, b []float32, tol float64) bool {
	if len(a) != len(b) {
		return true
	}
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > tol {
			return true
		}
	}
	return false
}
