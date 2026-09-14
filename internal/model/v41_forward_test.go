package model

// v41_forward_test.go — the independent oracle witness for the reduced DeepSeek
// V4.1 text forward assembled in v41_forward.go (issue #12901). It builds a tiny
// in-memory *Model with deterministic f32 weights and compares the production
// assembly against a plain in-order scalar transcription.
//
// Independence discipline (family_cpu_oracle_test.go): the reference below
// reuses NONE of the production forward machinery that it is checking — not
// matRows, not rmsnormCfg, not applyRopeRow, not v41MHCSplit/v41MHCPre/v41MHCPost,
// not v41Route/v41SharedExpertAdd, not V41SparseAttentionSink, not
// V41GroupedOutputProjection. It reads the fixture's own f32 bytes, hardcodes the
// reduced dataflow, and reuses only the weight-free scalar oracle helpers that
// v41_oracle_test.go already declares (oracleV41MHCKernel, oracleV41Route,
// oracleV41SharedExpertAdd, oracleV41Score) plus the generic cpuOracle* scalar
// primitives. Those helpers are themselves naive transcriptions, so this is a
// genuine cross-implementation comparison for the assembly wiring, not a
// production-vs-production tautology.
//
// Scope honesty. The reduced fixture exercises: embedding, mHC split/pre/post,
// the MLA-style q/kv projections + pairwise RoPE, the sink attention contraction,
// the grouped output projection, the router + routed/shared experts, the final
// norm and the head. Engram, compressor and the lightning indexer are NOT
// exercised here — their packed-row / compressed-stream inputs cannot be
// faithfully materialized in a weight-free in-memory fixture (see the scope note
// in v41_forward.go); the assembly omits them explicitly rather than silently.

import (
	"errors"
	"math"
	"testing"
)

// v41ReducedModel builds the tiny V4.1 fixture. The routed+shared leaf
// (v41_router.go) admits only the published 384-expert / top-6 envelope, so the
// fixture keeps 384 experts (each tiny) and top-6; the mHC mix width is
// (2+4)*4 = 24.
func v41ReducedModel(t *testing.T) *Model {
	t.Helper()
	cfg := v41TestReducedConfig(t, 1, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	// Pin a scaling-free RoPE so the scalar oracle can rebuild the table from
	// theta and the head width alone (see rope.go invFreqDenom / ropeAttentionFactor).
	cfg.RopeScaling = ""
	cfg.LongRope = nil
	cfg.RopeFactor = 0
	cfg.RopeOrigContext = 0
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	hd := cfg.HeadDim
	nH := cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups

	type ts = synthTensor
	tensors := []ts{
		{"model.embed_tokens.weight", []int{cfg.VocabSize, H}},
		{"lm_head.weight", []int{cfg.VocabSize, H}},
		{"model.norm.weight", []int{H}},
	}
	for l := 0; l < cfg.NumLayers; l++ {
		tensors = append(tensors,
			ts{layerName(l, "attn_norm.weight"), []int{H}},
			ts{layerName(l, "ffn_norm.weight"), []int{H}},
			ts{layerName(l, "mhc.mixes.weight"), []int{v41MHCMixWidth, H}},
			ts{layerName(l, "mhc.base"), []int{v41MHCMixWidth}},
			ts{layerName(l, "mhc.scale"), []int{3}},
			ts{layerName(l, "attn.wq_a.weight"), []int{cfg.QLoraRank, H}},
			ts{layerName(l, "attn.wq_b.weight"), []int{qHeadDim, cfg.QLoraRank}},
			ts{layerName(l, "attn.wkv.weight"), []int{v41KVLoraRankReduced(cfg), H}},
			ts{layerName(l, "attn.wo_a.weight"), []int{cfg.OLoraRank, qHeadDim}},
			ts{layerName(l, "attn.wo_b.weight"), []int{H, oDim}},
			ts{layerName(l, "attn.sink"), []int{nH}},
			ts{layerName(l, "ffn.gate.weight"), []int{cfg.NumExperts, H}},
			ts{layerName(l, "ffn.gate.e_score_correction_bias"), []int{cfg.NumExperts}},
			ts{layerName(l, "ffn.shared_experts.w1.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w3.weight"), []int{I, H}},
			ts{layerName(l, "ffn.shared_experts.w2.weight"), []int{H, I}},
		)
		for e := 0; e < cfg.NumExperts; e++ {
			stem := "ffn.experts." + itoa(e)
			tensors = append(tensors,
				ts{layerName(l, stem+".w1.weight"), []int{I, H}},
				ts{layerName(l, stem+".w3.weight"), []int{I, H}},
				ts{layerName(l, stem+".w2.weight"), []int{H, I}},
			)
		}
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
			return 1.0 // well-conditioned norms
		case hasSuffix(name, "mhc.scale"):
			return 1.0
		case hasSuffix(name, "mhc.base"):
			return 0.0
		case hasSuffix(name, "attn.sink"):
			return 0.25 * next()
		default:
			return synthMatmulFill(name, next)
		}
	})
	return &Model{Cfg: cfg, manifest: man, raw: raw}
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

// v41OracleForward is the independent scalar reference for the reduced forward.
func v41OracleForward(t *testing.T, m *Model, ids []int) [][]float32 {
	t.Helper()
	cfg := m.Cfg
	H, hd, nH := cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)

	tensor := func(name string) []float32 { return cpuOracleTensor(t, m, name) }
	embed := tensor("model.embed_tokens.weight")
	x := make([][]float32, seq)
	for tt, id := range ids {
		x[tt] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}

	scale := float32(1.0 / math.Sqrt(float64(hd)))
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
		attnNorm := tensor(layerName(l, "attn_norm.weight"))
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

		preByPos := make([][]float32, seq)
		pre := make([][]float64, seq)
		post := make([][]float64, seq)
		comb := make([][]float64, seq)
		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], attnNorm, eps)
			mixes := cpuOracleMatVec(wMix, xn, v41MHCMixWidth, H)
			mixF := toF64(mixes)
			baseF := toF64(mixBase)
			scaleF := toF64(mixScale)
			p, po, c := oracleV41MHCKernel(mixF, scaleF, baseF, 4, hcIters, hcEps)
			pre[tt], post[tt], comb[tt] = p, po, c
			streams := [][]float64{toF64(xn), toF64(xn), toF64(xn), toF64(xn)}
			collapsed := oracleV41MHCPre(streams, p)
			preByPos[tt] = toF32(collapsed)
		}

		qHeads := make([][]float32, seq)
		kvRows := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			c := preByPos[tt]
			qLat := cpuOracleMatVec(wQA, c, cfg.QLoraRank, H)
			q := cpuOracleMatVec(wQB, qLat, nH*hd, cfg.QLoraRank)
			kv := cpuOracleMatVec(wKV, c, hd, H)
			for h := 0; h < nH; h++ {
				cpuOracleRope(q[h*hd:(h+1)*hd], tt, hd, cfg.RopeTheta)
			}
			cpuOracleRope(kv, tt, hd, cfg.RopeTheta)
			qHeads[tt] = q
			kvRows[tt] = kv
		}

		attnOut := make([][]float32, seq)
		for tt := 0; tt < seq; tt++ {
			rows := tt + 1
			o := make([]float32, nH*hd)
			for h := 0; h < nH; h++ {
				qh := qHeads[tt][h*hd : (h+1)*hd]
				maxScore := float64(sink[h])
				dots := make([]float64, rows)
				for i := 0; i < rows; i++ {
					var d float64
					for j := 0; j < hd; j++ {
						d += float64(qh[j]) * float64(kvRows[i][j])
					}
					d *= float64(scale)
					dots[i] = d
					if d > maxScore {
						maxScore = d
					}
				}
				sum := math.Exp(float64(sink[h]) - maxScore)
				for i := 0; i < rows; i++ {
					sum += math.Exp(dots[i] - maxScore)
				}
				if sum == 0 {
					continue
				}
				for i := 0; i < rows; i++ {
					w := math.Exp(dots[i]-maxScore) / sum
					for j := 0; j < hd; j++ {
						o[h*hd+j] += float32(w * float64(kvRows[i][j]))
					}
				}
			}
			attnOut[tt] = v41OracleGroupedOutput(o, woA, woB, nH, hd, cfg.OGroups, cfg.OLoraRank, H)
		}

		for tt := 0; tt < seq; tt++ {
			xn := cpuOracleRMSNorm(x[tt], ffnNorm, eps)
			router := cpuOracleMatVec(wGate, xn, cfg.NumExperts, H)
			logits := toF64(router)
			bias := toF64(gateBias)
			picks, weights := oracleV41Route(logits, bias, cfg.NumExpertsPerTok, routeScale)
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
			streams := [][]float64{toF64(preByPos[tt]), toF64(preByPos[tt]), toF64(preByPos[tt]), toF64(preByPos[tt])}
			next := oracleV41MHCPost(delta, streams, post[tt], comb[tt], 4, false)
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
	return logits
}

// v41OracleGroupedOutput mirrors V41GroupedOutputProjection scalar.
func v41OracleGroupedOutput(o, woA, woB []float32, heads, headDim, groups, oLoRARank, dim int) []float32 {
	headsPerGroup := heads / groups
	aRow := headsPerGroup * headDim
	joined := make([]float64, groups*oLoRARank)
	for g := 0; g < groups; g++ {
		groupBase := g * aRow
		for r := 0; r < oLoRARank; r++ {
			aBase := (g*oLoRARank + r) * aRow
			var acc float64
			for k := 0; k < aRow; k++ {
				acc += float64(o[groupBase+k]) * float64(woA[aBase+k])
			}
			joined[g*oLoRARank+r] = acc
		}
	}
	out := make([]float32, dim)
	for d := 0; d < dim; d++ {
		bBase := d * groups * oLoRARank
		var acc float64
		for k := 0; k < groups*oLoRARank; k++ {
			acc += float64(woB[bBase+k]) * joined[k]
		}
		out[d] = float32(acc)
	}
	return out
}

func oracleSwiGLU(w1, w3, w2, xn []float32, I, H int) []float64 {
	h1 := cpuOracleMatVec(w1, xn, I, H)
	h3 := cpuOracleMatVec(w3, xn, I, H)
	h := make([]float32, I)
	for i := 0; i < I; i++ {
		h[i] = cpuOracleSilu(h1[i]) * h3[i]
	}
	down := cpuOracleMatVec(w2, h, H, I)
	out := make([]float64, H)
	for i := range down {
		out[i] = float64(down[i])
	}
	return out
}

func toF64(v []float32) []float64 {
	out := make([]float64, len(v))
	for i := range v {
		out[i] = float64(v[i])
	}
	return out
}

func toF32(v []float64) []float32 {
	out := make([]float32, len(v))
	for i := range v {
		out[i] = float32(v[i])
	}
	return out
}

func v41LogitsClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s logits length = %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if d := math.Abs(float64(got[i] - want[i])); d > cpuOracleTol {
			t.Fatalf("%s logits[%d] = %v, want %v (|delta| = %.3e > tol %.0e)", name, i, got[i], want[i], d, cpuOracleTol)
		}
	}
}

// TestV41Forward is the #12901 acceptance witness: the reduced assembly matches
// the independent scalar oracle, Prefill+Step is consistent with one longer
// Forward, and every missing-stage path fails closed with a typed error.
func TestV41Forward(t *testing.T) {
	m := v41ReducedModel(t)

	t.Run("logits match independent oracle", func(t *testing.T) {
		ids := []int{1, 3, 5}
		act := m.Forward(ids)
		if act == nil || len(act.Logits) != len(ids) {
			t.Fatalf("Forward returned %v positions, want %d", act, len(ids))
		}
		want := v41OracleForward(t, m, ids)
		for tPos := range ids {
			v41LogitsClose(t, "forward", act.Logits[tPos], want[tPos])
		}
	})

	t.Run("prefill step continuation", func(t *testing.T) {
		s := &Session{M: m}
		prefill := s.Prefill([]int{2, 4})
		if len(prefill) == 0 {
			t.Fatal("Prefill returned no logits")
		}
		step := s.Step(6)
		long := m.Forward([]int{2, 4, 6})
		v41LogitsClose(t, "step-vs-forward", step, long.Logits[2])
	})

	t.Run("missing stage fails closed", func(t *testing.T) {
		// Remove exactly one required stage weight: the assembly must refuse with
		// ErrV41ForwardStage and emit no logits.
		broken := v41ReducedModel(t)
		delete(broken.manifest, layerName(0, "attn.wq_b.weight"))
		if err := broken.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("missing-stage admission error = %v, want ErrV41ForwardStage", err)
		}
		if errors.Is(broken.v41ForwardAdmitted(), ErrV41NativeUnsupported) {
			t.Fatal("missing-stage error incorrectly reported ErrV41NativeUnsupported")
		}
		if err := panicAsError(func() { _ = broken.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
			t.Fatalf("Forward missing-stage panic = %v, want ErrV41ForwardStage", err)
		}

		// The weightless probe (the #12967 fence) must still yield the native-
		// unsupported verdict so v41_failclosed_test.go keeps passing.
		_, cfg := readDeepSeekV41Config(t)
		weightless := &Model{Cfg: cfg}
		if err := panicAsError(func() { _ = weightless.Forward([]int{1, 2, 3}) }); !errors.Is(err, ErrV41NativeUnsupported) {
			t.Fatalf("weightless Forward error = %v, want ErrV41NativeUnsupported", err)
		}
	})
}

// TestV41ForwardEngramDeclaredFailsClosed is the #13007 fail-closed witness: the
// reduced assembly does not execute the Engram stage, so a config that declares
// an Engram layer WITHIN the model's layer range must refuse rather than silently
// emit non-Engram logits as if Engram were absent. Out-of-range declared Engram
// layers (the reduced oracle fixture: NumLayers=1 with EngramLayerIDs [1,14]) stay
// admitted, because the assembly never reaches them.
func TestV41ForwardEngramDeclaredFailsClosed(t *testing.T) {
	// In-range declaration: layer 0 is inside the single reduced decoder layer.
	declared := v41ReducedModel(t)
	declared.Cfg.DeepSeekV41.EngramLayerIDs = []int{0}
	declared.Cfg.DeepSeekV41.EngramNumEmbeddings = []int{8}

	if err := declared.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range Engram admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = declared.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range Engram Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// The reduced oracle fixture declares only out-of-range Engram layers, so the
	// existing reduced forward must keep running (no regression).
	reduced := v41ReducedModel(t)
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced out-of-range Engram admission error = %v, want nil", err)
	}
}

// TestV41ForwardCompressIndexDeclaredFailsClosed is the #13006 fail-closed
// witness: the reduced assembly does not execute the CED/CSA2 compressor or the
// lightning indexer, so a config that declares a compressed layer WITHIN the
// model's layer range (CompressRatios[l] > 1), or an index-source layer within
// range, must refuse rather than silently emit reduced, non-compressed logits as
// if those stages were absent. Declarations that only touch out-of-range layers
// (the reduced oracle fixture derives from the published 40-layer config but
// narrows NumLayers to 1, so ratios above index 0 and index sources {2,8,...} are
// all unreachable) stay admitted, because the assembly never reaches them.
func TestV41ForwardCompressIndexDeclaredFailsClosed(t *testing.T) {
	// In-range compressed layer 0 (ratio 2 = CED/CSA2): must refuse.
	compressed := v41ReducedModel(t)
	compressed.Cfg.DeepSeekV41.CompressRatios = []int{2}
	if err := compressed.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range compressor admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = compressed.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range compressor Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// In-range index source 0: must refuse.
	indexed := v41ReducedModel(t)
	indexed.Cfg.DeepSeekV41.IndexSourceLayerIDs = []int{0}
	if err := indexed.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range indexer admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = indexed.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range indexer Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// The reduced oracle fixture declares only out-of-range compressor/indexer
	// axes (ratio 0 at layer 0; index sources >= 2), so the existing reduced
	// forward must keep running (no regression).
	reduced := v41ReducedModel(t)
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced out-of-range compress/index admission error = %v, want nil", err)
	}
}

// TestV41ForwardKVSourceDeclaredFailsClosed is the shared-KV-source fail-closed
// witness: the reduced assembly projects its own per-layer attn.wkv.weight and
// never consumes a KV source layer's shared key/value state, so a config that
// declares a shared-KV source layer WITHIN the model's layer range must refuse
// rather than silently emit logits from a per-layer KV cache as if the shared
// source were absent. Out-of-range declared KV sources (the reduced oracle
// fixture derives from the published 40-layer config but narrows NumLayers to 1,
// so kv sources {2,8,14,20} are all unreachable) stay admitted, because the
// assembly never reaches them.
func TestV41ForwardKVSourceDeclaredFailsClosed(t *testing.T) {
	// In-range KV source layer 0: must refuse.
	shared := v41ReducedModel(t)
	shared.Cfg.DeepSeekV41.KVSourceLayerIDs = []int{0}

	if err := shared.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range KV source admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = shared.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range KV source Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// The reduced oracle fixture declares only out-of-range KV sources (>= 2), so
	// the existing reduced forward must keep running (no regression).
	reduced := v41ReducedModel(t)
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced out-of-range KV source admission error = %v, want nil", err)
	}
}
