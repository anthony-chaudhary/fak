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
	"fmt"
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
// reduced assembly now EXECUTES the Engram stage when it is wired, so a config
// that declares an Engram layer WITHIN the model's layer range but carries no
// packed-row source must refuse rather than silently emit non-Engram logits as if
// Engram were absent. Out-of-range declared Engram layers (the reduced oracle
// fixture: NumLayers=1 with EngramLayerIDs [1,14]) stay admitted, because the
// assembly never reaches them.
//
// Error-class note (#13007): before this leaf the in-range declaration wrapped
// ErrV41ForwardStage. It now wraps ErrV41NativeUnsupported, because the stage
// cannot execute without its packed-row source — the same native-forward-
// unavailable class the #12967 weightless fence uses. The intent is unchanged:
// the model still fails closed and emits no logits.
func TestV41ForwardEngramDeclaredFailsClosed(t *testing.T) {
	// In-range declaration with no wired row source: layer 0 is inside the
	// single reduced decoder layer and has no stage, so it must refuse.
	declared := v41ReducedModel(t)
	declared.Cfg.DeepSeekV41.EngramLayerIDs = []int{0}
	declared.Cfg.DeepSeekV41.EngramNumEmbeddings = []int{8}

	if err := declared.v41ForwardAdmitted(); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("in-range Engram admission error = %v, want ErrV41NativeUnsupported", err)
	}
	if err := panicAsError(func() { _ = declared.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("in-range Engram Forward panic = %v, want ErrV41NativeUnsupported", err)
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

// TestV41ForwardHCMultDeclaredFailsClosed is the mHC-multiplicity fail-closed
// witness: the reduced assembly hardcodes the four-stream hyperconnection
// geometry (v41MHCSplit called with hc=4, four identical stand-in streams, and
// the width-24 mix projection v41MHCMixWidth), so it executes only the published
// hc_mult=4 layout. A config that declares a different hc_mult must refuse rather
// than silently run the four-stream geometry for a model the assembly never ran.
// hc_mult=4 (the published artifact and every reduced fixture) stays admitted.
func TestV41ForwardHCMultDeclaredFailsClosed(t *testing.T) {
	// A declared multiplicity other than 4: must refuse.
	other := v41ReducedModel(t)
	other.Cfg.DeepSeekV41.HCMult = 2
	if err := other.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("hc_mult=2 admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = other.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("hc_mult=2 Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// The published/reduced hc_mult=4 must keep running (no regression).
	reduced := v41ReducedModel(t)
	if reduced.Cfg.DeepSeekV41.HCMult != 4 {
		t.Fatalf("reduced fixture hc_mult = %d, want 4", reduced.Cfg.DeepSeekV41.HCMult)
	}
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("hc_mult=4 admission error = %v, want nil", err)
	}
}

// TestV41FullGeometryRouterConfigFailsClosed witnesses the #13009 router-geometry
// reduction: the real forward path must source its routed+shared geometry through
// v41RouterConfigFromConfig (the published 384/top-6/1-shared/1.5-scale envelope),
// not read an arbitrary live Config. A config whose router axis departs from the
// published envelope must fail closed with the typed ErrV41ForwardStage at the
// admission boundary -- never silently run a different MoE geometry -- while the
// published/reduced envelope keeps running.
func TestV41FullGeometryRouterConfigFailsClosed(t *testing.T) {
	// A non-published expert count must refuse.
	fewerExperts := v41ReducedModel(t)
	fewerExperts.Cfg.NumExperts = V41RouterExperts / 2
	if err := fewerExperts.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("NumExperts=%d admission error = %v, want ErrV41ForwardStage", fewerExperts.Cfg.NumExperts, err)
	}
	if err := panicAsError(func() { _ = fewerExperts.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("NumExperts=%d Forward panic = %v, want ErrV41ForwardStage", fewerExperts.Cfg.NumExperts, err)
	}

	// A non-published top-k must refuse.
	badTopK := v41ReducedModel(t)
	badTopK.Cfg.NumExpertsPerTok = V41RouterTopK + 1
	if err := badTopK.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("NumExpertsPerTok=%d admission error = %v, want ErrV41ForwardStage", badTopK.Cfg.NumExpertsPerTok, err)
	}

	// A non-published route scale must refuse.
	badScale := v41ReducedModel(t)
	badScale.Cfg.RoutedScalingFactor = 2.5
	if err := badScale.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("RoutedScalingFactor=%g admission error = %v, want ErrV41ForwardStage", badScale.Cfg.RoutedScalingFactor, err)
	}

	// The published/reduced envelope must keep running (no regression).
	reduced := v41ReducedModel(t)
	if reduced.Cfg.NumExperts != V41RouterExperts || reduced.Cfg.NumExpertsPerTok != V41RouterTopK ||
		reduced.Cfg.NSharedExperts != V41RouterSharedCount || reduced.Cfg.RoutedScalingFactor != float64(V41RouterRouteScale) {
		t.Fatalf("reduced fixture router envelope = experts %d topk %d shared %d scale %g, want published",
			reduced.Cfg.NumExperts, reduced.Cfg.NumExpertsPerTok, reduced.Cfg.NSharedExperts, reduced.Cfg.RoutedScalingFactor)
	}
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("published router envelope admission error = %v, want nil", err)
	}
}

// ---------------------------------------------------------------------------
// #13007 Engram retrieval witness
//
// This section wires a synthetic packed-row Engram stage into a variant of the
// reduced fixture and checks the production injection against an independent
// scalar transcription of the reference schedule (antirez/ds4 @ bd66c402,
// ds41_graph_before_attention + kernel_dsv41_engram_add). The reference's
// four-stream residual is reduced to the single stand-in vector exactly as
// v41_forward_engram.go documents: x[t][i] += sum_s bf16(gate_s * bf16(value[i])).
// The oracle re-transcribes the dequant/projection/gate/mix itself; it reuses
// only GatherV41EngramRows (the retrieval primitive under test) and the generic
// cpuOracle* scalar primitives.
// ---------------------------------------------------------------------------

// v41EngramTestLayout is the synthetic hash layout for the reduced Engram
// fixture: one table, MaxNgramSize=4, HeadsPerNgram=8 (so cols=24), and 24 primes
// of 2 giving 48 rows.
func v41EngramTestLayout(cfg Config) V41EngramLayout {
	primes := make([]uint32, 24)
	for i := range primes {
		primes[i] = 2
	}
	tokenMap := make([]uint32, cfg.VocabSize)
	for i := range tokenMap {
		tokenMap[i] = uint32(i % 4)
	}
	return V41EngramLayout{
		TokenMap:        tokenMap,
		CompressedVocab: 4,
		PadID:           0,
		Rows:            []uint32{48},
		Multipliers:     [][]uint64{{17, 13, 19, 11}},
		Primes:          [][]uint32{primes},
		MaxNgramSize:    4,
		HeadsPerNgram:   8,
	}
}

// v41EngramTestPackedRows fills rows valid packed rows: 256 E4M3 codes that are
// never the NaN byte, plus 8 E8M0 scale bytes in [120,127] that are never 0xff.
func v41EngramTestPackedRows(rows int) []byte {
	out := make([]byte, rows*V41EngramPackedRowBytes)
	for r := 0; r < rows; r++ {
		for j := 0; j < 256; j++ {
			code := byte((r*131 + j*7 + 3) & 0x7f)
			if code == 127 {
				code = 126
			}
			out[r*V41EngramPackedRowBytes+j] = code
		}
		for s := 0; s < 8; s++ {
			out[r*V41EngramPackedRowBytes+256+s] = byte(120 + (r+s)%8)
		}
	}
	return out
}

// v41EngramMemorySource is an in-memory V41EngramRowSource over fixed packed rows.
type v41EngramMemorySource struct {
	packed []byte
	rows   int
}

func (s *v41EngramMemorySource) RowBytes() int { return V41EngramPackedRowBytes }

func (s *v41EngramMemorySource) ReadRows(start, count int, dst []byte) (int, error) {
	if start < 0 || count <= 0 || start+count > s.rows {
		return 0, fmt.Errorf("engram memory source: range [%d,%d) outside [0,%d)", start, start+count, s.rows)
	}
	n := count * V41EngramPackedRowBytes
	copy(dst, s.packed[start*V41EngramPackedRowBytes:start*V41EngramPackedRowBytes+n])
	return n, nil
}

// v41ReducedEngramModel builds a two-layer reduced fixture whose in-range layer 1
// is a declared Engram layer with the three mixing tensors and a wired synthetic
// packed-row source. Engram geometry: MaxNgramSize=4, NHeads=8 (cols=24),
// HeadDim=H=64, hc=4.
func v41ReducedEngramModel(t *testing.T) (*Model, V41EngramLayout) {
	t.Helper()
	cfg := v41TestReducedConfig(t, 2, V41RouterExperts)
	cfg.NumExpertsPerTok = V41RouterTopK
	cfg.NSharedExperts = 1
	cfg.RoutedScalingFactor = 1.5
	cfg.RopeScaling = ""
	cfg.LongRope = nil
	cfg.RopeFactor = 0
	cfg.RopeOrigContext = 0
	if cfg.RopeTheta == 0 {
		cfg.RopeTheta = 10000
	}
	cfg.DeepSeekV41.EngramLayerIDs = []int{1}
	cfg.DeepSeekV41.EngramNumEmbeddings = []int{48}
	cfg.DeepSeekV41.EngramMaxNgramSize = 4
	cfg.DeepSeekV41.EngramNHeads = 8
	cfg.DeepSeekV41.EngramHeadDim = cfg.HiddenSize

	H := cfg.HiddenSize
	I := cfg.MoEIntermediateSize
	hd := cfg.HeadDim
	nH := cfg.NumHeads
	qHeadDim := nH * hd
	oDim := cfg.OLoraRank * cfg.OGroups
	cols := (cfg.DeepSeekV41.EngramMaxNgramSize - 1) * cfg.DeepSeekV41.EngramNHeads

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
	// The Engram mixing tensors at the in-range Engram layer.
	for _, id := range cfg.DeepSeekV41.EngramLayerIDs {
		tensors = append(tensors,
			ts{layerName(id, "engram_kv.weight"), []int{cols * cfg.DeepSeekV41.EngramHeadDim, (4 + 1) * H}},
			ts{layerName(id, "engram_q_norm.weight"), []int{4 * H}},
			ts{layerName(id, "engram_k_norm.weight"), []int{4 * H}},
		)
	}

	man, raw := synthBuildRaw(tensors, func(name string, next func() float32) float32 {
		switch {
		case name == "model.norm.weight" || hasSuffix(name, "attn_norm.weight") || hasSuffix(name, "ffn_norm.weight"):
			return 1.0
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
	m := &Model{Cfg: cfg, manifest: man, raw: raw}

	layout := v41EngramTestLayout(cfg)
	src := &v41EngramMemorySource{packed: v41EngramTestPackedRows(int(layout.Rows[0])), rows: int(layout.Rows[0])}
	if err := m.wireV41Engram(layout, []V41EngramRowSource{src}, int64(V41EngramPackedRowBytes)*4); err != nil {
		t.Fatalf("wire Engram stage: %v", err)
	}
	return m, layout
}

// v41OracleBF16 is an independent bf16 round-half-to-even (mirrors dsv41_bf16).
func v41OracleBF16(x float32) float32 {
	bits := math.Float32bits(x)
	if bits&0x7f800000 != 0x7f800000 {
		bits += 0x7fff + ((bits >> 16) & 1)
	}
	return math.Float32frombits(bits & 0xffff0000)
}

// v41OracleEngramDequant decodes one packed row independently of production.
func v41OracleEngramDequant(t *testing.T, row []byte, dim int) []float32 {
	t.Helper()
	if len(row) != V41EngramPackedRowBytes {
		t.Fatalf("packed row %d bytes, want %d", len(row), V41EngramPackedRowBytes)
	}
	out := make([]float32, dim)
	for j := 0; j < dim; j++ {
		code := row[j]
		scale := row[256+j/32]
		if code&127 == 127 || scale == 255 {
			t.Fatalf("invalid packed row byte at %d", j)
		}
		out[j] = v41OracleBF16(fp8E4M3ToF32(code) * float32(math.Ldexp(1, int(scale)-127)))
	}
	return out
}

// v41OracleEngramInject is the independent scalar transcription of the reference
// Engram add for one in-range layer, into the single stand-in residual vector.
func v41OracleEngramInject(t *testing.T, m *Model, layout V41EngramLayout, l int, x [][]float32, ids []int, eps float32) {
	t.Helper()
	cfg := m.Cfg
	H := cfg.HiddenSize
	dim := cfg.DeepSeekV41.EngramHeadDim
	hc := 4
	cols := (layout.MaxNgramSize - 1) * layout.HeadsPerNgram

	stage := m.v41EngramStageFor()
	if stage == nil {
		t.Fatal("oracle: Engram stage not wired")
	}
	hash, err := NewV41EngramHashState(layout)
	if err != nil {
		t.Fatal(err)
	}
	allRows, err := hash.Hash(ids, nil)
	if err != nil {
		t.Fatal(err)
	}
	layers := len(layout.Rows)
	cacheIdx := stage.cacheIndex(l)
	layerRows := make([]uint32, 0, len(ids)*cols)
	for tt := range ids {
		base := tt*layers*cols + cacheIdx*cols
		layerRows = append(layerRows, allRows[base:base+cols]...)
	}
	gathered, err := GatherV41EngramRows([]*V41EngramRowCache{stage.caches[cacheIdx]}, layerRows, cols)
	if err != nil {
		t.Fatal(err)
	}

	wKV := cpuOracleTensor(t, m, layerName(l, "engram_kv.weight"))
	qNorm := cpuOracleTensor(t, m, layerName(l, "engram_q_norm.weight"))
	kNorm := cpuOracleTensor(t, m, layerName(l, "engram_k_norm.weight"))

	for tt := range ids {
		rowVec := make([]float32, cols*dim)
		for c := 0; c < cols; c++ {
			copy(rowVec[c*dim:], v41OracleEngramDequant(t, gathered[tt*cols+c], dim))
		}
		projected := cpuOracleMatVec(wKV, rowVec, (hc+1)*H, cols*dim)
		value := make([]float32, H)
		for i := 0; i < H; i++ {
			value[i] = v41OracleBF16(projected[hc*H+i])
		}
		h := x[tt]
		acc := make([]float64, H)
		for s := 0; s < hc; s++ {
			key := make([]float32, H)
			var h2, k2, dot float64
			for i := 0; i < H; i++ {
				key[i] = v41OracleBF16(projected[s*H+i])
				hh := float64(h[i])
				kk := float64(key[i])
				h2 += hh * hh
				k2 += kk * kk
				dot += hh * float64(qNorm[s*H+i]) * float64(kNorm[s*H+i]) * kk
			}
			dot *= 1 / math.Sqrt(h2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(k2/float64(H)+float64(eps))
			dot *= 1 / math.Sqrt(float64(H))
			gate := 1 / (1 + math.Exp(-math.Copysign(math.Sqrt(math.Max(math.Abs(dot), 1e-6)), dot)))
			for i := 0; i < H; i++ {
				acc[i] += float64(v41OracleBF16(float32(gate) * value[i]))
			}
		}
		for i := 0; i < H; i++ {
			h[i] += float32(acc[i])
		}
	}
}

// v41OracleEngramForward mirrors v41OracleForward with the Engram injection at
// the start of every declared in-range Engram layer.
func v41OracleEngramForward(t *testing.T, m *Model, layout V41EngramLayout, ids []int) [][]float32 {
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
		if cfg.DeepSeekV41 != nil {
			for _, eng := range cfg.DeepSeekV41.EngramLayerIDs {
				if eng == l {
					v41OracleEngramInject(t, m, layout, l, x, ids, eps)
					break
				}
			}
		}

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

// TestV41EngramRetrievalForward is the #13007 acceptance witness: at a declared
// in-range Engram layer the reduced assembly retrieves, dequantizes, projects,
// gates, and mixes the packed rows, and its logits match the independent scalar
// oracle within cpuOracleTol. It also proves the injection is live (not a no-op)
// by comparing against the same model with the Engram layer moved out of range.
func TestV41EngramRetrievalForward(t *testing.T) {
	m, layout := v41ReducedEngramModel(t)
	ids := []int{1, 3, 5}

	if err := m.v41ForwardAdmitted(); err != nil {
		t.Fatalf("admission with wired Engram stage = %v, want nil", err)
	}
	act := m.Forward(ids)
	if act == nil || len(act.Logits) != len(ids) {
		t.Fatalf("Forward returned %v positions, want %d", act, len(ids))
	}
	want := v41OracleEngramForward(t, m, layout, ids)
	for tPos := range ids {
		v41LogitsClose(t, "engram-forward", act.Logits[tPos], want[tPos])
	}

	// Live-injection control: move the Engram layer out of range on the same
	// weights and confirm the logits DIFFER, so a silently skipped stage would
	// fail this test.
	off, _ := v41ReducedEngramModel(t)
	off.Cfg.DeepSeekV41.EngramLayerIDs = []int{99}
	off.Cfg.DeepSeekV41.EngramNumEmbeddings = []int{48}
	baseline := off.Forward(ids)
	if baseline == nil || len(baseline.Logits) != len(ids) {
		t.Fatalf("out-of-range Forward returned %v positions", baseline)
	}
	changed := false
	for tPos := range ids {
		for i := range act.Logits[tPos] {
			if math.Abs(float64(act.Logits[tPos][i]-baseline.Logits[tPos][i])) > cpuOracleTol {
				changed = true
			}
		}
	}
	if !changed {
		t.Fatal("Engram injection did not change the logits; the stage is a no-op")
	}
}

// TestV41EngramNoSourceFailsClosed is the #13007 negative witness: an in-range
// declared Engram layer with no wired row source must refuse at admission AND at
// Forward with ErrV41NativeUnsupported, and emit no logits.
func TestV41EngramNoSourceFailsClosed(t *testing.T) {
	m, _ := v41ReducedEngramModel(t)
	// Drop the stage to model an in-range declaration whose source is absent.
	v41EngramStages.Delete(m)

	if err := m.v41ForwardAdmitted(); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("unwired Engram admission error = %v, want ErrV41NativeUnsupported", err)
	}
	if errors.Is(m.v41ForwardAdmitted(), ErrV41ForwardStage) {
		t.Fatal("unwired Engram error incorrectly reported ErrV41ForwardStage")
	}
	if err := panicAsError(func() { _ = m.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41NativeUnsupported) {
		t.Fatalf("unwired Engram Forward panic = %v, want ErrV41NativeUnsupported", err)
	}
}

// TestV41ForwardCandidateSourceDeclaredFailsClosed is the candidate-source
// fail-closed witness: the reduced assembly runs its own full per-layer attention
// contraction and never executes the CED/CSA2 blocked-candidate selection, so a
// config that declares a candidate-source layer WITHIN the model's layer range
// must refuse rather than silently emit logits from an unblocked attention path
// as if the candidate selection were absent. Out-of-range declared candidate
// sources (the reduced oracle fixture derives from the published 40-layer config
// but narrows NumLayers to 1, so the candidate source 20 is unreachable) stay
// admitted, because the assembly never reaches them.
func TestV41ForwardCandidateSourceDeclaredFailsClosed(t *testing.T) {
	// In-range candidate source 0: must refuse.
	selected := v41ReducedModel(t)
	selected.Cfg.DeepSeekV41.CandidateSourceLayerID = 0

	if err := selected.v41ForwardAdmitted(); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range candidate source admission error = %v, want ErrV41ForwardStage", err)
	}
	if err := panicAsError(func() { _ = selected.Forward([]int{1, 2}) }); !errors.Is(err, ErrV41ForwardStage) {
		t.Fatalf("in-range candidate source Forward panic = %v, want ErrV41ForwardStage", err)
	}

	// The reduced oracle fixture declares only an out-of-range candidate source
	// (20), so the existing reduced forward must keep running (no regression).
	reduced := v41ReducedModel(t)
	if err := reduced.v41ForwardAdmitted(); err != nil {
		t.Fatalf("reduced out-of-range candidate source admission error = %v, want nil", err)
	}
}
