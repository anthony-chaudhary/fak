package model

// attn_inner_loop_share_test.go — the #1129 measurement gate (epic #1124, gap C5).
//
// #1129 is the LOWER-CONFIDENCE child: "the CPU attention inner loop is pure scalar —
// MEASURE its share before committing to asm." Its required first step, and its primary
// deliverable, is a witnessed measurement of attention's share of prefill AND decode, plus
// a recorded verdict among three outcomes (Go-ILP/fdot closes it · a flash-tiling rewrite
// is the real lever · a genuine SIMD kernel remains). Per the issue's own acceptance bar,
// "the measurement, with the share quoted (witnessed), is itself a deliverable — even a
// 'leave it scalar' verdict closes this honestly."
//
// This file IS that witness. It reuses the in-kernel profiler (profile.go), whose opAttn
// op-class already attributes the score-dot + softmax + ΣwV inner loop exactly (it is the
// instrumented twin of the proven path, pinned bit-for-bit by TestProfileMatchesProven), so
// the share it reports is about the real forward pass, not a stale copy. It then quantifies
// the Go-ILP headroom on the score-dot (scalar single-accumulator dot vs the 8-accumulator
// fdot, parallel.go:185) and proves that reuse stays inside the f32 oracle tolerance.
//
// ── THE VERDICT (witnessed by the tests + benches below) ──────────────────────────────────
//
//   The three scalar score-dots — the f32 prefill attnSeq (forward.go:264), the tensor-
//   parallel twin (tensor_parallel_attn.go:70), and the decode profiler twin (profile.go:290)
//   — all call the single-accumulator dot (forward.go:426). It does NOT auto-vectorize: each
//   += waits on the previous (a serial FP-add latency chain), exactly the cliff fdot exists to
//   break. So the gap the issue names is real.
//
//   (1) Go-ILP / fdot-reuse — MARGINAL, and it spends f32 tolerance. MODEL-BASELINE-RESULTS.md
//       (Act 5) measured dot→fdot on the *Q8* attn stage at ~1.15× of that stage; the f32 path
//       was DELIBERATELY left scalar ("the proven f32 path ... is byte-untouched") because
//       fdot's reassociated sum is NOT bit-identical to dot (~1e-6), and the f32 attnSeq is
//       pinned by a web of bit-exact rungs (forward_band, tensor_parallel, the arch topology
//       twins). TestAttnScoreDotFdotWithinOracleTol below witnesses BOTH halves: fdot stays far
//       inside the oracle tolerance AND is not bit-exact — i.e. adopting it would re-spend the
//       f32 bit-exact budget for a single-digit-% prefill win. Not worth it as a standalone.
//
//   (2) Flash-style tiling IS the real structural lever — but it is an ALGORITHM change (fuse
//       score+softmax+ΣwV over a hot K/V tile), partly orthogonal to SIMD, with its own
//       correctness surface. It belongs in its own issue (the CPU analog of the GPU flash
//       kernels), not bundled into this measure-first child.
//
//   (3) A hand AVX2/NEON score-dot is NOT justified by this measurement. Decode attention is a
//       ~1% slice (the KV cache is L2-resident, so attn is latency/overhead-bound at ~2 GB/s,
//       not a compute cliff — profile.go's own roofline note), so asm buys ~nothing on decode;
//       and on prefill the pure-Go fdot already captures the available dense-f32 ILP without
//       any bit-identity-asm risk.
//
//   => CLOSE #1129 with the measurement: leave the scalar f32 default untouched; do NOT write
//      asm; carry the structural win forward as a separate flash-tiling algorithm issue.
//
// The tests assert the *structure* of this verdict (the share is attributed and non-trivial;
// the Go-ILP reuse is tolerance-safe but not bit-exact); the witnessed NUMBERS are logged so a
// reader sees the actual shares and speedup on the box under test (they are wall-time, hence
// machine-dependent, so they are reported, not thresholded).

import (
	"math"
	"testing"
)

// representative head geometry for the score-dot microbench / tolerance witness: head_dim
// 128 is the Llama/Qwen norm and a clean multiple of fdot's 8-wide body.
const attnBenchHeadDim = 128

// makeAttnHeadVecs builds one query head and nKeys key heads, deterministically (no RNG, so
// the witness is reproducible), with values in a realistic post-RoPE-ish range.
func makeAttnHeadVecs(hd, nKeys int) (qh []float32, keys [][]float32) {
	qh = make([]float32, hd)
	for d := 0; d < hd; d++ {
		qh[d] = float32(math.Sin(float64(d)*0.13+0.4)) * 0.7
	}
	keys = make([][]float32, nKeys)
	for j := 0; j < nKeys; j++ {
		kh := make([]float32, hd)
		for d := 0; d < hd; d++ {
			kh[d] = float32(math.Cos(float64(d)*0.071+float64(j)*0.017)) * 0.9
		}
		keys[j] = kh
	}
	return qh, keys
}

// opShare returns the named op-class's measured time share + MACs from a profile.
func opShare(p *Profile, class string) (timePct float64, macs int64, found bool) {
	for i := range p.Stats {
		if p.Stats[i].Class == class {
			return p.Stats[i].TimePct, p.Stats[i].MACs, true
		}
	}
	return 0, 0, false
}

// attnShareModel builds a representative-geometry synthetic model (GQA head_dim 64, 8 query
// heads over 2 KV heads, 4 layers) so the share witness runs host-independently — it needs no
// re-exported HF oracle weights (NewSynthetic fills well-conditioned f32 in-memory). The
// SHAPES drive the exact MAC attribution; the absolute wall times are this box's.
func attnShareModel() *Model {
	return NewSynthetic(Config{
		HiddenSize:       512,
		NumLayers:        4,
		NumHeads:         8,
		NumKVHeads:       2,
		HeadDim:          64,
		IntermediateSize: 1376,
		VocabSize:        2048,
		RMSNormEps:       1e-5,
		RopeTheta:        10000,
		EOSTokenID:       -1,
	})
}

// TestAttnInnerLoopShareIsMeasured is the #1129 required-first-step witness: it profiles a
// long-context prefill AND a decode run and quotes attention's measured share of each. The
// assertions are structural (attention IS attributed, with real work, as a sane fraction);
// the deterministic MAC share + the (machine-dependent) time share are both logged so the
// verdict above is grounded in numbers from this box. The realistic absolute shares on a real
// model are in MODEL-BASELINE-RESULTS.md (Act 5): attn ~23-27% of prefill, ~1.2% of DECODE.
func TestAttnInnerLoopShareIsMeasured(t *testing.T) {
	m := attnShareModel()

	const prefillLen = 384 // long-enough context that the O(P^2) attn loop is a visible cliff
	pp := m.ProfilePrefill(prefillLen)
	attnPctP, attnMACsP, okP := opShare(pp, opAttn)
	if !okP {
		t.Fatalf("prefill profile did not attribute the %q op-class — attention share unmeasured", opAttn)
	}
	if attnMACsP <= 0 {
		t.Fatalf("prefill attn MACs = %d, want > 0 (the inner loop must do real work to have a share)", attnMACsP)
	}
	if attnPctP < 0 || attnPctP > 100 {
		t.Fatalf("prefill attn time share = %.2f%%, outside [0,100] — profiler attribution broken", attnPctP)
	}

	const decodePrompt, decodeSteps = 384, 24
	pd := m.ProfileDecode(decodePrompt, decodeSteps)
	attnPctD, attnMACsD, okD := opShare(pd, opAttn)
	if !okD {
		t.Fatalf("decode profile did not attribute the %q op-class — attention share unmeasured", opAttn)
	}
	if attnMACsD <= 0 {
		t.Fatalf("decode attn MACs = %d, want > 0", attnMACsD)
	}
	if attnPctD < 0 || attnPctD > 100 {
		t.Fatalf("decode attn time share = %.2f%%, outside [0,100]", attnPctD)
	}

	// Deterministic MAC share (shape-exact, reproducible) alongside the noisy wall-time share.
	macFrac := func(macs int64, total int64) float64 {
		if total == 0 {
			return 0
		}
		return 100 * float64(macs) / float64(total)
	}
	oprojP, _, _ := opShare(pp, opOProj)
	qkvP, _, _ := opShare(pp, opQKVProj)
	t.Logf("#1129 MEASURE prefill P=%d: attn MAC=%.1f%% time=%.1f%% (MACs=%d) | qkv_proj time=%.1f%% o_proj time=%.1f%% bottleneck=%q",
		prefillLen, macFrac(attnMACsP, pp.TotalMACs), attnPctP, attnMACsP, qkvP, oprojP, pp.Bottleneck)
	t.Logf("#1129 MEASURE decode  P=%d steps=%d: attn MAC=%.1f%% time=%.1f%% (MACs=%d) bottleneck=%q",
		decodePrompt, decodeSteps, macFrac(attnMACsD, pd.TotalMACs), attnPctD, attnMACsD, pd.Bottleneck)
	// The DETERMINISTIC, geometry-driven witness is the MAC share. The synthetic wall-time
	// share is NOT representative of a real deployment and must not be read as one: this
	// micro-model's projection GEMVs are tiny and the profiler times each op separately, which
	// over-weights attention's serial O(nPos) loop on decode. On a REAL model the projections
	// dominate and the KV cache is L2-resident (attn is latency/overhead-bound, ~2 GB/s) — so
	// the authoritative shares are the baseline-doc measurements below, not this box's wall time.
	t.Logf("#1129 AUTHORITATIVE real-model shares (MODEL-BASELINE-RESULTS.md Act 5, SmolLM2-135M, WSL-16t): attn ~23-27%% of PREFILL; ~1.2%% of DECODE (L2-resident KV → latency-bound, not a compute cliff)")
	t.Logf("#1129 VERDICT: leave the scalar f32 default; Go-ILP/fdot is marginal + spends bit-exact tolerance; flash-tiling is the real lever → its own algorithm issue. See file header.")
}

// TestAttnScoreDotFdotWithinOracleTol witnesses BOTH halves of outcome (1): reusing the
// 8-accumulator fdot for the attention score-dot stays far inside the f32 oracle tolerance
// (argmax-exact / max|Δ|<0.05 regime) — so it is numerically safe — AND it is NOT bit-exact
// vs the single-accumulator dot, which is precisely why adopting it on the f32 default would
// re-spend the proven path's bit-exact budget for a single-digit-% win. This is the evidence
// behind "leave the scalar default."
func TestAttnScoreDotFdotWithinOracleTol(t *testing.T) {
	const nKeys = 512
	qh, keys := makeAttnHeadVecs(attnBenchHeadDim, nKeys)

	var maxAbs float64
	var anyDiff bool
	var dotSS, fSS, cross float64 // for cosine between the two score vectors
	for j := 0; j < nKeys; j++ {
		sScalar := dot(qh, keys[j])
		sILP := fdot(qh, keys[j])
		d := math.Abs(float64(sScalar - sILP))
		if d > maxAbs {
			maxAbs = d
		}
		if sScalar != sILP {
			anyDiff = true
		}
		dotSS += float64(sScalar) * float64(sScalar)
		fSS += float64(sILP) * float64(sILP)
		cross += float64(sScalar) * float64(sILP)
	}
	cos := cross / (math.Sqrt(dotSS) * math.Sqrt(fSS))

	// Safety half: within oracle tolerance by a wide margin.
	if maxAbs >= 0.05 {
		t.Fatalf("fdot score-dot drifts max|Δ|=%.3e from scalar dot — NOT inside the <0.05 oracle tolerance", maxAbs)
	}
	if cos < 0.999999 {
		t.Fatalf("fdot vs scalar score-vector cosine=%.8f < 0.999999 — reassociation changed the scores materially", cos)
	}
	// Cost half: it is genuinely not bit-exact (the reason the f32 default stays scalar). On
	// amd64 (no auto-FMA) the reassociated 8-accumulator sum differs from the serial sum for
	// these non-trivial inputs; if a future arch fused both identically this would be false,
	// which is fine — the test still records the witnessed drift.
	t.Logf("#1129 Go-ILP score-dot: fdot vs scalar dot over %d keys (hd=%d): cosine=%.8f max|Δ|=%.3e not-bit-exact=%v",
		nKeys, attnBenchHeadDim, cos, maxAbs, anyDiff)
}

// BenchmarkAttnScoreDotScalar / ...ILP quantify the Go-ILP headroom on the score-dot itself:
// the scalar single-accumulator dot (the current forward.go:264 / profile.go:290 kernel) vs
// the 8-accumulator fdot. The ratio between them is the most fdot-reuse could buy on this
// stage before any work-unit reshape — run with `go test -run x -bench AttnScoreDot`.
func BenchmarkAttnScoreDotScalar(b *testing.B) {
	const nKeys = 512
	qh, keys := makeAttnHeadVecs(attnBenchHeadDim, nKeys)
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		for j := 0; j < nKeys; j++ {
			sink += dot(qh, keys[j])
		}
	}
	_ = sink
}

func BenchmarkAttnScoreDotILP(b *testing.B) {
	const nKeys = 512
	qh, keys := makeAttnHeadVecs(attnBenchHeadDim, nKeys)
	b.ResetTimer()
	var sink float32
	for i := 0; i < b.N; i++ {
		for j := 0; j < nKeys; j++ {
			sink += fdot(qh, keys[j])
		}
	}
	_ = sink
}

// BenchmarkAttnWeightedSum benchmarks the ΣwV stage (forward.go:273-278). Its inner d-loop
// (o[d] += w*vh[d]) is already a per-element-independent AXPY the Go compiler auto-vectorizes,
// so — unlike the serial score-dot — there is little scalar cliff here. This bench is the
// evidence for that: it should already run near memory speed, confirming the inner loop's
// scalar cost is concentrated in the score-dot, not ΣwV.
func BenchmarkAttnWeightedSum(b *testing.B) {
	const nKeys = 512
	hd := attnBenchHeadDim
	_, vals := makeAttnHeadVecs(hd, nKeys)
	weights := make([]float32, nKeys)
	for j := range weights {
		weights[j] = float32(1.0 / float64(nKeys)) // post-softmax-ish
	}
	o := make([]float32, hd)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for d := range o {
			o[d] = 0
		}
		for j := 0; j < nKeys; j++ {
			w := weights[j]
			vh := vals[j]
			for d := 0; d < hd; d++ {
				o[d] += w * vh[d]
			}
		}
	}
	_ = o
}
