package compute

// decode_throughput.go — the host-tractable witness for issue #3176 ("CPU decode throughput
// very low on EPYC: a ~500-token Qwen2.5-1.5B Q8 generation takes >10 min, warm or cold").
// The issue's real ask is diagnostic and observability, not a device kernel: (1) is the Q8
// SIMD decode lane engaged or did it fall back to the reference path, (2) does decode scale
// across cores, (3) expose decode tok/s so this is measurable without wall-clock guessing.
// Each of those is answerable WITHOUT a CUDA/AVX device and WITHOUT a timer — by counting the
// work exactly and reading the backend registry — which is what this file ships. The actual
// per-token wall-clock is a device measurement (deferred to the EPYC box); the CEILING it must
// be judged against, and the tier that would serve it, are host-computable and exact.
//
// The load-bearing fact, and why decode is slow on CPU where prefill is not: decode runs the
// SAME projection/FFN GEMMs as prefill but at a batch of ONE token, so there is no weight reuse
// to amortize the weight stream against. prefill.go's roofline shows prefill is compute-bound
// (weights streamed once, reused across P tokens); decode is its P=1 corner, where every GEMM
// collapses to a matrix-times-VECTOR that must read the whole model's weights from RAM for each
// token it emits. Decode is therefore MEMORY-BOUND on weight streaming: tok/s is ceilinged by
// (peak memory bandwidth) / (weight bytes per token), and reaching that ceiling REQUIRES
// streaming weights at full aggregate bandwidth — which a scalar, single-accumulator reference
// kernel on one core cannot do. A measured decode two orders of magnitude below the ceiling is
// the quantitative signature of the reference path, not of an intrinsically slow model.
//
// Counting work is exact and host-independent; only the two device peaks (bandwidth, FLOP/s)
// and the one wall-clock reading are the caller's — the same honesty discipline as prefill.go's
// caller-supplied ridge/peaks. No hardware constant is baked in here.

// DecodeRoofline is the analytic decode-throughput roofline for ONE emitted token at a model
// geometry (reusing PrefillGeometry — decode is the same shape evaluated one token at a time;
// g.P is ignored because weight streaming is per-token and P-independent). WeightBytesPerToken
// and FLOPsPerToken are EXACT counts of operands and multiply-adds, not measurements; Intensity
// = FLOPs/WeightBytes is the roofline arithmetic intensity (FLOP per weight byte streamed).
// For a Q8 dense model Intensity sits at ~1-2 FLOP/byte — far below any plausible CPU ridge
// point (peak FLOP/s ÷ peak bytes/s, typically tens to hundreds of FLOP/byte) — so decode is
// memory-bound at every geometry, which is the structural reason it is slow on CPU.
type DecodeRoofline struct {
	WeightBytesPerToken int64   // resident weight bytes streamed to emit one token (all layers + lm_head)
	FLOPsPerToken       int64   // multiply+add (=2 flops) over the weight GEMMs, per token
	Intensity           float64 // FLOPsPerToken / WeightBytesPerToken — FLOP per weight byte
}

// DecodeWeightBytes returns the EXACT resident weight bytes a dense-model decode streams to
// produce ONE token: every layer's attention (q/k/v/o) and FFN (gate/up/down) weight summed
// across NLayers, plus the LM head once, each in g.WeightDtype (Q8_0 charges codes + per-block
// scales via g.weightBytes, so a quantized decode's memory floor is not understated). With a
// batch of one token there is no reuse to amortize this stream against, so this is the decode
// memory-traffic denominator: the bytes that MUST cross the bus for each token emitted.
func DecodeWeightBytes(g PrefillGeometry) int64 {
	qOut := g.NHeads * g.HeadDim
	kvOut := g.NKVHeads * g.HeadDim
	perLayer := g.weightBytes(qOut, g.DModel) + // q_proj
		g.weightBytes(kvOut, g.DModel) + // k_proj
		g.weightBytes(kvOut, g.DModel) + // v_proj
		g.weightBytes(g.DModel, qOut) + // o_proj
		g.weightBytes(g.DFF, g.DModel) + // ffn_gate
		g.weightBytes(g.DFF, g.DModel) + // ffn_up
		g.weightBytes(g.DModel, g.DFF) // ffn_down
	return perLayer*int64(g.NLayers) + g.weightBytes(g.Vocab, g.DModel) // + lm_head (once)
}

// DecodeFLOPs returns the EXACT multiply-add FLOPs (mul+add counted as 2) the weight GEMMs of a
// dense decode perform per token: the same projection/FFN shapes as prefill at a single token
// (rows=1), summed across NLayers, plus the LM head. Attention-over-KV FLOPs are deliberately
// NOT included — that traffic is the KV-cache side modeled separately by decode_occupancy.go's
// DecodeHBMTraffic, and the two accountings answer different questions and must not be summed
// (the same discipline fusion_traffic.go keeps). For a small model the weight-GEMM term is the
// decode compute; this is the numerator of the roofline intensity.
func DecodeFLOPs(g PrefillGeometry) int64 {
	qOut := int64(g.NHeads * g.HeadDim)
	kvOut := int64(g.NKVHeads * g.HeadDim)
	d := int64(g.DModel)
	ff := int64(g.DFF)
	perLayer := int64(2) * (qOut*d + // q_proj
		kvOut*d + // k_proj
		kvOut*d + // v_proj
		d*qOut + // o_proj
		ff*d + // ffn_gate
		ff*d + // ffn_up
		d*ff) // ffn_down
	return perLayer*int64(g.NLayers) + int64(2)*int64(g.Vocab)*d // + lm_head
}

// DecodeProfile folds the exact per-token counts into the decode roofline for a geometry — the
// one-call entry point answering "why is decode slow, structurally, at this model shape?".
func DecodeProfile(g PrefillGeometry) DecodeRoofline {
	wb := DecodeWeightBytes(g)
	fl := DecodeFLOPs(g)
	return DecodeRoofline{WeightBytesPerToken: wb, FLOPsPerToken: fl, Intensity: ratio(fl, wb)}
}

// MemoryBound reports whether decode is memory-bound at a device ridge point (peak FLOP/s ÷
// peak bytes/s, in FLOP/byte). Decode is memory-bound when its intensity is below the ridge —
// the regime where tok/s is set by memory bandwidth and wider SIMD helps only by letting more
// cores stream weights in parallel. The ridge is the caller's (measured on the target device),
// so this bakes in no hardware constant, matching StageCost.Bound's discipline.
func (r DecodeRoofline) MemoryBound(ridge float64) bool { return r.Intensity < ridge }

// TokPerSecCeiling is the memory-bound UPPER bound on decode throughput (tokens/second) at a
// device's peak memory bandwidth (peakBytesPerSec, bytes/s): peakBytesPerSec ÷
// WeightBytesPerToken. It is a ceiling because a decoder cannot emit a token without streaming
// every weight at least once, so no kernel can beat it — which is exactly what makes it a
// diagnostic: a MEASURED tok/s far below this ceiling means the kernel is not reaching memory
// bandwidth (a scalar or single-threaded weight stream), not that the model is intrinsically
// this slow. peakBytesPerSec is the caller's (measured on the box); no bandwidth constant is
// baked in. A non-positive peak or an empty geometry yields 0 (no divide-by-zero).
func (r DecodeRoofline) TokPerSecCeiling(peakBytesPerSec float64) float64 {
	if peakBytesPerSec <= 0 || r.WeightBytesPerToken <= 0 {
		return 0
	}
	return peakBytesPerSec / float64(r.WeightBytesPerToken)
}

// ObservedTokPerSec converts a single wall-clock observation — tokens produced over a duration
// in seconds — into throughput. It is the ONE measurement issue #3176 reports (~500 tokens in
// >10 minutes ≈ 0.8 tok/s), and exists so an operator can compare a lone stopwatch reading
// against TokPerSecCeiling without re-deriving the division. Non-positive seconds or tokens → 0.
func ObservedTokPerSec(tokens int, seconds float64) float64 {
	if seconds <= 0 || tokens <= 0 {
		return 0
	}
	return float64(tokens) / seconds
}

// DecodeThroughputVerdict grades a measured decode throughput against the memory-bound ceiling,
// turning issue #3176's "is the SIMD lane engaged / does it parallelize" questions into a
// number an operator or a /metrics exporter can act on. Fraction = observed ÷ ceiling: a decode
// streaming weights at bandwidth lands near 1.0 (on the roofline); the issue's report — ~0.8
// tok/s against a Q8 1.5B ceiling of ~100+ tok/s at commodity EPYC bandwidth — lands at ~0.01,
// two orders below the floor, the signature of the scalar reference kernel rather than a
// bandwidth-bound SIMD lane. AtRoofline is true when Fraction ≥ atRooflineFraction.
type DecodeThroughputVerdict struct {
	ObservedTokPerSec float64
	CeilingTokPerSec  float64
	Fraction          float64 // ObservedTokPerSec ÷ CeilingTokPerSec (0 if ceiling unknown)
	AtRoofline        bool    // Fraction ≥ atRooflineFraction — consistent with a bandwidth-bound kernel
}

// atRooflineFraction is the share of the memory-bound ceiling a decode must reach to count as
// "bandwidth-bound" rather than "fell back to the scalar path". A well-threaded SIMD weight
// stream on a many-core box realistically reaches a large fraction of aggregate bandwidth; a
// scalar single-core reference kernel reaches a tiny sliver. 0.5 sits well between those regimes
// so the verdict is robust to the exact peak-bandwidth figure the caller supplies.
const atRooflineFraction = 0.5

// GradeDecodeThroughput builds the verdict from an emitted-token count, its wall-clock seconds,
// the geometry, and the device's peak memory bandwidth. It is the single call a /metrics decode
// exporter or a bench harness makes to answer "is this box's decode where the roofline says it
// should be, or did it fall back?" — no timer of its own, just the caller's one measurement
// against the exact host-computed ceiling.
func GradeDecodeThroughput(tokens int, seconds float64, g PrefillGeometry, peakBytesPerSec float64) DecodeThroughputVerdict {
	obs := ObservedTokPerSec(tokens, seconds)
	ceil := DecodeProfile(g).TokPerSecCeiling(peakBytesPerSec)
	v := DecodeThroughputVerdict{ObservedTokPerSec: obs, CeilingTokPerSec: ceil}
	if ceil > 0 {
		v.Fraction = obs / ceil // fractional ratio (ratio() is integer-only, so divide directly)
		v.AtRoofline = v.Fraction >= atRooflineFraction
	}
	return v
}

// ---- which decode tier is actually registered (fell back to reference, or not?) ----

// DecodeTier describes one registered backend's decode identity: its registry name, its Tier()
// probe label (e.g. "scalar", "avx512", "sm90"), and whether it is an accelerated lane
// (Class() != Reference) or the Reference floor. It is how issue #3176's first question — "is
// the Q8 SIMD decode lane engaged, or did it fall back to the reference path?" — is answered
// IN-PROCESS from the registry instead of by guesswork or wall-clock inference.
type DecodeTier struct {
	Name        string // registry name (Backend.Name())
	Tier        string // capability probe result (Backend.Tier())
	Accelerated bool   // Class() != Reference — a real acceleration over the scalar floor
}

// RegisteredDecodeTiers reports every registered backend's decode identity, Reference first (the
// order Registered() yields). On a build with no accelerator compiled/registered this is exactly
// one entry — cpu-ref, Tier "scalar", not accelerated — which is itself the answer to #3176:
// there is no SIMD lane to engage, so decode necessarily runs on the scalar reference floor
// regardless of the host's AVX-512 width. When a device backend is registered it appears here
// with Accelerated=true, and an operator can confirm the accelerated lane is present before
// blaming throughput on the model.
func RegisteredDecodeTiers() []DecodeTier {
	names := Registered()
	out := make([]DecodeTier, 0, len(names))
	for _, n := range names {
		b, ok := Lookup(n)
		if !ok {
			continue
		}
		out = append(out, DecodeTier{Name: b.Name(), Tier: b.Tier(), Accelerated: b.Class() != Reference})
	}
	return out
}

// HasAcceleratedDecodeTier reports whether ANY registered backend is an accelerated (non-
// Reference) lane. When false, decode necessarily runs on the scalar reference floor — the
// "fell back to the reference path" state issue #3176 asks about — and no SIMD/threaded Q8
// kernel exists to engage regardless of host ISA, so the fix is a kernel/registration, not a
// config knob. This is the one-bit form of the tier report a health check or /metrics gauge reads.
func HasAcceleratedDecodeTier() bool {
	for _, t := range RegisteredDecodeTiers() {
		if t.Accelerated {
			return true
		}
	}
	return false
}
