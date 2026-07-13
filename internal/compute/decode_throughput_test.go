package compute

import "testing"

// decodeFloatNear is a local tolerance check so the roofline ratios can be asserted without
// pulling a helper across test files (which would collide in this package).
func decodeFloatNear(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d < 1e-9
}

// smallDecodeGeom is a hand-computable F32 geometry: with F32 weightBytes(out,in)=out*in*4 the
// exact per-token counts fall out by arithmetic, so the golden numbers below are checkable by
// hand and pin DecodeWeightBytes / DecodeFLOPs against silent drift.
//
//	qOut = NHeads*HeadDim = 4 ; kvOut = NKVHeads*HeadDim = 2
//	per-layer bytes = 4·4·4 (q) + 2·4·4 (k) + 2·4·4 (v) + 4·4·4 (o)
//	                + 8·4·4 (gate) + 8·4·4 (up) + 4·8·4 (down) = 576
//	weight bytes    = 576·2 layers + 10·4·4 (lm_head) = 1312
//	per-layer flops = 2·(16+8+8+16+32+32+32) = 288 ; flops = 288·2 + 2·10·4 = 656
//	intensity       = 656 / 1312 = 0.5
func smallDecodeGeom() PrefillGeometry {
	return PrefillGeometry{
		DModel: 4, NHeads: 2, HeadDim: 2, NKVHeads: 1,
		DFF: 8, NLayers: 2, Vocab: 10, WeightDtype: F32,
	}
}

func TestDecodeWeightBytesExact(t *testing.T) {
	if got := DecodeWeightBytes(smallDecodeGeom()); got != 1312 {
		t.Fatalf("DecodeWeightBytes = %d, want 1312", got)
	}
}

func TestDecodeFLOPsExact(t *testing.T) {
	if got := DecodeFLOPs(smallDecodeGeom()); got != 656 {
		t.Fatalf("DecodeFLOPs = %d, want 656", got)
	}
}

func TestDecodeProfileIntensityAndMemoryBound(t *testing.T) {
	p := DecodeProfile(smallDecodeGeom())
	if p.WeightBytesPerToken != 1312 || p.FLOPsPerToken != 656 {
		t.Fatalf("profile counts = (%d,%d), want (1312,656)", p.WeightBytesPerToken, p.FLOPsPerToken)
	}
	if !decodeFloatNear(p.Intensity, 0.5) {
		t.Fatalf("Intensity = %v, want 0.5", p.Intensity)
	}
	// Decode is memory-bound at any realistic CPU ridge (tens–hundreds of FLOP/byte) and
	// compute-bound only below its own intensity — the crux of why decode is slow on CPU.
	if !p.MemoryBound(10) {
		t.Fatal("intensity 0.5 must be memory-bound at ridge 10")
	}
	if p.MemoryBound(0.1) {
		t.Fatal("intensity 0.5 must NOT be memory-bound at ridge 0.1")
	}
}

func TestTokPerSecCeiling(t *testing.T) {
	p := DecodeProfile(smallDecodeGeom()) // WeightBytesPerToken = 1312
	// peak == weight bytes/token ⇒ exactly one token/second.
	if got := p.TokPerSecCeiling(1312); !decodeFloatNear(got, 1.0) {
		t.Fatalf("TokPerSecCeiling(1312) = %v, want 1.0", got)
	}
	if got := p.TokPerSecCeiling(13120); !decodeFloatNear(got, 10.0) {
		t.Fatalf("TokPerSecCeiling(13120) = %v, want 10.0", got)
	}
	// Zero-guards: no bandwidth, or an empty geometry, yields 0 (never a divide-by-zero).
	if got := p.TokPerSecCeiling(0); got != 0 {
		t.Fatalf("TokPerSecCeiling(0) = %v, want 0", got)
	}
	if got := (DecodeRoofline{}).TokPerSecCeiling(1e9); got != 0 {
		t.Fatalf("empty-roofline ceiling = %v, want 0", got)
	}
}

func TestFLOPBoundTokPerSecCeiling(t *testing.T) {
	p := DecodeProfile(smallDecodeGeom()) // FLOPsPerToken = 656
	// peak FLOP/s == FLOPs/token ⇒ exactly one token/second (compute-bound).
	if got := p.FLOPBoundTokPerSecCeiling(656); !decodeFloatNear(got, 1.0) {
		t.Fatalf("FLOPBoundTokPerSecCeiling(656) = %v, want 1.0", got)
	}
	if got := p.FLOPBoundTokPerSecCeiling(6560); !decodeFloatNear(got, 10.0) {
		t.Fatalf("FLOPBoundTokPerSecCeiling(6560) = %v, want 10.0", got)
	}
	// Zero-guards: no compute peak, or an empty geometry, yields 0 (never a divide-by-zero).
	if got := p.FLOPBoundTokPerSecCeiling(0); got != 0 {
		t.Fatalf("FLOPBoundTokPerSecCeiling(0) = %v, want 0", got)
	}
	if got := (DecodeRoofline{}).FLOPBoundTokPerSecCeiling(1e9); got != 0 {
		t.Fatalf("empty-roofline FLOP ceiling = %v, want 0", got)
	}
}

func TestRooflineTokPerSecCeiling(t *testing.T) {
	p := DecodeProfile(smallDecodeGeom()) // FLOPsPerToken = 656, WeightBytesPerToken = 1312
	// Memory binds: compute ceiling 10 tok/s (6560 FLOP/s), memory ceiling 1 tok/s (1312 B/s).
	if got := p.RooflineTokPerSecCeiling(6560, 1312); !decodeFloatNear(got, 1.0) {
		t.Fatalf("roofline(compute 10, memory 1) = %v, want the min 1.0", got)
	}
	// Compute binds: compute ceiling 1 tok/s (656 FLOP/s), memory ceiling 10 tok/s (13120 B/s).
	if got := p.RooflineTokPerSecCeiling(656, 13120); !decodeFloatNear(got, 1.0) {
		t.Fatalf("roofline(compute 1, memory 10) = %v, want the min 1.0", got)
	}
	// One peak unknown ⇒ the other single bound stands (drop-out discipline).
	if got := p.RooflineTokPerSecCeiling(0, 1312); !decodeFloatNear(got, 1.0) {
		t.Fatalf("roofline(_, memory 1) = %v, want 1.0 (memory bound alone)", got)
	}
	if got := p.RooflineTokPerSecCeiling(656, 0); !decodeFloatNear(got, 1.0) {
		t.Fatalf("roofline(compute 1, _) = %v, want 1.0 (compute bound alone)", got)
	}
	// Both peaks unknown ⇒ 0 (no bound to report).
	if got := p.RooflineTokPerSecCeiling(0, 0); got != 0 {
		t.Fatalf("roofline(0, 0) = %v, want 0", got)
	}
}

func TestObservedTokPerSec(t *testing.T) {
	// The issue's one measurement: ~500 tokens in >10 minutes.
	if got := ObservedTokPerSec(500, 600); !decodeFloatNear(got, 500.0/600.0) {
		t.Fatalf("ObservedTokPerSec(500,600) = %v, want %v", got, 500.0/600.0)
	}
	if got := ObservedTokPerSec(0, 600); got != 0 {
		t.Fatalf("ObservedTokPerSec(0,600) = %v, want 0", got)
	}
	if got := ObservedTokPerSec(5, 0); got != 0 {
		t.Fatalf("ObservedTokPerSec(5,0) = %v, want 0", got)
	}
}

func TestGradeDecodeThroughputIssueScenario(t *testing.T) {
	g := smallDecodeGeom() // WeightBytesPerToken = 1312
	// Pick a peak bandwidth giving a 100 tok/s ceiling: 1312 * 100 = 131200 bytes/s.
	const peak = 131200.0
	// The issue: ~500 tokens over 600 s ≈ 0.833 tok/s — two orders below the ceiling. This is
	// the quantitative signature of the scalar reference path, not a bandwidth-bound SIMD lane.
	low := GradeDecodeThroughput(500, 600, g, peak)
	if !decodeFloatNear(low.CeilingTokPerSec, 100.0) {
		t.Fatalf("ceiling = %v, want 100", low.CeilingTokPerSec)
	}
	if low.AtRoofline {
		t.Fatalf("0.83/100 tok/s must NOT be at-roofline (fraction=%v)", low.Fraction)
	}
	if low.Fraction >= 0.05 {
		t.Fatalf("fraction = %v, want far below the roofline (<0.05)", low.Fraction)
	}
	// A decode streaming weights near bandwidth (60 tok/s of a 100 tok/s ceiling) lands on the
	// roofline — the "SIMD lane engaged and threads scale" regime.
	hi := GradeDecodeThroughput(60, 1, g, peak)
	if !hi.AtRoofline {
		t.Fatalf("60/100 tok/s must be at-roofline (fraction=%v)", hi.Fraction)
	}
	// Unknown bandwidth ⇒ no ceiling, no verdict (fraction 0, not at-roofline).
	if v := GradeDecodeThroughput(500, 600, g, 0); v.AtRoofline || v.Fraction != 0 {
		t.Fatalf("no-bandwidth grade = %+v, want fraction 0 / not at-roofline", v)
	}
}

func TestQ8DecodeStreamsFewerBytesThanF32(t *testing.T) {
	// Same shape, Q8_0 vs F32: the quantized decode must stream strictly fewer weight bytes per
	// token (that is the point of lean-Q8), and still yields a positive tok/s ceiling. This
	// exercises the Q8_0 weightBytes path without pinning its exact block-layout arithmetic.
	shape := PrefillGeometry{DModel: 1536, NHeads: 12, HeadDim: 128, NKVHeads: 2, DFF: 8960, NLayers: 28, Vocab: 151936}
	f32 := shape
	f32.WeightDtype = F32
	q8 := shape
	q8.WeightDtype = Q8_0
	bf, bq := DecodeWeightBytes(f32), DecodeWeightBytes(q8)
	if !(bq > 0 && bq < bf) {
		t.Fatalf("Q8 weight bytes = %d, F32 = %d; want 0 < Q8 < F32", bq, bf)
	}
	if got := DecodeProfile(q8).TokPerSecCeiling(200e9); !(got > 0) {
		t.Fatalf("Q8 ceiling at 200 GB/s = %v, want > 0", got)
	}
}

func TestRegisteredDecodeTiersReferenceFloor(t *testing.T) {
	ref := Default() // the Reference floor is always registered
	if ref == nil {
		t.Fatal("Default() returned nil — no backend registered")
	}
	tiers := RegisteredDecodeTiers()
	if len(tiers) == 0 {
		t.Fatal("RegisteredDecodeTiers() is empty")
	}
	var refTier *DecodeTier
	anyAccel := false
	for i := range tiers {
		if tiers[i].Accelerated {
			anyAccel = true
		}
		if tiers[i].Name == ref.Name() {
			refTier = &tiers[i]
		}
	}
	if refTier == nil {
		t.Fatalf("reference backend %q missing from tiers %+v", ref.Name(), tiers)
	}
	// The reference floor is, by definition, not an accelerated lane and carries a Tier label.
	if refTier.Accelerated {
		t.Fatalf("reference backend %q reported Accelerated=true", refTier.Name)
	}
	if refTier.Tier == "" {
		t.Fatalf("reference backend %q has empty Tier label", refTier.Name)
	}
	// HasAcceleratedDecodeTier must agree with the tier list (self-consistent, never flaky even
	// if a device backend is registered under a build tag on some host).
	if HasAcceleratedDecodeTier() != anyAccel {
		t.Fatalf("HasAcceleratedDecodeTier()=%v disagrees with tiers %+v", HasAcceleratedDecodeTier(), tiers)
	}
}
