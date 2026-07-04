package cachemeta

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// qwenKVSpanBytes returns the resident bytes of a `tokens`-long KV span at the exact
// f32 layout and at the requantized q8 tier, straight from compute's SHIPPED density
// arithmetic (EstimateKVStoreBytes) for a realistic geometry. Deriving both sizes from
// the real #1045/#1047 model — instead of a hand-picked ratio — is what makes this a
// #1474 witness and not a re-run of the generic #523 compress-demote test: the q8 size
// is honestly ~1.96x denser (the pre-RoPE K row stays f32 for evict-exactness), never
// a fabricated 4x.
func qwenKVSpanBytes(tokens int) (f32Bytes, q8Bytes int64) {
	geom := compute.KVConfig{NumLayers: 48, NumKVHeads: 8, HeadDim: 128}
	f32 := geom
	f32.Precision = compute.KVPrecisionF32
	q8 := geom
	q8.Precision = compute.KVPrecisionQ8
	return compute.EstimateKVStoreBytes(f32, tokens), compute.EstimateKVStoreBytes(q8, tokens)
}

// TestQuantizedDemoteKeepsSpanResidentUnderL2Pressure is the #1474 witness. A span
// whose EXACT f32 layout is too costly to retain even in the coldest attendable tier
// with room (CXL) EVICTS under exact-only economics — a full re-prefill on the next
// request. Supplied a quality-PROVEN q8 requantize-down target, the same span instead
// COMPRESS-DEMOTES: it stays resident and attendable in CXL at ~half the bytes, keeping
// its (quality-bounded) context per byte instead of dropping it. The refute guard: an
// unmeasured or over-bound quantization is refused and still evicts.
func TestQuantizedDemoteKeepsSpanResidentUnderL2Pressure(t *testing.T) {
	const tokens = 1000
	f32Bytes, q8Bytes := qwenKVSpanBytes(tokens)
	if !(q8Bytes > 0 && q8Bytes < f32Bytes) {
		t.Fatalf("compute density model must make q8 a real downgrade of f32: f32=%d q8=%d", f32Bytes, q8Bytes)
	}

	// Only CXL (and disk) have room: HBM/DRAM/NUMA-far are full, so the coldest tier
	// with room the planner walks to is CXL — byte-addressable and attendable in place.
	profiles := DefaultTierProfiles()
	if !profiles[TierCXL].AttendableInPlace() {
		t.Fatalf("precondition: CXL must be attendable in place for a requantize-AND-KEEP witness")
	}

	// Pick a recompute cost strictly BETWEEN staging the q8 span and staging the exact
	// span into CXL, derived from the real tier profile so the witness is not brittle to
	// the exact density ratio: exact staging must lose to recompute (→ exact evicts)
	// while q8 staging must beat it (→ q8 stays resident).
	cxl := profiles[TierCXL]
	stageExact := stageNanos(f32Bytes, cxl)
	stageQ8 := stageNanos(q8Bytes, cxl)
	perTokenPrefillNanos := ((stageExact + stageQ8) / 2) / tokens
	if !RetainCheaperThanRecompute(q8Bytes, tokens, perTokenPrefillNanos, cxl) ||
		RetainCheaperThanRecompute(f32Bytes, tokens, perTokenPrefillNanos, cxl) {
		t.Fatalf("cost setup wrong: want q8 retain-cheaper AND exact retain-costlier at %d ns/token (stageExact=%d stageQ8=%d)",
			perTokenPrefillNanos, stageExact, stageQ8)
	}

	base := baseReq()
	base.Pressure = TierPressure{TierHBM: 1.0, TierDRAM: 1.0, TierNUMAFar: 1.0}
	base.Tokens = tokens
	base.PerTokenPrefillNanos = perTokenPrefillNanos
	base.SizeBytes = f32Bytes

	// (1) Exact-only economics (no quantized target): the span evicts.
	if d := PlanPlacement(base); d.Action != ActionEvict {
		t.Fatalf("exact-only span should evict, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
	}

	// (2) A quality-proven q8 target flips evict -> compress_demote (stays resident in CXL).
	target := QuantizedDemoteTarget{
		To:            compute.KVPrecisionQ8,
		ResidentBytes: q8Bytes,
		Quality:       QualityEvidence{Measured: true, QualityDelta: 0.02, MaxQualityDelta: 0.05},
	}
	req := base
	if !target.ApplyTo(&req) {
		t.Fatalf("q8 target should arm the compress-demote lever (q8=%d < exact=%d)", q8Bytes, f32Bytes)
	}
	d := PlanPlacement(req)
	if d.Action != ActionCompressDemote || d.ToTier != TierCXL {
		t.Fatalf("quality-proven q8 span should compress-demote to CXL, got %s->%s (%s)", d.Action, d.ToTier, d.Reason)
	}
	if d.EstMoveBytes != q8Bytes {
		t.Fatalf("compress-demote should stage the requantized bytes %d, got %d", q8Bytes, d.EstMoveBytes)
	}

	// Track-1 effective-context-per-byte gain: q8 keeps ~1.96x more span per byte.
	if gain := target.EffectiveContextPerByteGain(f32Bytes); gain <= 1.5 {
		t.Fatalf("q8 requantize-down should keep >1.5x context per byte, got %.3fx", gain)
	}

	// (3) Refute: an unmeasured or over-bound quantization is not a proven hit — it
	// still evicts, never kept under a quality claim fak cannot stand behind.
	for _, tc := range []struct {
		name string
		q    QualityEvidence
	}{
		{name: "unmeasured", q: QualityEvidence{}},
		{name: "over_bound", q: QualityEvidence{Measured: true, QualityDelta: 0.20, MaxQualityDelta: 0.05}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			bad := QuantizedDemoteTarget{To: compute.KVPrecisionQ8, ResidentBytes: q8Bytes, Quality: tc.q}
			r := base
			if !bad.ApplyTo(&r) {
				t.Fatalf("target should still arm the size lever so PlanPlacement's quality gate decides")
			}
			if got := PlanPlacement(r); got.Action != ActionEvict {
				t.Fatalf("unproven quantization must evict, got %s->%s (%s)", got.Action, got.ToTier, got.Reason)
			}
		})
	}
}

// TestQuantizedDemoteTargetOnlyArmsRealDowngrade: a target that is not strictly smaller
// than the exact span buys no capacity, so ApplyTo refuses to arm the lever and leaves
// the request byte-identical (the planner behaves exactly as with no target).
func TestQuantizedDemoteTargetOnlyArmsRealDowngrade(t *testing.T) {
	req := baseReq()
	req.SizeBytes = 8 << 20

	for _, tc := range []struct {
		name     string
		resident int64
	}{
		{name: "equal_size", resident: 8 << 20},
		{name: "larger", resident: 16 << 20},
		{name: "zero", resident: 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			target := QuantizedDemoteTarget{
				To:            compute.KVPrecisionQ8,
				ResidentBytes: tc.resident,
				Quality:       QualityEvidence{Measured: true, QualityDelta: 0.01, MaxQualityDelta: 0.05},
			}
			if target.IsDowngrade(req.SizeBytes) {
				t.Fatalf("resident=%d is not a downgrade of %d", tc.resident, req.SizeBytes)
			}
			r := req
			if target.ApplyTo(&r) {
				t.Fatalf("non-downgrade target must not arm the lever")
			}
			if r.CompressedSizeBytes != 0 || r.Quality != (QualityEvidence{}) {
				t.Fatalf("non-downgrade target must leave the request byte-identical, got size=%d q=%+v", r.CompressedSizeBytes, r.Quality)
			}
			if gain := target.EffectiveContextPerByteGain(req.SizeBytes); gain != 0 {
				t.Fatalf("non-downgrade gain should be 0, got %.3f", gain)
			}
		})
	}
}
