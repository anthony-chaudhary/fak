package cachemeta

import "github.com/anthony-chaudhary/fak/internal/compute"

// quantized_demote.go ties the KV precision ladder (internal/compute.KVPrecision,
// #1045/#1047) to PlanPlacement's compress-and-demote lever (#523) so that a
// quantized-KV representation is a first-class L2 demote TARGET rather than an
// anonymous byte count (#1474).
//
// The move it enables: under host-memory (L2) pressure a span whose EXACT (f32,
// three-row) layout is too costly to retain even in the coldest tier with room can
// be REQUANTIZED DOWN to a denser tier (compute's q8_0 attended-row layout) and kept
// resident — byte-addressable and attendable in place — at a fraction of the bytes,
// instead of demoted-exact (which may not fit under the retain-vs-recompute bar) or
// evicted (which loses it to a full re-prefill). It is admitted ONLY on PROVEN
// quality (QualityEvidence.Acceptable()), exactly as ActionCompressDemote already
// gates: an unmeasured or over-bound requantization is refused and the span is
// evicted, never kept under a quality claim fak cannot stand behind.
//
// The fence (from #1474): cachemeta owns no bytes and never requantizes. The
// requantization codec and the MEASURED QualityEvidence live in the engine (#118);
// the resident byte size of the denser tier is compute's honest density arithmetic
// (compute.EstimateKVStoreBytes) — which deliberately keeps the pre-RoPE K row f32
// so Evict's single-rotation re-positioning stays bit-exact, making q8 ~2x denser,
// NOT the naive 4x of quantizing every row. This file only PACKAGES that
// engine/compute-supplied sized representation as the (CompressedSizeBytes, Quality)
// pair PlanPlacement reads, and records which precision tier it came from so the
// sweep decision is auditable. The lever stays inert until a caller supplies a real
// target, so a box that never quantizes plans byte-identically to before #1474.

// QuantizedDemoteTarget is the requantize-down demote target the live lowering
// supplies to PlanPlacement's compress-and-demote lever. It names the denser
// precision tier a span would be requantized into (To — provenance that ties the
// decision to the #1045/#1047 ladder), the engine/compute-measured RESIDENT byte
// size of that representation (ResidentBytes), and the engine-measured error bound
// that gates admission (Quality). It carries no bytes itself.
type QuantizedDemoteTarget struct {
	// To is the denser KV precision tier the span is requantized into (e.g.
	// compute.KVPrecisionQ8). It is provenance for the placement decision — which
	// tier bought the capacity — not a size; the size is ResidentBytes.
	To compute.KVPrecision
	// ResidentBytes is the resident byte size of the requantized representation, as
	// computed by compute.EstimateKVStoreBytes for the To tier (the engine realizes
	// the actual byte movement). It is only a real demote target when strictly
	// smaller than the exact span — see IsDowngrade.
	ResidentBytes int64
	// Quality is the engine-measured error bound of the requantized representation.
	// PlanPlacement admits the target ONLY when Quality.Acceptable(); an unmeasured
	// or over-bound target still evicts.
	Quality QualityEvidence
}

// IsDowngrade reports whether this target is a genuine requantize-DOWN of a span of
// exactBytes: the requantized representation must be resident (ResidentBytes > 0) and
// strictly smaller than the exact span. A target that is not smaller buys no capacity
// and is not offered to the compress-and-demote lever (requantizing across or up is
// not an L2 capacity win).
func (t QuantizedDemoteTarget) IsDowngrade(exactBytes int64) bool {
	return t.ResidentBytes > 0 && t.ResidentBytes < exactBytes
}

// EffectiveContextPerByteGain reports how many times more span the requantized
// representation keeps resident per byte versus the exact layout — exactBytes /
// ResidentBytes (≈1.96x for compute's f32→q8 tier, whose evict-exact pre-RoPE K row
// keeps it honestly below the naive 2x). It is the effective-context-per-byte gain
// the compress-and-demote lever buys under L2 pressure, and returns 0 when the target
// is not a real downgrade.
func (t QuantizedDemoteTarget) EffectiveContextPerByteGain(exactBytes int64) float64 {
	if !t.IsDowngrade(exactBytes) {
		return 0
	}
	return float64(exactBytes) / float64(t.ResidentBytes)
}

// ApplyTo arms PlanPlacement's compress-and-demote lever on req from this
// requantize-down target, using req.SizeBytes as the exact source size, and reports
// whether it armed. When the target is a real downgrade it sets req.CompressedSizeBytes
// to the requantized ResidentBytes and req.Quality to the measured evidence; when it
// is not (ResidentBytes not strictly smaller than the exact span), it leaves req
// byte-identical so the planner behaves exactly as it would with no target.
//
// It wires the MEASURED Quality through unchanged — an unmeasured or over-bound
// target still arms the size lever, but PlanPlacement's Quality.Acceptable() gate then
// refuses it and the span evicts. Keeping the quality gate in PlanPlacement (its one
// home) is what makes an unproven quantization evict for the right reason rather than
// silently disappearing behind a zeroed size.
func (t QuantizedDemoteTarget) ApplyTo(req *PlacementRequest) bool {
	if req == nil || !t.IsDowngrade(req.SizeBytes) {
		return false
	}
	req.CompressedSizeBytes = t.ResidentBytes
	req.Quality = t.Quality
	return true
}
