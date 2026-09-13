package model

// ExpertCacheReceiptSchema identifies the routed-expert cache measurement wire
// format. It is the locked schema every leaf of the MoE expert-cache benchmark
// suite emits and consumes (moe-expert-cache-benchmarks umbrella).
const ExpertCacheReceiptSchema = "fak.moe-expert-cache/v1"

// The locked provenance rungs. A synthetic or replayed trace is "simulated"; a
// physical appliance run is "hardware". The rung is a first-class field so the
// simulated-vs-hardware distinction is structural, never prose, and a synthetic
// receipt can never masquerade as a hardware measurement.
const (
	ExpertCacheMeasurementSimulated = "simulated"
	ExpertCacheMeasurementHardware  = "hardware"
)

// ExpertCacheReceipt is ONE routed-expert cache measurement at a model shape.
// The JSON tags are the locked schema the field-lock test
// (TestExpertCacheReceiptRequiredFields) pins - adding, renaming, or dropping a
// field without updating ExpertCacheRequiredFields fails that test on purpose,
// mirroring the deepseekbench.Row + RequiredFields discipline.
type ExpertCacheReceipt struct {
	// Provenance / honesty - read these BEFORE any hit rate or throughput number.
	Schema      string `json:"schema"`      // always ExpertCacheReceiptSchema
	Measurement string `json:"measurement"` // "simulated" | "hardware"

	// Model identity + precision/quant axis.
	ModelID   string `json:"model_id"`  // e.g. "deepseek-v4.1-flash"
	Arch      string `json:"arch"`      // e.g. "deepseek_v41_text"
	Precision string `json:"precision"` // e.g. "fp8" | "bf16" | "gguf-q2_k"
	Quant     string `json:"quant"`     // e.g. "q2_k" | "iq2_xxs" | "native-fp4" | "none"

	// Model shape - the axes the benchmark drives.
	Layers   int `json:"layers"`    // 40 for V4.1 text
	Experts  int `json:"experts"`   // 384 routed experts
	TopK     int `json:"top_k"`     // 6 experts/token
	MoeInter int `json:"moe_inter"` // 2304 expert intermediate width

	// Cache sizing + residency outcome.
	RingByteCap  int64   `json:"ring_byte_cap"`  // shipped expert-ring budget in bytes
	HitCount     int64   `json:"hit_count"`      // experts served resident
	MissCount    int64   `json:"miss_count"`     // experts streamed from the next tier
	HitRate      float64 `json:"hit_rate"`       // HitCount / (HitCount+MissCount)
	HitRateKnown bool    `json:"hit_rate_known"` // false when the denominator is zero

	// Per-tier bytes read per token.
	BytesReadDRAM int64 `json:"bytes_read_dram"` // bytes read from the resident tier
	BytesReadNVMe int64 `json:"bytes_read_nvme"` // bytes read from the streamed tier

	// Throughput / latency.
	PrefillToksPerSec float64 `json:"prefill_toks_per_s"`
	DecodeToksPerSec  float64 `json:"decode_toks_per_s"`
	TTFTMillis        float64 `json:"ttft_ms"`

	// Resource envelope.
	PeakMemoryBytes int64 `json:"peak_memory_bytes"`
}

// ExpertCacheRequiredFields is the locked set of JSON keys every emitted receipt
// MUST carry. The field-lock test marshals a fully populated receipt and asserts
// each key is present and that no other key appears, so the schema cannot
// silently drift.
func ExpertCacheRequiredFields() []string {
	return []string{
		"schema", "measurement",
		"model_id", "arch", "precision", "quant",
		"layers", "experts", "top_k", "moe_inter",
		"ring_byte_cap", "hit_count", "miss_count", "hit_rate", "hit_rate_known",
		"bytes_read_dram", "bytes_read_nvme",
		"prefill_toks_per_s", "decode_toks_per_s", "ttft_ms",
		"peak_memory_bytes",
	}
}

// ExpertCacheHitRateKnown reports whether a hit rate is meaningful for the given
// hit/miss counters. A zero denominator means there was no measured access, so
// the rate is unknown and must not be reported as a phantom 0% (or 1).
func ExpertCacheHitRateKnown(hitCount, missCount int64) bool {
	return hitCount+missCount > 0
}

// ExpertCacheHitRate returns the hit rate for the given counters. When the
// denominator is zero it returns 0 and known=false so callers cannot surface a
// fabricated rate.
func ExpertCacheHitRate(hitCount, missCount int64) (rate float64, known bool) {
	total := hitCount + missCount
	if total <= 0 {
		return 0, false
	}
	return float64(hitCount) / float64(total), true
}
