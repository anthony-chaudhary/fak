package compute

import "fmt"

// kvprecision.go — KV-cache precision tiers (#1047, child of #1045).
//
// The in-kernel KV is F32 with THREE rows per cached position per layer (pre-RoPE K,
// post-RoPE K, V — see EstimateKVStoreBytes / capacity.go pre-RoPE-K note), the heaviest
// possible layout (~0.75 MiB/token for Qwen3.6-27B). Auto-sizing (#1046) picks a context
// from kv_bytes_per_token, so halving that token cost ~doubles the context that fits the
// same box — the cheapest capacity win available, the llama.cpp `-ctk/-ctv q8_0` analogue.
//
// A KVPrecision tier names HOW dense the resident KV is. It is the dtype + row-count knob
// the issue asks EstimateKVStoreBytes / KVConfig to learn, so the planner's arithmetic
// matches the layout that will actually be resident. Crucially it preserves EVICT
// CORRECTNESS: the denser q8 tier quantizes only the two ATTENDED rows (post-RoPE K, V) to
// q8_0 and keeps the pre-RoPE K row F32, because Evict re-positions a survivor by a single
// rotation OF THE PRE-RoPE K (capacity.go pre-RoPE-K note) and that must stay bit-exact. So
// q8 is ~2x denser than f32 — not the naive 4x of quantizing every row — and it is honest
// about why: the exact-evict row cannot be lossy.
//
// This file is the PLANNER/arithmetic half, the same split kvresidency.go (#1048) draws: it
// teaches the byte estimate and exposes the selector + an auto-select policy, performing no
// allocation and holding no model state. The denser KVStore *realization* (the byte movement
// that makes a position's K/V actually q8_0-resident, and the live coherence run) is the
// engine half — analogous to the kvmmu adapter kvresidency defers — and is not in this file.

// KVPrecision is the storage density tier of the in-kernel KV cache: the dtype + row-count
// that EstimateKVStoreBytes uses, and the selector the serve path exposes. Its zero value is
// KVPrecisionF32, so a KVConfig that does not set it estimates byte-identically to before the
// tier existed.
type KVPrecision uint8

const (
	// KVPrecisionF32 is the reference layout: all THREE rows (pre-RoPE K, post-RoPE K, V)
	// stored F32. The heaviest and the most exact — the only tier the KVStore realizes today.
	KVPrecisionF32 KVPrecision = iota
	// KVPrecisionQ8 quantizes the two ATTENDED rows (post-RoPE K, V) to q8_0 while keeping the
	// pre-RoPE K row F32 so Evict's single-rotation re-positioning stays exact. ~2x denser than
	// f32, evict-correct by construction. The storage is MIXED (f32 + q8_0), hence StorageLabel.
	KVPrecisionQ8
)

// String renders the tier as its short selector token ("f32" | "q8"), or "kvprec?" for an
// unknown value. It is the token ParseKVPrecision accepts.
func (p KVPrecision) String() string {
	switch p {
	case KVPrecisionF32:
		return "f32"
	case KVPrecisionQ8:
		return "q8"
	default:
		return "kvprec?"
	}
}

// StorageLabel is the bounded dtype/storage label for a MemoryDemand.DType field. F32 is a
// uniform "f32"; the q8 tier is "mixed" (the documented vocabulary value), because it stores
// the pre-RoPE K row f32 and the attended K/V rows q8_0 — not a single uniform dtype.
func (p KVPrecision) StorageLabel() string {
	switch p {
	case KVPrecisionQ8:
		return "mixed"
	default:
		return F32.String()
	}
}

// ParseKVPrecision maps a serve-path selector token to a tier. It accepts "f32" and both "q8"
// and "q8_0" (the wire dtype the attended rows use). An empty string selects the F32 default;
// anything else is an error so a typo refuses rather than silently picking a tier.
func ParseKVPrecision(s string) (KVPrecision, error) {
	switch s {
	case "", "f32":
		return KVPrecisionF32, nil
	case "q8", "q8_0":
		return KVPrecisionQ8, nil
	default:
		return KVPrecisionF32, fmt.Errorf("compute: unknown kv precision %q (want f32 | q8)", s)
	}
}

// perTokenPerLayerBytes is the resident bytes for one cached position in one layer at this
// tier, given elemsPerRow = NumKVHeads*HeadDim elements per row. It is the heart of the
// dtype + row-count the issue asks EstimateKVStoreBytes to learn. A non-positive elemsPerRow
// (incomplete geometry) yields 0 — the fail-open floor EstimateKVStoreBytes already promises.
func (p KVPrecision) perTokenPerLayerBytes(elemsPerRow int64) int64 {
	if elemsPerRow <= 0 {
		return 0
	}
	switch p {
	case KVPrecisionQ8:
		// pre-RoPE K stays f32 (exact evict re-positioning); post-RoPE K and V are q8_0.
		kRaw := saturatingMulInt64(elemsPerRow, 4)
		q8Row := kvQ8RowBytes(elemsPerRow)
		return saturatingAddInt64(kRaw, q8Row, q8Row)
	default: // KVPrecisionF32 — three f32 rows (Kraw + K + V)
		return saturatingMulInt64(elemsPerRow, 3, 4)
	}
}

// kvQ8RowBytes is the resident bytes of one row of `elems` values stored as q8_0: one int8
// code per value plus one f16 scale per 32-value block — exactly llama.cpp's block_q8_0 wire
// size (34 bytes / 32 elems), the same {Block:32, Bits:8} scheme QuantSpec documents.
func kvQ8RowBytes(elems int64) int64 {
	if elems <= 0 {
		return 0
	}
	const block = 32
	blocks := (elems + block - 1) / block
	return saturatingAddInt64(elems, saturatingMulInt64(blocks, 2)) // int8 codes + per-block f16 scale
}

// AutoSelectKVPrecision is the "auto-select a denser KV when f32 would force a tiny context"
// policy. Given a per-pool KV budget (the headroom-adjusted bytes already free for KV) and a
// desired window, it KEEPS f32 (exact, heaviest) when f32 already fits wantTokens, and steps
// down to q8 only when f32 cannot — and only when q8 actually buys more tokens. It returns the
// chosen tier and a human reason naming the tradeoff for the operator log; it never logs or
// mutates anything itself (the planner owns the log line). It is fail-open: incomplete geometry
// or a non-positive budget/want keeps the exact f32 tier, never inventing a denser one.
func AutoSelectKVPrecision(cfg KVConfig, budget int64, wantTokens int) (KVPrecision, string) {
	f32Cfg := cfg
	f32Cfg.Precision = KVPrecisionF32
	f32PerToken := EstimateKVStoreBytes(f32Cfg, 1)
	if f32PerToken <= 0 || budget <= 0 || wantTokens <= 0 {
		return KVPrecisionF32, "kv-precision=f32 (fail-open: incomplete KV geometry or no budget; kept the exact tier)"
	}
	f32Fit := budget / f32PerToken
	if f32Fit >= int64(wantTokens) {
		return KVPrecisionF32, fmt.Sprintf("kv-precision=f32 (fits %d>=want %d tokens; kept the exact tier)", f32Fit, wantTokens)
	}
	q8Cfg := cfg
	q8Cfg.Precision = KVPrecisionQ8
	q8PerToken := EstimateKVStoreBytes(q8Cfg, 1)
	q8Fit := budget / q8PerToken
	if q8Fit <= f32Fit {
		return KVPrecisionF32, fmt.Sprintf("kv-precision=f32 (q8 would not raise the %d-token fit; kept the exact tier)", f32Fit)
	}
	return KVPrecisionQ8, fmt.Sprintf(
		"kv-precision=q8 (f32 forced a tiny %d-token context < want %d; q8 lifts it to %d, ~%.2fx — attended K/V are q8_0-lossy, pre-RoPE K kept f32 so evict stays exact)",
		f32Fit, wantTokens, q8Fit, float64(q8Fit)/float64(f32Fit),
	)
}
