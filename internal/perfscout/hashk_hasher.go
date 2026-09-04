package perfscout

// Salts defined by the HashK embedding compressor (airawatraj).
var HashKSalt = [2]uint64{
	0x517cc1b727220a95, // Subtable 0 salt
	0x9e3779b97f4a7c15, // Subtable 1 salt
}

const (
	// SplitMixConstant is the standard Golden Ratio 64-bit additive increment.
	SplitMixConstant uint64 = 0x9e3779b97f4a7c15
	// HashKMultiplier is the polynomial congruential multiplier.
	HashKMultiplier uint64 = 2862933555777941757
	// HashKHeadPrime is the prime step applied per head.
	HashKHeadPrime uint64 = 998244353
)

// SplitMix64 computes a 64-bit deterministic hash state step.
func SplitMix64(x uint64) uint64 {
	z := x + SplitMixConstant
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// ComputeHashKSlot computes the compressed subtable slot index for a given token index.
func ComputeHashKSlot(localIdx uint64, subtableIdx int, headIdx uint64, numSlots uint64) uint64 {
	if numSlots == 0 {
		return 0
	}
	salt := HashKSalt[subtableIdx%2]
	xSub := (localIdx+1)*HashKMultiplier + salt + headIdx*HashKHeadPrime
	hashed := SplitMix64(xSub)
	return hashed % numSlots
}

// HashKRouter manages dual-subtable slot mapping for large embedding table compression.
type HashKRouter struct {
	TotalVocabulary uint64 // e.g. 320,001,536 rows
	CompressionRate uint64 // e.g. 4 for 4x compression
	NumSlotsPerSub  uint64 // TotalVocabulary / CompressionRate
	SubtableDim     int    // e.g. 80 (for 160 total dims split into k=2)
}

// NewHashKRouter creates a router configured for R=4 compression.
func NewHashKRouter(totalVocab, compressionRate uint64, fullDim int) *HashKRouter {
	if compressionRate == 0 {
		compressionRate = 4
	}
	slots := (totalVocab + compressionRate - 1) / compressionRate
	subDim := fullDim / 2
	if subDim <= 0 {
		subDim = 80
	}
	return &HashKRouter{
		TotalVocabulary: totalVocab,
		CompressionRate: compressionRate,
		NumSlotsPerSub:  slots,
		SubtableDim:     subDim,
	}
}

// RouteToken computes the subtable 0 and subtable 1 slot addresses for a given token and head.
func (h *HashKRouter) RouteToken(tokenIdx uint64, headIdx uint64) (slot0, slot1 uint64) {
	slot0 = ComputeHashKSlot(tokenIdx, 0, headIdx, h.NumSlotsPerSub)
	slot1 = ComputeHashKSlot(tokenIdx, 1, headIdx, h.NumSlotsPerSub)
	return slot0, slot1
}
