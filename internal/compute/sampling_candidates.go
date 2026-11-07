package compute

import "math"

// SamplingCandidateMaxK bounds host transport, not the vocabulary on the device.
const SamplingCandidateMaxK = 64

// SamplingCandidatePacket holds raw globally selected logits. Its producer must
// inspect the entire row, preserve raw F32 payloads and return the global k+1
// maxima with lower original IDs winning numeric ties. No probability or RNG
// calculation belongs in the producer. The packet is caller-owned scratch.
type SamplingCandidatePacket struct {
	Vocab int
	Count int
	Valid bool
	IDs   [SamplingCandidateMaxK + 1]int
	Raw   [SamplingCandidateMaxK + 1]float32
	// Native words contain row flags, emitted count, then (ID, raw bits) pairs.
	// Keeping this pointer-free storage with the lane avoids per-token CGo buffers.
	nativeWords [2 + 2*(SamplingCandidateMaxK+1)]uint32
}

// SamplingCandidateBackend is optional. A false available result is legal only
// before candidate work is submitted and returns no retained owner. A successful
// transaction completes every candidate operation and optional retained copy
// before publishing either result. Failure is never a numeric fallback: all
// possibly live storage must remain protected by the backend's terminal rules.
type SamplingCandidateBackend interface {
	SamplingCandidates(row Tensor, k int, retain bool, out *SamplingCandidatePacket) (kept Tensor, available bool, err error)
}

// Certify converts a complete raw packet to the host sampler's compact row.
// This check cannot establish global completeness: that is the producer's
// obligation. A refusal leaves the caller responsible for a full read of the
// original row before recycling it. It never consumes randomness.
func (p *SamplingCandidatePacket) Certify(temp float64, k int) bool {
	if p == nil || !p.Valid || !(temp > 0) || math.IsInf(temp, 0) ||
		k < 1 || k > SamplingCandidateMaxK || p.Vocab <= k || p.Count != k+1 {
		return false
	}
	for i := 0; i < p.Count; i++ {
		if p.IDs[i] < 0 || p.IDs[i] >= p.Vocab || math.IsNaN(float64(p.Raw[i])) || math.IsInf(float64(p.Raw[i]), 1) {
			return false
		}
		for j := 0; j < i; j++ {
			if p.IDs[i] == p.IDs[j] {
				return false
			}
		}
	}
	// At most 65 elements; insertion sort retains raw float32 payloads and has
	// no closure or slice allocation. Device emission order is immaterial.
	for i := 1; i < p.Count; i++ {
		id, raw := p.IDs[i], p.Raw[i]
		j := i
		for j > 0 && (raw > p.Raw[j-1] || (raw == p.Raw[j-1] && id < p.IDs[j-1])) {
			p.IDs[j], p.Raw[j] = p.IDs[j-1], p.Raw[j-1]
			j--
		}
		p.IDs[j], p.Raw[j] = id, raw
	}
	maxL := p.Raw[0]
	if math.IsInf(float64(maxL), -1) {
		return false
	}
	// Match the served sampler's float32 subtraction before float64 division.
	weight := func(i int) float64 { return math.Exp(float64(p.Raw[i]-maxL) / temp) }
	if !(weight(k-1) > weight(k)) {
		return false
	}
	for i := 1; i < k; i++ {
		id, raw := p.IDs[i], p.Raw[i]
		j := i
		for j > 0 && id < p.IDs[j-1] {
			p.IDs[j], p.Raw[j] = p.IDs[j-1], p.Raw[j-1]
			j--
		}
		p.IDs[j], p.Raw[j] = id, raw
	}
	p.Count = k
	return true
}
