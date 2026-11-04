// sampling_topk.go — the device sampling-truncation family for wide-vocabulary
// rows: the host-runnable exact twin of the CUDA kernels in sampling_topk.cu,
// serving the samplingTruncationCluster path declared at
// internal/agent/inkernel_sampling.go (currently unwired; the successor
// packet wires the .cu into the backend and this face behind
// selectSamplingTruncationPath).
//
// The host applyCPUSamplingTruncation chain (heap/sort based) remains the
// correctness oracle and fallback; this face reproduces its closed contract —
// same kept sets, same summation orders, same sentinels — through the borrowed
// threshold/radix machinery instead, so the two paths are independent
// implementations of one contract.
//
// ADAPTED from vLLM (Apache-2.0; vllm-project/vllm at
// ae065918640a198a56f6f17eb45e251c3229fe41):
//   csrc/persistent_topk.cuh:868  persistent_topk_kernel — coarse-histogram +
//       byte-refinement threshold selection over monotone keys.
//   csrc/persistent_topk.cuh:1295 FilteredTopKUnifiedKernel — single-CTA-per-row
//       filtered top-k with a bounded shared stash.
//   csrc/sampled_topk.cuh:144     sampled_topk_kernel — sample-estimated cutoff,
//       survivor compaction, exact full-row fallback.
//   csrc/sampler.cu:16            apply_repetition_penalties_kernel — tiled
//       row-parallel pre-truncation penalty pass.
// No upstream source is copied. Deliberate deviations from upstream:
//   - ties at the selection cutoff resolve by ASCENDING token index (upstream
//     leaves them to atomic order); the host probSorter tieByIndex contract
//     requires the deterministic order.
//   - the repetition penalty is the additive OpenAI frequency/presence form the
//     host chain applies, not upstream's multiplicative GPT-2 form.

package compute

import (
	"errors"
	"math"
)

// ErrNonFiniteLogits refuses a row that cannot yield a valid sample: any
// weight NaN or ±Inf (the device-family twin of agent.ErrNonFiniteLogits), or
// a post-truncation mass that is zero or infinite. It is the single closed
// refusal of the family; callers fall back to the host CPU chain on it. A
// refused row is left unmodified.
var ErrNonFiniteLogits = errors.New("compute: sampling top-k: logits contain NaN or +Inf, or no finite entry")

// ApplyRepetitionPenalties applies the OpenAI-style frequency/presence
// repetition penalties IN PLACE before truncation:
//
//	logit[t] -= frequencyPenalty*count[t] + presencePenalty   (count[t] > 0)
//
// counts is indexed by token id; a nil/short slice treats every uncovered
// token as count 0, and entries with count <= 0 are skipped. Both penalties
// zero makes this a strict no-op. This is the device twin of the host
// penalty pass in agent.sampleLogitsWithPenaltyCoreWS (the borrowed shape is
// vLLM's tiled row-parallel apply_repetition_penalties_kernel; the arithmetic
// follows fak's additive host contract, not upstream's multiplicative form).
func ApplyRepetitionPenalties(logits []float32, frequencyPenalty, presencePenalty float64, counts []int32) {
	if frequencyPenalty == 0 && presencePenalty == 0 {
		return
	}
	for tok, c := range counts {
		if tok >= len(logits) || c <= 0 {
			continue
		}
		penalty := frequencyPenalty*float64(c) + presencePenalty
		logits[tok] -= float32(penalty)
	}
}

// DeviceSamplingTruncation runs the device-family truncation contract over one
// temperature-scaled probability row: top-k first, then top-p (nucleus), in
// that order, zeroing excluded weights IN PLACE and returning the surviving
// mass. It is the exact float64 twin of the host applyCPUSamplingTruncation
// chain — same kept sets, same summation orders, same sentinels — computed by
// the borrowed threshold/radix machinery instead of the host's heap/sort.
//
// Contract:
//   - probs holds non-negative temperature-scaled weights; sum is their total
//     mass before truncation (trusted, exactly as the host trusts it).
//   - topK > 0 && topK < len(probs) keeps exactly the topK highest weights
//     under the (value desc, index asc) total order; otherwise top-k is inert.
//   - topP in (0,1) keeps the smallest prefix of that total order whose mass
//     reaches topP times the post-top-k mass; rank 0 is always kept.
//   - A row holding any NaN or ±Inf weight, or whose post-truncation mass is
//     zero or infinite, is refused with ErrNonFiniteLogits and left unmodified.
func DeviceSamplingTruncation(probs []float64, sum, topP float64, topK int) (float64, error) {
	n := len(probs)
	if n == 0 {
		return 0, ErrNonFiniteLogits
	}
	keys := make([]uint64, n)
	for i, p := range probs {
		if math.IsNaN(p) || math.IsInf(p, 0) {
			return 0, ErrNonFiniteLogits
		}
		keys[i] = samplingWeightKey(p)
	}
	mass := sum
	if topK > 0 && topK < n {
		mass = truncateTopKInPlace(probs, keys, topK)
	}
	if topP > 0 && topP < 1 {
		mass = nucleusTruncateInPlace(probs, keys, mass, topP)
	}
	if !(mass > 0) || math.IsInf(mass, 0) {
		return 0, ErrNonFiniteLogits
	}
	return mass, nil
}

// DrawToken inverts one truncated probability row with a coin in [0,1) using
// the strict CDF boundary the host sampler uses: a zero-mass token is never
// selected (even at coin zero), a coin exactly on a CDF boundary belongs to
// the next positive interval, and an exhausted row falls back to the last
// nonzero token. The input row is left unchanged.
func DrawToken(probs []float64, mass, coin float64) int {
	target := coin * mass
	var cum float64
	for i, p := range probs {
		if p <= 0 {
			continue
		}
		cum += p
		if cum > target {
			return i
		}
	}
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return len(probs) - 1
}

// samplingWeightKey maps a non-negative float64 weight to a uint64 whose
// unsigned order equals the weight's numeric order (the standard sign-flip
// radix transform). NaN and infinities are out of contract: the face rejects
// non-finite rows before keying.
func samplingWeightKey(w float64) uint64 {
	b := math.Float64bits(w)
	if b&(1<<63) != 0 {
		return ^b
	}
	return b | (1 << 63)
}

// samplingZeroKey is the key of a zeroed (excluded) weight.
const samplingZeroKey = uint64(1) << 63

// truncateTopKInPlace keeps the k highest weights under the (value desc,
// index asc) total order, zeroes the rest in place, and returns the surviving
// mass summed in ascending index order — the bit-exact twin of the host
// topKTruncate kept-set and renormalization order. The selection is the
// borrowed threshold machinery: descend the 64-bit monotone key MSB-first,
// eight bits at a time, until the exact cutoff key is isolated; items strictly
// above it are kept, and the remaining slots fill the equal-key group in
// ascending index order (the deterministic tie-break).
func truncateTopKInPlace(probs []float64, keys []uint64, k int) float64 {
	cutoff := samplingCutoffKey(keys, k)
	above := 0
	for _, key := range keys {
		if key > cutoff {
			above++
		}
	}
	need := k - above
	var mass float64
	keptTies := 0
	for i := range probs {
		keep := false
		switch {
		case keys[i] > cutoff:
			keep = true
		case keys[i] == cutoff && keptTies < need:
			keep = true
			keptTies++
		}
		if keep {
			mass += probs[i] // ascending index order: the host renormalization order
		} else {
			probs[i] = 0
			keys[i] = samplingZeroKey
		}
	}
	return mass
}

// samplingCutoffKey isolates the key of the k-th item under key-descending
// order: the unique key c with #{key > c} < k <= #{key >= c}. It descends the
// 64-bit key MSB-first, eight bits at a time (the borrowed coarse-histogram +
// byte-refinement shape), histogramming the next byte over the items matching
// the fixed prefix each round. offset carries the count of items strictly
// above the current prefix level so each round's boundary test stays global.
func samplingCutoffKey(keys []uint64, k int) uint64 {
	prefix := uint64(0)
	offset := 0
	var hist [256]int
	for shift := 56; shift >= 0; shift -= 8 {
		for b := range hist {
			hist[b] = 0
		}
		// fixedMask covers exactly the bits fixed so far (the bytes above the
		// current one); the first round fixes nothing, so every item matches.
		fixedMask := ^uint64(0) << uint(shift+8)
		for _, key := range keys {
			if key&fixedMask == prefix&fixedMask {
				hist[int((key>>uint(shift))&0xFF)]++
			}
		}
		rem := k - offset
		suf := 0
		boundary := uint64(0)
		for b := 255; b >= 0; b-- {
			if suf < rem && suf+hist[b] >= rem {
				boundary = uint64(b)
				break
			}
			suf += hist[b]
		}
		offset += suf
		prefix |= boundary << uint(shift)
	}
	return prefix
}

// nucleusTruncateInPlace keeps the smallest prefix of the (value desc,
// index asc) total order whose mass reaches topP times the incoming mass —
// rank 0 always kept — zeroes the tail in place, and returns the surviving
// mass summed in ascending index order. The total order comes from a stable
// LSD radix sort over the monotone keys (the borrowed final-pass radix
// ordering), which makes the walk bit-identical to the host nucleus walk:
// same value sequence, same accumulation order, same stop rule.
func nucleusTruncateInPlace(probs []float64, keys []uint64, mass, topP float64) float64 {
	order := stableKeyDescOrder(keys)
	target := topP * mass
	var cum float64
	cutoff := len(order)
	for rank, idx := range order {
		if rank > 0 && cum >= target {
			cutoff = rank
			break
		}
		cum += probs[idx]
	}
	for _, idx := range order[cutoff:] {
		probs[idx] = 0
		keys[idx] = samplingZeroKey
	}
	var newMass float64
	for _, p := range probs {
		newMass += p
	}
	return newMass
}

// stableKeyDescOrder returns the index permutation sorting keys descending,
// ties by ascending index: a stable LSD radix sort (least byte first) over
// the inverted keys, seeded with natural index order so stability yields the
// index-asc tie-break. Two n-sized scratch buffers; eight counting passes.
func stableKeyDescOrder(keys []uint64) []int {
	n := len(keys)
	src := make([]int, n)
	dst := make([]int, n)
	for i := range src {
		src[i] = i
	}
	for bytePos := 0; bytePos < 8; bytePos++ {
		shift := uint(bytePos * 8)
		var cnt [256]int
		for _, idx := range src {
			cnt[int((^keys[idx]>>shift)&0xFF)]++
		}
		sum := 0
		for b := 0; b < 256; b++ {
			c := cnt[b]
			cnt[b] = sum
			sum += c
		}
		for _, idx := range src {
			b := int((^keys[idx] >> shift) & 0xFF)
			dst[cnt[b]] = idx
			cnt[b]++
		}
		src, dst = dst, src
	}
	return src
}
