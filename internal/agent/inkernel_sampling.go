package agent

import (
	"math"
	"math/rand"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// sampleLogits mirrors cmd/fakchat.sample: argmax when temp<=0, else a
// temperature-scaled softmax draw. topK then topP truncate the stochastic path, in
// that order (the standard top-k → top-p pipeline): top-k keeps only the k
// highest-probability tokens, then nucleus (top-p) keeps the smallest set whose
// cumulative mass reaches topP. The tail each step excludes is zeroed before the
// draw. A topK<=0 or topK>=len(logits) disables top-k; a topP<=0 or topP>=1 disables
// nucleus — with both off the draw is the full softmax, byte-for-byte the pre-seam
// behavior. The single most-probable token is always kept so neither cutoff can
// empty the candidate set. Both shape only the stochastic path: temp<=0 stays pure
// argmax (top-k/top-p never change the argmax winner).
func sampleLogits(logits []float32, temp, topP float64, topK int, rng *rand.Rand) int {
	if temp <= 0 {
		best, bi := float32(-math.MaxFloat32), 0
		for i, x := range logits {
			if x > best {
				best, bi = x, i
			}
		}
		return bi
	}
	maxL := float32(-math.MaxFloat32)
	for _, x := range logits {
		if x > maxL {
			maxL = x
		}
	}
	var sum float64
	probs := make([]float64, len(logits))
	for i, x := range logits {
		p := math.Exp(float64(x-maxL) / temp)
		probs[i] = p
		sum += p
	}
	if topK > 0 && topK < len(probs) {
		sum = topKTruncate(probs, sum, topK)
	}
	if topP > 0 && topP < 1 {
		sum = nucleusTruncate(probs, sum, topP)
	}
	r := rng.Float64() * sum
	for i, p := range probs {
		r -= p
		if r <= 0 {
			return i
		}
	}
	// Fall back to the last token with nonzero mass (nucleus zeroed the tail).
	for i := len(probs) - 1; i >= 0; i-- {
		if probs[i] > 0 {
			return i
		}
	}
	return len(logits) - 1
}

// sampleLogitsWithBias applies the OpenAI logit_bias map before the existing in-kernel
// sampler. The nil/empty map is a strict no-op, preserving the historical argmax /
// stochastic path byte-for-byte. Biases are clamped to the same [-100, 100] bound the
// native model constraint sink uses.
func sampleLogitsWithBias(logits []float32, temp, topP float64, topK int, bias model.LogitBias, rng *rand.Rand) int {
	if len(bias) == 0 {
		return sampleLogits(logits, temp, topP, topK, rng)
	}
	eff := append([]float32(nil), logits...)
	for tok, b := range bias {
		if tok < 0 || tok >= len(eff) {
			continue
		}
		if b > model.LogitBiasClamp {
			b = model.LogitBiasClamp
		} else if b < -model.LogitBiasClamp {
			b = -model.LogitBiasClamp
		}
		eff[tok] += float32(b)
	}
	return sampleLogits(eff, temp, topP, topK, rng)
}

// nucleusTruncate zeroes every probability outside the top-p nucleus in place and
// returns the surviving mass (the new normalization sum). The nucleus is the
// smallest set of highest-probability tokens whose cumulative mass reaches topP;
// the single most-probable token is always kept so the nucleus is never empty.
// probs is unsorted on entry and stays index-aligned to the caller's logits.
func nucleusTruncate(probs []float64, sum, topP float64) float64 {
	order := descProbOrder(probs, func(i, j int) bool { return probs[i] > probs[j] })
	target := topP * sum
	var cum float64
	kept := make(map[int]bool, len(order))
	for rank, idx := range order {
		// Stop BEFORE adding this token once the nucleus already reached the target —
		// the kept set is the minimal prefix whose mass >= target. Rank 0 is always
		// kept (the head token) so the nucleus is never empty.
		if rank > 0 && cum >= target {
			break
		}
		kept[idx] = true
		cum += probs[idx]
	}
	return maskKept(probs, kept)
}

// descProbOrder returns the indices of probs ordered by the caller's less comparator, which
// ranks two ELEMENT indices (not positions in the returned slice). It is the shared
// highest-probability-first index permutation nucleusTruncate and topKTruncate sort on; each
// passes its own tie-break (nucleus leaves equal masses in arbitrary order; topK breaks ties by
// the lower index for a stable kept set).
func descProbOrder(probs []float64, less func(i, j int) bool) []int {
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	sort.Slice(order, func(a, b int) bool { return less(order[a], order[b]) })
	return order
}

// maskKept zeroes every probability whose index is not in kept (in place) and returns the
// surviving mass (the new normalization sum) — the shared renormalization tail of
// nucleusTruncate and topKTruncate.
func maskKept(probs []float64, kept map[int]bool) float64 {
	var newSum float64
	for i := range probs {
		if kept[i] {
			newSum += probs[i]
		} else {
			probs[i] = 0
		}
	}
	return newSum
}

// topKTruncate zeroes every probability outside the top-k highest-probability
// tokens in place and returns the surviving mass (the new normalization sum). Ties
// at the k-th rank are broken by index order (the sort is stable on equal probs via
// the index comparator), so the kept set is deterministic. probs is unsorted on
// entry and stays index-aligned to the caller's logits. The caller guarantees
// 0 < k < len(probs); k>=len(probs) is a no-op handled before the call so the full
// distribution stays byte-for-byte the pre-seam draw.
func topKTruncate(probs []float64, sum float64, k int) float64 {
	// Highest probability first; ties resolve to the lower index so the kept set is
	// stable and reproducible across runs.
	order := descProbOrder(probs, func(i, j int) bool {
		if probs[i] != probs[j] {
			return probs[i] > probs[j]
		}
		return i < j
	})
	kept := make(map[int]bool, k)
	for rank := 0; rank < k; rank++ {
		kept[order[rank]] = true
	}
	return maskKept(probs, kept)
}
