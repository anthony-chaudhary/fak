package kvbudget

// This file adds an OPT-IN entropy-gated chunk budgeting mode (#13062): a pure,
// GPU-free policy that reallocates a conserved retained-token budget across
// chunks instead of spending a fixed per-chunk share. It is an ADAPT of
// EntropyGatedChunkKVPress from the NVIDIA/kvpress study
// (pin 71640b4f9061054a7630c5049bb9ee659a01523c, Apache-2.0,
// `kvpress/presses/entropy_gated_chunkkv_press.py:91-141`); no upstream bytes
// are copied.
//
// # The waste this removes
//
// A chunked KV eviction that keeps a FIXED per-chunk share spends a whole
// chunk's worth of cache slots to preserve one useful token when a chunk's
// importance is concentrated in a single hot position. The share is allocated
// by count, not by where the information is.
//
// # What the mode does
//
// Each chunk is scored (its per-position importance) and summarized by two
// numbers: its MEAN score (how important the chunk is overall) and its
// NORMALIZED WITHIN-CHUNK ENTROPY (how evenly that importance is spread). A
// chunk that is important but SPIKEY — high mean, low entropy — is capped at a
// smaller LowEntropyChunkLength instead of kept whole; the slots it gives back
// are re-spent on MORE chunks in a greedy descending-mean pass, and a
// deterministic ascending-chunk-index top-up makes the retained token count
// EXACTLY (1 - CompressionRatio) * KVLen.
//
// # The default path is untouched
//
// The mode is a separate value type with an Enabled switch; nothing in the
// existing package consults it, so every existing caller's behavior is
// bit-for-bit unchanged. A disabled config returns the identity allocation
// (every chunk at its full length) rather than an error.

import "math"

// ChunkScores is the per-chunk importance input the mode scores: one inner
// slice of per-position scores per chunk, in KV order. Chunks MAY have unequal
// lengths (a partial trailing chunk), and an empty chunk is legal. Scores are
// produced elsewhere (a scorer, a press, or a test fixture) — this package does
// not compute attention.
type ChunkScores [][]float64

// KVLen is the total number of scored positions across all chunks.
func (c ChunkScores) KVLen() int {
	n := 0
	for _, ch := range c {
		n += len(ch)
	}
	return n
}

// ChunkStat is one chunk's two-number summary and the cap decision the greedy
// pass makes from it. Mean is the arithmetic mean of the chunk's scores (0 for
// an empty chunk); Entropy is the NORMALIZED within-chunk score entropy in
// [0,1] (0 = all importance on one position, 1 = perfectly even; 0 for a chunk
// of fewer than two positions, which has no distribution to spread). Capped is
// true when the chunk is both important (Mean >= ScoreThreshold) and spiky
// (Entropy <= EntropyThreshold), so its length is limited to the configured cap.
type ChunkStat struct {
	Mean    float64
	Entropy float64
	Capped  bool
	Length  int
}

// EntropyGatedConfig is the opt-in switch plus its gates and the conserved
// budget ratio. The zero value is DISABLED (Enabled == false) and leaves every
// existing caller untouched. When enabled, a negative CompressionRatio,
// ScoreThreshold, or EntropyThreshold is clamped to 0, and a
// LowEntropyChunkLength <= 0 disables the cap (no chunk is truncated, so the
// allocation is the identity) — each a conservative, fail-open-to-identity
// choice, never a silent quality regression.
type EntropyGatedConfig struct {
	// Enabled turns the entropy-gated reallocation on. False (the zero value)
	// means Allocate returns the identity allocation.
	Enabled bool
	// CompressionRatio is the fraction of KVLen to DROP: the retained target is
	// (1 - CompressionRatio) * KVLen. Clamped to [0, 1].
	CompressionRatio float64
	// ScoreThreshold is the mean-score floor a chunk must reach to be eligible
	// for capping: chunks below it are not "important" and keep their length.
	ScoreThreshold float64
	// EntropyThreshold is the normalized-entropy ceiling: a chunk at or below it
	// is "spiky" and eligible for capping. An entropy above it is evenly spread
	// and is not capped.
	EntropyThreshold float64
	// LowEntropyChunkLength is the per-chunk length a capped (important, spiky)
	// chunk is reduced to. A value <= 0 disables capping entirely.
	LowEntropyChunkLength int
}

// EntropyGatedAllocation is the result of one allocation: the retained length
// chosen for each chunk, the retained total (exactly the target when the mode
// ran and the target is reachable), and the per-chunk statistics the decision
// was made from.
type EntropyGatedAllocation struct {
	// Lengths[i] is the number of retained positions of chunk i, in [0, len].
	Lengths []int
	// Retained is the sum of Lengths — the conserved token budget actually spent.
	Retained int
	// Target is the requested retained total,
	// round((1 - CompressionRatio) * KVLen).
	Target int
	// Stats[i] is chunk i's mean/entropy/cap decision.
	Stats []ChunkStat
}

// normalizedEntropy is the within-chunk score entropy normalized to [0,1]:
// H(p) / ln(n) with p the chunk's scores normalized to sum 1. It returns 0 for
// a chunk of fewer than two positions or a non-positive total score (no
// distribution to speak of), which the gate reads as maximally spiky — the
// conservative direction (a chunk with nothing to spread is never treated as
// evenly-important).
func normalizedEntropy(scores []float64) float64 {
	n := len(scores)
	if n < 2 {
		return 0
	}
	total := 0.0
	for _, s := range scores {
		if s > 0 {
			total += s
		}
	}
	if total <= 0 {
		return 0
	}
	h := 0.0
	for _, s := range scores {
		if s <= 0 {
			continue
		}
		p := s / total
		h -= p * math.Log(p)
	}
	denom := math.Log(float64(n))
	if denom <= 0 {
		return 0
	}
	e := h / denom
	if e < 0 {
		return 0
	}
	if e > 1 {
		return 1
	}
	return e
}

// meanScore is the arithmetic mean of a chunk's scores (0 for an empty chunk).
func meanScore(scores []float64) float64 {
	if len(scores) == 0 {
		return 0
	}
	sum := 0.0
	for _, s := range scores {
		sum += s
	}
	return sum / float64(len(scores))
}

// Allocate computes the entropy-gated chunk allocation under cfg. With the mode
// disabled it returns the identity allocation (every chunk at its full length).
// When enabled it:
//
//  1. summarizes each chunk (mean score, normalized entropy);
//  2. flags a chunk Capped iff LowEntropyChunkLength > 0 AND mean >=
//     ScoreThreshold AND entropy <= EntropyThreshold;
//  3. spends the target budget greedily in descending mean-score order, giving
//     each chunk up to its cap (capped) or its full length;
//  4. tops up — in ascending chunk index order — any unspent budget on chunks
//     that still have room.
//
// The target is a conserved FLOOR: Retained == Target always, for any target in
// [0, KVLen]. The cap is a reallocation PREFERENCE, not a second ceiling — if
// capping every eligible chunk would leave the target unreachable (the caps sum
// below the target), the highest-mean capped chunks are relaxed back toward
// their full length until the target is exactly met. Conservation therefore
// holds unconditionally; the cap decides only WHERE the retained tokens sit.
//
// It is deterministic and allocates no hardware resource.
func (cfg EntropyGatedConfig) Allocate(c ChunkScores) EntropyGatedAllocation {
	n := len(c)
	alloc := EntropyGatedAllocation{
		Lengths: make([]int, n),
		Stats:   make([]ChunkStat, n),
	}
	if !cfg.Enabled {
		for i, ch := range c {
			alloc.Lengths[i] = len(ch)
			alloc.Retained += len(ch)
		}
		alloc.Target = alloc.Retained
		return alloc
	}

	ratio := cfg.CompressionRatio
	if ratio < 0 {
		ratio = 0
	}
	if ratio > 1 {
		ratio = 1
	}
	scoreThr := cfg.ScoreThreshold
	if scoreThr < 0 {
		scoreThr = 0
	}
	entThr := cfg.EntropyThreshold
	if entThr < 0 {
		entThr = 0
	}

	kvLen := c.KVLen()
	// Retain exactly round((1-ratio)*kvLen) tokens. Round-to-nearest keeps the
	// conserved count stable against float noise at the half boundary.
	alloc.Target = int(math.Round((1 - ratio) * float64(kvLen)))
	if alloc.Target > kvLen {
		alloc.Target = kvLen
	}
	if alloc.Target < 0 {
		alloc.Target = 0
	}

	// (1) summarize, (2) flag caps.
	for i, ch := range c {
		m := meanScore(ch)
		e := normalizedEntropy(ch)
		capped := cfg.LowEntropyChunkLength > 0 && m >= scoreThr && e <= entThr
		alloc.Stats[i] = ChunkStat{Mean: m, Entropy: e, Capped: capped, Length: len(ch)}
	}

	// The per-chunk spend ceiling starts at the cap for a capped chunk and the
	// full length otherwise. A relaxed map tracks chunks whose cap was lifted to
	// meet an unreachable target.
	ceiling := make([]int, n)
	for i := range c {
		if alloc.Stats[i].Capped && cfg.LowEntropyChunkLength < len(c[i]) {
			ceiling[i] = cfg.LowEntropyChunkLength
		} else {
			ceiling[i] = len(c[i])
		}
	}

	// (2b) The cap may not starve the conserved target: if the capped ceilings
	// cannot seat the target, relax the HIGHEST-MEAN capped chunks (the ones we
	// would most want to keep whole) back to their full length until the target
	// is reachable. Ties break by ascending index for determinism.
	totalCeiling := 0
	for i := range c {
		totalCeiling += ceiling[i]
	}
	if totalCeiling < alloc.Target {
		relax := make([]int, 0, n)
		for i := range c {
			if alloc.Stats[i].Capped && ceiling[i] < len(c[i]) {
				relax = append(relax, i)
			}
		}
		sortByMeanDescIndexAsc(relax, alloc.Stats)
		for _, i := range relax {
			if totalCeiling >= alloc.Target {
				break
			}
			totalCeiling += len(c[i]) - ceiling[i]
			ceiling[i] = len(c[i])
			alloc.Stats[i].Capped = false // cap lifted: report the chunk as uncapped
		}
	}

	// (3) greedy descending mean-score pass; ties broken by ascending index for
	// determinism.
	order := make([]int, n)
	for i := range order {
		order[i] = i
	}
	sortByMeanDescIndexAsc(order, alloc.Stats)

	remaining := alloc.Target
	for _, i := range order {
		if remaining <= 0 {
			break
		}
		give := ceiling[i]
		if give > remaining {
			give = remaining
		}
		if give < 0 {
			give = 0
		}
		alloc.Lengths[i] = give
		remaining -= give
	}

	// (4) deterministic ascending-index top-up so the budget is EXACTLY
	// conserved: distribute any remainder. Each chunk's ceiling is bounded by its
	// length, so no chunk can over-allocate.
	if remaining > 0 {
		for i := 0; i < n && remaining > 0; i++ {
			room := ceiling[i] - alloc.Lengths[i]
			if room <= 0 {
				continue
			}
			give := room
			if give > remaining {
				give = remaining
			}
			alloc.Lengths[i] += give
			remaining -= give
		}
	}

	for _, l := range alloc.Lengths {
		alloc.Retained += l
	}
	return alloc
}

// sortByMeanDescIndexAsc orders indices by descending mean score, then by
// ascending index — the deterministic greedy order (ties never depend on input
// order). Insertion sort keeps it dependency-free and is fine for the chunk
// counts a prefill produces.
func sortByMeanDescIndexAsc(order []int, stats []ChunkStat) {
	for i := 1; i < len(order); i++ {
		j := i
		for j > 0 {
			a, b := order[j-1], order[j]
			less := stats[b].Mean > stats[a].Mean ||
				(stats[b].Mean == stats[a].Mean && b < a)
			if !less {
				break
			}
			order[j-1], order[j] = order[j], order[j-1]
			j--
		}
	}
}
