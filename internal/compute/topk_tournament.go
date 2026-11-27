// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0
// Adapted at revision 48903c68b22d8ef52d6294b5af8f2855928ec355; see THIRD_PARTY_NOTICES.md ("FlashInfer").

package compute

import (
	"errors"
	"fmt"
	"math"
)

const (
	// TopkTournamentGroupLanes is the sub-warp lane width per tournament group.
	TopkTournamentGroupLanes = 16
	// TopkTournamentLaneStride is the number of candidates staged per lane (4-bit local index 0..15).
	TopkTournamentLaneStride = 16
	// TopkTournamentGroupCapacity is the single-pass candidate count per group (16 * 16 = 256).
	TopkTournamentGroupCapacity = TopkTournamentGroupLanes * TopkTournamentLaneStride
	// TopkTournamentScratchStride is the padded scratch row width per lane (+1 bank-conflict pad).
	TopkTournamentScratchStride = TopkTournamentLaneStride + 1
	// TopkTournamentMaxRefinementRounds bounds the warp-vote-gated refinement rounds (3 odd-even + 1 full sort).
	TopkTournamentMaxRefinementRounds = 4
	// TopkTournamentMaxK is the maximum k supported by one 16-wide tournament selection block.
	TopkTournamentMaxK = 16
)

var (
	// ErrTopkTournamentEmpty reports an empty candidate score slice.
	ErrTopkTournamentEmpty = errors.New("compute: topk tournament scores must not be empty")
	// ErrTopkTournamentInvalidK reports k <= 0, k > len(scores), or k > TopkTournamentMaxK.
	ErrTopkTournamentInvalidK = errors.New("compute: invalid k for topk tournament")
)

// TopkTournamentEntry pairs a selected score with its original candidate index.
type TopkTournamentEntry struct {
	Score float32 `json:"score"`
	Index int32   `json:"index"`
}

// TopkTournamentReceipt records deterministic accounting for a single-pass
// tournament top-k run (dense fast path vs multi-group merge, flagged low-bit
// collisions, and warp-vote early-stop refinement rounds).
type TopkTournamentReceipt struct {
	N                   int  `json:"n"`
	K                   int  `json:"k"`
	Groups              int  `json:"groups"`
	DenseFastPath       bool `json:"dense_fast_path"`
	FlaggedLanes        int  `json:"flagged_lanes"`
	RefinementRounds    int  `json:"refinement_rounds"`
	MaxRefinementRounds int  `json:"max_refinement_rounds"`
	WarpVotes           int  `json:"warp_votes"`
}

const (
	topkPackMaxFinite   float32 = 8.507059e+37  // 2^126 (0x7E800000)
	topkPackMinFinite   float32 = -8.507059e+37 // -2^126 (0xFE800000)
	topkPackNegInfKey   uint32  = 0xFF000000    // valid -Inf: strictly below finite keys, ordered by small idx
	topkPackSentinelKey uint32  = 0xFF000200    // NaN / OOB: strictly below valid -Inf
)

// topkPackKey packs a float32 score (top 28 bits) with its 4-bit local lane
// index [0..15] into one uint32 key whose float32 representation preserves
// coarse descending order. Valid -Inf and NaN/OOB entries use distinct +17
// spaced sentinel bands so unused or invalid lanes never trigger false collisions.
func topkPackKey(score float32, localIdx uint32, valid bool) uint32 {
	idx := localIdx & 0xF
	if !valid || math.IsNaN(float64(score)) {
		// 17*idx = (idx<<4)|idx separates every sentinel key by >= 16 while preserving low-4-bit localIdx.
		return topkPackSentinelKey + 17*idx
	}
	if math.IsInf(float64(score), -1) {
		return topkPackNegInfKey + 17*idx
	}
	sc := score
	if sc < topkPackMinFinite {
		sc = topkPackMinFinite
	} else if sc > topkPackMaxFinite {
		sc = topkPackMaxFinite
	}
	bits := math.Float32bits(sc)
	if bits == 0x80000000 {
		bits = 0
	}
	return (bits & 0xFFFFFFF0) | idx
}

func topkKeyMayCollide(a, b uint32) bool {
	if a >= topkPackNegInfKey && b >= topkPackNegInfKey {
		return false
	}
	if (a ^ b) < 16 {
		return true
	}
	// Subnormal zero/near-zero boundary: +0.0 vs -tiny can share float equality.
	return ((a | b) & 0x7FFFFFF0) == 0
}

// topkExactOutranks defines the total order over staged candidates:
// valid > invalid/NaN, finite/Inf descending by score, ties broken by smaller original index.
func topkExactOutranks(aScore float32, aIdx int32, aValid bool, bScore float32, bIdx int32, bValid bool) bool {
	aNaN := !aValid || math.IsNaN(float64(aScore))
	bNaN := !bValid || math.IsNaN(float64(bScore))
	if aNaN != bNaN {
		return !aNaN
	}
	if aNaN && bNaN {
		return aIdx < bIdx
	}
	if aScore != bScore {
		return aScore > bScore
	}
	return aIdx < bIdx
}

// 16-element Green sorting network pairs (60 comparators).
var topk16NetworkPairs = [][2]int{
	{0, 13}, {1, 12}, {2, 15}, {3, 14}, {4, 8}, {5, 6}, {7, 11}, {9, 10},
	{0, 5}, {1, 7}, {2, 9}, {3, 4}, {6, 13}, {8, 14}, {10, 15}, {11, 12},
	{0, 1}, {2, 3}, {4, 5}, {6, 8}, {7, 9}, {10, 11}, {12, 13}, {14, 15},
	{0, 2}, {1, 3}, {4, 10}, {5, 11}, {6, 7}, {8, 9}, {12, 14}, {13, 15},
	{1, 2}, {3, 12}, {4, 6}, {5, 7}, {8, 10}, {9, 11}, {13, 14},
	{1, 4}, {2, 6}, {5, 8}, {7, 10}, {9, 13}, {11, 14},
	{2, 4}, {3, 6}, {9, 12}, {11, 13},
	{3, 5}, {6, 8}, {7, 9}, {10, 12},
	{3, 4}, {5, 6}, {7, 8}, {9, 10}, {11, 12},
	{6, 7}, {8, 9},
}

func sortPackedKeys16(keys *[TopkTournamentLaneStride]uint32) {
	for _, p := range topk16NetworkPairs {
		i, j := p[0], p[1]
		fi := math.Float32frombits(keys[i])
		fj := math.Float32frombits(keys[j])
		if fi < fj {
			keys[i], keys[j] = keys[j], keys[i]
		}
	}
}

func oddEvenRefinePass16(
	vals *[TopkTournamentLaneStride]float32,
	idxs *[TopkTournamentLaneStride]int32,
	valids *[TopkTournamentLaneStride]bool,
	localOrder *[TopkTournamentLaneStride]uint8,
) bool {
	swapped := false
	swapIfNeeded := func(i, j int) {
		if topkExactOutranks(vals[j], idxs[j], valids[j], vals[i], idxs[i], valids[i]) {
			vals[i], vals[j] = vals[j], vals[i]
			idxs[i], idxs[j] = idxs[j], idxs[i]
			valids[i], valids[j] = valids[j], valids[i]
			localOrder[i], localOrder[j] = localOrder[j], localOrder[i]
			swapped = true
		}
	}
	for i := 0; i+1 < TopkTournamentLaneStride; i += 2 {
		swapIfNeeded(i, i+1)
	}
	for i := 1; i+1 < TopkTournamentLaneStride; i += 2 {
		swapIfNeeded(i, i+1)
	}
	return swapped
}

func fullRefineNetwork16(
	vals *[TopkTournamentLaneStride]float32,
	idxs *[TopkTournamentLaneStride]int32,
	valids *[TopkTournamentLaneStride]bool,
	localOrder *[TopkTournamentLaneStride]uint8,
) {
	for _, p := range topk16NetworkPairs {
		i, j := p[0], p[1]
		if topkExactOutranks(vals[j], idxs[j], valids[j], vals[i], idxs[i], valids[i]) {
			vals[i], vals[j] = vals[j], vals[i]
			idxs[i], idxs[j] = idxs[j], idxs[i]
			valids[i], valids[j] = valids[j], valids[i]
			localOrder[i], localOrder[j] = localOrder[j], localOrder[i]
		}
	}
}

type topkCandidateSlot struct {
	score float32
	index int32
	valid bool
}

// mergeSortedTop16 merges two descending top-16 candidate lists into the exact
// top-16 descending result under topkExactOutranks, mirroring the pairwise
// warp-shuffle bitonic merge across 16-lane sub-warps.
func mergeSortedTop16(a, b [TopkTournamentMaxK]topkCandidateSlot) [TopkTournamentMaxK]topkCandidateSlot {
	var out [TopkTournamentMaxK]topkCandidateSlot
	ia, ib := 0, 0
	for k := 0; k < TopkTournamentMaxK; k++ {
		if topkExactOutranks(a[ia].score, a[ia].index, a[ia].valid, b[ib].score, b[ib].index, b[ib].valid) {
			out[k] = a[ia]
			ia++
		} else {
			out[k] = b[ib]
			ib++
		}
	}
	return out
}

func packNibbleWords(order [TopkTournamentLaneStride]uint8) [2]uint32 {
	var w0, w1 uint32
	for i := 0; i < 8; i++ {
		w0 |= uint32(order[i]&0xF) << uint(i*4)
		w1 |= uint32(order[8+i]&0xF) << uint(i*4)
	}
	return [2]uint32{w0, w1}
}

func unpackNibbleAt(words [2]uint32, pos int) uint8 {
	return uint8((words[pos>>3] >> uint((pos&7)*4)) & 0xF)
}

// SelectTopkTournament selects the exact top-k entries from scores using a
// single-pass bit-packed tournament select with shared-row staging, dense fast
// path when candidates fit one pass (<=256), and warp-vote early-stopped
// refinement when low-bit key collisions occur.
func SelectTopkTournament(scores []float32, k int) ([]TopkTournamentEntry, TopkTournamentReceipt, error) {
	n := len(scores)
	if n == 0 {
		return nil, TopkTournamentReceipt{}, ErrTopkTournamentEmpty
	}
	if k <= 0 || k > n || k > TopkTournamentMaxK {
		return nil, TopkTournamentReceipt{}, fmt.Errorf("%w: k=%d n=%d max=%d",
			ErrTopkTournamentInvalidK, k, n, TopkTournamentMaxK)
	}

	groups := (n + TopkTournamentGroupCapacity - 1) / TopkTournamentGroupCapacity
	receipt := TopkTournamentReceipt{
		N:                   n,
		K:                   k,
		Groups:              groups,
		DenseFastPath:       groups == 1,
		MaxRefinementRounds: TopkTournamentMaxRefinementRounds,
	}

	var globalTop [TopkTournamentMaxK]topkCandidateSlot
	for i := range globalTop {
		globalTop[i] = topkCandidateSlot{score: float32(math.Inf(-1)), index: math.MaxInt32, valid: false}
	}

	for g := 0; g < groups; g++ {
		// Shared scratch staging with +1 stride per lane plus packed nibble words.
		var scr [TopkTournamentGroupLanes * TopkTournamentScratchStride]float32
		var scrValid [TopkTournamentGroupLanes * TopkTournamentScratchStride]bool
		var scrIdx [TopkTournamentGroupLanes * TopkTournamentScratchStride]int32
		var nib [TopkTournamentGroupLanes][2]uint32

		var laneKeys [TopkTournamentGroupLanes][TopkTournamentLaneStride]uint32
		var laneOrder [TopkTournamentGroupLanes][TopkTournamentLaneStride]uint8
		anyFlagged := false

		for lane := 0; lane < TopkTournamentGroupLanes; lane++ {
			base := g*TopkTournamentGroupCapacity + lane*TopkTournamentLaneStride
			rowBase := lane * TopkTournamentScratchStride
			for pos := 0; pos < TopkTournamentLaneStride; pos++ {
				globalIdx := base + pos
				if globalIdx < n {
					v := scores[globalIdx]
					scr[rowBase+pos] = v
					scrIdx[rowBase+pos] = int32(globalIdx)
					scrValid[rowBase+pos] = !math.IsNaN(float64(v))
					laneKeys[lane][pos] = topkPackKey(v, uint32(pos), true)
				} else {
					scr[rowBase+pos] = float32(math.Inf(-1))
					scrIdx[rowBase+pos] = int32(globalIdx)
					scrValid[rowBase+pos] = false
					laneKeys[lane][pos] = topkPackKey(0, uint32(pos), false)
				}
			}

			sortPackedKeys16(&laneKeys[lane])
			laneFlagged := false
			for pos := 0; pos < TopkTournamentLaneStride; pos++ {
				laneOrder[lane][pos] = uint8(laneKeys[lane][pos] & 0xF)
				if pos > 0 && topkKeyMayCollide(laneKeys[lane][pos-1], laneKeys[lane][pos]) {
					laneFlagged = true
				}
			}
			if laneFlagged {
				receipt.FlaggedLanes++
				anyFlagged = true
			}
		}

		// Initial warp vote: skip scratch reloads and refinement completely when no
		// lane hit a coarse-key collision.
		receipt.WarpVotes++
		if anyFlagged {
			var vals [TopkTournamentGroupLanes][TopkTournamentLaneStride]float32
			var idxs [TopkTournamentGroupLanes][TopkTournamentLaneStride]int32
			var valids [TopkTournamentGroupLanes][TopkTournamentLaneStride]bool

			for lane := 0; lane < TopkTournamentGroupLanes; lane++ {
				rowBase := lane * TopkTournamentScratchStride
				for pos := 0; pos < TopkTournamentLaneStride; pos++ {
					lid := int(laneOrder[lane][pos])
					vals[lane][pos] = scr[rowBase+lid]
					idxs[lane][pos] = scrIdx[rowBase+lid]
					valids[lane][pos] = scrValid[rowBase+lid]
				}
			}

			more := true
			for round := 0; round < TopkTournamentMaxRefinementRounds-1 && more; round++ {
				receipt.RefinementRounds++
				swappedAny := false
				for lane := 0; lane < TopkTournamentGroupLanes; lane++ {
					if oddEvenRefinePass16(&vals[lane], &idxs[lane], &valids[lane], &laneOrder[lane]) {
						swappedAny = true
					}
				}
				receipt.WarpVotes++
				more = swappedAny
			}
			if more {
				receipt.RefinementRounds++
				for lane := 0; lane < TopkTournamentGroupLanes; lane++ {
					fullRefineNetwork16(&vals[lane], &idxs[lane], &valids[lane], &laneOrder[lane])
				}
			}
		}

		// Pack per-lane sorted local indices as 4-bit nibbles, then run the
		// tree-structured 16-lane tournament merge down to the group's top-16.
		var laneLists [TopkTournamentGroupLanes][TopkTournamentMaxK]topkCandidateSlot
		for lane := 0; lane < TopkTournamentGroupLanes; lane++ {
			nib[lane] = packNibbleWords(laneOrder[lane])
			rowBase := lane * TopkTournamentScratchStride
			for pos := 0; pos < TopkTournamentMaxK; pos++ {
				lid := int(unpackNibbleAt(nib[lane], pos))
				laneLists[lane][pos] = topkCandidateSlot{
					score: scr[rowBase+lid],
					index: scrIdx[rowBase+lid],
					valid: scrValid[rowBase+lid],
				}
			}
		}

		for stride := 8; stride >= 1; stride >>= 1 {
			for lane := 0; lane < stride; lane++ {
				laneLists[lane] = mergeSortedTop16(laneLists[lane], laneLists[lane+stride])
			}
		}

		if g == 0 {
			globalTop = laneLists[0]
		} else {
			globalTop = mergeSortedTop16(globalTop, laneLists[0])
		}
	}

	out := make([]TopkTournamentEntry, k)
	for i := 0; i < k; i++ {
		out[i] = TopkTournamentEntry{
			Score: globalTop[i].score,
			Index: globalTop[i].index,
		}
	}
	return out, receipt, nil
}
