package modelengine

// Adapted from modular/modular max/python/max/pipelines/kv_cache/paged_kv_cache/_mixed_step_split.py@88ab8913af6b9298fe914e0f20d87a49e4172d0d (Apache-2.0 WITH LLVM-exception; Copyright (c) 2026 Modular Inc.). Adapted to fak-owned Go.
//
// Some attention ops pick one route for the whole batch from its longest row,
// so in a step that batches decode rows with prefill rows every decode row
// takes the prefill route. The split record cuts each replica batch into a
// head and a tail view, so the decode rows take the decode routes and the
// prefill rows the prefill route, in one op call. This leaf ports the pure
// host-side record builder only; native scheduling today has only coarse
// step-level prefillInterval cadence, and wiring this record into per-op
// routing is a later leaf.

import (
	"errors"
	"fmt"
)

// MixedStepSplitDecodeCap is the widest query the split-route ops run on
// their decode routes. A wider bound would also take short prefill rows,
// which the persistent decode scratch, sized for maxBatch*(1+K), does not
// cover. A batch whose widest row stays at or under the cap never splits.
const MixedStepSplitDecodeCap = 8

// MixedStepSplit is one replica batch's host uint32[9] max-lengths array.
// Slots 0 and 1 hold the batch max query and cache lengths; slots 2-8 hold
// the split record. The head view is rows [0, r), the tail view the rest; an
// unsplit record (slot 2 == 0, all record slots zero) sends every row down
// the unsplit route.
//
//	Slot  Value
//	----  -------------------------------------------------
//	0     Batch max query length
//	1     Batch max cache length
//	2     Head rows r (0 means no split)
//	3     Head query tokens
//	4     Head max query length
//	5     Head max cache length
//	6     Tail max query length
//	7     Tail max cache length
//	8     Total context length of the prefill view
type MixedStepSplit [9]uint32

// Closed slot indices of MixedStepSplit.
const (
	MixedStepSlotMaxQueryLength  = 0
	MixedStepSlotMaxCacheLength  = 1
	MixedStepSlotHeadRows        = 2
	MixedStepSlotHeadQueryTokens = 3
	MixedStepSlotHeadMaxQuery    = 4
	MixedStepSlotHeadMaxCache    = 5
	MixedStepSlotTailMaxQuery    = 6
	MixedStepSlotTailMaxCache    = 7
	MixedStepSlotPrefillContext  = 8
)

// ErrMixedStepSplitMismatch reports widths and validLengths disagreeing in
// length. Callers may use errors.Is.
var ErrMixedStepSplitMismatch = errors.New("modelengine: mixed-step split: widths and valid lengths disagree")

// MixedStepDecodeWidth returns the widest row the split puts in the decode
// view. A decode row is 1+K tokens wide (K = 0 without spec); the result is
// capped at MixedStepSplitDecodeCap.
func MixedStepDecodeWidth(numDraftTokens int) int {
	return min(1+numDraftTokens, MixedStepSplitDecodeCap)
}

// IsUnsplit reports whether the record carries no split: every record slot
// is zero, so a caller keeps the unsplit route for the whole batch.
func (m MixedStepSplit) IsUnsplit() bool {
	return m[MixedStepSlotHeadRows] == 0
}

// HeadRows returns r, the head-view row count: the head view is rows
// [0, r), the tail view rows [r, n). It is 0 when the record is unsplit, in
// which case the whole batch rides the unsplit route.
func (m MixedStepSplit) HeadRows() int {
	return int(m[MixedStepSlotHeadRows])
}

// MixedStepSplitRecord builds one replica batch's host max-lengths array in
// the batch's row order. widths are each row's query tokens this step,
// drafts included; validLengths are each row's cache bound for this step;
// decodeWidth is the widest decode row, from MixedStepDecodeWidth. Both
// slices must be non-negative token counts of equal length.
//
// The decode view is the longer of the two end runs of rows at most
// decodeWidth wide (ties go to the head). A decode row past the first
// prefill row stays in the prefill view: correct, only slower. The record
// is unsplit when the batch has no prefill row, no decode row at either
// end, or no rows at all; ops fail open to the unsplit route for such
// records. Slot values wrap modulo 2^32, matching the host-array ABI.
func MixedStepSplitRecord(widths, validLengths []int, decodeWidth int) (MixedStepSplit, error) {
	var m MixedStepSplit
	if len(widths) != len(validLengths) {
		return m, fmt.Errorf("%w: %d widths but %d valid lengths", ErrMixedStepSplitMismatch, len(widths), len(validLengths))
	}
	n := len(widths)
	if n == 0 {
		return m, nil
	}
	maxQuery, maxCache := 0, 0
	for i, w := range widths {
		if w > maxQuery {
			maxQuery = w
		}
		if validLengths[i] > maxCache {
			maxCache = validLengths[i]
		}
	}
	m[MixedStepSlotMaxQueryLength] = uint32(maxQuery)
	m[MixedStepSlotMaxCacheLength] = uint32(maxCache)
	// No row wider than the decode cap means no prefill row at all.
	if maxQuery <= MixedStepSplitDecodeCap {
		return m, nil
	}
	leading := leadingRun(widths, decodeWidth)
	trailing := trailingRun(widths, decodeWidth)
	if leading == 0 && trailing == 0 {
		return m, nil
	}
	headRows := leading
	decodeLeads := leading >= trailing
	if !decodeLeads {
		headRows = n - trailing
	}
	headW, headV := widths[:headRows], validLengths[:headRows]
	tailW, tailV := widths[headRows:], validLengths[headRows:]
	prefillV := tailV
	if !decodeLeads {
		prefillV = headV
	}
	headTokens, headMaxQ, headMaxC := sumMax(headW, headV)
	_, tailMaxQ, tailMaxC := sumMax(tailW, tailV)
	m[MixedStepSlotHeadRows] = uint32(headRows)
	m[MixedStepSlotHeadQueryTokens] = uint32(headTokens)
	m[MixedStepSlotHeadMaxQuery] = uint32(headMaxQ)
	m[MixedStepSlotHeadMaxCache] = uint32(headMaxC)
	m[MixedStepSlotTailMaxQuery] = uint32(tailMaxQ)
	m[MixedStepSlotTailMaxCache] = uint32(tailMaxC)
	m[MixedStepSlotPrefillContext] = uint32(sumOf(prefillV))
	return m, nil
}

// leadingRun counts the run of rows at the front of widths that are at most
// limit wide.
func leadingRun(widths []int, limit int) int {
	count := 0
	for _, w := range widths {
		if w > limit {
			break
		}
		count++
	}
	return count
}

// trailingRun counts the run of rows at the back of widths that are at most
// limit wide.
func trailingRun(widths []int, limit int) int {
	count := 0
	for i := len(widths) - 1; i >= 0; i-- {
		if widths[i] > limit {
			break
		}
		count++
	}
	return count
}

// sumMax returns the sum of ws and the maxima of ws and vs.
func sumMax(ws, vs []int) (sum, maxW, maxV int) {
	for i, w := range ws {
		sum += w
		if w > maxW {
			maxW = w
		}
		if vs[i] > maxV {
			maxV = vs[i]
		}
	}
	return sum, maxW, maxV
}

// sumOf adds vs element-wise.
func sumOf(vs []int) (sum int) {
	for _, v := range vs {
		sum += v
	}
	return sum
}
