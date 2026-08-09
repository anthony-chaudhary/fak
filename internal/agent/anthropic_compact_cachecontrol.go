package agent

// anthropic_compact_cachecontrol.go — the cache_control concern of the byte-level Anthropic
// compaction rewrite: WHERE the provider's prompt-cache breakpoints sit in a request body, and
// WHETHER bursting one of them ever repays itself.
//
// This is a seam because it is the only part of anthropic_compact.go that answers questions about
// the CACHE rather than about the splice. Nothing here reads an elementSpan, mints a stub, or
// touches the outbound bytes: the breakpoint scanners (firstBreakpointMessage /
// lastBreakpointMessage / rangeHasCacheControl / messageHasCacheControl / rawHasCacheControl) are
// pure JSON shape predicates over messages[] elements, and the CacheBurst* pricers are pure
// arithmetic over token counts and the provider's read/write multipliers. Together they are the
// inputs the compaction gates consume — anchorCompactablePrefixMode asks the scanners where the
// protected prefix ends, and headBurstGate asks the pricers whether a head-anchored drop may fire
// — so isolating them keeps the anchor/economics vocabulary readable on its own and leaves
// anthropic_compact.go to the splice it exists for.
//
// Pure code motion out of anthropic_compact.go: same package, same declarations, no behavior
// change.

import (
	"encoding/json"
	"math"
)

// lastBreakpointMessage returns the index of the last messages[] element whose content
// carries a cache_control breakpoint, or -1 if none does.
func lastBreakpointMessage(elems []json.RawMessage) int {
	last := -1
	for i, el := range elems {
		if messageHasCacheControl(el) {
			last = i
		}
	}
	return last
}

// firstBreakpointMessage returns the index of the FIRST messages[] element whose content
// carries a cache_control breakpoint, or -1 if none does. This is the anchor for the protected
// prefix: the earliest message breakpoint marks the stable cached HEAD the provider reuses
// every turn (the growing-conversation layout marks the static head AND recent turns; only the
// head's prefix is byte-stable across turns). Anchoring here — not on the last breakpoint —
// is what lets compaction drop the un-cacheable MIDDLE on real multi-breakpoint traffic.
func firstBreakpointMessage(elems []json.RawMessage) int {
	for i, el := range elems {
		if messageHasCacheControl(el) {
			return i
		}
	}
	return -1
}

func rangeHasCacheControl(elems []json.RawMessage, start, end int) bool {
	if start < 0 {
		start = 0
	}
	if end > len(elems) {
		end = len(elems)
	}
	for i := start; i < end; i++ {
		if messageHasCacheControl(elems[i]) {
			return true
		}
	}
	return false
}

func messageHasCacheControl(el json.RawMessage) bool {
	var m struct {
		Content json.RawMessage `json:"content"`
	}
	if json.Unmarshal(el, &m) != nil {
		return false
	}
	return rawHasCacheControl(m.Content)
}

// CacheBurstBreakEvenTurns prices an explicit cache-burst rewrite. If a compaction would
// delete already cache_control-marked tokens, the immediate penalty is the cached suffix that
// must be written cold again; the future saving is only the provider's discounted read cost for
// the deleted cached tokens. It returns the minimum future turns needed to repay that burst.
// A return of 0 means there is no one-time suffix penalty; MaxInt means the rewrite never
// breaks even under the supplied multipliers.
func CacheBurstBreakEvenTurns(droppedCachedTokens, invalidatedSuffixTokens int, readMult, writeMult float64) int {
	if invalidatedSuffixTokens <= 0 {
		return 0
	}
	perTurnSaving := float64(droppedCachedTokens) * readMult
	oneTimePenalty := float64(invalidatedSuffixTokens) * (writeMult - readMult)
	if oneTimePenalty <= 0 {
		return 0
	}
	if perTurnSaving <= 0 {
		return int(^uint(0) >> 1)
	}
	return int(math.Ceil(oneTimePenalty / perTurnSaving))
}

// CacheBurstPaysBack reports whether an explicit cache-burst rewrite has enough future
// turns left in this session to repay itself. currentTurn is 1-based and "now": in a
// 50-turn session at currentTurn=20, there are 30 future turns left (21..50). Unknown or
// exhausted horizons return false unless the burst has no one-time penalty. It is
// CacheBurstPaysBackWithMargin at the untuned zero margin (fire whenever the burst repays at
// all), so every caller predating the fed-back threshold is byte-for-byte unchanged.
func CacheBurstPaysBack(totalTurns, currentTurn, droppedCachedTokens, invalidatedSuffixTokens int, readMult, writeMult float64) bool {
	return CacheBurstPaysBackWithMargin(totalTurns, currentTurn, droppedCachedTokens, invalidatedSuffixTokens, readMult, writeMult, 0)
}

// CacheBurstPaysBackWithMargin is CacheBurstPaysBack with the fed-back fire/bail threshold
// (#2817): the burst fires only when the remaining horizon clears the break-even by at least
// minHorizonMargin extra turns — remainingTurns >= breakEven + minHorizonMargin. A positive
// margin (learned OFFLINE by rsiloop.TuneFirePolicy over scored per-fire receipts) hedges the
// break-even estimate's error by bailing the thin-headroom fires whose realized net most often
// goes negative when the session ends earlier than predicted. The comparison is the exact live-gate
// twin of rsiloop.FirePolicy.Fires: since a receipt's PredictedHorizonMargin is
// remainingTurns − breakEven, "remainingTurns >= breakEven + margin" is "PredictedHorizonMargin >=
// margin", so the offline tuner and this gate agree by construction. The margin does NOT relax the
// penalty-free short-circuit: a burst with no one-time penalty (breakEven 0, e.g. an observed-cold
// suffix) fires horizon-free regardless of the margin — there is no break-even error to hedge.
// A negative margin is clamped to 0 (never below the untuned gate). minHorizonMargin 0 reproduces
// CacheBurstPaysBack exactly.
func CacheBurstPaysBackWithMargin(totalTurns, currentTurn, droppedCachedTokens, invalidatedSuffixTokens int, readMult, writeMult float64, minHorizonMargin int) bool {
	breakEven := CacheBurstBreakEvenTurns(droppedCachedTokens, invalidatedSuffixTokens, readMult, writeMult)
	if breakEven == 0 {
		return true
	}
	if totalTurns <= 0 || currentTurn <= 0 {
		return false
	}
	if minHorizonMargin < 0 {
		minHorizonMargin = 0
	}
	remainingTurns := totalTurns - currentTurn
	return remainingTurns >= breakEven+minHorizonMargin
}

// rawHasCacheControl reports whether a `system` or message `content` value (a bare string,
// a single block object, or an array of blocks) carries a cache_control breakpoint on any
// block. A bare string has no blocks → no breakpoint.
func rawHasCacheControl(content json.RawMessage) bool {
	if len(content) == 0 {
		return false
	}
	// Array of blocks (the common Claude Code shape).
	var blocks []map[string]json.RawMessage
	if json.Unmarshal(content, &blocks) == nil {
		for _, b := range blocks {
			if _, ok := b["cache_control"]; ok {
				return true
			}
		}
		return false
	}
	// A single block object.
	var block map[string]json.RawMessage
	if json.Unmarshal(content, &block) == nil {
		_, ok := block["cache_control"]
		return ok
	}
	return false
}
