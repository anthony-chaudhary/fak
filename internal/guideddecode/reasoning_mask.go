package guideddecode

// reasoning_arm.go — a reasoning-boundary latch that decides WHEN the byte-level
// constrainer in guideddecode.go may begin applying its mask.
//
// A reasoning model (GLM-5.2, Qwen-3.6, DeepSeek V4) emits its chain of thought
// inline, wrapped in <think>…</think>, ahead of the user-visible answer, and only
// AFTER that span closes does it emit the tool-call envelope the constrainer knows
// how to shape. If the tool-envelope mask were applied from byte 0 it would force
// the envelope grammar DURING the think span — corrupting the reasoning or
// dead-ending the matcher. This latch defers the mask until the reasoning-end
// marker has been emitted.
//
// The latch is a ONE-WAY arm: before the reasoning-end marker the mask is inactive
// (free generation); the token that completes the marker arms it; every token after
// keeps the mask active. Once armed it never re-disables, even if a later <think>
// re-appears — a second reasoning span cannot un-constrain a stream that has already
// entered its answer.
//
// Pure and wall-clock-free: same tokens in, same result out, so a caller's prefix
// reuse stays stable. This file deliberately keeps the marker a caller-supplied
// value (defaulting to the qwen3-style close tag) rather than importing the harness
// so guideddecode stays free of model/tokenizer deps, as its package doc requires.

import "strings"

// defaultMarker is the qwen3-style reasoning-end marker used when a caller passes an
// empty marker. It mirrors the sibling reasoning strip in the harness, whose close
// tag is the same literal.
const defaultMarker = "</think>"

// NoMarkerVerdict selects what the latch reports for a stream in which the
// reasoning-end marker never appears at all (e.g. a truncated reasoning span, or a
// non-reasoning turn that never opened a think block).
type NoMarkerVerdict int

const (
	// InactiveUntilMarker keeps the mask INACTIVE for the whole stream when no
	// reasoning-end marker is ever emitted. This mirrors the reasoning strip's
	// truncated-span rule: an unterminated reasoning span is treated as still
	// reasoning, so the constrainer is never armed and free generation is left
	// untouched. This is the fail-open default.
	InactiveUntilMarker NoMarkerVerdict = iota

	// ActiveWhenNoMarker treats a markerless stream as a non-reasoning turn and
	// reports the mask ACTIVE from the first token — byte-identical to constraining
	// the envelope from byte 0 the way guideddecode does today. This is the
	// fail-closed default for callers that want a markerless turn constrained.
	ActiveWhenNoMarker
)

// MaskArm is the incremental one-way latch. It observes emitted tokens one at a
// time and reports whether the constrained-decode mask must be active for the NEXT
// decode step. The zero value is not ready; build one with NewMaskArm.
//
// Semantics of Emit's return, position by position:
//   - a token entirely BEFORE the reasoning-end marker  => false (mask inactive)
//   - the token that COMPLETES the reasoning-end marker  => true  (latch arms)
//   - every token AFTER the marker                       => true  (mask active)
//
// The NoMarkerVerdict does not affect the incremental latch (a streaming latch
// cannot see whether a marker still lies ahead, so it optimistically stays inactive
// while reasoning); it is applied by MaskActiveByToken over a completed stream.
type MaskArm struct {
	markerLower string          // reasoning-end marker, lower-cased for matching
	noMark      NoMarkerVerdict // reported by the whole-stream helper on a markerless stream
	armed       bool            // one-way: true once the marker has been emitted
	tail        string          // rolling lower-cased suffix of pre-marker text (< len(marker) bytes)
}

// NewMaskArm builds a latch for one decode stream. An empty marker defaults to the
// qwen3-style reasoning-end tag. Matching is case-insensitive, mirroring the
// harness reasoning strip that recognises <Think>/<THINK> variants.
func NewMaskArm(marker string, noMark NoMarkerVerdict) *MaskArm {
	if marker == "" {
		marker = defaultMarker
	}
	return &MaskArm{markerLower: strings.ToLower(marker), noMark: noMark}
}

// Emit records that tok was just emitted and returns whether the mask must be
// active for the next decode step. It is a one-way arm: once the reasoning-end
// marker has been seen it always returns true and ignores tok's content.
func (a *MaskArm) Emit(tok string) bool {
	if a.armed {
		return true
	}
	// Join the carried pre-marker suffix with the new token so a marker split
	// across a token boundary is still detected, then search for the marker.
	win := a.tail + strings.ToLower(tok)
	if strings.Contains(win, a.markerLower) {
		a.armed = true
		a.tail = ""
		return true
	}
	// No marker yet: keep only the trailing bytes that could still begin it.
	if n := len(a.markerLower) - 1; n > 0 && len(win) > n {
		win = win[len(win)-n:]
	}
	a.tail = win
	return false
}

// Armed reports whether the reasoning-end marker has already been observed. Once
// true it stays true for the life of the latch.
func (a *MaskArm) Armed() bool { return a.armed }

// MaskActiveByToken evaluates a whole token stream and returns, for each token
// position, whether the constrained-decode mask is active at that position.
//
// When the reasoning-end marker appears, positions before it are inactive and the
// marker token plus every position after it are active. When the marker never
// appears anywhere in the stream, every position takes the NoMarkerVerdict default.
// The result is deterministic and wall-clock-free.
func MaskActiveByToken(tokens []string, marker string, noMark NoMarkerVerdict) []bool {
	a := NewMaskArm(marker, noMark)
	out := make([]bool, len(tokens))
	seen := false
	for i, tok := range tokens {
		if a.Emit(tok) {
			seen = true
			out[i] = true
		}
	}
	if !seen {
		fill := noMark == ActiveWhenNoMarker
		for i := range out {
			out[i] = fill
		}
	}
	return out
}
