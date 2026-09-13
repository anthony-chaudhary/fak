package model

import (
	"errors"
	"fmt"
)

// ErrV41SpecDecodeUnqualified is the typed fail-closed refusal for a V4.1 Flash
// speculative-decode request that fails the STATIC admission gate. The adapter
// is DISABLED BY DEFAULT: the zero-value config never engages, and an enabled
// config that fails the layout/length check is refused here.
//
// IMPORTANT: this refusal is not a losslessness proof. Validate only checks the
// static layout and DraftLen window; it does NOT execute any losslessness
// verifier. Executing the losslessness verifier (the companion internal/mtptune
// losslessness invariant plus its K=1..32 sweep) is a precondition the serving
// consumer must satisfy before it may engage speculative decode. This adapter
// does not itself establish or execute that proof.
var ErrV41SpecDecodeUnqualified = errors.New("model: DeepSeek V4.1 Flash speculative decode is not admitted; static layout/length gate failed")

// ErrV41SpecDecodeAnchorCorruption is the typed refusal for a spec-decode
// layout whose anchor token is not the first drafted token. An anchor-first
// layout corrupts the 1+N bonus accounting: the verified prefix and the
// accepted bonus token would no longer be bit-exact with base greedy decoding,
// so it is rejected outright. It is defined inside internal/model so no
// cross-package import is needed to witness the refusal.
var ErrV41SpecDecodeAnchorCorruption = errors.New("model: DeepSeek V4.1 Flash spec-decode layout places a non-anchor token first")

// V41SpecDecodeLayout names the draft/verify token layout a V4.1 spec-decode
// request proposes. Only V41SpecDecodeLayoutBonus1N is admissible; the zero
// value is deliberately invalid so a zero-value config fails closed.
type V41SpecDecodeLayout int

const (
	// V41SpecDecodeLayoutInvalid is the zero value. It names no admitted layout,
	// so a config that leaves Layout unset must not engage spec-decode.
	V41SpecDecodeLayoutInvalid V41SpecDecodeLayout = iota
	// V41SpecDecodeLayoutBonus1N is the default and only admitted layout: one
	// anchor token followed by N drafted tokens, where accepting all N drafts
	// yields one bonus token. Its acceptance accounting is bit-exact with base
	// greedy decoding (see v41SpecDecodeOracleFromLogits).
	V41SpecDecodeLayoutBonus1N
	// V41SpecDecodeLayoutAnchorFirst is the explicitly-refused layout in which a
	// non-anchor token is placed first. It is retained as a named counterexample
	// for the parity guard; it is never admissible.
	V41SpecDecodeLayoutAnchorFirst
)

// V41SpecDecodeConfig is the typed admission gate for V4.1 Flash speculative
// decode. The zero value (Enabled=false) must never engage: spec-decode is
// disabled by default and only an Enabled config that declares the default
// 1+N bonus layout and a DraftLen in 1..32 may pass Validate.
type V41SpecDecodeConfig struct {
	// Enabled requests speculative decode. It defaults to false; a false value
	// keeps the base text path byte-for-byte unchanged.
	Enabled bool
	// DraftLen is the number N of drafted tokens in the bonus 1+N layout. It is
	// admitted only in the inclusive range 1..32.
	DraftLen int
	// Layout is the draft/verify token layout. An Enabled config must declare
	// V41SpecDecodeLayoutBonus1N.
	Layout V41SpecDecodeLayout
}

// V41SpecDecodeMinDraftLen and V41SpecDecodeMaxDraftLen bound DraftLen. The
// lower bound keeps the layout at least one draft deep; the upper bound is the
// losslessness sweep range the companion invariant covers.
const (
	V41SpecDecodeMinDraftLen = 1
	V41SpecDecodeMaxDraftLen = 32
)

// Validate applies the fail-closed STATIC admission gate for a V4.1 spec-decode
// request. A disabled config (Enabled=false) is always valid and inert,
// regardless of the other fields, so disabling spec-decode can never be
// refused. An enabled config must declare the default 1+N bonus layout (any
// other layout, including the zero value, is refused with
// ErrV41SpecDecodeAnchorCorruption) and a DraftLen in 1..32; otherwise it is
// refused with ErrV41SpecDecodeUnqualified.
//
// Validate checks layout and length only. It does NOT prove losslessness and
// never runs the losslessness verifier: a config that passes here is admitted
// as a layout/length shape, not as a verified-safe draft/verify loop. The
// serving consumer must separately execute the companion internal/mtptune
// losslessness invariant (including its K=1..32 sweep) before engaging
// speculative decode.
func (c V41SpecDecodeConfig) Validate() error {
	if !c.Enabled {
		return nil
	}
	if c.Layout != V41SpecDecodeLayoutBonus1N {
		return fmt.Errorf("%w: layout=%d is not the anchored bonus 1+N layout", ErrV41SpecDecodeAnchorCorruption, c.Layout)
	}
	if c.DraftLen < V41SpecDecodeMinDraftLen || c.DraftLen > V41SpecDecodeMaxDraftLen {
		return fmt.Errorf("%w: DraftLen=%d is outside [%d,%d]", ErrV41SpecDecodeUnqualified, c.DraftLen, V41SpecDecodeMinDraftLen, V41SpecDecodeMaxDraftLen)
	}
	return nil
}

// RefuseDeepSeekV41SpecDecodeUnqualified is the admission hook the V4.1
// dispatch seam calls before any speculative work. It is nil (no-op) for every
// non-V4.1 family and for a V4.1 config that does not request spec-decode, so
// the base text path is byte-for-byte unchanged. Only a V4.1 spec-decode
// request that is enabled but unqualified is refused with
// ErrV41SpecDecodeUnqualified.
//
// This composes with, and does not replace, RefuseDeepSeekV41DSpark: a config
// carrying DSpark metadata is refused by that hook on its own terms, and a
// caller that separately requests this spec-decode adapter gets this refusal.
func (c Config) RefuseDeepSeekV41SpecDecodeUnqualified(sc V41SpecDecodeConfig) error {
	if !c.IsDeepSeekV41() || !sc.Enabled {
		return nil
	}
	return sc.Validate()
}

// v41SpecDecodeOracleFromLogits is an INDEPENDENT base greedy oracle. It takes
// the raw logits rows the base target forward emits at each position and
// computes the greedy token stream by argmax over each row. It shares no code
// with the speculative accept path, so comparing a spec-decode result against
// it witnesses parity against a genuinely independent computation rather than a
// copy of the adapter's own input. Ties break toward the lowest token index.
//
// An empty logits row names no token. Rather than panic on row[0], it is
// fail-closed: the row emits no token, so the returned stream is shorter than
// len(rows). Callers that require a token per row must pass non-empty rows.
func v41SpecDecodeOracleFromLogits(rows [][]float64) []int {
	out := make([]int, 0, len(rows))
	for _, row := range rows {
		if len(row) == 0 {
			continue
		}
		best := 0
		bestVal := row[0]
		for i := 1; i < len(row); i++ {
			if row[i] > bestVal {
				bestVal = row[i]
				best = i
			}
		}
		out = append(out, best)
	}
	return out
}

// v41SpecDecodeAcceptedUnderBonus1N is the sealed acceptance function for the
// default 1+N bonus layout. Given the anchor token and the draft tokens, plus a
// verdict slice reporting whether the base greedy target accepted each draft,
// it returns the accepted prefix under the bonus 1+N accounting: the anchor,
// then every draft accepted in order up to the first rejection. When every
// draft is accepted it emits exactly one bonus token drawn from the base stream
// at position len(draft) (the next oracle token), which is what makes the
// accounting bit-exact with base greedy decoding. It assumes the layout has
// been attested as the anchored bonus 1+N layout.
func v41SpecDecodeAcceptedUnderBonus1N(anchor int, draft []int, accept []bool, base []int) []int {
	out := []int{anchor}
	for i, tok := range draft {
		if i >= len(accept) || !accept[i] {
			return out
		}
		out = append(out, tok)
	}
	// All drafts accepted: consume the next oracle token as the single bonus.
	// Draft i corresponds to base[i+1] (base[0] is the anchor), so the bonus is
	// the token after the last draft, base[len(draft)+1]. Emitting it makes the
	// accepted prefix exactly the oracle's prefix of the same length.
	if len(draft)+1 < len(base) {
		out = append(out, base[len(draft)+1])
	}
	return out
}

// v41SpecDecodeLayoutIsBonus1N reports whether layout is the admitted default.
// It exists so both Validate and the parity seam read layout admissibility from
// one place.
func v41SpecDecodeLayoutIsBonus1N(layout V41SpecDecodeLayout) bool {
	return layout == V41SpecDecodeLayoutBonus1N
}
