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

// ErrV41SpecDecodeHostMirrorMismatch is the typed fail-closed refusal for a
// host-mirror trim that the device scatter cannot be in agreement with: a width
// outside the mirrored proposals, or an accepted count beyond the verified width.
// A silently-clamped mirror would drift from the device, so the trim is refused
// instead. It is defined inside internal/model so no cross-package import is
// needed to witness the refusal.
var ErrV41SpecDecodeHostMirrorMismatch = errors.New("model: DeepSeek V4.1 Flash spec-decode host mirror disagrees with the device scatter width")

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

// ---------------------------------------------------------------------------
// Per-step verify width (issue #13148)
//
// The admission seam above gates a whole spec-decode REQUEST. This section owns
// the per-STEP contract that request runs under: how many of an already-drafted
// block this step verifies, how a host mirror of the previous step's proposals is
// trimmed to agree with the device scatter, and how unverified drafts carry
// forward across a length-cap/EOS boundary.
//
// The width is a PURE FUNCTION OF THE BATCH. Both the execute path and the async
// FSM callback call the same v41SpecDecodeVerifyWidth and therefore agree without
// communicating: neither may derive a width from its own private state. That is
// the invariant a fixed draft depth with a strict prefix verified depends on; a
// width chosen independently on either side is how a broadcast shape error is
// born.
// ---------------------------------------------------------------------------

// V41SpecDecodeStepInput is the per-step batch view the verify width derives
// from. It is deliberately a plain value with no session, backend, or device
// handle: the width must be computable identically on the execute path and in the
// async FSM callback, neither of which may reach into the other's state.
type V41SpecDecodeStepInput struct {
	// DraftLen is the declared per-step draft depth (the #13148 DraftLen). It is
	// the requested width before the batch narrows it.
	DraftLen int
	// Shape is the concrete batch shape this step runs under. Only its
	// sequence/decode members are read, so a shape that has not yet been
	// scheduler-adjudicated still yields a deterministic (if conservative) width.
	Shape SpeculativeShape
	// MixedPrefill is true when this step also carries prefill remainder tokens.
	// A mixed step takes the narrower width; see v41SpecDecodeVerifyWidth.
	MixedPrefill bool
	// MaxWidth caps the width for this step (a scheduler or backend bound). A
	// non-positive value means "no additional cap".
	MaxWidth int
}

// V41SpecDecodeVerifyWidth derives the number of already-drafted tokens this step
// verifies. It is a PURE FUNCTION OF THE BATCH: same input, same width, with no
// clock, no counter, and no acceptance history. The execute path and the async
// FSM callback both call it, so they agree by construction rather than by
// convention.
//
// The narrowing ladder, in order:
//
//  1. Start from DraftLen (0 -> nothing to verify).
//  2. A mixed prefill+decode step takes the NARROWER of DraftLen and the decode
//     width the prefill remainder leaves room for: a step spending its budget on
//     prefill cannot verify a full-depth block, and this override beats the
//     batch-size schedule below.
//  3. Otherwise the width is min(DraftLen, per-sequence decode width from Shape).
//     A shape spanning several sequences verifies the per-sequence depth, not the
//     whole batch's decode total, so the width does not grow with N_seq.
//  4. Apply MaxWidth when it narrows further.
//
// The result is never negative and never exceeds DraftLen: the host mirror holds
// at most the drafts this step proposed, so a width greater than DraftLen would
// demand a trim of tokens that were never drafted.
func v41SpecDecodeVerifyWidth(in V41SpecDecodeStepInput) int {
	width := in.DraftLen
	if width <= 0 {
		return 0
	}
	if in.MixedPrefill {
		// Narrower width for a mixed step: the prefill remainder consumes the
		// step's capacity, so the verify width is capped by the room left for it.
		if mixed := v41SpecDecodeMixedDecodeWidth(in.Shape); mixed < width {
			width = mixed
		}
	} else if perSeq := v41SpecDecodePerSequenceWidth(in.Shape); perSeq > 0 && perSeq < width {
		width = perSeq
	}
	if in.MaxWidth > 0 && in.MaxWidth < width {
		width = in.MaxWidth
	}
	if width < 0 {
		return 0
	}
	return width
}

// v41SpecDecodePerSequenceWidth returns the per-sequence decode depth a shape
// implies (that is, the widest single sequence's token count, K_i + 1). It reads
// TokensPerSeq first and falls back to MaxTokensPerSeq; a shape that carries only
// aggregate counts has no per-sequence view and returns 0 (meaning "no narrowing
// from the shape"), which keeps a not-yet-adjudicated shape deterministic rather
// than guessing a width.
func v41SpecDecodePerSequenceWidth(shape SpeculativeShape) int {
	if len(shape.TokensPerSeq) > 0 {
		width := 0
		for _, n := range shape.TokensPerSeq {
			if n > width {
				width = n
			}
		}
		return width
	}
	return shape.MaxTokensPerSeq
}

// v41SpecDecodeMixedDecodeWidth is the decode width a mixed prefill+decode step
// can afford. A step's total token budget is bounded by the combined decode and
// prefill width it must process; the prefill remainder is served first, so only
// the leftover is available to speculative verification.
//
// The result is capped by v41SpecDecodePerSequenceWidth so a mixed step can only
// ever be NARROWER than the same step without prefill. Reading the shape's
// aggregate decode total for a single sequence can exceed that sequence's own
// MaxTokensPerSeq (a shape may state TotalTokens independently of the per-seq
// schedule), and a mixed step that came out wider than the non-mixed one would
// invert the narrowing rather than override the schedule with a smaller width.
// When the shape carries no per-sequence view the cap does not apply: an
// unadjudicated shape keeps its aggregate estimate rather than collapsing.
func v41SpecDecodeMixedDecodeWidth(shape SpeculativeShape) int {
	decode := shape.DecodeTokens()
	if decode <= 0 {
		return v41SpecDecodePerSequenceWidth(shape)
	}
	mixed := decode
	if shape.NumSequences > 1 {
		mixed = decode / shape.NumSequences
	}
	if perSeq := v41SpecDecodePerSequenceWidth(shape); perSeq > 0 && mixed > perSeq {
		mixed = perSeq
	}
	return mixed
}

// ---------------------------------------------------------------------------
// Host mirror (issue #13148)
//
// The device scatter commits exactly the verified prefix of a step's drafts; the
// host mirror that tracks those same drafts must trim to the SAME width or the
// two sides disagree and the next step's shapes no longer line up. The mirror
// below is the host side of that pair. It is fail-closed: a trim that cannot be
// explained by the step's own width and acceptance is refused, never silently
// clamped, because a silently-clamped mirror is exactly the drift this contract
// exists to prevent.
// ---------------------------------------------------------------------------

// V41SpecDecodeHostMirror holds the host-side copy of the previous step's
// proposals plus the cursor the device scatter committed to. TrimTo is the only
// mutation: it moves the mirror to a narrower width, matching the device.
type V41SpecDecodeHostMirror struct {
	proposals []int
	committed int
}

// NewV41SpecDecodeHostMirror returns a mirror over proposals with nothing
// committed yet.
func NewV41SpecDecodeHostMirror(proposals []int) *V41SpecDecodeHostMirror {
	return &V41SpecDecodeHostMirror{proposals: append([]int(nil), proposals...)}
}

// Proposals returns a copy of the mirrored proposals.
func (m *V41SpecDecodeHostMirror) Proposals() []int {
	if m == nil {
		return nil
	}
	return append([]int(nil), m.proposals...)
}

// Committed is the width the mirror currently agrees the device holds.
func (m *V41SpecDecodeHostMirror) Committed() int {
	if m == nil {
		return 0
	}
	return m.committed
}

// TrimTo narrows the mirror to width and reports the tokens it dropped. Width
// must be in [committed, len(proposals)]: committing an already-committed prefix
// is a no-op, and a width beyond the mirror's own proposals is refused because
// the host does not hold that many drafts to trim. This is the host-side twin of
// the device scatter: both keep exactly the verified prefix, and a mismatch is an
// error on the host rather than a silent truncation to whatever the device did.
func (m *V41SpecDecodeHostMirror) TrimTo(width int) ([]int, error) {
	if m == nil {
		return nil, fmt.Errorf("%w: host mirror is nil", ErrV41SpecDecodeHostMirrorMismatch)
	}
	if width < m.committed || width > len(m.proposals) {
		return nil, fmt.Errorf("%w: trim to %d outside [%d,%d]", ErrV41SpecDecodeHostMirrorMismatch, width, m.committed, len(m.proposals))
	}
	dropStart := m.committed
	m.committed = width
	if dropStart >= width {
		return nil, nil
	}
	dropped := append([]int(nil), m.proposals[dropStart:width]...)
	return dropped, nil
}

// V41SpecDecodeMirrorTrimForStep is the single call site that keeps the host
// mirror and the device scatter in agreement for one step. Given the step's
// already-drafted proposals, the width this step verified, and how many of the
// verified drafts the target accepted, it trims the mirror to the accepted width
// and reports the tokens the device also dropped.
//
// It trims ONCE, directly to accepted. Trimming to verifyWidth first and then
// back down to accepted would move the mirror forward past the device's own
// cursor and then demand a reverse move the mirror rightly refuses — the two
// sides would no longer agree on a strict-prefix verify, which is the normal
// case (accepted < verifyWidth). The dropped tokens are the verified tail the
// device scatter discarded: proposals[accepted:verifyWidth]. The unverified tail
// beyond verifyWidth was never a device position at all, so it is not reported
// as dropped here; it is the draft carry-forward's to keep.
//
// It fails closed when the verified width exceeds the mirrored proposals: that
// means the width and the mirror were derived from different proposal sets,
// which is precisely the broadcast shape error the contract forbids.
func V41SpecDecodeMirrorTrimForStep(proposals []int, verifyWidth, accepted int) (mirror *V41SpecDecodeHostMirror, dropped []int, err error) {
	if verifyWidth < 0 || accepted < 0 || accepted > verifyWidth {
		return nil, nil, fmt.Errorf("%w: accepted %d of verify width %d", ErrV41SpecDecodeHostMirrorMismatch, accepted, verifyWidth)
	}
	if verifyWidth > len(proposals) {
		return nil, nil, fmt.Errorf("%w: verify width %d exceeds %d mirrored proposals", ErrV41SpecDecodeHostMirrorMismatch, verifyWidth, len(proposals))
	}
	mirror = NewV41SpecDecodeHostMirror(proposals)
	if _, err = mirror.TrimTo(accepted); err != nil {
		return nil, nil, err
	}
	dropped = append([]int(nil), proposals[accepted:verifyWidth]...)
	return mirror, dropped, nil
}

// ---------------------------------------------------------------------------
// Draft carry-forward (issue #13148)
//
// A step that cannot verify its whole draft (a length cap or an EOS boundary) has
// already done the drafting work for the unverified tail. Carry-forward keeps
// that tail for the next step instead of discarding it, but only while the
// context is still open: a DONE context (the length cap reached, or the stop
// token committed) clears the carry so no stale proposals outlive the sequence.
// EOS outranks the cap: when a step both hits the length cap and commits the stop
// token, the done reason is EOS.
// ---------------------------------------------------------------------------

// V41SpecDecodeDoneReason names why a spec-decode carry was cleared. It is a
// closed set so a caller can distinguish "the sequence really finished" (EOS)
// from "we ran out of budget" (the cap).
type V41SpecDecodeDoneReason int

const (
	// V41SpecDecodeNotDone is the zero value: the context is live and carries
	// forward.
	V41SpecDecodeNotDone V41SpecDecodeDoneReason = iota
	// V41SpecDecodeDoneCap is a done context because the length cap was reached.
	V41SpecDecodeDoneCap
	// V41SpecDecodeDoneEOS is a done context because the stop token was
	// committed. It outranks the cap.
	V41SpecDecodeDoneEOS
)

// String renders the reason for receipts and test messages.
func (r V41SpecDecodeDoneReason) String() string {
	switch r {
	case V41SpecDecodeDoneCap:
		return "length-cap"
	case V41SpecDecodeDoneEOS:
		return "eos"
	default:
		return "not-done"
	}
}

// V41SpecDecodeCarry is the carry-forward state between steps: the drafts a
// previous step could not verify (because its width narrowed or its context
// ended) plus the reason the context is done, if it is.
type V41SpecDecodeCarry struct {
	pending    []int
	doneReason V41SpecDecodeDoneReason
}

// V41SpecDecodeCarryInput is the pure per-step input to the carry decision.
type V41SpecDecodeCarryInput struct {
	// Drafted is the full draft this step drafted.
	Drafted []int
	// VerifyWidth is how many of Drafted this step verified. The unverified tail
	// is what carries forward.
	VerifyWidth int
	// Emitted is how many tokens this step committed to the sequence.
	Emitted int
	// LengthCap is the maximum sequence length; reaching it marks the context
	// done with V41SpecDecodeDoneCap.
	LengthCap int
	// EmittedTotal is the sequence's committed length AFTER this step. It is the
	// value compared against LengthCap.
	EmittedTotal int
	// StopEnabled is true when a stop token is configured.
	StopEnabled bool
	// StopToken is the configured stop token id.
	StopToken int
	// CommittedStop is true when this step committed the stop token. EOS outranks
	// the length cap when both fire.
	CommittedStop bool
}

// StepV41SpecDecodeCarry advances the carry across one step: the unverified tail
// of this step's drafts carries forward, the tail already consumed by a previous
// carry is not re-carried, and a DONE context clears the carry outright. The done
// reason is EOS when the stop token was committed, even if the length cap was
// also reached in the same step.
//
// This is a pure function of its input plus the receiver's prior state: same
// state and input yield the same next state, which is what lets the execute path
// and the async FSM callback share one carry without coordinating.
func (c *V41SpecDecodeCarry) Step(in V41SpecDecodeCarryInput) V41SpecDecodeCarry {
	incoming := c.pending
	done := V41SpecDecodeNotDone
	switch {
	case in.CommittedStop && in.StopEnabled:
		done = V41SpecDecodeDoneEOS
	case in.LengthCap > 0 && in.EmittedTotal >= in.LengthCap:
		done = V41SpecDecodeDoneCap
	}
	if done != V41SpecDecodeNotDone {
		// A done context clears the carry: no stale proposals outlive the
		// sequence. EOS is recorded as the reason when it outranked the cap.
		return V41SpecDecodeCarry{doneReason: done}
	}

	// Carry the drafts this step drafted but did not verify. The step's draft is
	// assumed to begin with the previously-carried tokens (the drafter continues
	// from the carried head), so the incoming carry occupies Drafted[0:len(incoming)].
	// The verified prefix is Drafted[0:width]:
	//
	//   - width >= len(incoming): the verified prefix absorbed the whole incoming
	//     carry, so the entire unverified tail is new drafts and carries.
	//   - width < len(incoming): the verified prefix stopped short, so the
	//     unconsumed incoming[width:] sits at the head of the tail and is dropped
	//     before carrying.
	//
	// Subtracting len(incoming) from the tail unconditionally (as an earlier
	// revision did) would silently drop genuinely new drafts whenever the incoming
	// carry was shorter than the verified width.
	width := in.VerifyWidth
	if width < 0 {
		width = 0
	}
	if width > len(in.Drafted) {
		width = len(in.Drafted)
	}
	tail := in.Drafted[width:]
	carryStart := 0
	if len(incoming) > width {
		// The verified prefix stopped short of the whole incoming carry, so
		// incoming[width:] is still sitting at the head of the tail and must not
		// be re-carried.
		carryStart = len(incoming) - width
	}
	if carryStart > len(tail) {
		carryStart = len(tail)
	}
	next := append([]int(nil), tail[carryStart:]...)
	return V41SpecDecodeCarry{pending: next}
}

// Pending returns a copy of the drafts carrying forward.
func (c *V41SpecDecodeCarry) Pending() []int {
	if c == nil {
		return nil
	}
	return append([]int(nil), c.pending...)
}

// Done reports whether the context is done and why. A done carry holds no
// pending drafts.
func (c *V41SpecDecodeCarry) Done() (bool, V41SpecDecodeDoneReason) {
	if c == nil {
		return false, V41SpecDecodeNotDone
	}
	return c.doneReason != V41SpecDecodeNotDone, c.doneReason
}
