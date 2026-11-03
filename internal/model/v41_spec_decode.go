package model

import (
	"errors"
	"fmt"
)

// ErrV41SpecDecodeUnqualified is the typed fail-closed refusal for a V4.1 Flash
// speculative-decode request that fails the static admission gate. The adapter is disabled by
// default: the zero-value config never engages.
//
// This refusal is not a losslessness proof: Validate only checks the static layout and DraftLen
// window. Executing the losslessness verifier (the companion internal/mtptune invariant plus its
// K=1..32 sweep) is a precondition the serving consumer must satisfy before engaging spec-decode.
var ErrV41SpecDecodeUnqualified = errors.New("model: DeepSeek V4.1 Flash speculative decode is not admitted; static layout/length gate failed")

// ErrV41SpecDecodeAnchorCorruption is the typed refusal for a spec-decode layout whose anchor
// token is not the first drafted token: an anchor-first layout corrupts the 1+N bonus accounting
// (the verified prefix and accepted bonus token would no longer be bit-exact with base greedy
// decoding), so it is rejected outright.
var ErrV41SpecDecodeAnchorCorruption = errors.New("model: DeepSeek V4.1 Flash spec-decode layout places a non-anchor token first")

// ErrV41SpecDecodeHostMirrorMismatch is the typed fail-closed refusal for a host-mirror trim the
// device scatter cannot agree with: a width outside the mirrored proposals, or an accepted count
// beyond the verified width. A silently-clamped mirror would drift from the device, so the trim
// is refused instead.
var ErrV41SpecDecodeHostMirrorMismatch = errors.New("model: DeepSeek V4.1 Flash spec-decode host mirror disagrees with the device scatter width")

// V41SpecDecodeLayout names the draft/verify token layout a V4.1 spec-decode request proposes.
// Only Bonus1N is admissible; the zero value is deliberately invalid so a zero-value config fails
// closed.
type V41SpecDecodeLayout int

const (
	// Invalid is the zero value: no admitted layout, so an unset Layout must not engage spec-decode.
	V41SpecDecodeLayoutInvalid V41SpecDecodeLayout = iota
	// Bonus1N is the default and only admitted layout: one anchor token followed by N drafted
	// tokens, where accepting all N drafts yields one bonus token, bit-exact with base greedy
	// decoding (see v41SpecDecodeOracleFromLogits).
	V41SpecDecodeLayoutBonus1N
	// AnchorFirst is the explicitly-refused layout with a non-anchor token first, retained as a
	// named counterexample for the parity guard; never admissible.
	V41SpecDecodeLayoutAnchorFirst
)

// V41SpecDecodeConfig is the typed admission gate for V4.1 Flash speculative decode. The zero
// value (Enabled=false) must never engage: only an Enabled config declaring the 1+N bonus layout
// and a DraftLen in 1..32 may pass Validate.
type V41SpecDecodeConfig struct {
	// Enabled requests speculative decode; false keeps the base text path byte-for-byte unchanged.
	Enabled bool
	// DraftLen is the number N of drafted tokens in the bonus 1+N layout, admitted only in 1..32.
	DraftLen int
	// Layout is the draft/verify token layout; an Enabled config must declare V41SpecDecodeLayoutBonus1N.
	Layout V41SpecDecodeLayout
	// RequestKey is the per-request keyed-coin identity the verify path draws its accept/bonus
	// coins from. Zero names no identity, so an unkeyed config fails closed on the keyed seam.
	// Arm it with WithRequestKey.
	RequestKey V41SpecDecodeRequestKey
}

// V41SpecDecodeMinDraftLen and V41SpecDecodeMaxDraftLen bound DraftLen; the upper bound is the
// losslessness sweep range the companion invariant covers.
const (
	V41SpecDecodeMinDraftLen = 1
	V41SpecDecodeMaxDraftLen = 32
)

// Validate applies the fail-closed static admission gate. A disabled config is always valid and
// inert; an enabled config must declare the 1+N bonus layout (any other layout, including the
// zero value, is refused with ErrV41SpecDecodeAnchorCorruption) and a DraftLen in 1..32, else it
// is refused with ErrV41SpecDecodeUnqualified.
//
// It checks layout and length only and never runs the losslessness verifier: passing here admits
// a layout/length shape, not a verified-safe draft/verify loop.
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

// RefuseDeepSeekV41SpecDecodeUnqualified is the admission hook the V4.1 dispatch seam calls
// before any speculative work. It is a no-op for every non-V4.1 family and for a config that does
// not request spec-decode, so the base text path is byte-for-byte unchanged. It composes with,
// and does not replace, RefuseDeepSeekV41DSpark.
func (c Config) RefuseDeepSeekV41SpecDecodeUnqualified(sc V41SpecDecodeConfig) error {
	if !c.IsDeepSeekV41() || !sc.Enabled {
		return nil
	}
	return sc.Validate()
}

// v41SpecDecodeOracleFromLogits is an independent base greedy oracle: it computes the greedy
// token stream by argmax over the raw logits rows the base target forward emits, sharing no code
// with the speculative accept path, so comparing a spec-decode result against it witnesses parity
// against a genuinely independent computation. Ties break toward the lowest token index.
//
// An empty logits row names no token and emits none (fail-closed rather than panicking), so the
// returned stream is shorter than len(rows); callers requiring a token per row must pass
// non-empty rows.
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

// v41SpecDecodeAcceptedUnderBonus1N is the sealed acceptance function for the default 1+N bonus
// layout: the anchor, then every draft accepted in order up to the first rejection; when every
// draft is accepted it emits exactly one bonus token (the next oracle token), which is what makes
// the accounting bit-exact with base greedy decoding. It assumes the layout has been attested as
// the anchored bonus 1+N layout.
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

// v41SpecDecodeLayoutIsBonus1N reports whether layout is the admitted default, so both Validate
// and the parity seam read layout admissibility from one place.
func v41SpecDecodeLayoutIsBonus1N(layout V41SpecDecodeLayout) bool {
	return layout == V41SpecDecodeLayoutBonus1N
}

// ---------------------------------------------------------------------------
// Per-step verify width (issue #13148)
//
// The admission seam above gates a whole spec-decode REQUEST; this section owns the per-STEP
// contract: how many of an already-drafted block this step verifies, how a host mirror trims to
// agree with the device scatter, and how unverified drafts carry forward across a
// length-cap/EOS boundary.
//
// The width is a PURE FUNCTION OF THE BATCH: the execute path and the async FSM callback call the
// same v41SpecDecodeVerifyWidth and agree without communicating. A width chosen independently on
// either side is how a broadcast shape error is born.
// ---------------------------------------------------------------------------

// V41SpecDecodeStepInput is the per-step batch view the verify width derives from. It is a plain
// value with no session, backend, or device handle: the width must be computable identically on
// the execute path and in the async FSM callback.
type V41SpecDecodeStepInput struct {
	// DraftLen is the declared per-step draft depth: the requested width before the batch narrows it.
	DraftLen int
	// Shape is the concrete batch shape this step runs under; only its sequence/decode members are
	// read, so a not-yet-adjudicated shape still yields a deterministic width.
	Shape SpeculativeShape
	// MixedPrefill is true when this step also carries prefill remainder tokens; a mixed step takes
	// the narrower width.
	MixedPrefill bool
	// MaxWidth caps the width for this step; non-positive means "no additional cap".
	MaxWidth int
}

// V41SpecDecodeVerifyWidth derives the number of already-drafted tokens this step verifies. It is
// a pure function of the batch: same input, same width, with no clock, counter, or acceptance
// history.
//
// Narrowing ladder: start from DraftLen; a mixed prefill+decode step takes the narrower of
// DraftLen and the decode width the prefill remainder leaves room for (this override beats the
// batch-size schedule); otherwise min(DraftLen, per-sequence decode width from Shape) — a shape
// spanning several sequences verifies the per-sequence depth, not the batch's decode total; then
// apply MaxWidth when it narrows further.
//
// The result is never negative and never exceeds DraftLen: the host mirror holds at most the
// drafts this step proposed.
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

// v41SpecDecodePerSequenceWidth returns the per-sequence decode depth a shape implies (the widest
// single sequence's token count, K_i + 1): TokensPerSeq first, falling back to MaxTokensPerSeq;
// a shape with only aggregate counts returns 0 ("no narrowing"), keeping an unadjudicated shape
// deterministic rather than guessing a width.
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

// v41SpecDecodeMixedDecodeWidth is the decode width a mixed prefill+decode step can afford: the
// prefill remainder is served first, so only the leftover is available to speculative
// verification.
//
// The result is capped by v41SpecDecodePerSequenceWidth so a mixed step can only ever be NARROWER
// than the same step without prefill — the shape's aggregate decode total can exceed a sequence's
// own MaxTokensPerSeq, and a mixed step wider than the non-mixed one would invert the narrowing.
// With no per-sequence view the cap does not apply: an unadjudicated shape keeps its aggregate
// estimate rather than collapsing.
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
// The device scatter commits exactly the verified prefix of a step's drafts; the host mirror that
// tracks those same drafts must trim to the SAME width or the two sides disagree and the next
// step's shapes no longer line up. Fail-closed: a trim that cannot be explained by the step's own
// width and acceptance is refused, never silently clamped — a clamped mirror is exactly the drift
// this contract exists to prevent.
// ---------------------------------------------------------------------------

// V41SpecDecodeHostMirror holds the host-side copy of the previous step's proposals plus the
// cursor the device scatter committed to. TrimTo is the only mutation: it moves the mirror to a
// narrower width, matching the device.
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

// TrimTo narrows the mirror to width and reports the tokens it dropped. Width must be in
// [committed, len(proposals)]: committing an already-committed prefix is a no-op, and a width
// beyond the mirror's own proposals is refused because the host does not hold that many drafts to
// trim — a mismatch is an error on the host rather than a silent truncation to whatever the
// device did.
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

// V41SpecDecodeMirrorTrimForStep is the single call site that keeps the host mirror and the
// device scatter in agreement for one step: given the step's proposals, the width this step
// verified, and how many verified drafts the target accepted, it trims the mirror to the accepted
// width and reports the tokens the device also dropped.
//
// It trims ONCE, directly to accepted: trimming to verifyWidth first and back down would move the
// mirror past the device's own cursor and then demand a reverse move the mirror rightly refuses.
// The dropped tokens are the verified tail the device scatter discarded,
// proposals[accepted:verifyWidth]; the unverified tail beyond verifyWidth was never a device
// position, so it is the draft carry-forward's to keep.
//
// It fails closed when the verified width exceeds the mirrored proposals: the width and the mirror
// were derived from different proposal sets, precisely the broadcast shape error the contract
// forbids.
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
// A step that cannot verify its whole draft (length cap or EOS boundary) has already done the
// drafting work for the unverified tail; carry-forward keeps that tail for the next step, but a
// DONE context (cap reached, or stop token committed) clears the carry so no stale proposals
// outlive the sequence. EOS outranks the cap.
// ---------------------------------------------------------------------------

// V41SpecDecodeDoneReason names why a spec-decode carry was cleared: a closed set so a caller can
// distinguish "the sequence really finished" (EOS) from "we ran out of budget" (the cap).
type V41SpecDecodeDoneReason int

const (
	// NotDone is the zero value: the context is live and carries forward.
	V41SpecDecodeNotDone V41SpecDecodeDoneReason = iota
	// DoneCap: the length cap was reached.
	V41SpecDecodeDoneCap
	// DoneEOS: the stop token was committed. It outranks the cap.
	V41SpecDecodeDoneEOS
)

// String renders the reason for receipts.
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

// V41SpecDecodeCarry is the carry-forward state between steps: the drafts a previous step could
// not verify plus the reason the context is done, if it is.
type V41SpecDecodeCarry struct {
	pending    []int
	doneReason V41SpecDecodeDoneReason
}

// V41SpecDecodeCarryInput is the pure per-step input to the carry decision.
type V41SpecDecodeCarryInput struct {
	// Drafted is the full draft this step drafted.
	Drafted []int
	// VerifyWidth is how many of Drafted this step verified; the unverified tail is what carries.
	VerifyWidth int
	// Emitted is how many tokens this step committed to the sequence.
	Emitted int
	// LengthCap is the maximum sequence length; reaching it marks the context done (DoneCap).
	LengthCap int
	// EmittedTotal is the sequence's committed length AFTER this step, compared against LengthCap.
	EmittedTotal int
	// StopEnabled / StopToken configure the stop token.
	StopEnabled bool
	StopToken   int
	// CommittedStop is true when this step committed the stop token; EOS outranks the cap.
	CommittedStop bool
}

// Step advances the carry across one step: the unverified tail of this step's drafts carries
// forward, the tail already consumed by a previous carry is not re-carried, and a DONE context
// clears the carry outright (done reason EOS when the stop token was committed, even if the cap
// was also reached).
//
// Pure in its input plus the receiver's prior state, so the execute path and the async FSM
// callback share one carry without coordinating.
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
		// A done context clears the carry: no stale proposals outlive the sequence.
		return V41SpecDecodeCarry{doneReason: done}
	}

	// Carry the drafts this step drafted but did not verify. The step's draft begins with the
	// previously-carried tokens, so the incoming carry occupies Drafted[0:len(incoming)] and the
	// verified prefix is Drafted[0:width]:
	//   - width >= len(incoming): the verified prefix absorbed the whole carry; the entire
	//     unverified tail is new drafts and carries.
	//   - width < len(incoming): the unconsumed incoming[width:] sits at the head of the tail and
	//     is dropped before carrying.
	// Subtracting len(incoming) from the tail unconditionally would silently drop genuinely new
	// drafts whenever the incoming carry was shorter than the verified width.
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
		// incoming[width:] is still at the head of the tail and must not be re-carried.
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

// ---------------------------------------------------------------------------
// Keyed request-identity coins (issue #13195)
//
// When accept verdicts are drawn from batch-slot- or call-order-derived randomness, the same
// request re-admitted into a different slot yields a different accepted-token sequence — exactly
// the divergence the keyed coin (landed for qwen_mtp.go in #13146) removes. This section wires the
// same keyedcoin.go primitive into the V4.1 verify path: the accept coin for draft i and the
// bonus draw derive from the request identity alone, so a re-admitted request reproduces the
// identical accepted-token sequence regardless of slot. No per-slot or call-order state is
// consulted.
// ---------------------------------------------------------------------------

// V41SpecDecodeRequestKey is the V4.1 spec-decode request identity: the FNV-1a hash of the
// request id. A zero key names no identity, so the keyed seam fails closed rather than drawing
// from an unkeyed stream.
type V41SpecDecodeRequestKey uint64

// V41SpecDecodeKeyFromRequest folds a request id to its V4.1 spec-decode key; an empty id folds
// to 0, which the keyed seam treats as "no key".
func V41SpecDecodeKeyFromRequest(requestID string) V41SpecDecodeRequestKey {
	return V41SpecDecodeRequestKey(HashRequestID(requestID))
}

// Keyed reports whether the key names a real request identity; a false key must not be used for
// a keyed draw.
func (k V41SpecDecodeRequestKey) Keyed() bool {
	return k != 0
}

// V41SpecDecodeAcceptCoin is the keyed accept coin for draft index step, in [0,1). Pure in
// (request key, step): no batch slot, no clock, no acceptance history, so the same request draws
// the same coin for the same draft index no matter where it runs. A zero key fails closed with
// ok=false.
func V41SpecDecodeAcceptCoin(key V41SpecDecodeRequestKey, step int) (float32, bool) {
	if !key.Keyed() {
		return 0, false
	}
	return KeyedUniform(uint64(key), DomainAcceptCoin, step), true
}

// V41SpecDecodeBonusCoin is the keyed bonus draw for a fully-accepted step, in [0,1),
// domain-separated from the accept coin so the two never collide on the same (request, step) key.
// A zero key fails closed with ok=false.
func V41SpecDecodeBonusCoin(key V41SpecDecodeRequestKey, step int) (float32, bool) {
	if !key.Keyed() {
		return 0, false
	}
	return KeyedUniform(uint64(key), DomainBonusToken, step), true
}

// V41SpecDecodeKeyedAccept derives the accept verdicts for a whole draft block from the request
// identity alone, using the Leviathan-Chen rule (accept iff coin <= alpha, matching
// RejectionSampler.VerifyToken). A position with no alpha (alpha shorter than the draft) fails
// closed as a rejection.
//
// The returned slice is exactly what v41SpecDecodeAcceptedUnderBonus1N consumes, so keying this
// derivation makes the committed prefix a function of the request identity and the draft index.
func V41SpecDecodeKeyedAccept(key V41SpecDecodeRequestKey, draft []int, alpha []float32) ([]bool, bool) {
	if !key.Keyed() {
		return nil, false
	}
	accept := make([]bool, len(draft))
	for i := range draft {
		coin, ok := V41SpecDecodeAcceptCoin(key, i)
		if !ok {
			return nil, false
		}
		if i >= len(alpha) {
			// No acceptance probability for this position: fail closed as a rejection.
			accept[i] = false
			continue
		}
		accept[i] = coin <= alpha[i]
	}
	return accept, true
}

// V41SpecDecodeKeyedAcceptedPrefix is the end-to-end keyed acceptance for the default 1+N bonus
// layout: derive the accept verdicts from the request identity, then commit the accepted prefix.
// Pure in (key, anchor, draft, alpha, base), so a re-admitted request reproduces the identical
// accepted-token sequence. A zero key fails closed with ok=false.
func V41SpecDecodeKeyedAcceptedPrefix(key V41SpecDecodeRequestKey, anchor int, draft []int, alpha []float32, base []int) ([]int, bool) {
	accept, ok := V41SpecDecodeKeyedAccept(key, draft, alpha)
	if !ok {
		return nil, false
	}
	return v41SpecDecodeAcceptedUnderBonus1N(anchor, draft, accept, base), true
}

// ---------------------------------------------------------------------------
// Keyed capability surface (issue #13195)
//
// The keyed coin functions above are free functions; these thin methods expose the same
// capability on V41SpecDecodeConfig, so the red-then-green symptom witness reaches it through an
// interface type-assertion: the witness compiles against the parent commit (the methods are simply
// absent there) and fails at runtime, rather than failing to build against the old source.
// ---------------------------------------------------------------------------

// V41SpecDecodeKeyedCapability names the per-request keyed-coin capability the V4.1 spec-decode
// verify path exposes, so the symptom witness can assert its presence at runtime and a consumer
// can depend on the surface without naming every concrete function.
type V41SpecDecodeKeyedCapability interface {
	KeyFromRequest(requestID string) V41SpecDecodeRequestKey
	AcceptCoin(step int) (float32, bool)
	BonusCoin(step int) (float32, bool)
	KeyedAccept(draft []int, alpha []float32) ([]bool, bool)
	KeyedAcceptedPrefix(anchor int, draft []int, alpha []float32, base []int) ([]int, bool)
}

// KeyFromRequest folds a request id to the keyed-coin identity this config's spec-decode path
// draws from.
func (c V41SpecDecodeConfig) KeyFromRequest(requestID string) V41SpecDecodeRequestKey {
	return V41SpecDecodeKeyFromRequest(requestID)
}

// KeyFold is KeyFromRequest returning the identity as a plain uint64, for consumers (or the
// symptom witness) that must not name the keyed type. A zero value names no identity.
func (c V41SpecDecodeConfig) KeyFold(requestID string) uint64 {
	return uint64(V41SpecDecodeKeyFromRequest(requestID))
}

// AcceptCoin is the keyed accept coin for draft index step under the config's request identity;
// it fails closed (ok=false) when no key is armed.
func (c V41SpecDecodeConfig) AcceptCoin(step int) (float32, bool) {
	return V41SpecDecodeAcceptCoin(c.RequestKey, step)
}

// BonusCoin is the keyed bonus draw for a fully-accepted step under the config's request
// identity; it fails closed (ok=false) when no key is armed.
func (c V41SpecDecodeConfig) BonusCoin(step int) (float32, bool) {
	return V41SpecDecodeBonusCoin(c.RequestKey, step)
}

// KeyedAccept derives the accept verdicts for a draft block from the config's request identity;
// a zero key fails closed.
func (c V41SpecDecodeConfig) KeyedAccept(draft []int, alpha []float32) ([]bool, bool) {
	return V41SpecDecodeKeyedAccept(c.RequestKey, draft, alpha)
}

// KeyedAcceptedPrefix commits the accepted prefix under the bonus 1+N layout from the config's
// request identity; a zero key fails closed.
func (c V41SpecDecodeConfig) KeyedAcceptedPrefix(anchor int, draft []int, alpha []float32, base []int) ([]int, bool) {
	return V41SpecDecodeKeyedAcceptedPrefix(c.RequestKey, anchor, draft, alpha, base)
}

// WithRequestKey returns a copy of the config with the request-id keyed coin identity armed; an
// empty request id clears the key, so the keyed seam fails closed.
func (c V41SpecDecodeConfig) WithRequestKey(requestID string) V41SpecDecodeConfig {
	c.RequestKey = V41SpecDecodeKeyFromRequest(requestID)
	return c
}
