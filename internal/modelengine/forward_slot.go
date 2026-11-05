package modelengine

// forward_slot.go — Continual Engine phase 0 (ForwardSlot): the atomic
// forward-pass indirection for the in-kernel engine's decode entry.
//
// One atomic pointer load per decode step: a scheduler step loads the live
// ForwardGen once, before its forward call, and runs the whole step on that
// generation. Perf-neutral by contract — no locks, no allocation on the hot
// path; Install/Revert take a mutex that readers never touch.
//
// Whole-old/whole-new swap: a caller that loaded a generation finishes its
// step on that generation (in-flight continuity); new steps see the new
// generation. The swap is NOT a close: weight-lifetime guards
// (model.ModelWeightsClosedError) are untouched and out of scope here.
//
// This generalizes internal/modelroute.Live (whole-manifest atomic pointer
// swap + operator-confirmable reload/reject counters + last-good) and
// internal/model.SpeculativeControl (lock-free parameter hot-swap with
// Snapshot-style consistent reads) to the forward entry. All results this
// package witnesses are [SW-VERIFIED] class only; no hardware or performance
// claims are made.

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"time"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// Sentinel errors for ForwardSlot operations. Callers assert with errors.Is,
// never error prose.
var (
	// ErrForwardSlotBaseMismatch reports an Install whose delta Base does not
	// equal the live generation. The delta is refused, never merged.
	ErrForwardSlotBaseMismatch = errors.New("modelengine: forward slot base mismatch")
	// ErrForwardSlotNilImpl reports a nil slot or an impl with a nil Step or
	// StepBatch func.
	ErrForwardSlotNilImpl = errors.New("modelengine: forward slot impl is nil or incomplete")
	// ErrForwardSlotVerify wraps the error returned by an Install Verify hook
	// that refused the candidate.
	ErrForwardSlotVerify = errors.New("modelengine: forward slot verify gate refused the swap")
	// ErrForwardSlotNoLastGood reports a Revert with nothing to revert to.
	ErrForwardSlotNoLastGood = errors.New("modelengine: forward slot has no last-known-good generation")
)

// Swap outcome markers carried in ForwardSwapReceipt.Outcome.
const (
	ForwardSwapPromoted = "promoted"
	ForwardSwapReverted = "reverted"
	ForwardSwapRefused  = "refused"
)

// ForwardImpl is one immutable forward-pass implementation generation.
// Both funcs must be non-nil at install. Implementations must not mutate
// shared state observable by other generations after installation.
type ForwardImpl struct {
	Name      string
	Step      func(sess *model.Session, id int) []float32         // single-lane decode forward
	StepBatch func(bs *model.BatchSession, ids []int) [][]float32 // shared batched decode forward
}

// ForwardGen is one installed generation record {generation, impl, graphHandle}.
type ForwardGen struct {
	Generation  uint64
	Impl        ForwardImpl
	GraphHandle any // opaque graph-capture handle; nil in phase 0 (graph recapture is a later phase)
}

// ForwardDelta is a proposed hot swap. Base is the CAS identity: it must equal
// the live generation or the delta is refused, never merged (fail-closed).
type ForwardDelta struct {
	Base        uint64
	Impl        ForwardImpl
	GraphHandle any
	Verify      func(candidate, incumbent ForwardImpl) error // optional gate run BEFORE the store; error refuses the swap
	Reason      string
}

// ForwardSwapReceipt witnesses one apply or refusal — a swap without a receipt
// did not happen. Refusals return a receipt alongside their error.
type ForwardSwapReceipt struct {
	From    uint64 // generation before the apply
	To      uint64 // generation after (equal to From on refusal)
	Impl    string // Impl.Name of the generation promoted ("" on refusal)
	Outcome string // "promoted" | "reverted" | "refused"
	Reason  string // delta Reason, or refusal cause
	At      time.Time
}

// ForwardSlotStats are operator-confirmable cumulative counts (the
// Live.Reloads/Rejects pattern).
type ForwardSlotStats struct {
	Installs uint64
	Reverts  uint64
	Rejects  uint64
}

// ForwardSlot is the atomic forward-pass indirection: cur is the live
// generation, prev is the one-deep last-known-good revert handle. Reads are
// lock-free; mu serializes Install/Revert only.
type ForwardSlot struct {
	cur                        atomic.Pointer[ForwardGen] // live generation
	prev                       atomic.Pointer[ForwardGen] // one-deep last-known-good (revert handle)
	mu                         sync.Mutex                 // serializes Install/Revert only; reads are lock-free
	installs, reverts, rejects atomic.Uint64
	lastReceipt                atomic.Pointer[ForwardSwapReceipt]
}

// DefaultForwardImpl is the exact pre-slot behavior: the engine's own
// Session.Step / BatchSession.StepBatch decode forwards.
func DefaultForwardImpl() ForwardImpl {
	return ForwardImpl{
		Name: "incumbent",
		Step: func(sess *model.Session, id int) []float32 {
			return sess.Step(id)
		},
		StepBatch: func(bs *model.BatchSession, ids []int) [][]float32 {
			return bs.StepBatch(ids)
		},
	}
}

// NewForwardSlot installs initial as generation 1. If initial has nil funcs,
// the defaults are substituted for those funcs (the constructor is
// code-controlled; it stays total).
func NewForwardSlot(initial ForwardImpl) *ForwardSlot {
	def := DefaultForwardImpl()
	if initial.Step == nil {
		initial.Step = def.Step
	}
	if initial.StepBatch == nil {
		initial.StepBatch = def.StepBatch
	}
	s := &ForwardSlot{}
	s.cur.Store(&ForwardGen{Generation: 1, Impl: initial})
	r := ForwardSwapReceipt{
		From:    0,
		To:      1,
		Impl:    initial.Name,
		Outcome: ForwardSwapPromoted,
		Reason:  "initial install",
		At:      time.Now(),
	}
	s.lastReceipt.Store(&r)
	return s
}

// Load is the hot path: exactly one atomic pointer load, returning the live
// record by value. Callers hold the returned Impl for the duration of one
// step (in-flight continuity across a concurrent swap). A nil slot returns
// the zero ForwardGen; the engine always initializes its slot, so this cannot
// happen on the user path.
func (s *ForwardSlot) Load() ForwardGen {
	if s == nil {
		return ForwardGen{}
	}
	g := s.cur.Load()
	if g == nil {
		return ForwardGen{}
	}
	return *g
}

// Generation returns the live generation number.
func (s *ForwardSlot) Generation() uint64 {
	if s == nil {
		return 0
	}
	if g := s.cur.Load(); g != nil {
		return g.Generation
	}
	return 0
}

// Install proposes one hot swap, in order:
//  1. nil-impl check -> refuse (ErrForwardSlotNilImpl), rejects++;
//  2. CAS check d.Base == live generation else refuse
//     (ErrForwardSlotBaseMismatch), rejects++;
//  3. Verify hook (if set) run against the incumbent; error -> refuse
//     (wrapped ErrForwardSlotVerify), rejects++;
//  4. atomic whole swap: prev <- cur, cur <- new generation (strictly
//     monotonic across both Install and Revert, no ABA);
//  5. installs++, receipt Outcome "promoted".
//
// Refusal receipts are returned alongside the error: a refused swap is
// observable too.
func (s *ForwardSlot) Install(d ForwardDelta) (ForwardSwapReceipt, error) {
	if s == nil {
		return refusalReceipt(0, "nil-impl"), ErrForwardSlotNilImpl
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	live := s.cur.Load().Generation
	if d.Impl.Step == nil || d.Impl.StepBatch == nil {
		rec := s.recordRefusal(live, "nil-impl")
		return rec, ErrForwardSlotNilImpl
	}
	if d.Base != live {
		rec := s.recordRefusal(live, "base-mismatch")
		return rec, ErrForwardSlotBaseMismatch
	}
	if d.Verify != nil {
		if err := d.Verify(d.Impl, s.cur.Load().Impl); err != nil {
			rec := s.recordRefusal(live, fmt.Sprintf("verify: %v", err))
			return rec, fmt.Errorf("%w: %w", ErrForwardSlotVerify, err)
		}
	}
	next := &ForwardGen{Generation: live + 1, Impl: d.Impl, GraphHandle: d.GraphHandle}
	s.prev.Store(s.cur.Load())
	s.cur.Store(next)
	s.installs.Add(1)
	rec := ForwardSwapReceipt{
		From:    live,
		To:      live + 1,
		Impl:    d.Impl.Name,
		Outcome: ForwardSwapPromoted,
		Reason:  d.Reason,
		At:      time.Now(),
	}
	s.lastReceipt.Store(&rec)
	return rec, nil
}

// Revert promotes the one-deep prev impl as a NEW generation (gen+1, strictly
// monotonic) and clears prev. A second Revert with nothing to revert to is
// refused with ErrForwardSlotNoLastGood — never flip-flop back to a rejected
// generation.
func (s *ForwardSlot) Revert() (ForwardSwapReceipt, error) {
	if s == nil {
		return refusalReceipt(0, "nil-impl"), ErrForwardSlotNilImpl
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	cur := s.cur.Load()
	prev := s.prev.Load()
	live := cur.Generation
	if prev == nil {
		rec := s.recordRefusal(live, "no-last-good")
		return rec, ErrForwardSlotNoLastGood
	}
	next := &ForwardGen{Generation: live + 1, Impl: prev.Impl, GraphHandle: prev.GraphHandle}
	s.prev.Store(nil)
	s.cur.Store(next)
	s.reverts.Add(1)
	rec := ForwardSwapReceipt{
		From:    live,
		To:      live + 1,
		Impl:    prev.Impl.Name,
		Outcome: ForwardSwapReverted,
		Reason:  "revert-to-last-good",
		At:      time.Now(),
	}
	s.lastReceipt.Store(&rec)
	return rec, nil
}

// Stats reports the operator-confirmable cumulative counts.
func (s *ForwardSlot) Stats() ForwardSlotStats {
	if s == nil {
		return ForwardSlotStats{}
	}
	return ForwardSlotStats{
		Installs: s.installs.Load(),
		Reverts:  s.reverts.Load(),
		Rejects:  s.rejects.Load(),
	}
}

// LastReceipt returns the most recent apply-or-refusal receipt (observability
// for tests and operators). ok is false only for a nil slot.
func (s *ForwardSlot) LastReceipt() (ForwardSwapReceipt, bool) {
	if s == nil {
		return ForwardSwapReceipt{}, false
	}
	r := s.lastReceipt.Load()
	if r == nil {
		return ForwardSwapReceipt{}, false
	}
	return *r, true
}

// recordRefusal bumps the reject counter, publishes the refusal receipt, and
// returns it. Callers hold mu.
func (s *ForwardSlot) recordRefusal(live uint64, cause string) ForwardSwapReceipt {
	s.rejects.Add(1)
	rec := ForwardSwapReceipt{
		From:    live,
		To:      live,
		Impl:    "",
		Outcome: ForwardSwapRefused,
		Reason:  cause,
		At:      time.Now(),
	}
	s.lastReceipt.Store(&rec)
	return rec
}

// refusalReceipt builds a refusal receipt for a nil receiver, which has no
// counters or receipt store to update.
func refusalReceipt(live uint64, cause string) ForwardSwapReceipt {
	return ForwardSwapReceipt{
		From:    live,
		To:      live,
		Impl:    "",
		Outcome: ForwardSwapRefused,
		Reason:  cause,
		At:      time.Now(),
	}
}
