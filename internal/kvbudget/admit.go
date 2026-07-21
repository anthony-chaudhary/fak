package kvbudget

// This file adds a pure, GPU-free admission fold on top of the static KV-cache
// sizing above: a guaranteed-no-drop admission verdict. It reserves each
// already-admitted stream's worst-case blocks-to-completion up front, so a
// stream, once admitted, can always be driven to completion without being
// dropped or paused mid-decode to make room for a later arrival.
//
// The axis mirrors TensorRT-LLM's never-drop-an-admitted-request capacity scheme (INSPIRE,
// clean-room — issue #5258 / epic #5256): admit a new decode stream only if the
// blocks needed to complete EVERY running stream PLUS this arrival's own
// worst-case all fit the free block pool. The budget is on the MAXIMUM length a
// request can reach (prompt + max new tokens), not its current length, so a
// later, larger arrival is refused while the free pool is already reserved by
// earlier admits — that refusal is the guarantee holding.
//
// Everything here is a deterministic integer fold: no hardware, no wall clock,
// no network. Block counts are KV blocks (tokens ÷ block size, rounded up).

// Stream is one decode request measured for admission: its token span and the
// blocks it already holds. Worst-case blocks-to-completion are derived from the
// MAXIMUM length it can reach (PromptTokens + MaxNewTokens), not its current
// length, which is what makes the admission a no-drop guarantee.
type Stream struct {
	// PromptTokens is the tokens already present in the request's prompt.
	PromptTokens int
	// MaxNewTokens is the worst-case additional tokens the request may decode
	// before it completes (its max output length). The reservation is sized
	// against prompt+MaxNewTokens so completion never needs an unbudgeted block.
	MaxNewTokens int
	// BlockSize is the number of tokens one KV block holds (> 0). A block is the
	// unit the pool is reserved in.
	BlockSize int
	// HeldBlocks is the blocks already allocated to this stream (subtracted from
	// the worst-case reservation — those blocks are already accounted for).
	HeldBlocks int
	// RetainedBlocks is ref-held reusable blocks this stream can reuse without a
	// fresh allocation (e.g. a shared/prefix cache). They are subtracted from the
	// reservation but, deliberately, are NOT added back to the free pool here —
	// mirroring the source's choice not to double-count already-resident blocks.
	RetainedBlocks int
}

// ceilDiv is integer ceiling division for non-negative a and positive b.
func ceilDiv(a, b int) int {
	if b <= 0 || a <= 0 {
		return 0
	}
	return (a + b - 1) / b
}

// valid reports whether the stream's fields are self-consistent enough to size
// a reservation. A non-positive block size or any negative count is rejected so
// admission fails closed rather than reserving a nonsense amount.
func (s Stream) valid() bool {
	return s.BlockSize > 0 &&
		s.PromptTokens >= 0 &&
		s.MaxNewTokens >= 0 &&
		s.HeldBlocks >= 0 &&
		s.RetainedBlocks >= 0
}

// WorstCaseBlocks is the total KV blocks the stream needs to run to completion:
// ceilDiv(PromptTokens+MaxNewTokens, BlockSize). This is the whole footprint at
// the maximum length, independent of what it already holds.
func (s Stream) WorstCaseBlocks() int {
	return ceilDiv(s.PromptTokens+s.MaxNewTokens, s.BlockSize)
}

// ReserveBlocks is the ADDITIONAL free blocks that must be reserved to guarantee
// this stream reaches completion: worst-case minus what it already holds minus
// the ref-held reusable blocks. Clamped at zero (a stream that already holds its
// whole worst case reserves nothing further).
func (s Stream) ReserveBlocks() int {
	n := s.WorstCaseBlocks() - s.HeldBlocks - s.RetainedBlocks
	if n < 0 {
		return 0
	}
	return n
}

// Reason is a closed, typed refusal vocabulary for an admission verdict. The
// empty Reason means admitted; a non-empty Reason means the request was refused
// and the budget was left unchanged.
type Reason string

const (
	// ReasonAdmitted is the empty reason carried by an admitted verdict.
	ReasonAdmitted Reason = ""
	// ReasonNoRoomToRetain refuses because the arrival's worst-case reservation
	// would not fit the blocks still free after the running streams are reserved
	// — admitting it could force a running stream to be dropped mid-decode, which
	// the no-drop guarantee forbids.
	ReasonNoRoomToRetain Reason = "no_room_to_retain_worst_case"
	// ReasonInvalidRequest refuses a self-inconsistent request (non-positive
	// block size or a negative count) without touching the budget.
	ReasonInvalidRequest Reason = "invalid_request"
)

// Verdict is the outcome of an admission attempt. On admission Admitted is true,
// ReservedBlocks records how much the budget was decremented (feed it to Release
// to return exactly that reservation), and Reason is empty. On refusal Admitted
// is false, ReservedBlocks is zero, the budget is unchanged, and Reason carries
// the typed cause.
type Verdict struct {
	Admitted        bool
	Reason          Reason
	ReservedBlocks  int // blocks reserved for this stream (0 if refused)
	WorstCaseBlocks int // the stream's total worst-case blocks
	AvailableBlocks int // free-minus-reserved blocks at the moment of the attempt
}

// Reservation is the running admission budget: a fixed free block pool and the
// blocks already reserved to complete admitted streams. It is a plain value —
// copy it to fork a what-if, keep one to track a live serve.
type Reservation struct {
	// FreeBlocks is the total KV blocks the pool can ever hand out.
	FreeBlocks int
	// ReservedBlocks is the blocks committed to completing already-admitted
	// streams. Available room is FreeBlocks - ReservedBlocks.
	ReservedBlocks int
}

// Available is the free blocks not yet reserved to complete an admitted stream —
// the room a new arrival's worst case must fit within.
func (r Reservation) Available() int { return r.FreeBlocks - r.ReservedBlocks }

// Admit decides whether the stream can be admitted under the no-drop guarantee
// and, on success, decrements the budget by the stream's worst-case reservation.
// It admits iff the arrival's ReserveBlocks fits the currently-Available room
// (exact fit admits). On refusal the budget is left untouched and the Verdict
// carries a typed Reason. Deterministic; no hardware, no clock.
func (r *Reservation) Admit(s Stream) Verdict {
	avail := r.Available()
	if !s.valid() {
		return Verdict{Reason: ReasonInvalidRequest, AvailableBlocks: avail}
	}
	need := s.ReserveBlocks()
	v := Verdict{
		WorstCaseBlocks: s.WorstCaseBlocks(),
		AvailableBlocks: avail,
	}
	if need > avail {
		v.Reason = ReasonNoRoomToRetain
		return v
	}
	r.ReservedBlocks += need
	v.Admitted = true
	v.ReservedBlocks = need
	return v
}

// Release returns an admitted verdict's reservation to the free pool (call it
// when the stream completes). A refused verdict releases nothing. The reserved
// total is clamped at zero so a double release cannot drive it negative.
func (r *Reservation) Release(v Verdict) {
	if !v.Admitted {
		return
	}
	r.ReservedBlocks -= v.ReservedBlocks
	if r.ReservedBlocks < 0 {
		r.ReservedBlocks = 0
	}
}

// AdmitAll folds a sequence of arrivals against a fresh free-block pool, in
// order, returning one Verdict per stream. Earlier admits reserve their worst
// case, so a later arrival is refused once the remaining free room can no longer
// retain its worst case — the guarantee holding across a batch. The final
// Reservation is returned so a caller can read the committed total.
func AdmitAll(freeBlocks int, streams []Stream) ([]Verdict, Reservation) {
	r := Reservation{FreeBlocks: freeBlocks}
	out := make([]Verdict, len(streams))
	for i, s := range streams {
		out[i] = r.Admit(s)
	}
	return out, r
}
