package kvbudget

// This file adds the anti-hog preallocation ceiling on top of the no-drop
// admission fold in admit.go (issue #5268, epic #2236; field-borrow from
// HuggingFace text-generation-inference v3). The no-drop admission in admit.go
// reserves each stream's FULL worst case (prompt + MaxNewTokens) up front, so a
// single request declaring a huge MaxNewTokens reserves proportionally huge
// budget and can shed or hog the whole free pool.
//
// The borrow decouples the RESERVED footprint from the CHARGED generation. TGI
// caps the preallocated generation chunk at DEFAULT_GENERATION_LENGTH = 1024
// tokens (router/src/validation.rs) while tracking the true requested max
// separately, re-admitting the next chunk as generation continues. So a
// "10-token prompt asking for 120k output" only reserves up to the cap up front,
// yet is still charged for exactly what it generates.
//
// Here the RESERVATION admission uses min(worst-case, cap) new tokens, while the
// CHARGE tracks the real generated tokens on a separate axis. The cap bounds how
// much one request may preallocate; it never bounds what it is charged. Same
// contract as admit.go: a deterministic integer fold, no hardware, no wall clock,
// no network.

// DefaultPreallocNewTokens is the per-request ceiling on the NEW generation
// tokens a single admission may reserve up front, mirroring TGI's
// DEFAULT_GENERATION_LENGTH. A request declaring more new tokens than this only
// reserves up to this many at admission; the remainder is topped up per chunk as
// it generates. 1024 is the borrowed default.
const DefaultPreallocNewTokens = 1024

// ReasonInvalidCap refuses an admission whose preallocation cap is non-positive.
// A zero or negative cap is a self-inconsistent request — it names no room to
// preallocate — so admission fails closed and the budget is left unchanged,
// exactly like ReasonInvalidRequest for a malformed Stream.
const ReasonInvalidCap Reason = "invalid_prealloc_cap"

// validCap reports whether a preallocation cap (in new tokens) is usable. A
// non-positive cap fails closed rather than reserving a nonsense amount.
func validCap(capNewTokens int) bool { return capNewTokens > 0 }

// cappedNewTokens is the effective new-token count this stream preallocates at
// admission: min(MaxNewTokens, capNewTokens). When the worst case is below the
// cap the cap is inactive and the full MaxNewTokens is preallocated; when the
// worst case is above the cap only the cap is preallocated (the hog is stopped).
// Assumes a positive cap — the admission path guards that first via validCap.
func (s Stream) cappedNewTokens(capNewTokens int) int {
	if capNewTokens < s.MaxNewTokens {
		return capNewTokens
	}
	return s.MaxNewTokens
}

// CapActive reports whether the preallocation cap bites for this stream: true
// when MaxNewTokens exceeds the cap (so the reservation is throttled below the
// worst case), false when the whole worst case already fits under the cap.
func (s Stream) CapActive(capNewTokens int) bool {
	return capNewTokens > 0 && s.MaxNewTokens > capNewTokens
}

// CappedWorstCaseBlocks is the KV blocks this stream would need to reach the
// CAPPED length prompt + min(MaxNewTokens, cap), rounded up per block. It is the
// preallocation footprint charged to the budget at admission — never the full
// prompt + MaxNewTokens footprint when the cap is active.
func (s Stream) CappedWorstCaseBlocks(capNewTokens int) int {
	return ceilDiv(s.PromptTokens+s.cappedNewTokens(capNewTokens), s.BlockSize)
}

// CappedReserveBlocks is the ADDITIONAL free blocks reserved at admission under
// the preallocation cap: the capped footprint minus already-held minus ref-held
// reusable blocks, clamped at zero. This is what admit.go's ReserveBlocks would
// return if MaxNewTokens were min(MaxNewTokens, cap). Assumes a positive cap.
func (s Stream) CappedReserveBlocks(capNewTokens int) int {
	n := s.CappedWorstCaseBlocks(capNewTokens) - s.HeldBlocks - s.RetainedBlocks
	if n < 0 {
		return 0
	}
	return n
}

// ChargedBlocks is the KV blocks a stream actually occupies for genNewTokens of
// real generation: ceilDiv(PromptTokens + genNewTokens, BlockSize), minus what
// it already holds and its ref-held reusable blocks, clamped at zero. This is
// the CHARGE axis — it tracks true generated tokens and is deliberately
// independent of the preallocation cap: a request charged for what it generates,
// not for what it reserved. genNewTokens below zero is treated as zero.
func (s Stream) ChargedBlocks(genNewTokens int) int {
	if genNewTokens < 0 {
		genNewTokens = 0
	}
	n := ceilDiv(s.PromptTokens+genNewTokens, s.BlockSize) - s.HeldBlocks - s.RetainedBlocks
	if n < 0 {
		return 0
	}
	return n
}

// AdmitCapped admits a stream against the budget under the anti-hog
// preallocation cap: it reserves min(worst-case, cap) blocks rather than the
// full worst case, so one long-max-tokens request cannot reserve or shed the
// whole pool. On success the budget is decremented by the CAPPED reservation and
// the Verdict's ReservedBlocks is that capped amount while WorstCaseBlocks still
// reports the full (uncapped) worst case, so a caller can see the cap bit. A
// non-positive cap fails closed with ReasonInvalidCap; a malformed stream fails
// closed with ReasonInvalidRequest; neither touches the budget. Deterministic;
// no hardware, no clock.
func (r *Reservation) AdmitCapped(s Stream, capNewTokens int) Verdict {
	avail := r.Available()
	if !s.valid() {
		return Verdict{Reason: ReasonInvalidRequest, AvailableBlocks: avail}
	}
	if !validCap(capNewTokens) {
		return Verdict{Reason: ReasonInvalidCap, AvailableBlocks: avail}
	}
	need := s.CappedReserveBlocks(capNewTokens)
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

// AdmitAllCapped folds a sequence of arrivals against a fresh free-block pool,
// in order, each admitted under the preallocation cap capNewTokens, returning
// one Verdict per stream and the final Reservation. Because each admit reserves
// only its capped footprint, a batch of long-max-tokens requests that could not
// co-admit under their full worst cases can co-admit here — the anti-hog
// property holding across a batch.
func AdmitAllCapped(freeBlocks, capNewTokens int, streams []Stream) ([]Verdict, Reservation) {
	r := Reservation{FreeBlocks: freeBlocks}
	out := make([]Verdict, len(streams))
	for i, s := range streams {
		out[i] = r.AdmitCapped(s, capNewTokens)
	}
	return out, r
}
