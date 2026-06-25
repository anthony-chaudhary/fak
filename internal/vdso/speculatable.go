package vdso

import "github.com/anthony-chaudhary/fak/internal/abi"

// speculatable.go is the public, executable form of the one law that gates
// speculative agent-loop execution (epic #809): DEFAULT-DENY ON EFFECTS. A
// speculative tool call — one the kernel runs ahead of the model's authoritative
// emission, during a tool's idle latency window — may COMPUTE and WARM anything
// but may COMMIT nothing externally visible. So a tool is admissible for
// speculation ONLY if it is provably effect-free: read-only, idempotent, and not
// write-shaped. Anything that can mutate external state is NEVER speculated; it is
// buffered as a declared intent and committed only when the model asks for it.
//
// This is the CPU store-buffer discipline at the agent layer (Spectre's lesson:
// "architecturally rolled back" is not "never happened" — a speculative store that
// escapes corrupts state a squash cannot undo). Unlike a CPU, our "stores" hit
// other people's systems (payments, emails, file deletes) that have no rollback,
// so the gate fails CLOSED: an unstamped or ambiguous call is non-speculatable.
//
// The predicate is intentionally the SAME one tier-1/tier-2 Lookup uses to admit a
// pure result to the cache (readOnlyHint && idempotentHint && !destructive). Sharing
// it is the point: the bytes a call is allowed to SERVE from cache and the bytes it
// is allowed to SPECULATE are governed by one decision, so the two can never drift
// into the gap where one says safe and the other says effectful. Pure and
// allocation-free; safe to call on the hot path before any speculative dispatch.

// SpecReason is the closed verdict for why a tool call may or may not be speculated.
// It mirrors the structured-refusal posture of the rest of the kernel: a "no" is a
// typed value a caller can route on, not a bare bool.
type SpecReason uint8

const (
	// SpecOK — the call is provably effect-free (read-only + idempotent + not
	// write-shaped) and may be executed speculatively. Its result is a cost/latency
	// win only; correctness never depends on the speculation landing.
	SpecOK SpecReason = iota
	// SpecRefusedNilCall — a nil call carries no hints to clear the default-deny
	// floor. Fails closed.
	SpecRefusedNilCall
	// SpecRefusedNotReadOnly — the call did not assert readOnlyHint=true. The floor
	// is default-deny, so the ABSENCE of the hint refuses, exactly as the vDSO cache
	// gate refuses to serve it.
	SpecRefusedNotReadOnly
	// SpecRefusedNotIdempotent — the call did not assert idempotentHint=true. A
	// non-idempotent read can have an observable second-call effect (a cursor, a
	// rate-limit tick), so it is not speculatable even though it does not "write".
	SpecRefusedNotIdempotent
	// SpecRefusedDestructive — the call is write-shaped or explicitly destructive.
	// This is the store that must never escape the buffer; NEVER speculated.
	SpecRefusedDestructive
)

// Refused reports whether the reason denies speculation (anything but SpecOK).
func (r SpecReason) Refused() bool { return r != SpecOK }

// String renders the reason as a stable token for logs and structured refusals.
func (r SpecReason) String() string {
	switch r {
	case SpecOK:
		return "SPEC_OK"
	case SpecRefusedNilCall:
		return "SPEC_REFUSED_NIL_CALL"
	case SpecRefusedNotReadOnly:
		return "SPEC_REFUSED_NOT_READ_ONLY"
	case SpecRefusedNotIdempotent:
		return "SPEC_REFUSED_NOT_IDEMPOTENT"
	case SpecRefusedDestructive:
		return "SPEC_REFUSED_DESTRUCTIVE"
	default:
		return "SPEC_REFUSED_UNCLASSIFIED"
	}
}

// Speculatable reports whether a tool call may be executed speculatively under the
// default-deny-on-effects law, returning the closed reason that decided it. It is
// the gate a speculative dispatcher (epic #809 children #812/#813) MUST clear before
// running any tool ahead of the model. The decision is the same one the vDSO cache
// uses to admit a pure result — read-only AND idempotent AND not destructive — so a
// call that cannot be cached can never be speculated, and vice versa.
//
// The checks run in escalating-trust order so the returned reason names the FIRST
// floor the call failed to clear: nil, then the two positive hints, then the
// write-shape/destructive refusal. A call that clears all three returns SpecOK.
func Speculatable(c *abi.ToolCall) (SpecReason, bool) {
	if c == nil {
		return SpecRefusedNilCall, false
	}
	if !metaTrue(c, "readOnlyHint") {
		return SpecRefusedNotReadOnly, false
	}
	if !metaTrue(c, "idempotentHint") {
		return SpecRefusedNotIdempotent, false
	}
	if destructive(c) {
		return SpecRefusedDestructive, false
	}
	return SpecOK, true
}

// CanSpeculate is the bare-bool form of Speculatable for call sites that only branch
// on admit/deny and do not surface the reason. It is exactly Speculatable's second
// return, kept as a named helper so the default-deny intent reads at the call site.
func CanSpeculate(c *abi.ToolCall) bool {
	_, ok := Speculatable(c)
	return ok
}
