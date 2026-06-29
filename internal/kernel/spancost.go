package kernel

import "github.com/anthony-chaudhary/fak/internal/abi"

// spancost.go is L0 of the self-tax plane (#1147 / #1149): the kernel stamps
// per-span COST — elapsed-ns and a signed token-delta — onto the OPEN
// abi.Event.Fields telemetry channel of the lifecycle events it already emits, so a
// passive observer (internal/rungobs) can fold cost, not just verdict, and the
// offline read-out reconciles with the live
// fak_gateway_operation_duration_seconds{adjudicator-rung} twin.
//
// The carrier is Event.Fields ("OPEN; unknown ignored — non-label telemetry only"),
// never a new field on the FROZEN abi message structs: the spine is additive-only and
// human-owned, and Fields is exactly the channel internal/journal and
// internal/trajectory already read off these events.
//
// Which SPAN an elapsed-ns belongs to is disambiguated by the Event.Kind it rides:
//   - EvDecide / EvDeny carry the EvSubmit->EvDecide adjudication tax;
//   - EvComplete carries the EvDispatch->EvComplete engine cost.
//
// FieldTokenDelta is SIGNED: tokens ADDED by a transform/quarantine re-emit are
// positive (a mediation cost); tokens SAVED by a vDSO/radix local hit are negative (a
// round-trip the kernel did not have to pay).
const (
	FieldElapsedNanos = "elapsed_ns"  // int64: span wall-clock, nanoseconds
	FieldTokenDelta   = "token_delta" // int64: + added (transform/quarantine) / - saved (vdso/radix)
)

// refLen is a Ref's payload length, falling back to the inline byte count when the
// backend left Len unset (a copy-CAS Put always sets it; a hand-built inline Ref in a
// test sometimes does not). It is the token proxy the span cost is denominated in.
func refLen(r abi.Ref) int64 {
	if r.Len > 0 {
		return r.Len
	}
	return int64(len(r.Inline))
}

// transformTokenDelta is the signed token-delta a TRANSFORM verdict introduces: the
// re-emitted (redacted/rewritten) args measured against the originals. It is zero for
// any non-transform verdict or a malformed payload, so an allow/deny carries no token
// cost.
func transformTokenDelta(v abi.Verdict, orig abi.Ref) int64 {
	if v.Kind != abi.VerdictTransform {
		return 0
	}
	tp, ok := v.Payload.(abi.TransformPayload)
	if !ok {
		return 0
	}
	return refLen(tp.NewArgs) - refLen(orig)
}

// costFields builds the OPEN telemetry map the kernel stamps on a lifecycle event.
// token_delta is omitted when zero so a pure allow/deny event carries only its
// elapsed-ns, keeping the map minimal on the common path.
func costFields(elapsedNs, tokenDelta int64) map[string]any {
	f := map[string]any{FieldElapsedNanos: elapsedNs}
	if tokenDelta != 0 {
		f[FieldTokenDelta] = tokenDelta
	}
	return f
}
