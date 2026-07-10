package screen

// screen.go — the taint-safe-outbound + per-principal scope FLOOR of the session read
// plane (epic #4176, worst-first child C1 #4192). A read plane is an exfiltration
// surface: a read that emits a quarantined or sealed span is the outbound dual of a
// prompt injection, and a read that crosses the per-principal boundary without a right
// is an unauthorized cross-tenant peek. This package is the reusable kernel that closes
// both, named by the S vocabulary (internal/sessionread) so every read seam adopts one
// screen and one scope check instead of re-inventing them.
//
// Two pure adjudications, no I/O, no gateway dependency (so the floor is testable in
// isolation and immune to the churn of the seams that will call it):
//
//  1. ScreenOutbound — the outbound taint screen. It is the exact mirror of the inbound
//     steer screen (ctxmmu.ScreenBytes in internal/gateway/steer_class.go): a span whose
//     suppression gate is set — Sealed (trust quarantine) or Tombstoned (context control),
//     the same two flags gateway.restoreEntry / ctxplan.Span carry — NEVER has its bytes
//     cross the boundary. It refuses with READ_TAINT_WITHHELD and returns nil. A clean
//     span's bytes are returned byte-exact. This is the disclosure-path guarantee #4192
//     requires: the handle recovers a dropped span, never a suppressed one.
//
//  2. Authorize — the per-principal scope check. The gap #4192 names: gateway reads only
//     engage auth when RequireKey is configured, so on the common no-key loopback `fak
//     guard` every read seam — including fak_context_restore, which returns the verbatim
//     compaction-dropped first user turn — is reachable unauthenticated by any loopback
//     process, with no per-principal ACL. Authorize is the floor: a read-self op may read
//     ONLY the caller's own trace; a read-fleet op needs the (not-yet-granted) fleet
//     right. A cross-principal read refuses READ_SCOPE_DENIED — a closed reason, not an
//     undocumented serve.
//
// Both refusals are drawn from the closed read-refusal vocabulary the S spine defines
// (sessionread.ReadRefusalTokens); this package invents no token of its own. The kernel
// is landed and red-team-tested here; wiring it into the live gateway restore/read call
// sites is the follow-on adoption (it must wait on the gateway package, which this floor
// deliberately does not import).

import (
	"errors"

	"github.com/anthony-chaudhary/fak/internal/sessionread"
)

// Span is the minimal outbound-screen view of a stored or dropped span: its verbatim
// bytes plus the two suppression gates. The fields mirror gateway.restoreEntry exactly —
// Sealed is the operator trust-quarantine gate (ctxplan.ErrSealed), Tombstoned is the
// context-control gate (ctxplan.ErrTombstoned) — so a gateway caller can screen a
// restoreEntry by copying its bytes and two bools into this shape with no translation.
type Span struct {
	// Bytes is the verbatim payload the read would disclose (the dropped originating turn).
	Bytes []byte
	// Sealed marks a trust-quarantined span: its bytes must never cross the boundary.
	Sealed bool
	// Tombstoned marks a context-controlled span: its bytes must never cross the boundary.
	Tombstoned bool
}

// Refusal is a closed read-refusal carried as an error. Reason is one of the
// sessionread read-refusal tokens (READ_TAINT_WITHHELD / READ_SCOPE_DENIED); Detail is a
// non-leaking human/audit aid that never contains span bytes. It implements error so a
// call site can propagate it, and RefusalReason recovers the closed token for the wire.
type Refusal struct {
	// Reason is the closed read-refusal token — always a member of ReadRefusalTokens().
	Reason string
	// Detail is a bounded, byte-free explanation. It must never echo screened content.
	Detail string
}

func (r *Refusal) Error() string {
	if r == nil {
		return "<nil screen.Refusal>"
	}
	if r.Detail == "" {
		return r.Reason
	}
	return r.Reason + ": " + r.Detail
}

// RefusalReason returns the closed read-refusal token carried by err, or "" if err is not
// a screen refusal. A caller maps the token onto its wire refusal envelope.
func RefusalReason(err error) string {
	var r *Refusal
	if errors.As(err, &r) {
		return r.Reason
	}
	return ""
}

// ScreenOutbound is the outbound taint screen: it returns the span's verbatim bytes ONLY
// when no suppression gate is set. A Sealed or Tombstoned span refuses with
// READ_TAINT_WITHHELD and returns nil — its bytes never cross the boundary. The returned
// slice for a clean span is the SAME bytes, unmodified (byte-exact disclosure); the
// screen adds nothing and removes nothing from a span it clears. This is the outbound
// mirror of the inbound steer screen: identical taint stamps, opposite direction.
func ScreenOutbound(sp Span) ([]byte, error) {
	if sp.Sealed {
		return nil, &Refusal{Reason: sessionread.ReasonReadTaintWithheld, Detail: "sealed span (trust quarantine) withheld from the outbound read"}
	}
	if sp.Tombstoned {
		return nil, &Refusal{Reason: sessionread.ReasonReadTaintWithheld, Detail: "tombstoned span (context control) withheld from the outbound read"}
	}
	return sp.Bytes, nil
}

// ScopeRequest is a per-principal scope query for one read op. Caller is the principal
// submitting the read (its own trace/session id); TargetOwner is the principal that owns
// the addressed state (after the gateway's default-trace resolution — a bare read
// resolves TargetOwner to Caller). FleetGrant records whether the caller holds the
// read-fleet right (today granted to no principal, which is exactly the worst-first
// exposure this floor makes legible).
type ScopeRequest struct {
	// Op is the read op being submitted; its registered Capability decides the rule.
	Op sessionread.ReadOp
	// Caller is the principal submitting the read.
	Caller string
	// TargetOwner is the principal owning the addressed state.
	TargetOwner string
	// FleetGrant is whether the caller holds the read-fleet right.
	FleetGrant bool
}

// Authorize applies the per-principal scope floor named by the op's capability. It fails
// closed: an unknown op, an empty caller, or an empty target owner refuses.
//
//   - read-self op: the caller may read ONLY its own trace — Caller must equal
//     TargetOwner. A cross-principal read (Caller != TargetOwner) refuses
//     READ_SCOPE_DENIED. This is the guarantee that, on a no-RequireKey loopback guard,
//     one principal cannot read another's dropped originating task.
//   - read-fleet op: the read crosses the per-principal boundary and requires the fleet
//     grant. Without it the read refuses READ_SCOPE_DENIED — the floor treats an
//     unauthorized fleet read as denied rather than silently serving it.
//
// Authorize is the scope half of the floor; a full read still passes any disclosed span
// bytes through ScreenOutbound — the two are defense-in-depth, not alternatives.
func Authorize(req ScopeRequest) error {
	spec, ok := sessionread.Spec(req.Op)
	if !ok {
		return &Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "unknown read op refused by the scope floor"}
	}
	if req.Caller == "" {
		return &Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "empty caller principal — the scope floor fails closed"}
	}
	switch spec.Capability {
	case sessionread.CapReadSelf:
		if req.TargetOwner == "" || req.Caller != req.TargetOwner {
			return &Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "read-self op may read only the caller's own trace"}
		}
		return nil
	case sessionread.CapReadFleet:
		if !req.FleetGrant {
			return &Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "read-fleet op requires the fleet right, which the caller does not hold"}
		}
		return nil
	default:
		// A new capability must extend this switch; until then it is denied, not defaulted
		// open — the floor never serves a right it cannot reason about.
		return &Refusal{Reason: sessionread.ReasonReadScopeDenied, Detail: "un-scoped capability refused by the floor"}
	}
}
