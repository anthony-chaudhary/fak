// crashdrive.go — the crash-journal ↔ identity-map join. A sessionjournal CRASHED
// record (internal/sessionjournal, #3785) is keyed by whichever id its source used — a
// transcript UUID (the watchdog / `claude --resume` keyspace) or a gateway/guard trace
// (the operator control-plane keyspace) — and the two keyspaces do not join on their own
// (drivestate.go documents the wall). This resolver joins each CRASHED record across that
// wall through the durable cluster-A identity map (identitymap.go, #4112): it resolves the
// counterpart id, attaches the DriveCarry the record carries (#4129), and attaches the
// operator hold recorded for the session (drivestate.go). It RETURNS the resolved set; it
// does not launch anything — the relaunch is C4 (#3788).
//
// Pure by construction, mirroring FoldIdentity / FoldDriveStates: same inputs → same
// output, no clock, no I/O, and it never imports internal/session — it consumes only the
// carried DriveCarry and the injected join/hold maps. An unmappable id degrades to empty
// counterparts, never a panic; a nil map answers "no join" (fail-open), so an absent
// identity store leaves a crashed row carried through rather than stranded.
package resume

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/sessionjournal"
)

// IdentityJoin is the read-side of the cluster-A transcript-uuid ↔ gateway-trace identity
// map (identitymap.go, #4112) this resolver joins through: the two folded lookup directions
// FoldIdentity / LoadIdentity return. It is injected as plain data (not internal/session,
// not a live store) so ResolveCrashedDrive stays pure and total — the same discipline
// FoldIdentity and FoldDriveStates hold. Either map may be nil; a nil map answers "no join"
// for every query (fail-open) rather than stranding a crashed row.
type IdentityJoin struct {
	TraceByUUID map[string]string // transcript UUID -> gateway trace
	UUIDByTrace map[string]string // gateway trace   -> transcript UUID
}

// ResolvedCrash is one CRASHED session resolved across the keyspace wall: the id the record
// carried, both counterpart ids the identity map paired (empty when unmappable), the carried
// drive-state (nil when the record had none — e.g. a legacy guard-index row), and the
// operator hold recorded for the session (empty = not held).
type ResolvedCrash struct {
	SessionID      string                     // the id the CRASHED record was keyed on
	GatewayTrace   string                     // the gateway / guard trace (from the join, or the id itself)
	TranscriptUUID string                     // the Claude transcript UUID (from the join, or the id itself)
	Drive          *sessionjournal.DriveCarry // the carried remaining drive-state (nil = none)
	Hold           WatchdogDriveState         // the operator hold for this session (empty = not held)
}

// ResolveCrashedDrive joins each CRASHED record to the identity map and its drive/hold.
// Pure and total: same inputs → same output, no clock, no I/O; an unmappable id yields a
// ResolvedCrash with empty counterpart ids (never a panic), and a nil join / nil holds
// simply resolve nothing. The record's own id is placed in the right keyspace slot by which
// direction of the map knows it: a known UUID fills GatewayTrace from TraceByUUID; a known
// trace fills TranscriptUUID from UUIDByTrace; an id known to neither is carried through as
// SessionID alone with empty counterparts.
func ResolveCrashedDrive(crashed []sessionjournal.Classified, join IdentityJoin, holds map[string]WatchdogDriveState) []ResolvedCrash {
	out := make([]ResolvedCrash, 0, len(crashed))
	for _, c := range crashed {
		id := strings.TrimSpace(c.ID)
		rc := ResolvedCrash{SessionID: id, Drive: c.Drive}
		switch {
		case id == "":
			// No id to join — carry the record through untouched.
		case identityKnown(join.TraceByUUID, id):
			rc.TranscriptUUID = id
			rc.GatewayTrace = join.TraceByUUID[id]
		case identityKnown(join.UUIDByTrace, id):
			rc.GatewayTrace = id
			rc.TranscriptUUID = join.UUIDByTrace[id]
		}
		// Attach the operator hold under whichever key the store recorded it. The hold
		// store keys on the UUID keyspace (drivestate.go re-records across the wall), so
		// the resolved UUID is the primary key; the raw id and the trace are tried too so a
		// hold recorded under either still binds.
		rc.Hold = resolveHold(holds, rc.TranscriptUUID, id, rc.GatewayTrace)
		out = append(out, rc)
	}
	return out
}

// identityKnown reports whether m pairs key to a non-empty counterpart. A nil map (an absent
// identity store) answers false for every key — the fail-open the resolver depends on.
func identityKnown(m map[string]string, key string) bool {
	v, ok := m[key]
	return ok && strings.TrimSpace(v) != ""
}

// resolveHold returns the first non-empty operator hold recorded under any of keys, in
// order; a session with no recorded state (or a nil holds map) reads "" — never held (the
// per-key fail-open the watchdog depends on).
func resolveHold(holds map[string]WatchdogDriveState, keys ...string) WatchdogDriveState {
	for _, k := range keys {
		if k = strings.TrimSpace(k); k == "" {
			continue
		}
		if st, ok := holds[k]; ok && st != "" {
			return st
		}
	}
	return ""
}
