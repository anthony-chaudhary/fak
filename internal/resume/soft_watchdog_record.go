package resume

// soft_watchdog_record.go — issue #5287, a leaf of the supervision epic (#4748):
// the DURABLE half of the soft watchdog.
//
// soft_watchdog.go decides WHEN an alive-but-stalled session deserves a
// diagnostic dump, but a dump that only ever exists in memory dies with the tick
// that observed it — the evidence of WHY the session wedged is still discarded
// before the intervention runs, which is exactly the gap the issue names
// ("writes a state/stack snapshot to the durable session record"). This file
// adds the two pure pieces the shell needs to make that dump durable, in this
// package's usual division of labour: the row shape, the path/fold, and the
// derivation live here; the append itself lives at the cmd/fak call site next to
// every other resume ledger write.
//
//   - SoftObservationFromCurve derives the observation from the resume anchor the
//     trajectory watchdog ALREADY reads on the alive-but-stalled branch, so the
//     soft path costs no extra IO: the newest witnessed progress point's
//     timestamp is the stall clock the soft timeout is measured against, and the
//     curve's detail is the last-progress marker.
//   - SoftDumpRow is the schema-pinned, session-keyed ledger row (phase
//     "soft_state_dump") that lands in the same durable session record the
//     nudge/revive decision lands in, so an operator reading one file sees the
//     evidence immediately before the intervention it explains.
//   - FoldSoftDumps reads the newest dump per session back out, last-row-wins,
//     matching how every other resume store is consumed.

import (
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

const (
	// SoftDumpSchema pins the durable soft-dump row so a consumer that sees any
	// other value knows it is reading a foreign or newer shape.
	SoftDumpSchema = "fak-resume-soft-dump/1"
	// SoftDumpPhase is the ledger phase an operator greps for. It is deliberately
	// distinct from the hard path's "trajectory_decision" phase: soft observes,
	// hard decides, and the two must never be confused in the audit trail.
	SoftDumpPhase = "soft_state_dump"
)

// SoftDumpRow is one soft-watchdog capture as it lands in the durable session
// record. TS is left empty by the pure constructor and stamped by the shell that
// writes it, the same discipline RelaunchResetRow and DriveCarryRow follow.
type SoftDumpRow struct {
	Schema  string        `json:"schema"`
	TS      string        `json:"ts,omitempty"`
	Session string        `json:"session"`
	Phase   string        `json:"phase"`
	Trace   string        `json:"trace,omitempty"`
	Dump    SoftStateDump `json:"dump"`
}

// NewSoftDumpRow wraps a captured dump in its durable, schema-pinned envelope.
// Pure: no clock, no IO — the caller stamps TS at write time.
func NewSoftDumpRow(dump SoftStateDump, trace string) SoftDumpRow {
	return SoftDumpRow{
		Schema:  SoftDumpSchema,
		Session: dump.SessionID,
		Phase:   SoftDumpPhase,
		Trace:   trace,
		Dump:    dump,
	}
}

// CurveLastProgress is the stall clock: the newest timestamped point across every
// method curve. A curve with no timestamped point returns the zero time, which
// DecideSoftStateDump reads as "stall clock unknown" and refuses to dump on.
func CurveLastProgress(curve *trajctl.ObjectiveCurve) time.Time {
	if curve == nil {
		return time.Time{}
	}
	var newest int64
	for _, m := range curve.Methods {
		for _, p := range m.Points {
			if p.UnixMillis > newest {
				newest = p.UnixMillis
			}
		}
	}
	if newest <= 0 {
		return time.Time{}
	}
	return time.UnixMilli(newest).UTC()
}

// SoftObservationFromCurve builds the soft watchdog's input out of the facts the
// trajectory watchdog already holds on the alive-but-stalled branch: liveness
// from the process scan, the witnessed signal and last-progress marker from the
// resume anchor's curve, and pending (the stalled session's live process command
// line) as the worker-side state snapshot. Pure — the caller injects now.
func SoftObservationFromCurve(session string, alive bool, curve *trajctl.ObjectiveCurve, pending string, now time.Time) SoftWatchdogObservation {
	out := SoftWatchdogObservation{
		SessionID:     session,
		Alive:         alive,
		PendingAction: pending,
		Now:           now,
	}
	if curve == nil {
		return out
	}
	out.Signal = curve.Signal
	out.LastProgressMarker = curve.Detail
	out.LastProgressAt = CurveLastProgress(curve)
	return out
}

// FoldSoftDumps reduces a read-back ledger to the newest dump per session
// (last row wins, matching every other resume store's fold). Rows that name no
// session are skipped rather than bucketed under "".
func FoldSoftDumps(rows []SoftDumpRow) map[string]SoftDumpRow {
	out := make(map[string]SoftDumpRow, len(rows))
	for _, r := range rows {
		if r.Session == "" {
			continue
		}
		out[r.Session] = r
	}
	return out
}
