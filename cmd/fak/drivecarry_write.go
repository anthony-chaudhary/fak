package main

import (
	"os"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/session"
)

// drivecarry_write.go — the PRODUCER half of the drive-carry channel (#4118). At the
// PreCompact reset boundary and at Stop, the guard-installed hook process is the one place
// the live REMAINING budget is knowable AND the transcript UUID is in the env: it projects
// the session's remaining allotment into a transcript-UUID-keyed DriveCarryRow so a later
// `claude --resume` re-seeds at the carried budget (rwLoadDriveCarry, #4119) instead of a
// fresh, full cap. Without this producer the read side has nothing to read.
//
// The two keyspaces are disjoint — the hook holds the Claude transcript UUID
// (CLAUDE_CODE_SESSION_ID), while the live drive State/Descriptor is keyed by the guard
// trace — so the write JOINS them through the A1 identity store (resume_identity.jsonl) the
// SessionStart hook records (uuid <-> trace, #4112), then reads the durable descriptor
// registry (session-registry.json) the guard mirrors on transitions for that trace's State.
//
// FAIL-OPEN at every step, matching the hooks' own contract: no transcript UUID, no identity
// join, no persisted descriptor, or an unbounded budget writes NOTHING — byte-identical to
// today. It never changes a hook's exit code.
func writeDriveCarryFailOpen(now time.Time) {
	uuid := strings.TrimSpace(os.Getenv("CLAUDE_CODE_SESSION_ID"))
	if uuid == "" {
		return // a resumed child has this stripped; only the original session carries it
	}
	regDir := resolveSweepRegDir("")
	traceByUUID, _ := resume.LoadIdentity(regDir)
	trace := strings.TrimSpace(traceByUUID[uuid])
	if trace == "" {
		return // no uuid<->trace join yet: nothing to key the live State by
	}
	reg := session.NewRegistry(session.NewFileStore(defaultSessionRegistryPath()))
	d, ok, err := reg.Get(trace)
	if err != nil || !ok {
		return // no persisted descriptor for this trace
	}
	st := d.RestoredState()
	if !driveCarryBounded(st) {
		return // an unbounded session has no remaining allotment worth carrying
	}
	rwAppendLedger(rwDriveCarryLedger(regDir), driveCarryRowFromState(uuid, st, now))
}

// driveCarryBounded reports whether the State carries any configured budget cap worth
// projecting. An unbounded default (turns/tokens == Unbounded(-1), context/spend/time
// unconfigured) yields false, so a plain attended session writes no carry row — exactly the
// fail-open the done-condition requires for an unbounded budget.
func driveCarryBounded(st session.State) bool {
	b := st.Budget
	return b.TurnsLeft >= 0 || b.TokensLeft >= 0 || b.ContextTokensLeft > 0 ||
		b.SpendMicroCentsLeft > 0 || st.Time.Bounded()
}

// driveCarryRowFromState projects the live drive State's REMAINING axes onto the
// transcript-UUID-keyed DriveCarryRow the re-seed reads. Axes are copied verbatim so the
// record round-trips through rwDriveCarryEnvelope's encoding: an Unbounded(-1) turn/token
// axis carries as -1 (rendered "unbounded" on re-seed), a not-configured context/spend axis
// stays 0 (dropped by omitempty), and a bounded wall-clock budget carries its remaining
// nanos.
func driveCarryRowFromState(uuid string, st session.State, now time.Time) resume.DriveCarryRow {
	row := resume.DriveCarryRow{
		TS:                   now.UTC().Format(time.RFC3339),
		Session:              uuid,
		TurnsLeft:            int64(st.Budget.TurnsLeft),
		TokensLeft:           int64(st.Budget.TokensLeft),
		ContextTokensLeft:    int64(st.Budget.ContextTokensLeft),
		SpendMicroCentsLeft:  st.Budget.SpendMicroCentsLeft,
		Priority:             st.Priority,
		PaceMaxTokensPerTurn: st.Pace.MaxTokensPerTurn,
		PaceMinTurnGapMs:     st.Pace.MinTurnGapMs,
		Generation:           st.Generation,
	}
	if rem, ok := st.Time.Remaining(now); ok {
		row.TimeLeftNanos = rem.Nanoseconds()
	}
	return row
}
