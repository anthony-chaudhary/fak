// relaunch_reset.go — the OS-relaunch analogue of internal/session's in-process
// ResetTransaction audit row, recorded in the transcript-UUID keyspace the watchdog
// actually keys on.
//
// # Why a resume-side reset row at all
//
// A context-budget reset that happens IN PROCESS stamps a replayable
// session.ResetTransaction (old-trace -> new-trace + budget re-arm) and appends it to
// the ResetTransactionLog audit chain (issue #1582). An OS relaunch — the watchdog
// running `claude --resume <uuid>` after a session died — IS just as much a context
// reset (fresh drive-state, fresh gateway trace), yet it records only a bare
// phase:"launched" ledger row and is invisible to that reset-transaction chain. A
// long-lived goal that survives N hidden relaunches therefore has N missing links in
// its reset history. This leaf closes that gap with the OS-relaunch counterpart of
// the in-process row.
//
// # Why it lives here and mirrors drivestate.go
//
// Exactly like WatchdogDriveState / DriveStateRow, this row is keyed on the Claude
// transcript UUID (== CLAUDE_CODE_SESSION_ID, the id `claude --resume` takes) — the
// one identity the watchdog and the relaunch actually share. The in-process
// ResetTransaction keys on the GATEWAY TRACE (State.TraceID), a DISJOINT keyspace the
// relaunched child never carries (its harness identity is stripped, see
// WatchdogChildEnvDrop). So the two rows cannot be the same object, and this one must
// NOT import internal/session: it mirrors the facts as scalars/strings so this
// foundation leaf can never be reddened by churn in the session package — the same
// keyspace-wall discipline drivestate.go documents.
//
// # Pure by construction
//
// NewRelaunchResetRow derives the row from the WatchdogPlanRow the tick already holds;
// there is no clock in the core, so TS (audit-only, ignored by the fold) is left for
// the shell to stamp at write time. FoldRelaunchResets reduces an append-only slice to
// the latest reset per session — same rows in, same map out, no I/O.
package resume

import "strings"

// RelaunchResetSchema is the stable schema token for an OS-relaunch reset audit row —
// the resume-keyspace sibling of session.ResetTransactionSchema.
const RelaunchResetSchema = "fak.resume.relaunch_reset.v1"

// RelaunchResetRow is the replayable row for one OS relaunch of a session. It is the
// transcript-UUID-keyed analogue of session.ResetTransaction: where the in-process row
// records OldTrace -> NewTrace, the relaunch preserves the transcript UUID and instead
// records the account re-home (PriorAccount -> RelaunchAccount) that IS the observable
// state change across the process boundary, plus the cause that put the session on the
// plan and the relaunch attempt count. Scalar fields only — no internal/session import.
type RelaunchResetRow struct {
	// Schema is the stable schema token (RelaunchResetSchema) marking a well-formed row.
	Schema string `json:"schema,omitempty"`
	// TS is the row's ISO-8601 write time (audit only; the fold ignores it, and the pure
	// constructor leaves it "" for the shell to stamp — no clock in the core).
	TS string `json:"ts,omitempty"`
	// Session is the Claude transcript UUID this relaunch reset applies to (the key).
	Session string `json:"session"`
	// Cause is the classifier disposition that put the session on the plan (the ledger
	// row's cause, e.g. STOPPED_MIDTOOL) — the OS-relaunch analogue of "why we reset".
	Cause string `json:"cause,omitempty"`
	// Attempt is the relaunch attempt count for this session (0-based floor).
	Attempt int `json:"attempt,omitempty"`
	// PriorAccount is the owning account the session ran under before the relaunch.
	PriorAccount string `json:"prior_account,omitempty"`
	// RelaunchAccount is the account the relaunch re-homed onto (== PriorAccount when the
	// plan did not re-home). Together with PriorAccount it is the prior->relaunched marker.
	RelaunchAccount string `json:"relaunch_account,omitempty"`
	// Rehomed marks that the transcript was copied onto a new config dir for this relaunch.
	Rehomed bool `json:"rehomed,omitempty"`
}

// NewRelaunchResetRow derives a relaunch reset row from the plan row the watchdog tick
// already has and the relaunch attempt count. Total over any input: a blank plan row
// yields a schema-stamped row with an empty Session (which the fold then skips), a
// negative attempt is floored to 0, and a re-home with no explicit ResumeAccount keeps
// RelaunchAccount == PriorAccount. TS is intentionally left "" — the pure core carries
// no clock; the shell stamps the write time.
func NewRelaunchResetRow(row WatchdogPlanRow, attempt int) RelaunchResetRow {
	if attempt < 0 {
		attempt = 0
	}
	prior := strings.TrimSpace(row.Account)
	relaunch := strings.TrimSpace(row.ResumeAccount)
	if relaunch == "" {
		relaunch = prior
	}
	return RelaunchResetRow{
		Schema:          RelaunchResetSchema,
		Session:         strings.TrimSpace(row.Session),
		Cause:           strings.TrimSpace(row.Disp),
		Attempt:         attempt,
		PriorAccount:    prior,
		RelaunchAccount: relaunch,
		Rehomed:         row.Rehomed,
	}
}

// WellFormed reports whether the row carries the schema token and a session id — the
// same shape-of-the-audit-trail check session.ResetTransactionLog.Replay applies to the
// in-process row (schema present + trace ids present), adapted to this keyspace where
// the transcript UUID is the identity.
func (r RelaunchResetRow) WellFormed() bool {
	return r.Schema == RelaunchResetSchema && strings.TrimSpace(r.Session) != ""
}

// Rehome reports the prior->relaunched account marker and whether the relaunch actually
// changed account. A caller auditing a cross-relaunch reset chain uses `changed` to spot
// the resets that moved a session to a different account.
func (r RelaunchResetRow) Rehome() (from, to string, changed bool) {
	return r.PriorAccount, r.RelaunchAccount, r.PriorAccount != r.RelaunchAccount
}

// FoldRelaunchResets folds append-only relaunch reset rows into the latest reset per
// session: last row per session wins (the store is append-only, so slice order is write
// order). A row with a blank Session has no key and is skipped so it never clobbers a
// prior valid reset. Total over any input: nil/empty rows yield an empty map.
func FoldRelaunchResets(rows []RelaunchResetRow) map[string]RelaunchResetRow {
	out := make(map[string]RelaunchResetRow, len(rows))
	for _, r := range rows {
		sid := strings.TrimSpace(r.Session)
		if sid == "" {
			continue
		}
		r.Session = sid
		out[sid] = r
	}
	return out
}
