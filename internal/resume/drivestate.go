// drivestate.go — the OPERATOR-HOLD vocabulary + durable-store fold the resume
// watchdog reads so it never resurrects a session an operator deliberately paused,
// drained, or stopped. It is the pure leaf behind the guard step in watchdog.go
// (DecideWatchdogRow) and the `fak resume hold` / `fak resume release` operator
// surface.
//
// # Why this lives in the Claude-UUID keyspace
//
// The watchdog keys every decision on the plan row's Session — the Claude Code
// transcript UUID (== CLAUDE_CODE_SESSION_ID, the id `claude --resume` takes). The
// out-of-band operator control plane (internal/session.Table, `fak session
// pause/stop <id>`) keys on the GATEWAY TRACE / guard --session-id instead — a
// DISJOINT namespace that never carries the transcript UUID (a resumed child even
// has CLAUDE_CODE_SESSION_ID stripped, see WatchdogChildEnvDrop). So an operator's
// `fak session stop` records durable intent the watchdog CANNOT join to a plan row.
// This store closes that gap: a hold recorded in the ONE key the operator surface
// and the watchdog actually share.
//
// # Why NOT the Descriptor registry
//
// Two reasons the descriptor drive-state can't be the source: (1) no join key
// (above); (2) a Stopped Descriptor is TTL-GC'd after 30 minutes — a terminal-Stop
// veto stored there would evaporate. This store is append-only and never swept, so a
// Stop persists (durable across GC) until the operator explicitly releases it.
//
// # Pure by construction
//
// The shell reads resume_drivestate.jsonl and hands FoldDriveStates the parsed rows;
// the fold returns the one current drive-state per session. No clock, no I/O — same
// rows in, same map out. The store is APPEND-ONLY, so slice order IS write order and
// the fold is "last row per session wins": a later `running` row RELEASES an earlier
// hold, which is exactly what `fak resume release` writes. A hold is reversed ONLY by
// the operator, never by the watchdog.
package resume

import "strings"

// WatchdogDriveState is the operator's recorded drive-state for one session, in the
// same closed vocabulary as internal/session.RunState (mirrored here as strings so
// this foundation leaf does not import the session package and cannot be reddened by
// churn in it). The empty value is "no opinion" — a session with no recorded state is
// never held.
type WatchdogDriveState string

const (
	// DriveRunning: live / released. A `running` row LIFTS a prior paused/draining/stopped
	// hold (the reversible-release write `fak resume release` appends). Never a hold.
	DriveRunning WatchdogDriveState = "running"
	// DriveThrottled: still advancing under a tightened pace — not a hold (the watchdog only
	// fires on a DEAD session anyway; a throttled-but-live one is not on the plan).
	DriveThrottled WatchdogDriveState = "throttled"
	// DrivePaused: an operator held the session at a boundary — a reversible hold.
	DrivePaused WatchdogDriveState = "paused"
	// DriveDraining: an operator is winding the session down gracefully — do not relaunch.
	DriveDraining WatchdogDriveState = "draining"
	// DriveStopped: an operator terminated the session — terminal intent. Durable (the store
	// has no TTL, so it outlives the descriptor registry's 30-min GC) and reversed only by an
	// explicit operator release, never by the watchdog.
	DriveStopped WatchdogDriveState = "stopped"
)

// normalizeDriveState maps a raw stored token to the closed vocabulary, returning ""
// (inert) for any unrecognized value — an unknown token never holds a session and
// never clobbers a prior valid state in the fold.
func normalizeDriveState(s string) WatchdogDriveState {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "running":
		return DriveRunning
	case "throttled":
		return DriveThrottled
	case "paused":
		return DrivePaused
	case "draining":
		return DriveDraining
	case "stopped":
		return DriveStopped
	default:
		return ""
	}
}

// HeldByOperator reports whether this drive-state means the operator has deliberately
// held the session, so the watchdog must NOT auto-resume it. Only paused/draining/
// stopped hold; running/throttled/"" (the zero value) fall through to today's behavior
// — the per-key fail-open the guard depends on (an absent session reads "" → not held).
// Exported so the operator `fak resume hold --list` surface and this guard share one
// predicate.
func (s WatchdogDriveState) HeldByOperator() bool {
	return s == DrivePaused || s == DriveDraining || s == DriveStopped
}

// HoldReason is the closed human one-liner the watchdog logs when it stands down on an
// operator hold. Each names its drive-state so a log grep never conflates a per-session
// operator hold with the account-level "tombstoned" (policy) skip. Exported so the guard
// (watchdog.go) and any operator surface render one wording.
func (s WatchdogDriveState) HoldReason() string {
	switch s {
	case DriveStopped:
		return "operator stopped this session (durable drive-state hold) — a session the operator terminated is never auto-resumed"
	case DriveDraining:
		return "operator is draining this session (drive-state hold) — do not relaunch a session being wound down"
	case DrivePaused:
		return "operator paused this session (drive-state hold) — releasing it is the operator's call (`fak resume release`), not the watchdog's"
	default:
		return "operator drive-state hold"
	}
}

// DriveCarry is the resume-keyspace projection of state that must survive an
// OS-process relaunch. Scalar fields keep this package independent of session.
type DriveCarry struct {
	TurnsLeft           int64  `json:"turns_left,omitempty"`
	TokensLeft          int64  `json:"tokens_left,omitempty"`
	ContextTokensLeft   int64  `json:"context_tokens_left,omitempty"`
	SpendMicroCentsLeft int64  `json:"spend_micro_cents_left,omitempty"`
	TimeLeftNanos       int64  `json:"time_left_nanos,omitempty"`
	Generation          int    `json:"generation,omitempty"`
	ObjectivePinID      string `json:"objective_pin_id,omitempty"`
	ObjectiveText       string `json:"objective_text,omitempty"`
	ObjectiveDigest     string `json:"objective_digest,omitempty"`
}

// DriveStateRow is one append-only line of the resume_drivestate.jsonl store, reduced
// to the typed facts the fold reads. The shell parses the JSONL (jsonlledger.Parse);
// unknown fields are dropped, not trusted. TS is carried for humans/audit only — the
// fold orders by FILE order (append-only ⇒ chronological), never by parsing a timestamp.
type DriveStateRow struct {
	// TS is the row's ISO-8601 write time (audit only; the fold ignores it).
	TS string `json:"ts,omitempty"`
	// Session is the Claude transcript UUID this state applies to (the watchdog's key).
	Session string `json:"session"`
	// State is the drive-state token (running/throttled/paused/draining/stopped).
	State string `json:"state"`
	// Reason is the optional human note recorded with the hold.
	Reason string `json:"reason,omitempty"`
	// Via names what wrote the row (e.g. "fak resume hold"), for provenance.
	Via string `json:"via,omitempty"`

	// Carry fields are additive and top-level so old JSONL remains readable and
	// producers can append a carry row without manufacturing a hold token.
	TurnsLeft           *int64 `json:"turns_left,omitempty"`
	TokensLeft          *int64 `json:"tokens_left,omitempty"`
	ContextTokensLeft   *int64 `json:"context_tokens_left,omitempty"`
	SpendMicroCentsLeft *int64 `json:"spend_micro_cents_left,omitempty"`
	TimeLeftNanos       *int64 `json:"time_left_nanos,omitempty"`
	Generation          *int   `json:"generation,omitempty"`
	ObjectivePinID      string `json:"objective_pin_id,omitempty"`
	ObjectiveText       string `json:"objective_text,omitempty"`
	ObjectiveDigest     string `json:"objective_digest,omitempty"`

	// ReArm is the operator's explicit re-grant directive. It is the ONE thing that
	// lets a carry row RAISE a remaining axis under ReconcileDriveCarry's non-regrant
	// clamp — mirroring drivestate.go's sticky-Stopped precedence (only an operator
	// release reverses it) and Recontinue's explicit-fresh-axis-wins rule
	// (table.go:550-553, where a stated fresh spend axis overrides the carried one).
	// Absent (the zero value) a row can only lower a remaining axis, never refill it.
	ReArm bool `json:"re_arm,omitempty"`
}

// FoldDriveStates folds the append-only rows into the one current drive-state per
// session the guard reads: last row per session wins (the store is append-only, so
// slice order is write order), and an unknown/blank state token is ignored so it never
// clobbers a prior valid state. A later `running` row therefore releases an earlier
// hold — the reversible-release contract `fak resume release` relies on. Total over any
// input: nil/empty rows yield an empty map (no holds).
func FoldDriveStates(rows []DriveStateRow) map[string]WatchdogDriveState {
	out := make(map[string]WatchdogDriveState, len(rows))
	for _, r := range rows {
		sid := strings.TrimSpace(r.Session)
		if sid == "" {
			continue
		}
		st := normalizeDriveState(r.State)
		if st == "" {
			continue // unknown token: leave any prior state for this session intact
		}
		out[sid] = st
	}
	return out
}

// FoldDriveCarry returns the latest explicit carry for each session. Hold-only
// rows do not clobber a prior carry, and carry-only rows do not clobber holds.
func FoldDriveCarry(rows []DriveStateRow) map[string]DriveCarry {
	out := make(map[string]DriveCarry)
	for _, row := range rows {
		sid := strings.TrimSpace(row.Session)
		if sid == "" || !row.hasCarry() {
			continue
		}
		out[sid] = row.driveCarry()
	}
	return out
}

// ReconcileDriveCarry folds the append-only carry rows into the SAFE remaining budget
// per session under a non-regrant invariant: for each remaining axis (turns, tokens,
// context, spend, time) the folded value is monotone-non-increasing across rows, so a
// stale or higher-numbered row can never silently REFILL a budget the session already
// spent down. This is the OS-relaunch analogue of Recontinue's honest-money rule
// (RecontinueAtWithTransaction, table.go:514-566): a spend-drained parent carries
// Left=0 under a positive cap, and no reset re-grants it.
//
// The ONLY thing that raises an axis is a row with ReArm set — an explicit operator
// re-arm, adopted wholesale (mirroring the explicit-fresh-axis-wins precedence at
// table.go:550-553). Generation and the objective fields are NOT remaining budget, so
// they take the latest carry row's value (last-wins), never a min-clamp.
//
// Pure and total, same shape as FoldDriveCarry: same append-only rows in, same map out;
// no clock, no I/O. Each carry row is treated as a COMPLETE remaining-budget snapshot
// (which is what the PreCompact/Stop producer writes), so an axis a row omits carries
// the prior folded value forward rather than reading as zero. FoldDriveCarry is left
// intact for consumers that want the raw last-written value (e.g. audit/render); the
// re-seed path consumes THIS clamped fold so a relaunch can never over-grant.
func ReconcileDriveCarry(rows []DriveStateRow) map[string]DriveCarry {
	out := make(map[string]DriveCarry)
	for _, row := range rows {
		sid := strings.TrimSpace(row.Session)
		if sid == "" || !row.hasCarry() {
			continue
		}
		prev, seen := out[sid]
		if !seen || row.ReArm {
			// First carry for the session, or an explicit operator re-arm: adopt the
			// row's snapshot as-is. Re-arm is the sanctioned way to raise an axis.
			out[sid] = row.driveCarry()
			continue
		}
		out[sid] = row.clampNonRegrant(prev)
	}
	return out
}

// clampNonRegrant projects this row onto prev under the non-regrant rule: an axis the
// row sets may only LOWER the prior folded remaining value (min), an axis the row omits
// keeps prev's value, and generation/objective take the row's latest value.
func (row DriveStateRow) clampNonRegrant(prev DriveCarry) DriveCarry {
	out := prev // carry forward every axis the row does not restate
	if row.TurnsLeft != nil {
		out.TurnsLeft = minInt64(prev.TurnsLeft, *row.TurnsLeft)
	}
	if row.TokensLeft != nil {
		out.TokensLeft = minInt64(prev.TokensLeft, *row.TokensLeft)
	}
	if row.ContextTokensLeft != nil {
		out.ContextTokensLeft = minInt64(prev.ContextTokensLeft, *row.ContextTokensLeft)
	}
	if row.SpendMicroCentsLeft != nil {
		out.SpendMicroCentsLeft = minInt64(prev.SpendMicroCentsLeft, *row.SpendMicroCentsLeft)
	}
	if row.TimeLeftNanos != nil {
		out.TimeLeftNanos = minInt64(prev.TimeLeftNanos, *row.TimeLeftNanos)
	}
	if row.Generation != nil {
		out.Generation = *row.Generation // generation counts up across resets — not a remaining axis
	}
	// Objective fields are identity, not budget: the latest non-empty value wins so a
	// re-pin is never clamped away (see ReconcileDriveCarry's objective note).
	if row.ObjectivePinID != "" {
		out.ObjectivePinID = row.ObjectivePinID
	}
	if row.ObjectiveText != "" {
		out.ObjectiveText = row.ObjectiveText
	}
	if row.ObjectiveDigest != "" {
		out.ObjectiveDigest = row.ObjectiveDigest
	}
	return out
}

func minInt64(a, b int64) int64 {
	if b < a {
		return b
	}
	return a
}

func (row DriveStateRow) hasCarry() bool {
	return row.TurnsLeft != nil || row.TokensLeft != nil || row.ContextTokensLeft != nil ||
		row.SpendMicroCentsLeft != nil || row.TimeLeftNanos != nil || row.Generation != nil ||
		row.ObjectivePinID != "" || row.ObjectiveText != "" || row.ObjectiveDigest != ""
}

func (row DriveStateRow) driveCarry() DriveCarry {
	carry := DriveCarry{
		ObjectivePinID:  row.ObjectivePinID,
		ObjectiveText:   row.ObjectiveText,
		ObjectiveDigest: row.ObjectiveDigest,
	}
	if row.TurnsLeft != nil {
		carry.TurnsLeft = *row.TurnsLeft
	}
	if row.TokensLeft != nil {
		carry.TokensLeft = *row.TokensLeft
	}
	if row.ContextTokensLeft != nil {
		carry.ContextTokensLeft = *row.ContextTokensLeft
	}
	if row.SpendMicroCentsLeft != nil {
		carry.SpendMicroCentsLeft = *row.SpendMicroCentsLeft
	}
	if row.TimeLeftNanos != nil {
		carry.TimeLeftNanos = *row.TimeLeftNanos
	}
	if row.Generation != nil {
		carry.Generation = *row.Generation
	}
	return carry
}
