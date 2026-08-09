package toolproc

import "strings"

// bgident.go — the brokered background-job identity (#5880).
//
// THE DEFECT. HookEvents bridges a background launch into a SECOND journaled
// proc, keyed on the id the harness announces ("Command running in background
// with ID: 8", {"shellId": "..."}). That id is allocated PER SESSION and
// restarts at 1 for each new session. But `.fak/toolproc/journal.jsonl` is ONE
// SHARED FILE for the whole workspace by design (cmd/fak/guard_toolproc_hooks.go:
// "One shared journal per workspace: events carry the harness session id, so
// concurrent guarded sessions interleave without colliding"). Writing the bare
// id as the call identity puts a per-WRITER key into a many-writer file — the
// #5524 / #5876 shape. Fold refuses a duplicate spawn, and it refuses the WHOLE
// table, so one collision between two sessions blinds every session's rows.
//
// THE WITNESS (this workspace's live journal, 30,882 rows, 2,348 sessions,
// 2026-07-16 .. 2026-08-07):
//
//   - 18,903 spawn rows carry 18,889 distinct identities; 11 of those identities
//     are spawned more than once, over 14 duplicate rows.
//   - ALL 11 are "bg:" ids (bg:1 .. bg:11) and ALL 11 span more than one session.
//     Of the 659 distinct "bg:" identities on the file, 1.7% collide.
//   - The other 18,230 identities are harness tool_use_ids, which are globally
//     unique by construction. "Zero non-bg collisions" is therefore what that key
//     space CANNOT do, not evidence that this bridge is fine — the collisions are
//     concentrated in the one key space this package mints itself.
//
// THE CURE IS ON THE WRITER. Fold's duplicate-spawn refusal is a real check for
// a real class, and #5524 settled the precedent for this defect shape: cure the
// writer, never weaken the check. Two sessions genuinely CAN both own a job
// called "8"; the journal simply has to say whose. backgroundCallID qualifies
// the id with the session that owns it, matching hook.go's existing minted
// fallback identity ("hk:<session>:<tool>:<digest>") — the one other place this
// package has to name a call the harness did not name for it. The full session
// id is used rather than a short digest of it: a truncated qualifier would
// reintroduce exactly this bug at a lower rate, which is the shape of defect
// being cured, not a cure for it.
//
// The poll bridge resolves through the same constructor, so a poll can only ever
// pulse a job in its OWN session. Before this, a poll naming id "8" resolved
// against whichever session's "8" the journal happened to be holding — a silent
// mis-correlation, worse than the loud refusal above because nothing reports it.
//
// SCOPE — HISTORICAL ROWS. Same position as #3152: this repairs journals written
// from here on and does not retro-fix rows already on disk. A journal that
// already carries two unqualified "bg:8" spawns from different sessions still
// refuses, correctly, because nothing in it records which session owned which;
// compaction (CompactJournal) reclaims that history as those calls go terminal.
// What is deliberately NOT done is to make the reader tolerant of unqualified
// ids: that would blind the check on every journal, forever, to repair a bounded
// backlog. Counts.BackgroundIDsUnqualified reports the size of that backlog
// instead, so the position is measurable rather than merely asserted.
//
// A job launched under the old identity and polled under the new one loses its
// pulses: the poll finds no such call and bridges nothing, which is the fail-open
// degrade hookpulse.go already documents for an un-journaled id. Falling back to
// an unqualified lookup would restore those pulses at the price of re-admitting
// the cross-session mis-correlation, so it is refused.

// backgroundCallIDPrefix tags a brokered background job's call identity.
const backgroundCallIDPrefix = "bg:"

// backgroundCallID mints the journal identity for the background job `id` that
// `session` announced (launch) or is polling. Launch and poll must agree on the
// identity, so both bridges resolve through this one constructor.
//
// An empty session yields "bg::<id>": the qualifier is visibly absent rather
// than silently dropped, matching hook.go, which likewise formats an empty
// session into its fallback identity instead of special-casing it.
func backgroundCallID(session, id string) string {
	return backgroundCallIDPrefix + session + ":" + id
}

// isUnqualifiedBackgroundCallID reports whether callID is a background identity
// written before the session qualifier landed — "bg:<id>" with no session
// segment. It reads shape only and never changes how such a row folds.
func isUnqualifiedBackgroundCallID(callID string) bool {
	rest, ok := strings.CutPrefix(callID, backgroundCallIDPrefix)
	return ok && !strings.Contains(rest, ":")
}
