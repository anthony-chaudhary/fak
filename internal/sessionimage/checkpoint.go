package sessionimage

// checkpoint.go — the typed on-demand CHECKPOINT op of the operator-control epic (#2760,
// child of #2753). A checkpoint CAPTURES a running session: it writes the session's
// boundary-consistent drive state + context image into a durable, addressable snapshot —
// WITHOUT stopping, pausing, or otherwise mutating the source. Durable checkpointed state
// is the substrate fork, safe experimentation, and crash-durable resume all build on.
//
// # Why the op lives here (tier: integrator), not with the sessionctl control ops
//
// The other operator-control ops (redirect, constraint) work purely on a session's drive
// STATE — they compose adjudicator + abi and belong in tier-3 sessionctl. A checkpoint is a
// different animal: it is an operator-initiated READ that produces a durable session-IMAGE
// ARTIFACT. Its capture is literally the in-memory twin of DumpDir (see snapshot.go's
// SnapshotDir, the on-disk twin), so it couples to the integrator types this package owns —
// Input, Meta, DumpDir. That coupling is why it is an integrator-tier op: a tier-3 composer
// may not import this tier-4 package (the layered-DAG import rule), so the checkpoint op
// lives here, next to the DumpDir it wraps, and the CLI verb (`fak session checkpoint`)
// binds transport in cmd/fak.
//
// # Why this op is NOT in the loop-consumed vocabulary spine (#2766)
//
// The nine spine ops (steer/redirect/pause/resume/cancel/terminate/throttle/budget/
// priority) share one shape: the LOOP consumes them at a boundary — a splice into the turn
// input, a hold, a stop, a sampling cap, a scheduler read. Their witness-of-applied is that
// loop-side consumption. A checkpoint has no loop-consumption half: nothing is spliced,
// nothing halts, the drive is not written. Its witness is the round-trip RESTORE — dump the
// live session, load it back, and prove the restored drive state is equivalent while the
// source keeps running (checkpoint_test.go, this issue's named witness). Because it has no
// loop-consumption half it registers no #2766 row and is deliberately absent from the closed
// spine registry. A future boundary-DRIVEN auto-checkpoint (out of scope for #2760) would be
// the spine citizen; this on-demand capture is not.

import (
	"fmt"
	"strings"
)

// checkpointReasonLabel is the image label a checkpoint stamps its operator reason under,
// so a restored snapshot can answer "why was this taken?" from the recorded Meta rather
// than a guess. Absent when the op carried no reason.
const checkpointReasonLabel = "checkpoint_reason"

// Checkpoint is the typed on-demand snapshot op payload: WHERE to write the durable image
// and an optional operator note. The op carries no session identity of its own — the
// session it captures is supplied at Snapshot time (the live drive + content), so one op
// value cannot silently target the wrong session.
type Checkpoint struct {
	// Dest is the destination bundle directory the snapshot is written to — required. A
	// checkpoint with nowhere to write has nothing to capture TO and is refused malformed.
	Dest string `json:"dest"`
	// Reason is an optional operator note recorded on the snapshot's image Meta.
	Reason string `json:"reason,omitempty"`
}

// CheckpointRefuseReason is the closed refusal vocabulary for a checkpoint op — the same
// closed-reason discipline as sessionctl's RedirectRefuseReason and every other fak
// refusal. These are op-shape reasons, distinct from the drive-state control vocabulary
// (session.ControlRefusalTokens): a checkpoint is a READ, so it has no "terminal session"
// refusal — a stopped session is a perfectly legal thing to snapshot (that is exactly the
// crash-durable-resume case).
type CheckpointRefuseReason string

const (
	// CheckpointMalformed refuses a checkpoint whose shape cannot be applied — today, an
	// empty destination (nowhere to write the snapshot).
	CheckpointMalformed CheckpointRefuseReason = "CHECKPOINT_MALFORMED"
	// CheckpointNoSession refuses a checkpoint of a session with no identity to key the
	// snapshot on (neither a SessionID nor a Drive.TraceID) — there is nothing to capture.
	CheckpointNoSession CheckpointRefuseReason = "CHECKPOINT_NO_SESSION"
)

// CheckpointRefusal is a structured, closed-reason refusal of one checkpoint op. It
// implements error so plumbing can thread it, but callers should switch on Reason (via
// errors.As), never parse Detail.
type CheckpointRefusal struct {
	Reason CheckpointRefuseReason `json:"reason"`
	Detail string                 `json:"detail,omitempty"`
}

// refusalString renders a closed-reason refusal the one way every *Refusal
// error in this package does: the reason token alone, or "REASON: detail"
// when a detail is attached. The string form is for logs and error plumbing;
// callers switch on Reason, never parse Detail (each refusal type's doc).
func refusalString(reason, detail string) string {
	if detail == "" {
		return reason
	}
	return reason + ": " + detail
}

func (r *CheckpointRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// checkpointRefuse builds a CheckpointRefusal in one line.
func checkpointRefuse(reason CheckpointRefuseReason, format string, args ...any) *CheckpointRefusal {
	return &CheckpointRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// Validate checks the op's SHAPE — parse, don't validate later: an empty destination is
// refused synchronously at the producer edge. Whether the SESSION being captured is
// well-formed (has an identity) is decided against the supplied live session in Snapshot,
// not here, since this payload carries no session.
func (c Checkpoint) Validate() *CheckpointRefusal {
	if strings.TrimSpace(c.Dest) == "" {
		return checkpointRefuse(CheckpointMalformed, "checkpoint needs a non-empty destination directory")
	}
	return nil
}

// Snapshot captures a live session into a durable image at c.Dest and returns the persisted
// Meta. `live` is the session as observed at a turn boundary — its drive state plus the
// optional content primitives (recall page table, ctxplan index, trajectory). It is taken
// BY VALUE, and the capture only READS it: the drive is copied, the recorder is persisted
// (its logical content is written out, never mutated), and the caller's Labels map is cloned
// before the reason is stamped. So the running session the operator checkpointed is provably
// unaffected — the "keeps running / source unaffected" guarantee this op exists to make.
//
// A shape refusal (empty dest) or an identity refusal (no SessionID/TraceID) surfaces as a
// *CheckpointRefusal carrying its closed reason; a disk/IO failure surfaces as a wrapped
// error. Callers branch the two with errors.As.
func (c Checkpoint) Snapshot(live Input) (Meta, error) {
	if r := c.Validate(); r != nil {
		return Meta{}, r
	}
	if strings.TrimSpace(live.SessionID) == "" && strings.TrimSpace(live.Drive.TraceID) == "" {
		return Meta{}, checkpointRefuse(CheckpointNoSession,
			"a checkpoint needs a session identity (SessionID or Drive.TraceID) to key the snapshot on")
	}
	if note := strings.TrimSpace(c.Reason); note != "" {
		live.Labels = withCheckpointLabel(live.Labels, checkpointReasonLabel, note)
	}
	meta, err := DumpDir(strings.TrimSpace(c.Dest), live)
	if err != nil {
		return Meta{}, fmt.Errorf("sessionimage: checkpoint snapshot: %w", err)
	}
	return meta, nil
}

// withCheckpointLabel returns a COPY of src with key=val set — never mutating the caller's
// map, so a checkpoint leaves the live session's Labels (and everything else it was handed)
// untouched.
func withCheckpointLabel(src map[string]string, key, val string) map[string]string {
	out := make(map[string]string, len(src)+1)
	for k, v := range src {
		out[k] = v
	}
	out[key] = val
	return out
}
