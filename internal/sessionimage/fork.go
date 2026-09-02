package sessionimage

// fork.go — the typed FORK op of the out-of-band operator control epic (#2761, child of
// #2753): SNAPSHOT-AND-BRANCH a running session into a divergent continuation while the
// original keeps its place. Fork is the exploration primitive the epic names — an operator
// can spawn a divergent line (e.g. after a redirect) without losing the current one.
//
// # Fork = checkpoint (#2760) then branch (#1200), as one op
//
// The two lifecycle primitives fork composes already exist:
//
//   - SnapshotDir (#2760) PINS the running session's current state as a durable, immutable
//     CHECKPOINT — the branch point — reading the source read-only so the live session is
//     unaffected.
//   - BranchDir (#1200) FORKS that checkpoint into a SECOND session under a fresh trace id,
//     sharing the content-addressed pages copy-on-write and recording the parent_id lineage.
//
// Fork is exactly their composition: pin the branch point, then diverge from it. That is why
// it is the child that was "blocked-by the checkpoint child" — checkpoint is the durable
// point fork branches from. Neither primitive alone is a fork: a checkpoint is the SAME
// session captured (no divergence), a bare branch forks from whatever is on disk without
// first freezing an immutable branch point. Fork does both so the operator gets a named,
// durable branch point AND a divergent fork in one move, the original untouched.
//
// # Why the op lives here (tier: integrator), not in sessionctl
//
// Like checkpoint (see checkpoint.go), fork produces a durable session-IMAGE ARTIFACT and
// couples to the integrator types this package owns (Meta, LoadDir, SnapshotDir, BranchDir).
// The tier-3 sessionctl control ops (redirect, constraint) work purely on a session's drive
// STATE and may not import this tier-4 package (the layered-DAG import rule), so the fork op
// lives here, next to the primitives it wraps, and the CLI verb (`fak session fork`) binds
// transport in cmd/fak. And like checkpoint it is NOT in the loop-consumed vocabulary spine
// (#2766): fork has no loop-consumption half — nothing is spliced into a turn, nothing halts,
// the running drive is not written — so it registers no spine row. Its witness is the
// divergence: two independent sessions that share history up to the branch point and diverge
// after (fork_test.go / cmd/fak session_fork_test.go, this issue's named witness).

import (
	"fmt"
	"strings"
)

// ForkOptions configures a snapshot-and-branch fork. ForkID is the fork's fresh durable id
// and trace — required, and it must differ from the parent so the two are distinct sessions
// in the live table and the C1 registry. ToModel / ToHost / Account / Residency, when set,
// re-home the fork's identity metadata (a fork may explore under a different model or host);
// unset fields inherit the parent's. Reason is an optional operator note folded into both the
// checkpoint's and the fork's migration-log entries. Now is an injected unix clock for
// deterministic stamps (0 => wall time).
type ForkOptions struct {
	ForkID    string
	ToModel   string
	ToHost    string
	Account   string
	Residency string
	Reason    string
	Now       int64
}

// ForkResult is what a completed fork produced: the durable BranchPoint checkpoint both lines
// share (same id as the parent — a checkpoint preserves the id) and the divergent Fork (a
// fresh id with a parent_id link back to the original). An operator can restore either.
type ForkResult struct {
	// BranchPoint is the pinned, immutable checkpoint the fork branched from — the point the
	// two sessions share history up to. Its SessionID is the parent's (a checkpoint is the
	// same session captured, not a fork).
	BranchPoint Meta `json:"branch_point"`
	// Fork is the divergent continuation: a fresh trace id, a parent_id link to the original,
	// and a "branched from <parent> at <sha>" lineage entry.
	Fork Meta `json:"fork"`
}

// ForkRefuseReason is the closed refusal vocabulary for a fork op — the same closed-reason
// discipline as sessionctl's RedirectRefuseReason and checkpoint.go's CheckpointRefuseReason.
// These are op-shape reasons: a fork is a READ of the source, so — like a checkpoint — it has
// no "terminal session" refusal (forking a stopped session is a perfectly legal explore).
type ForkRefuseReason string

const (
	// ForkMalformed refuses a fork whose shape cannot be applied: an empty fork id (nothing to
	// re-key the divergent trace to), or a checkpoint/fork/parent directory that collides with
	// another (the three must be distinct so the pin, the fork, and the source never overwrite).
	ForkMalformed ForkRefuseReason = "FORK_MALFORMED"
	// ForkSameID refuses a fork whose id equals the parent's — a fork must be a SECOND, distinct
	// session (a same-id "fork" is a checkpoint, not a divergence).
	ForkSameID ForkRefuseReason = "FORK_SAME_ID"
)

// ForkRefusal is a structured, closed-reason refusal of one fork op. It implements error so
// plumbing can thread it, but callers should switch on Reason (via errors.As), never parse
// Detail.
type ForkRefusal struct {
	Reason ForkRefuseReason `json:"reason"`
	Detail string           `json:"detail,omitempty"`
}

func (r *ForkRefusal) Error() string { return refusalString(string(r.Reason), r.Detail) }

// forkRefuse builds a ForkRefusal in one line.
func forkRefuse(reason ForkRefuseReason, format string, args ...any) *ForkRefusal {
	return &ForkRefusal{Reason: reason, Detail: fmt.Sprintf(format, args...)}
}

// ForkDir snapshot-and-branches the session bundle in parentDir into a divergent continuation.
// It first pins the parent's current state as an immutable checkpoint at checkpointDir (the
// branch point — SnapshotDir reads the source read-only, so the live session is unaffected),
// then forks that checkpoint into a fresh-trace session at forkDir (BranchDir — copy-on-write
// share of the pages, a parent_id link, and the recorded lineage). It returns the persisted
// Meta of both artifacts. The parent bundle is only read, never written.
//
// A shape refusal (empty fork id, colliding dirs, or a fork id equal to the parent) surfaces
// as a *ForkRefusal carrying its closed reason before anything is written; a load/integrity or
// disk failure surfaces as a wrapped error. Callers branch the two with errors.As.
func ForkDir(parentDir, checkpointDir, forkDir string, opts ForkOptions) (ForkResult, error) {
	forkID := strings.TrimSpace(opts.ForkID)
	if forkID == "" {
		return ForkResult{}, forkRefuse(ForkMalformed, "fork needs a non-empty fork id to re-key the divergent trace")
	}
	// The three dirs must be pairwise distinct: the source is read-only, the checkpoint pins
	// it, and the fork diverges from the checkpoint — any overlap would overwrite one with
	// another (SnapshotDir/BranchDir each reject src==dst too, but naming it here is the honest
	// edge refusal, before either op writes a byte).
	if parentDir == checkpointDir || parentDir == forkDir || checkpointDir == forkDir {
		return ForkResult{}, forkRefuse(ForkMalformed,
			"parent, checkpoint, and fork directories must be distinct (parent=%q checkpoint=%q fork=%q)", parentDir, checkpointDir, forkDir)
	}

	// Load the parent once, up front, so a bad fork id can be refused against the REAL parent id
	// (not merely a self-collision) before any artifact is written. LoadDir fails closed on a
	// truncated or tampered source, so a fork always starts from an integrity-verified bundle.
	parent, err := LoadDir(parentDir)
	if err != nil {
		return ForkResult{}, fmt.Errorf("sessionimage: fork: load parent: %w", err)
	}
	if forkID == parent.Meta.SessionID {
		return ForkResult{}, forkRefuse(ForkSameID,
			"fork id %q equals the parent id; a fork must be a distinct second session (use checkpoint for a same-id capture)", forkID)
	}

	// (1) Pin the branch point: a durable, immutable checkpoint of the parent's current state.
	// SnapshotDir preserves the id and reads the source read-only, so the running session the
	// operator forked is provably unaffected.
	branchPoint, err := SnapshotDir(parentDir, checkpointDir, SnapshotOptions{Reason: opts.Reason, Now: opts.Now})
	if err != nil {
		return ForkResult{}, fmt.Errorf("sessionimage: fork: pin branch point: %w", err)
	}

	// (2) Diverge: fork the pinned checkpoint into a fresh-trace session. BranchDir re-keys the
	// drive to forkID, shares the checkpoint's content-addressed pages copy-on-write, and
	// records the parent_id link + "branched from <parent> at <sha>" lineage.
	fork, err := BranchDir(checkpointDir, forkDir, BranchOptions{
		BranchID:  forkID,
		ToModel:   opts.ToModel,
		ToHost:    opts.ToHost,
		Account:   opts.Account,
		Residency: opts.Residency,
		Reason:    opts.Reason,
		Now:       opts.Now,
	})
	if err != nil {
		return ForkResult{}, fmt.Errorf("sessionimage: fork: branch from checkpoint: %w", err)
	}

	return ForkResult{BranchPoint: branchPoint, Fork: fork}, nil
}
