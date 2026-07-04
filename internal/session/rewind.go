package session

// rewind.go — workspace restore ("time travel") gated by the DOS lane arbiter.
//
// A workspace restore is a BULK WRITE: it re-applies a checkpoint tree onto the
// live workspace. On a shared multi-session trunk — fak's operating reality, and
// the failure class witnessed in #2320, where a stale snapshot silently reverted
// a peer's just-pushed files — that bulk write can clobber a fleet-mate mid-flight.
// Before any tree mutation, Rewind consults the DOS lane arbiter
// (internal/laneadmit, the in-binary twin of `dos arbitrate`, over the dos.toml
// lane taxonomy) over the change set — the tree the restore would touch — and
// refuses with the arbiter's closed lane-conflict reason (COLLISION_RISK) when a
// live intersecting lease holds it, naming the holder. The Applier is NOT called
// on a refusal, so zero files are modified. An operator force path clears the
// geometric / same-lane conflicts but still respects a live EXCLUSIVE lane (the
// arbiter's force semantics). Every refusal, force, and admission is journaled.
//
// This is the same admission discipline applied to worker dispatch (#1501),
// applied to time travel: rollback structurally cannot clobber a fleet-mate. The
// change set itself (checkpoint tree vs current tree) is the caller's concern —
// Rewind arbitrates the tree it is handed, so the gate composes with any
// checkpoint format the snapshot/sessionimage layers produce.

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/laneadmit"
)

// SurfaceRestore is the execution-surface name a workspace restore identifies as
// when it asks the lane arbiter for admission (the restore/time-travel surface,
// alongside laneadmit's SurfaceDispatch / SurfaceLoop / SurfaceManual).
const SurfaceRestore = "restore"

// Closed-vocabulary event kinds recorded on the rewind ledger.
const (
	EvRewindRefused  = "refused"  // the arbiter refused the change set; zero files modified
	EvRewindForced   = "forced"   // an operator force cleared the non-exclusive conflicts and the restore proceeded
	EvRewindAdmitted = "admitted" // the change set was disjoint (or the caller's own lease); the restore proceeded
)

// RewindVerdict is the workspace-restore decision. A refusal carries the
// arbiter's closed lane-conflict reason (COLLISION_RISK) plus the conflicting
// live leases as evidence — each naming its holder, the surface a refused
// operator reads (#2297).
type RewindVerdict struct {
	Admit     bool                 `json:"admit"`
	Reason    string               `json:"reason,omitempty"` // the arbiter's closed lane-conflict reason (COLLISION_RISK) on a refusal
	Detail    string               `json:"detail,omitempty"`
	Tree      []string             `json:"tree,omitempty"` // the change set the decision was made against (after taxonomy fallback)
	Conflicts []laneadmit.Conflict `json:"conflicts,omitempty"`
	Forced    bool                 `json:"forced,omitempty"` // true when an operator force path cleared the non-exclusive conflicts
}

// RewindApplier is the bulk-write step — the actual re-application of the
// checkpoint tree onto the workspace. Rewind calls Apply ONLY after the arbiter
// admits the change set (or an operator force clears it), so a refused restore
// modifies zero files. A nil Applier makes Rewind a pure admission decision.
type RewindApplier interface {
	Apply() error
}

// RewindJournal is the ledger a refusal / force / admission is recorded on. It
// is the same shape every other fak execution surface journals through (an
// append-only event log); Rewind never invents a free-text kind.
type RewindJournal interface {
	Record(e RewindEvent) error
}

// RewindEvent is one journaled rewind record.
type RewindEvent struct {
	Kind   string    `json:"kind"`             // EvRewindRefused / EvRewindForced / EvRewindAdmitted
	Holder string    `json:"holder,omitempty"` // the operator/agent that requested the restore
	Tree   []string  `json:"tree,omitempty"`   // the change set the decision was made against
	Reason string    `json:"reason,omitempty"` // the closed lane-conflict reason on a refusal
	At     time.Time `json:"at"`               // when the decision was made
}

// RewindInput configures a workspace restore.
type RewindInput struct {
	Holder   string             // the operator/agent requesting the restore
	Lane     string             // named dos.toml lane the restore acts on ("" = a tree-only request)
	LeaseID  string             // the restore's own lease id; a live lease with this id is the caller's own and never conflicts
	Tree     []string           // the change set: repo-relative globs the restore would touch
	Leases   []laneadmit.Lease  // the live leases (projected from refs/fak/locks/* via leaseref)
	Taxonomy laneadmit.Taxonomy // the dos.toml lane taxonomy
	Force    bool               // operator force: clears geometric/same-lane conflicts, still refuses over a live EXCLUSIVE lane
	Applier  RewindApplier      // the bulk-write step (nil => admission decision only, no apply)
	Journal  RewindJournal      // the ledger (nil => no journaling)
	Now      time.Time          // injected clock for deterministic journal stamps (zero => time.Now)
}

// Rewind is the workspace-restore handler. It consults the DOS lane arbiter over
// the change set BEFORE any tree mutation:
//
//   - a live intersecting lease refuses the restore with the arbiter's closed
//     lane-conflict reason (COLLISION_RISK), naming the holder, and zero files
//     are modified (the Applier is not called);
//   - a lease over a disjoint tree does not block, and the Applier runs;
//   - an operator force (Force=true) clears the geometric / same-lane conflicts
//     but still refuses over a live EXCLUSIVE lane.
//
// The returned verdict is the decision; a nil error means the gate ran. An
// Apply error is returned verbatim and only after the arbiter admitted the
// change set (so an Apply error is proof the restore was permitted to start).
func Rewind(in RewindInput) (*RewindVerdict, error) {
	now := in.Now
	if now.IsZero() {
		now = time.Now()
	}

	v := laneadmit.Decide(laneadmit.Request{
		Surface: SurfaceRestore,
		Lane:    in.Lane,
		Tree:    in.Tree,
		Holder:  in.Holder,
		LeaseID: in.LeaseID,
	}, in.Leases, in.Taxonomy)

	out := &RewindVerdict{
		Admit:     v.Admit,
		Reason:    v.Reason,
		Detail:    v.Detail,
		Tree:      v.Tree,
		Conflicts: v.Conflicts,
	}

	if out.Admit {
		if in.Journal != nil {
			if err := in.Journal.Record(RewindEvent{Kind: EvRewindAdmitted, Holder: in.Holder, Tree: v.Tree, At: now}); err != nil {
				return out, err
			}
		}
		if in.Applier != nil {
			if err := in.Applier.Apply(); err != nil {
				return out, err
			}
		}
		return out, nil
	}

	// Refused. An operator force path clears the geometric / same-lane conflicts
	// but still respects a live EXCLUSIVE lane — the arbiter's force semantics
	// ("still refuses over a live exclusive lane"). A live exclusive holder is the
	// one peer a forced restore must never race.
	if in.Force {
		kept := make([]laneadmit.Conflict, 0, len(out.Conflicts))
		for _, c := range out.Conflicts {
			if c.Kind == laneadmit.ConflictExclusiveLane {
				kept = append(kept, c)
			}
		}
		if len(kept) == 0 {
			out.Admit = true
			out.Forced = true
			out.Reason = ""
			out.Detail = ""
			out.Conflicts = nil
			if in.Journal != nil {
				if err := in.Journal.Record(RewindEvent{Kind: EvRewindForced, Holder: in.Holder, Tree: v.Tree, At: now}); err != nil {
					return out, err
				}
			}
			if in.Applier != nil {
				if err := in.Applier.Apply(); err != nil {
					return out, err
				}
			}
			return out, nil
		}
		out.Conflicts = kept
		out.Detail = fmt.Sprintf(
			"forced workspace restore over tree %v still refused: live exclusive lease %s (holder %q) runs alone",
			v.Tree, kept[0].LeaseID, kept[0].Holder)
	}

	// Hard refusal: the Applier is never called, so zero files are modified.
	if in.Journal != nil {
		if err := in.Journal.Record(RewindEvent{Kind: EvRewindRefused, Holder: in.Holder, Tree: v.Tree, Reason: v.Reason, At: now}); err != nil {
			return out, err
		}
	}
	return out, nil
}
