// Package wiprecon is the pure reconciliation core for #3875 (C3). When a session
// CRASHES — its lease vanishes from refs/fak/locks/* — the WIP checkpoint it left
// under refs/fak/wip/* is orphaned work no live owner is tending. This core decides,
// per checkpoint, the ONE safe action:
//
//   - DISCARD_WITNESSED — the delta already LANDED in HEAD (git-witnessed); dropping
//     the ref loses nothing and the landing commit is the witness.
//   - RECLAIM           — the delta is unlanded but applies cleanly to the current
//     tree; a successor can safely re-materialize it.
//   - QUARANTINE        — the delta is unlanded AND does not apply cleanly (it collides
//     with the tree HEAD moved to, or with live work); set aside for a human, never
//     auto-applied and never dropped.
//   - SKIP              — the owner is still LIVE; its WIP is not ours to reconcile.
//
// The fold is total (one decision per candidate), deterministic (sorted by session),
// and FAIL-SAFE: the only path to DISCARD_WITNESSED requires Landed==true, and any
// unrecognized owner state falls to QUARANTINE. The core never discards or reclaims
// without a positive git-witnessed signal. Pure — no git, no I/O; the cmd shell
// resolves liveness (leaseref), landing, and clean-apply, then folds here.
package wiprecon

import "sort"

// Owner is the resolved liveness of a checkpoint's owning session.
type Owner string

const (
	OwnerLive    Owner = "LIVE"
	OwnerCrashed Owner = "CRASHED"
)

// Action is the closed reconciliation vocabulary.
type Action string

const (
	ActSkip             Action = "SKIP"
	ActDiscardWitnessed Action = "DISCARD_WITNESSED"
	ActReclaim          Action = "RECLAIM"
	ActQuarantine       Action = "QUARANTINE"
)

// Candidate is one checkpoint plus the git-witnessed facts the decision turns on.
// Landed and Applies are consulted only when the owner has CRASHED.
type Candidate struct {
	Session       string
	Owner         Owner
	Landed        bool // the delta is already present in HEAD (git-witnessed)
	Applies       bool // the delta applies cleanly to the current working tree (git apply --check)
	DivergedPaths int  // payload files present on HEAD with different bytes
}

// Decision is the per-candidate verdict.
type Decision struct {
	Session         string   `json:"session"`
	Action          Action   `json:"action"`
	Reason          string   `json:"reason"`
	CheckpointClass string   `json:"checkpoint_class,omitempty"`
	Replication     string   `json:"replication,omitempty"`
	AbsentPaths     int      `json:"absent_paths,omitempty"`
	DivergedPaths   int      `json:"diverged_paths,omitempty"`
	LandedPaths     int      `json:"landed_paths,omitempty"`
	NextCommand     string   `json:"next_command,omitempty"`
	ReviewCommands  []string `json:"review_commands,omitempty"`
}

// Decide is the per-candidate rule. Fail-safe: DISCARD_WITNESSED requires Landed, and
// any owner state that is neither LIVE nor CRASHED falls closed to QUARANTINE.
func Decide(c Candidate) Decision {
	d := Decision{Session: c.Session}
	switch c.Owner {
	case OwnerLive:
		d.Action, d.Reason = ActSkip, "owner still live — not reconciled"
	case OwnerCrashed:
		switch {
		case c.DivergedPaths > 0:
			d.Action, d.Reason = ActQuarantine, "DIVERGED payload differs from HEAD - refuse checkout/apply; inspect git diff HEAD...<checkpoint> -- <path>"
		case c.Landed:
			d.Action, d.Reason = ActDiscardWitnessed, "delta landed in HEAD — safe to drop, witnessed"
		case c.Applies:
			d.Action, d.Reason = ActReclaim, "delta unlanded but applies cleanly — reclaimable"
		default:
			d.Action, d.Reason = ActQuarantine, "delta unlanded and does not apply cleanly — quarantined"
		}
	default:
		d.Action, d.Reason = ActQuarantine, "owner state unknown — quarantined (fail-safe)"
	}
	return d
}

// Reconcile folds every candidate to a decision, sorted by session for determinism.
func Reconcile(cands []Candidate) []Decision {
	out := make([]Decision, 0, len(cands))
	for _, c := range cands {
		out = append(out, Decide(c))
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].Session < out[j].Session })
	return out
}
