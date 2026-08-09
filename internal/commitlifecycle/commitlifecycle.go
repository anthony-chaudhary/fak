// Package commitlifecycle folds independently witnessed repository facts into the
// one next safe move from agent-authored edits to a remotely witnessed ship.
// It deliberately performs no git, filesystem, lease, or process I/O.
package commitlifecycle

// State is a closed lifecycle state. Unknown is a fail-closed contradiction or
// a row for which the caller has not supplied enough witnessed facts.
type State string

const (
	Editing           State = "EDITING"
	CommitReady       State = "COMMIT_READY"
	CommittedUnpushed State = "COMMITTED_UNPUSHED"
	Parked            State = "PARKED"
	Reclaim           State = "RECLAIM"
	LandReady         State = "LAND_READY"
	LandedUnpushed    State = "LANDED_UNPUSHED"
	Shipped           State = "SHIPPED"
	Unknown           State = "UNKNOWN"
)

// Action is argv, not a shell command. NeedsOperator means that advancing the
// row requires information or a decision which may not be synthesized safely.
type Action struct {
	Tool          string   `json:"tool,omitempty"`
	Args          []string `json:"args,omitempty"`
	NeedsOperator bool     `json:"needs_operator,omitempty"`
	Reason        string   `json:"reason,omitempty"`
}

// Facts contains only facts witnessed by callers. A boolean never asks this
// package to infer git state. IDs are carried into existing commands verbatim.
type Facts struct {
	DirtyPaths      []string
	CommitArgs      []string // complete args after "fak commit"; empty means no safe contract
	LocalCommit     string
	LocalOnRemote   bool
	Checkpoint      string // refs/fak/wip session id accepted by `fak wip reconcile adopt`
	CheckpointLive  bool
	CheckpointApply bool
	// CheckpointAdoptedBy is the successor session holding this checkpoint's adoption
	// receipt (#5998); "" means unclaimed. CheckpointAdoptMine narrows that to "and it is
	// me", and CheckpointAdoptExpired to "and its claim lapsed with its holder gone".
	// Together they are what keeps this queue from handing one checkpoint to two agents:
	// a row a live peer already claimed is real work, but not the reader's work.
	CheckpointAdoptedBy    string
	CheckpointAdoptMine    bool
	CheckpointAdoptExpired bool
	WorkerPath             string
	WorkerLandReady        bool
	LandedCommit           string
	LandedOnRemote         bool
}

// Row is the fold result. Every non-terminal row has either executable argv or
// an explicit operator gate. SHIPPED alone has neither.
type Row struct {
	State  State  `json:"state"`
	Action Action `json:"action,omitempty"`
}

// Fold applies terminal-first precedence after rejecting contradictory facts.
func Fold(f Facts) Row {
	if reason := contradiction(f); reason != "" {
		return gated(Unknown, reason)
	}
	if f.LocalOnRemote || f.LandedOnRemote {
		return Row{State: Shipped}
	}
	if f.LandedCommit != "" {
		return Row{State: LandedUnpushed, Action: Action{Tool: "fak", Args: []string{"sync", "push"}}}
	}
	if f.WorkerLandReady {
		return Row{State: LandReady, Action: Action{Tool: "fak", Args: []string{"worktree", "worker", "land", "--path", f.WorkerPath}}}
	}
	if f.Checkpoint != "" {
		if f.CheckpointLive {
			return gated(Parked, "checkpoint owner is live; wait for its lease to close or intervene explicitly")
		}
		if f.CheckpointApply {
			// The argv this row names is the ADOPTION command (#5998), not a bare land.
			// Two reasons, and the second is why the old `wip land <id> --apply` here was
			// worse than merely wrong: a shared queue that tells every reader to land the
			// same checkpoint invites two of them to do it, and `--apply` is a flag of
			// `fak wip restore`, not of `fak wip land`, so that argv exits 2 whichever way
			// it is ordered — unparsed as a second positional, undefined once reordered.
			// A printed command that cannot run is indistinguishable, to a dispatcher,
			// from one that can.
			switch {
			case f.CheckpointAdoptMine:
				return Row{State: LandReady, Action: Action{Tool: "fak", Args: []string{"wip", "reconcile", "resume", f.Checkpoint}}}
			case f.CheckpointAdoptedBy != "" && !f.CheckpointAdoptExpired:
				return gated(Parked, "checkpoint is adopted by "+f.CheckpointAdoptedBy+"; wait for that claim to lapse or take it over explicitly")
			default:
				return Row{State: LandReady, Action: Action{Tool: "fak", Args: []string{"wip", "reconcile", "adopt", f.Checkpoint}}}
			}
		}
		return Row{State: Reclaim, Action: Action{Tool: "fak", Args: []string{"wip", "reconcile", "--reclaim"}}}
	}
	if f.LocalCommit != "" {
		return Row{State: CommittedUnpushed, Action: Action{Tool: "fak", Args: []string{"sync", "push"}}}
	}
	if len(f.DirtyPaths) > 0 {
		if len(f.CommitArgs) == 0 {
			return gated(Editing, "dirty paths have no witnessed issue, message, and explicit path contract")
		}
		return Row{State: CommitReady, Action: Action{Tool: "fak", Args: append([]string{"commit"}, f.CommitArgs...)}}
	}
	return gated(Unknown, "no edit, commit, checkpoint, worker-land, or remote-ancestry witness")
}

func contradiction(f Facts) string {
	if f.LocalOnRemote && f.LocalCommit == "" {
		return "local remote-ancestry witness has no local commit"
	}
	if f.LandedOnRemote && f.LandedCommit == "" {
		return "landed remote-ancestry witness has no landed commit"
	}
	if (f.CheckpointLive || f.CheckpointApply) && f.Checkpoint == "" {
		return "checkpoint state has no checkpoint identity"
	}
	if (f.CheckpointAdoptedBy != "" || f.CheckpointAdoptMine || f.CheckpointAdoptExpired) && f.Checkpoint == "" {
		return "adoption state has no checkpoint identity"
	}
	if (f.CheckpointAdoptMine || f.CheckpointAdoptExpired) && f.CheckpointAdoptedBy == "" {
		return "adoption state has no adopting successor"
	}
	if f.WorkerLandReady && f.WorkerPath == "" {
		return "worker land witness has no worker path"
	}
	if len(f.CommitArgs) > 0 && len(f.DirtyPaths) == 0 {
		return "commit contract has no dirty paths"
	}
	if f.CheckpointLive && f.CheckpointApply {
		return "live checkpoint must not be offered to the shared-tree lander"
	}
	return ""
}

func gated(state State, reason string) Row {
	return Row{State: state, Action: Action{NeedsOperator: true, Reason: reason}}
}
