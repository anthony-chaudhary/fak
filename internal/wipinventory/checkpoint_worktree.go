package wipinventory

import (
	"fmt"
	"sort"
	"strings"
)

// CheckpointSnapshotScope describes what a checkpoint captured. A shared-tree
// snapshot is never evidence that its session owns every contained path.
type CheckpointSnapshotScope string

const (
	CheckpointSnapshotOwned  CheckpointSnapshotScope = "owned_paths"
	CheckpointSnapshotShared CheckpointSnapshotScope = "shared_tree"
)

// CheckpointWIPBinding declares checkpoint ownership through session and lease
// provenance. Paths are diagnostic only and never participate in a join key.
type CheckpointWIPBinding struct {
	CheckpointID string                  `json:"checkpoint_id"`
	WIPUnitID    WIPUnitID               `json:"wip_unit_id"`
	SessionID    string                  `json:"session_id"`
	Lane         string                  `json:"lane"`
	LeaseID      string                  `json:"lease_id"`
	Registered   bool                    `json:"registered"`
	Scope        CheckpointSnapshotScope `json:"scope"`
	Paths        []string                `json:"paths,omitempty"`
	ForeignPaths []string                `json:"foreign_paths,omitempty"`
}

// LiveLaneLease is authoritative lease provenance supplied by the lease
// registry. Live is explicit; timestamps are not interpreted here.
type LiveLaneLease struct {
	LeaseID   string    `json:"lease_id"`
	Lane      string    `json:"lane"`
	WIPUnitID WIPUnitID `json:"wip_unit_id"`
	SessionID string    `json:"session_id,omitempty"`
	WorkerID  string    `json:"worker_id,omitempty"`
	Live      bool      `json:"live"`
}

// CheckpointWorktreeDebtKind is a stable classification for provenance that
// cannot safely be deduplicated.
type CheckpointWorktreeDebtKind string

const (
	DebtSharedSnapshot      CheckpointWorktreeDebtKind = "shared_snapshot"
	DebtForeignPaths        CheckpointWorktreeDebtKind = "foreign_paths"
	DebtMissingRegistration CheckpointWorktreeDebtKind = "missing_registration"
	DebtStaleLease          CheckpointWorktreeDebtKind = "stale_lease"
	DebtConflictingOwners   CheckpointWorktreeDebtKind = "conflicting_owners"
)

// CheckpointWorktreeDebt preserves an artifact as visible typed debt.
type CheckpointWorktreeDebt struct {
	Kind       CheckpointWorktreeDebtKind `json:"kind"`
	Artifact   string                     `json:"artifact"`
	WIPUnitIDs []WIPUnitID                `json:"wip_unit_ids,omitempty"`
	Detail     string                     `json:"detail"`
}

// CheckpointWorktreeUnit is one deduplicated logical unit and its explicitly
// joined representations.
type CheckpointWorktreeUnit struct {
	WIPUnitID  WIPUnitID `json:"wip_unit_id,omitempty"`
	Checkpoint string    `json:"checkpoint,omitempty"`
	Lease      string    `json:"lease,omitempty"`
	Worktree   string    `json:"worktree,omitempty"`
}

// CheckpointWorktreeJoin is a deterministic, read-only reconciliation result.
type CheckpointWorktreeJoin struct {
	LogicalCount int                      `json:"logical_count"`
	Units        []CheckpointWorktreeUnit `json:"units"`
	Debt         []CheckpointWorktreeDebt `json:"debt,omitempty"`
}

// ManagedWorktreeWIPBinding is the narrow provenance projection required by the
// inventory join. Implementations expose declared receipt fields only.
type ManagedWorktreeWIPBinding interface {
	WIPWorktreeID() string
	WIPUnit() string
	WIPWorkerID() string
	WIPLane() string
	WIPLeaseID() string
	WIPRegistered() bool
}

// JoinCheckpointWorktrees joins only explicit checkpoint, lease, and managed
// worktree receipts. It does not inspect or mutate repositories or artifacts.
func JoinCheckpointWorktrees[T ManagedWorktreeWIPBinding](checkpoints []CheckpointWIPBinding, leases []LiveLaneLease, worktrees []T) CheckpointWorktreeJoin {
	result := CheckpointWorktreeJoin{}
	units := make(map[string]*CheckpointWorktreeUnit)
	leaseByKey := make(map[string]LiveLaneLease)
	conflictingLease := make(map[string]bool)

	addUnit := func(key string, id WIPUnitID) *CheckpointWorktreeUnit {
		if units[key] == nil {
			units[key] = &CheckpointWorktreeUnit{WIPUnitID: id}
		}
		return units[key]
	}
	addDebt := func(kind CheckpointWorktreeDebtKind, artifact, detail string, ids ...WIPUnitID) {
		ids = compactSortedIDs(ids)
		result.Debt = append(result.Debt, CheckpointWorktreeDebt{Kind: kind, Artifact: artifact, WIPUnitIDs: ids, Detail: detail})
	}

	// Detect conflicting authoritative identifiers before joining anything to
	// them. Once a lane/lease pair names multiple owners, none of its
	// representations may be collapsed.
	for _, lease := range leases {
		if lease.LeaseID == "" || lease.Lane == "" || lease.WIPUnitID.Validate() != nil {
			continue
		}
		key := leaseKey(lease.Lane, lease.LeaseID)
		if prior, ok := leaseByKey[key]; ok && (prior.WIPUnitID != lease.WIPUnitID || incompatibleOwner(prior.SessionID, lease.SessionID) || incompatibleOwner(prior.WorkerID, lease.WorkerID)) {
			conflictingLease[key] = true
		}
		leaseByKey[key] = lease
	}

	for i, lease := range leases {
		artifact := fmt.Sprintf("lease:%s#%d", lease.LeaseID, i)
		if lease.LeaseID == "" || lease.Lane == "" || lease.WIPUnitID.Validate() != nil {
			addUnit("artifact:"+artifact, "").Lease = lease.LeaseID
			addDebt(DebtMissingRegistration, artifact, "lease lacks an authoritative lane, lease ID, or WIP unit registration")
			continue
		}
		key := leaseKey(lease.Lane, lease.LeaseID)
		if conflictingLease[key] {
			addUnit("artifact:"+artifact, lease.WIPUnitID).Lease = lease.LeaseID
			owners := []WIPUnitID{lease.WIPUnitID}
			for _, other := range leases {
				if leaseKey(other.Lane, other.LeaseID) == key {
					owners = append(owners, other.WIPUnitID)
				}
			}
			addDebt(DebtConflictingOwners, artifact, "the same lane lease declares multiple owners", owners...)
			continue
		}
		if !lease.Live {
			addUnit("artifact:"+artifact, lease.WIPUnitID).Lease = lease.LeaseID
			addDebt(DebtStaleLease, artifact, "lease registry marks the lease stale", lease.WIPUnitID)
			continue
		}
		addUnit("unit:"+string(lease.WIPUnitID), lease.WIPUnitID).Lease = lease.LeaseID
	}

	for _, checkpoint := range checkpoints {
		artifact := "checkpoint:" + checkpoint.CheckpointID
		valid := checkpoint.Registered && checkpoint.CheckpointID != "" && checkpoint.SessionID != "" && checkpoint.Lane != "" && checkpoint.LeaseID != "" && checkpoint.WIPUnitID.Validate() == nil
		if !valid {
			addUnit("artifact:"+artifact, checkpoint.WIPUnitID).Checkpoint = checkpoint.CheckpointID
			addDebt(DebtMissingRegistration, artifact, "checkpoint lacks explicit session/lease/WIP registration")
			continue
		}
		if checkpoint.Scope == CheckpointSnapshotShared {
			addDebt(DebtSharedSnapshot, artifact, "shared-tree contents are diagnostic and do not establish path ownership", checkpoint.WIPUnitID)
		}
		if len(checkpoint.ForeignPaths) > 0 {
			paths := append([]string(nil), checkpoint.ForeignPaths...)
			sort.Strings(paths)
			addDebt(DebtForeignPaths, artifact, "foreign paths are not owned by this checkpoint: "+strings.Join(paths, ", "), checkpoint.WIPUnitID)
		}
		key := leaseKey(checkpoint.Lane, checkpoint.LeaseID)
		lease, ok := leaseByKey[key]
		switch {
		case conflictingLease[key]:
			addUnit("artifact:"+artifact, checkpoint.WIPUnitID).Checkpoint = checkpoint.CheckpointID
			addDebt(DebtConflictingOwners, artifact, "checkpoint references a lease with conflicting owners", checkpoint.WIPUnitID)
		case !ok || !lease.Live:
			addUnit("artifact:"+artifact, checkpoint.WIPUnitID).Checkpoint = checkpoint.CheckpointID
			addDebt(DebtStaleLease, artifact, "checkpoint has no matching live lease", checkpoint.WIPUnitID)
		case lease.WIPUnitID != checkpoint.WIPUnitID || incompatibleOwner(lease.SessionID, checkpoint.SessionID):
			addUnit("artifact:"+artifact, checkpoint.WIPUnitID).Checkpoint = checkpoint.CheckpointID
			addDebt(DebtConflictingOwners, artifact, "checkpoint and lease declare conflicting owners", checkpoint.WIPUnitID, lease.WIPUnitID)
		default:
			addUnit("unit:"+string(checkpoint.WIPUnitID), checkpoint.WIPUnitID).Checkpoint = checkpoint.CheckpointID
		}
	}

	for _, worktree := range worktrees {
		artifact := "worktree:" + worktree.WIPWorktreeID()
		id, err := ParseWIPUnitID(worktree.WIPUnit())
		valid := worktree.WIPRegistered() && worktree.WIPWorktreeID() != "" && worktree.WIPWorkerID() != "" && worktree.WIPLane() != "" && worktree.WIPLeaseID() != "" && err == nil
		if !valid {
			addUnit("artifact:"+artifact, id).Worktree = worktree.WIPWorktreeID()
			addDebt(DebtMissingRegistration, artifact, "managed worktree lacks explicit worker/lane/lease/WIP registration")
			continue
		}
		key := leaseKey(worktree.WIPLane(), worktree.WIPLeaseID())
		lease, ok := leaseByKey[key]
		switch {
		case conflictingLease[key]:
			addUnit("artifact:"+artifact, id).Worktree = worktree.WIPWorktreeID()
			addDebt(DebtConflictingOwners, artifact, "managed worktree references a lease with conflicting owners", id)
		case !ok || !lease.Live:
			addUnit("artifact:"+artifact, id).Worktree = worktree.WIPWorktreeID()
			addDebt(DebtStaleLease, artifact, "managed worktree has no matching live lease", id)
		case lease.WIPUnitID != id || incompatibleOwner(lease.WorkerID, worktree.WIPWorkerID()):
			addUnit("artifact:"+artifact, id).Worktree = worktree.WIPWorktreeID()
			addDebt(DebtConflictingOwners, artifact, "managed worktree and lease declare conflicting owners", id, lease.WIPUnitID)
		default:
			addUnit("unit:"+string(id), id).Worktree = worktree.WIPWorktreeID()
		}
	}

	keys := make([]string, 0, len(units))
	for key := range units {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	for _, key := range keys {
		result.Units = append(result.Units, *units[key])
	}
	result.LogicalCount = len(result.Units)
	sort.Slice(result.Debt, func(i, j int) bool {
		a, b := result.Debt[i], result.Debt[j]
		return fmt.Sprintf("%s\x00%s\x00%s", a.Kind, a.Artifact, a.Detail) < fmt.Sprintf("%s\x00%s\x00%s", b.Kind, b.Artifact, b.Detail)
	})
	return result
}

func incompatibleOwner(authoritative, declared string) bool {
	return authoritative != "" && declared != "" && authoritative != declared
}

func leaseKey(lane, lease string) string { return lane + "\x00" + lease }

func compactSortedIDs(ids []WIPUnitID) []WIPUnitID {
	seen := make(map[WIPUnitID]bool)
	out := make([]WIPUnitID, 0, len(ids))
	for _, id := range ids {
		if id != "" && !seen[id] {
			seen[id] = true
			out = append(out, id)
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}
