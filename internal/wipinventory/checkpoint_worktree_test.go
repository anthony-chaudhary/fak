package wipinventory

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	unitA = WIPUnitID("wip:v1:11111111111111111111111111111111")
	unitB = WIPUnitID("wip:v1:22222222222222222222222222222222")
)

func TestJoinCheckpointWorktreesCountsExplicitLiveChainOnce(t *testing.T) {
	got := JoinCheckpointWorktrees(
		[]CheckpointWIPBinding{{CheckpointID: "cp-1", WIPUnitID: unitA, SessionID: "session-1", Lane: "lane-a", LeaseID: "lease-1", Registered: true, Scope: CheckpointSnapshotOwned}},
		[]LiveLaneLease{{LeaseID: "lease-1", Lane: "lane-a", WIPUnitID: unitA, SessionID: "session-1", WorkerID: "worker-1", Live: true}},
		[]workerworktree.WIPBinding{{Schema: workerworktree.WIPBindingSchema, WorktreeID: "wt-1", WIPUnitID: string(unitA), WorkerID: "worker-1", Lane: "lane-a", LeaseID: "lease-1", Registered: true, Dirty: true}},
	)
	if got.LogicalCount != 1 || len(got.Units) != 1 {
		t.Fatalf("joined count = %d, units = %#v", got.LogicalCount, got.Units)
	}
	want := CheckpointWorktreeUnit{WIPUnitID: unitA, Checkpoint: "cp-1", Lease: "lease-1", Worktree: "wt-1"}
	if got.Units[0] != want {
		t.Fatalf("unit = %#v, want %#v", got.Units[0], want)
	}
	if len(got.Debt) != 0 {
		t.Fatalf("unexpected debt: %#v", got.Debt)
	}
}

func TestJoinCheckpointWorktreesClassifiesSharedForeignStaleAndOwnerless(t *testing.T) {
	got := JoinCheckpointWorktrees(
		[]CheckpointWIPBinding{
			{CheckpointID: "cp-shared", WIPUnitID: unitA, SessionID: "session-1", Lane: "lane-a", LeaseID: "lease-stale", Registered: true, Scope: CheckpointSnapshotShared, Paths: []string{"owned.go", "peer.go"}, ForeignPaths: []string{"peer.go"}},
			{CheckpointID: "cp-ownerless"},
		},
		[]LiveLaneLease{{LeaseID: "lease-stale", Lane: "lane-a", WIPUnitID: unitA, SessionID: "session-1", Live: false}},
		[]workerworktree.WIPBinding{{Schema: workerworktree.WIPBindingSchema, WorktreeID: "wt-ownerless", Dirty: true}},
	)
	if got.LogicalCount != 4 {
		t.Fatalf("logical count = %d, want 4: %#v", got.LogicalCount, got.Units)
	}
	assertDebtKinds(t, got, DebtForeignPaths, DebtMissingRegistration, DebtSharedSnapshot, DebtStaleLease)
}

func TestJoinCheckpointWorktreesConflictingOwnersDoNotDeduplicate(t *testing.T) {
	got := JoinCheckpointWorktrees(
		[]CheckpointWIPBinding{{CheckpointID: "cp", WIPUnitID: unitA, SessionID: "session-a", Lane: "lane", LeaseID: "lease", Registered: true, Scope: CheckpointSnapshotOwned}},
		[]LiveLaneLease{{LeaseID: "lease", Lane: "lane", WIPUnitID: unitB, SessionID: "session-b", WorkerID: "worker-b", Live: true}},
		[]workerworktree.WIPBinding{{Schema: workerworktree.WIPBindingSchema, WorktreeID: "wt", WIPUnitID: string(unitB), WorkerID: "worker-b", Lane: "lane", LeaseID: "lease", Registered: true, Dirty: true}},
	)
	if got.LogicalCount != 2 {
		t.Fatalf("conflicting owners deduplicated: count=%d units=%#v", got.LogicalCount, got.Units)
	}
	assertDebtKinds(t, got, DebtConflictingOwners)
}

func TestJoinCheckpointWorktreesDeterministicAndIgnoresSimilarity(t *testing.T) {
	checkpoints := []CheckpointWIPBinding{
		{CheckpointID: "z", WIPUnitID: unitB, SessionID: "s2", Lane: "lane-b", LeaseID: "l2", Registered: true, Scope: CheckpointSnapshotOwned, Paths: []string{"same.go"}},
		{CheckpointID: "a", WIPUnitID: unitA, SessionID: "s1", Lane: "lane-a", LeaseID: "l1", Registered: true, Scope: CheckpointSnapshotOwned, Paths: []string{"same.go"}},
	}
	leases := []LiveLaneLease{
		{LeaseID: "l2", Lane: "lane-b", WIPUnitID: unitB, SessionID: "s2", WorkerID: "w2", Live: true},
		{LeaseID: "l1", Lane: "lane-a", WIPUnitID: unitA, SessionID: "s1", WorkerID: "w1", Live: true},
	}
	worktrees := []workerworktree.WIPBinding{
		{Schema: workerworktree.WIPBindingSchema, WorktreeID: "wt-z", WIPUnitID: string(unitB), WorkerID: "w2", Lane: "lane-b", LeaseID: "l2", Registered: true, Dirty: true},
		{Schema: workerworktree.WIPBindingSchema, WorktreeID: "wt-a", WIPUnitID: string(unitA), WorkerID: "w1", Lane: "lane-a", LeaseID: "l1", Registered: true, Dirty: true},
	}
	first := JoinCheckpointWorktrees(checkpoints, leases, worktrees)
	second := JoinCheckpointWorktrees(
		[]CheckpointWIPBinding{checkpoints[1], checkpoints[0]},
		[]LiveLaneLease{leases[1], leases[0]},
		[]workerworktree.WIPBinding{worktrees[1], worktrees[0]},
	)
	firstJSON, _ := json.Marshal(first)
	secondJSON, _ := json.Marshal(second)
	if !reflect.DeepEqual(firstJSON, secondJSON) {
		t.Fatalf("nondeterministic:\n%s\n%s", firstJSON, secondJSON)
	}
	if first.LogicalCount != 2 {
		t.Fatalf("matching paths incorrectly established ownership: %#v", first)
	}
}

func assertDebtKinds(t *testing.T, got CheckpointWorktreeJoin, wants ...CheckpointWorktreeDebtKind) {
	t.Helper()
	seen := make(map[CheckpointWorktreeDebtKind]bool)
	for _, debt := range got.Debt {
		seen[debt.Kind] = true
	}
	for _, want := range wants {
		if !seen[want] {
			t.Errorf("missing debt %q in %#v", want, got.Debt)
		}
	}
}
