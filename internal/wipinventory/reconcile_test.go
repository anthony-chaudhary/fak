package wipinventory

import (
	"bytes"
	"context"
	"encoding/json"
	"os/exec"
	"strings"
	"testing"
	"time"
)

func TestReconcileInventoryMixedFixture(t *testing.T) {
	fixedTime := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	unitJoined := WIPUnitID("wip:v1:11111111111111111111111111111111")
	unitTerminal := WIPUnitID("wip:v1:22222222222222222222222222222222")
	unitStale := WIPUnitID("wip:v1:33333333333333333333333333333333")
	unitConflict1 := WIPUnitID("wip:v1:44444444444444444444444444444444")
	unitConflict2 := WIPUnitID("wip:v1:54444444444444444444444444444444")
	unitSingle := WIPUnitID("wip:v1:66666666666666666666666666666666")

	histories := []History{
		{
			Schema: WIPUnitSchema,
			Transitions: []Transition{
				// unitJoined lifecycle
				{Kind: TransitionCreate, Timestamp: fixedTime, Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Successors: []WIPUnitID{unitJoined}, Witness: "w-create-1"},
				{Kind: TransitionBind, Timestamp: fixedTime.Add(time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Predecessors: []WIPUnitID{unitJoined}, Successors: []WIPUnitID{unitJoined}, References: []SurfaceReference{{Kind: SurfaceIssue, Issue: &IssueReference{Repository: "fak-repo", Number: 101}}}, Witness: "w-bind-1"},
				{Kind: TransitionHandoff, Timestamp: fixedTime.Add(2 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Predecessors: []WIPUnitID{unitJoined}, Successors: []WIPUnitID{unitJoined}, References: []SurfaceReference{{Kind: SurfaceDispatchSession, DispatchSession: &DispatchSessionReference{SessionID: "sess-joined"}}}, Witness: "w-handoff-1"},

				// unitTerminal lifecycle
				{Kind: TransitionCreate, Timestamp: fixedTime.Add(3 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Successors: []WIPUnitID{unitTerminal}, Witness: "w-create-2"},
				{Kind: TransitionLand, Timestamp: fixedTime.Add(4 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Predecessors: []WIPUnitID{unitTerminal}, Successors: []WIPUnitID{unitTerminal}, References: []SurfaceReference{{Kind: SurfaceWitnessedRetirement, WitnessedRetirement: &WitnessedRetirementReference{RetirementID: "retire-1", Witness: "commit:abc"}}}, Witness: "w-land-2"},

				// other units
				{Kind: TransitionCreate, Timestamp: fixedTime.Add(5 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Successors: []WIPUnitID{unitConflict1}, Witness: "w-create-3"},
				{Kind: TransitionCreate, Timestamp: fixedTime.Add(6 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Successors: []WIPUnitID{unitConflict2}, Witness: "w-create-4"},
				{Kind: TransitionCreate, Timestamp: fixedTime.Add(7 * time.Minute), Source: "fixture", Provenance: Provenance{Actor: "agent", Mechanism: "test"}, Successors: []WIPUnitID{unitSingle}, Witness: "w-create-5"},
			},
		},
	}

	bindings := ExecutionBindingReport{
		Bindings: []ExecutionBinding{
			{
				RootRegistrationID: "reg-joined",
				Issue:              &ExecutionIssueIdentity{Repository: "fak-repo", Number: 101},
				SessionIDs:         []string{"sess-joined"},
				Status:             ExecutionBindingJoined,
			},
			{
				RootRegistrationID: "reg-unjoined",
				SessionIDs:         []string{"sess-unjoined"},
				Status:             ExecutionBindingMissing,
				Details:            []string{"unjoined worker session"},
			},
		},
	}

	checkpoints := []CheckpointWIPBinding{
		{
			CheckpointID: "cp-joined",
			WIPUnitID:    unitJoined,
			SessionID:    "sess-joined",
			Lane:         "lane-1",
			LeaseID:      "lease-1",
			Registered:   true,
			Scope:        CheckpointSnapshotOwned,
		},
		{
			CheckpointID: "cp-ownerless",
			Registered:   false,
		},
		{
			CheckpointID: "cp-conflict",
			WIPUnitID:    unitConflict1,
			SessionID:    "sess-conflict",
			Lane:         "lane-conflict",
			LeaseID:      "lease-conflict",
			Registered:   true,
			Scope:        CheckpointSnapshotOwned,
		},
	}

	leases := []LiveLaneLease{
		{
			LeaseID:   "lease-1",
			Lane:      "lane-1",
			WIPUnitID: unitJoined,
			SessionID: "sess-joined",
			WorkerID:  "worker-1",
			Live:      true,
		},
		{
			LeaseID:   "lease-stale",
			Lane:      "lane-stale",
			WIPUnitID: unitStale,
			SessionID: "sess-stale",
			Live:      false,
		},
		{
			LeaseID:   "lease-conflict",
			Lane:      "lane-conflict",
			WIPUnitID: unitConflict2,
			SessionID: "sess-conflict-b",
			WorkerID:  "worker-b",
			Live:      true,
		},
	}

	worktrees := []ManagedWorktreeBinding{
		{
			WorktreeID: "wt-joined",
			WIPUnitID:  string(unitJoined),
			WorkerID:   "worker-1",
			Lane:       "lane-1",
			LeaseID:    "lease-1",
			Registered: true,
		},
		{
			WorktreeID: "wt-ownerless",
			Registered: false,
		},
	}

	inputs := InventoryInputs{
		Histories:     histories,
		Bindings:      bindings,
		Checkpoints:   checkpoints,
		Leases:        leases,
		Worktrees:     worktrees,
		UnlinkedFiles: []string{"unlinked_foo.go"},
		UnlinkedDirs:  []string{"_scratch/orphan_run"},
		Now:           fixedTime,
		HEAD:          "deadbeef1234",
	}

	report, err := ReconcileInventory(context.Background(), "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("ReconcileInventory failed: %v", err)
	}

	// Verify Schema, Repo, HEAD, ObservedAt
	if report.Schema != ReconcileSchema {
		t.Errorf("schema = %q, want %q", report.Schema, ReconcileSchema)
	}
	if report.Repo != "/fak/repo" {
		t.Errorf("repo = %q, want /fak/repo", report.Repo)
	}
	if report.HEAD != "deadbeef1234" {
		t.Errorf("HEAD = %q, want deadbeef1234", report.HEAD)
	}
	if !report.ObservedAt.Equal(fixedTime) {
		t.Errorf("ObservedAt = %v, want %v", report.ObservedAt, fixedTime)
	}

	// Exact raw surfaces checks
	wantRaw := RawSurfacesReport{
		Issues:           1,
		Sessions:         4, // sess-joined, sess-unjoined, sess-stale, sess-conflict (or sess-conflict-b)
		Checkpoints:      3,
		LaneLeases:       3,
		ManagedWorktrees: 2,
		UnlinkedFiles:    1,
		UnlinkedDirs:     1,
	}
	if report.RawSurfaces.Issues != wantRaw.Issues {
		t.Errorf("RawSurfaces.Issues = %d, want %d", report.RawSurfaces.Issues, wantRaw.Issues)
	}
	if report.RawSurfaces.Checkpoints != wantRaw.Checkpoints {
		t.Errorf("RawSurfaces.Checkpoints = %d, want %d", report.RawSurfaces.Checkpoints, wantRaw.Checkpoints)
	}
	if report.RawSurfaces.LaneLeases != wantRaw.LaneLeases {
		t.Errorf("RawSurfaces.LaneLeases = %d, want %d", report.RawSurfaces.LaneLeases, wantRaw.LaneLeases)
	}
	if report.RawSurfaces.ManagedWorktrees != wantRaw.ManagedWorktrees {
		t.Errorf("RawSurfaces.ManagedWorktrees = %d, want %d", report.RawSurfaces.ManagedWorktrees, wantRaw.ManagedWorktrees)
	}
	if report.RawSurfaces.UnlinkedFiles != wantRaw.UnlinkedFiles {
		t.Errorf("RawSurfaces.UnlinkedFiles = %d, want %d", report.RawSurfaces.UnlinkedFiles, wantRaw.UnlinkedFiles)
	}
	if report.RawSurfaces.UnlinkedDirs != wantRaw.UnlinkedDirs {
		t.Errorf("RawSurfaces.UnlinkedDirs = %d, want %d", report.RawSurfaces.UnlinkedDirs, wantRaw.UnlinkedDirs)
	}

	// Exact logical totals checks
	if report.LogicalUnits.Total != 6 {
		t.Errorf("LogicalUnits.Total = %d, want 6", report.LogicalUnits.Total)
	}
	if report.LogicalUnits.Terminal != 1 {
		t.Errorf("LogicalUnits.Terminal = %d, want 1 (unitTerminal)", report.LogicalUnits.Terminal)
	}
	if report.LogicalUnits.Active != 5 {
		t.Errorf("LogicalUnits.Active = %d, want 5", report.LogicalUnits.Active)
	}
	if report.LogicalUnits.Total != report.LogicalUnits.Active+report.LogicalUnits.Terminal {
		t.Errorf("Total != Active + Terminal: %d != %d + %d",
			report.LogicalUnits.Total, report.LogicalUnits.Active, report.LogicalUnits.Terminal)
	}
	if report.LogicalUnits.Total != report.LogicalUnits.SingleRepresentation+report.LogicalUnits.SplitRepresentations {
		t.Errorf("Total != Single + Split: %d != %d + %d",
			report.LogicalUnits.Total, report.LogicalUnits.SingleRepresentation, report.LogicalUnits.SplitRepresentations)
	}

	// Transition counts
	if report.TransitionCounts[TransitionCreate] != 5 {
		t.Errorf("TransitionCounts[create] = %d, want 5", report.TransitionCounts[TransitionCreate])
	}
	if report.TransitionCounts[TransitionLand] != 1 {
		t.Errorf("TransitionCounts[land] = %d, want 1", report.TransitionCounts[TransitionLand])
	}
	if report.TransitionCounts[TransitionBind] != 1 {
		t.Errorf("TransitionCounts[bind] = %d, want 1", report.TransitionCounts[TransitionBind])
	}

	// Debt buckets checks
	reasons := make(map[string]int)
	for _, d := range report.UnresolvedJoinDebt {
		reasons[d.Reason]++
		if d.Reason == "" || d.Surface == "" || d.Details == "" {
			t.Errorf("invalid debt item: %#v", d)
		}
	}

	wantDebtReasons := []string{
		"unjoined_sessions",
		"orphaned_checkpoints",
		"unlinked_worktrees",
		"stale_leases",
		"conflicting_parentage",
		"unlinked_files",
		"unlinked_dirs",
	}
	for _, r := range wantDebtReasons {
		if reasons[r] == 0 {
			t.Errorf("missing expected debt bucket %q; got reasons: %#v", r, reasons)
		}
	}

	// Deterministic JSON output check
	b1, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON marshal failed: %v", err)
	}
	b2, err := report.JSON()
	if err != nil {
		t.Fatalf("JSON marshal failed: %v", err)
	}
	if !bytes.Equal(b1, b2) {
		t.Fatalf("non-deterministic JSON output:\nFirst:\n%s\nSecond:\n%s", string(b1), string(b2))
	}

	var parsed ReconciliationReport
	if err := json.Unmarshal(b1, &parsed); err != nil {
		t.Fatalf("unmarshal JSON failed: %v", err)
	}
	if parsed.Schema != ReconcileSchema {
		t.Errorf("parsed schema = %q, want %q", parsed.Schema, ReconcileSchema)
	}

	// Compact human text output check
	summary := report.SummaryText()
	lines := strings.Split(strings.TrimSpace(summary), "\n")
	if len(lines) < 4 {
		t.Fatalf("human summary too short: %q", summary)
	}
	if !strings.HasPrefix(lines[0], "Active logical WIP: 5") {
		t.Errorf("human summary first line must lead with active logical WIP: %q", lines[0])
	}
	if !strings.Contains(summary, "Raw surfaces:") {
		t.Errorf("human summary missing 'Raw surfaces:': %q", summary)
	}
	if !strings.Contains(summary, "Transitions:") {
		t.Errorf("human summary missing 'Transitions:': %q", summary)
	}
	if !strings.Contains(summary, "Unresolved join debt") {
		t.Errorf("human summary missing 'Unresolved join debt': %q", summary)
	}
}

func TestReconcileInventoryReadOnly(t *testing.T) {
	dir := t.TempDir()

	// Initialize git repository
	runGit := func(args ...string) string {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s failed: %v\nOutput: %s", strings.Join(args, " "), err, string(out))
		}
		return strings.TrimSpace(string(out))
	}

	runGit("init")
	runGit("config", "user.name", "Test Runner")
	runGit("config", "user.email", "test@example.com")
	runGit("commit", "--allow-empty", "-m", "initial commit")
	headBefore := runGit("rev-parse", "HEAD")
	refsBefore := runGit("for-each-ref")
	statusBefore := runGit("status", "--porcelain")

	// Run ReconcileInventory with empty inputs pointing at repo
	report, err := ReconcileInventory(context.Background(), dir, InventoryInputs{})
	if err != nil {
		t.Fatalf("ReconcileInventory failed: %v", err)
	}

	if report.HEAD != headBefore {
		t.Errorf("HEAD = %q, want %q", report.HEAD, headBefore)
	}

	// Verify repo state is completely unchanged
	headAfter := runGit("rev-parse", "HEAD")
	refsAfter := runGit("for-each-ref")
	statusAfter := runGit("status", "--porcelain")

	if headAfter != headBefore {
		t.Errorf("HEAD mutated: before %q, after %q", headBefore, headAfter)
	}
	if refsAfter != refsBefore {
		t.Errorf("refs mutated: before %q, after %q", refsBefore, refsAfter)
	}
	if statusAfter != statusBefore {
		t.Errorf("status mutated: before %q, after %q", statusBefore, statusAfter)
	}
}
