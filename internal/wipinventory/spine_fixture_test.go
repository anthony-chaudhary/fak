package wipinventory

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestSpineLifecycleOneWIPUnit(t *testing.T) {
	ctx := context.Background()
	t0 := time.Date(2026, 9, 4, 12, 0, 0, 0, time.UTC)

	unit1 := id(1) // represents issue #10419 / item-1
	unit2 := id(2) // split child 1
	unit3 := id(3) // split child 2

	// Stage 1: Starts with 1 issue (#10419/item-1)
	trCreate1 := tr(TransitionCreate, nil, []WIPUnitID{unit1})
	trCreate1.Timestamp = t0
	trCreate1.Witness = "witness:create-10419"

	trBindIssue := tr(TransitionBind, []WIPUnitID{unit1}, []WIPUnitID{unit1})
	trBindIssue.Timestamp = t0.Add(time.Minute)
	trBindIssue.References = []SurfaceReference{
		{
			Kind: SurfaceIssue,
			Issue: &IssueReference{
				Repository: "anthony-chaudhary/fak",
				Number:     10419,
			},
		},
	}
	trBindIssue.Witness = "witness:bind-issue-10419"

	transitions := []Transition{trCreate1, trBindIssue}

	inputs := InventoryInputs{
		Histories: []History{{Schema: WIPUnitSchema, Transitions: transitions}},
		Issues:    []IssueReference{{Repository: "anthony-chaudhary/fak", Number: 10419}},
		Now:       t0.Add(time.Minute),
		HEAD:      "1041900000000000000000000000000000001041",
	}

	rep1, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 1 ReconcileInventory failed: %v", err)
	}
	if rep1.RawSurfaces.Issues != 1 {
		t.Errorf("Stage 1: RawSurfaces.Issues = %d, want 1", rep1.RawSurfaces.Issues)
	}
	if rep1.LogicalUnits.Active != 1 {
		t.Errorf("Stage 1: LogicalUnits.Active = %d, want 1", rep1.LogicalUnits.Active)
	}
	if rep1.LogicalUnits.Total != 1 {
		t.Errorf("Stage 1: LogicalUnits.Total = %d, want 1", rep1.LogicalUnits.Total)
	}
	if rep1.LogicalUnits.SingleRepresentation != 1 {
		t.Errorf("Stage 1: LogicalUnits.SingleRepresentation = %d, want 1", rep1.LogicalUnits.SingleRepresentation)
	}
	if rep1.LogicalUnits.SplitRepresentations != 0 {
		t.Errorf("Stage 1: LogicalUnits.SplitRepresentations = %d, want 0", rep1.LogicalUnits.SplitRepresentations)
	}

	// Stage 2: Joins to dispatch attempt & session
	trHandoffSess := tr(TransitionHandoff, []WIPUnitID{unit1}, []WIPUnitID{unit1})
	trHandoffSess.Timestamp = t0.Add(2 * time.Minute)
	trHandoffSess.References = []SurfaceReference{
		{
			Kind: SurfaceDispatchSession,
			DispatchSession: &DispatchSessionReference{
				DispatchID: "dispatch-10419-1",
				SessionID:  "sess-10419-1",
			},
		},
	}
	trHandoffSess.Witness = "witness:handoff-sess-10419"
	transitions = append(transitions, trHandoffSess)

	inputs.Histories[0].Transitions = transitions
	inputs.Bindings = ExecutionBindingReport{
		Bindings: []ExecutionBinding{
			{
				RootRegistrationID: "reg-10419",
				Issue: &ExecutionIssueIdentity{
					Repository: "anthony-chaudhary/fak",
					Number:     10419,
				},
				RegistrationIDs: []string{"reg-10419"},
				AttemptIDs:      []string{"attempt-10419-1"},
				SessionIDs:      []string{"sess-10419-1"},
				Status:          ExecutionBindingJoined,
			},
		},
	}
	inputs.Sessions = []string{"sess-10419-1"}
	inputs.Now = t0.Add(2 * time.Minute)

	rep2, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 2 ReconcileInventory failed: %v", err)
	}
	rawTotal2 := rep2.RawSurfaces.Issues + rep2.RawSurfaces.Sessions
	if rawTotal2 <= 1 {
		t.Errorf("Stage 2: raw surface representations = %d, want > 1", rawTotal2)
	}
	if rep2.LogicalUnits.Active != 1 {
		t.Errorf("Stage 2: LogicalUnits.Active = %d, want 1", rep2.LogicalUnits.Active)
	}
	if rep2.LogicalUnits.Total != 1 {
		t.Errorf("Stage 2: LogicalUnits.Total = %d, want 1", rep2.LogicalUnits.Total)
	}
	if rep2.LogicalUnits.SplitRepresentations != 1 {
		t.Errorf("Stage 2: LogicalUnits.SplitRepresentations = %d, want 1", rep2.LogicalUnits.SplitRepresentations)
	}

	// Stage 3: Checkpoint joined
	trBindCP := tr(TransitionBind, []WIPUnitID{unit1}, []WIPUnitID{unit1})
	trBindCP.Timestamp = t0.Add(3 * time.Minute)
	trBindCP.References = []SurfaceReference{
		{
			Kind: SurfaceCheckpoint,
			Checkpoint: &CheckpointReference{
				CheckpointID: "cp-10419-1",
			},
		},
	}
	trBindCP.Witness = "witness:bind-cp-10419"
	transitions = append(transitions, trBindCP)

	inputs.Histories[0].Transitions = transitions
	inputs.Checkpoints = []CheckpointWIPBinding{
		{
			CheckpointID: "cp-10419-1",
			WIPUnitID:    unit1,
			SessionID:    "sess-10419-1",
			Lane:         "wipinventory",
			LeaseID:      "lease-10419-1",
			Registered:   true,
			Scope:        CheckpointSnapshotOwned,
		},
	}
	inputs.Now = t0.Add(3 * time.Minute)

	rep3, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 3 ReconcileInventory failed: %v", err)
	}
	rawTotal3 := rep3.RawSurfaces.Issues + rep3.RawSurfaces.Sessions + rep3.RawSurfaces.Checkpoints
	if rawTotal3 != 3 {
		t.Errorf("Stage 3: raw surface representations = %d, want 3", rawTotal3)
	}
	if rep3.LogicalUnits.Active != 1 {
		t.Errorf("Stage 3: LogicalUnits.Active = %d, want 1", rep3.LogicalUnits.Active)
	}
	if rep3.LogicalUnits.Total != 1 {
		t.Errorf("Stage 3: LogicalUnits.Total = %d, want 1", rep3.LogicalUnits.Total)
	}

	// Stage 4: Lane lease joined
	trBindLease := tr(TransitionBind, []WIPUnitID{unit1}, []WIPUnitID{unit1})
	trBindLease.Timestamp = t0.Add(4 * time.Minute)
	trBindLease.References = []SurfaceReference{
		{
			Kind: SurfaceLaneLease,
			LaneLease: &LaneLeaseReference{
				Lane:    "wipinventory",
				LeaseID: "lease-10419-1",
			},
		},
	}
	trBindLease.Witness = "witness:bind-lease-10419"
	transitions = append(transitions, trBindLease)

	inputs.Histories[0].Transitions = transitions
	inputs.Leases = []LiveLaneLease{
		{
			LeaseID:   "lease-10419-1",
			Lane:      "wipinventory",
			WIPUnitID: unit1,
			SessionID: "sess-10419-1",
			WorkerID:  "worker-10419-1",
			Live:      true,
		},
	}
	inputs.Now = t0.Add(4 * time.Minute)

	rep4, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 4 ReconcileInventory failed: %v", err)
	}
	rawTotal4 := rep4.RawSurfaces.Issues + rep4.RawSurfaces.Sessions + rep4.RawSurfaces.Checkpoints + rep4.RawSurfaces.LaneLeases
	if rawTotal4 != 4 {
		t.Errorf("Stage 4: raw surface representations = %d, want 4", rawTotal4)
	}
	if rep4.LogicalUnits.Active != 1 {
		t.Errorf("Stage 4: LogicalUnits.Active = %d, want 1", rep4.LogicalUnits.Active)
	}
	if rep4.LogicalUnits.Total != 1 {
		t.Errorf("Stage 4: LogicalUnits.Total = %d, want 1", rep4.LogicalUnits.Total)
	}

	// Stage 5: Managed worktree joined
	trBindWT := tr(TransitionBind, []WIPUnitID{unit1}, []WIPUnitID{unit1})
	trBindWT.Timestamp = t0.Add(5 * time.Minute)
	trBindWT.References = []SurfaceReference{
		{
			Kind: SurfaceManagedWorktree,
			ManagedWorktree: &ManagedWorktreeReference{
				WorktreeID: "wt-10419-1",
			},
		},
	}
	trBindWT.Witness = "witness:bind-wt-10419"
	transitions = append(transitions, trBindWT)

	inputs.Histories[0].Transitions = transitions
	inputs.Worktrees = []ManagedWorktreeBinding{
		{
			WorktreeID: "wt-10419-1",
			WIPUnitID:  string(unit1),
			WorkerID:   "worker-10419-1",
			Lane:       "wipinventory",
			LeaseID:    "lease-10419-1",
			Registered: true,
		},
	}
	inputs.Now = t0.Add(5 * time.Minute)

	rep5, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 5 ReconcileInventory failed: %v", err)
	}

	// Prove: raw surface counts = 5 (1 issue, 1 session, 1 checkpoint, 1 lease, 1 worktree)
	if rep5.RawSurfaces.Issues != 1 || rep5.RawSurfaces.Sessions != 1 || rep5.RawSurfaces.Checkpoints != 1 ||
		rep5.RawSurfaces.LaneLeases != 1 || rep5.RawSurfaces.ManagedWorktrees != 1 {
		t.Errorf("Stage 5: unexpected raw surfaces: %+v", rep5.RawSurfaces)
	}
	rawTotal5 := rep5.RawSurfaces.Issues + rep5.RawSurfaces.Sessions + rep5.RawSurfaces.Checkpoints +
		rep5.RawSurfaces.LaneLeases + rep5.RawSurfaces.ManagedWorktrees
	if rawTotal5 != 5 {
		t.Errorf("Stage 5: raw surface representations = %d, want 5", rawTotal5)
	}

	// Prove: logical active WIP count remains EXACTLY 1
	if rep5.LogicalUnits.Active != 1 {
		t.Errorf("Stage 5: LogicalUnits.Active = %d, want 1", rep5.LogicalUnits.Active)
	}
	if rep5.LogicalUnits.Total != 1 {
		t.Errorf("Stage 5: LogicalUnits.Total = %d, want 1", rep5.LogicalUnits.Total)
	}
	if rep5.LogicalUnits.SplitRepresentations != 1 {
		t.Errorf("Stage 5: LogicalUnits.SplitRepresentations = %d, want 1", rep5.LogicalUnits.SplitRepresentations)
	}
	if rep5.LogicalUnits.SingleRepresentation != 0 {
		t.Errorf("Stage 5: LogicalUnits.SingleRepresentation = %d, want 0", rep5.LogicalUnits.SingleRepresentation)
	}
	if len(rep5.UnresolvedJoinDebt) != 0 {
		t.Errorf("Stage 5: expected zero debt, got %d items: %+v", len(rep5.UnresolvedJoinDebt), rep5.UnresolvedJoinDebt)
	}

	// Stage 6: Explicit split transition: splits unit-1 into two units (unit-2 and unit-3)
	trCreate2 := tr(TransitionCreate, nil, []WIPUnitID{unit2})
	trCreate2.Timestamp = t0.Add(6 * time.Minute)
	trCreate2.Witness = "witness:create-10419-a"

	trCreate3 := tr(TransitionCreate, nil, []WIPUnitID{unit3})
	trCreate3.Timestamp = t0.Add(7 * time.Minute)
	trCreate3.Witness = "witness:create-10419-b"

	trSplit := tr(TransitionSplit, []WIPUnitID{unit1}, []WIPUnitID{unit2, unit3})
	trSplit.Timestamp = t0.Add(8 * time.Minute)
	trSplit.Witness = "witness:split-10419"

	transitions = append(transitions, trCreate2, trCreate3, trSplit)
	inputs.Histories[0].Transitions = transitions

	// Update raw physical surfaces to reflect the active child units
	inputs.Bindings = ExecutionBindingReport{
		Bindings: []ExecutionBinding{
			{
				RootRegistrationID: "reg-10419",
				Issue: &ExecutionIssueIdentity{
					Repository: "anthony-chaudhary/fak",
					Number:     10419,
				},
				RegistrationIDs: []string{"reg-10419"},
				AttemptIDs:      []string{"attempt-10419-a", "attempt-10419-b"},
				SessionIDs:      []string{"sess-10419-a", "sess-10419-b"},
				Status:          ExecutionBindingJoined,
			},
		},
	}
	inputs.Sessions = []string{"sess-10419-a", "sess-10419-b"}
	inputs.Checkpoints = []CheckpointWIPBinding{
		{
			CheckpointID: "cp-10419-a",
			WIPUnitID:    unit2,
			SessionID:    "sess-10419-a",
			Lane:         "wipinventory-a",
			LeaseID:      "lease-10419-a",
			Registered:   true,
			Scope:        CheckpointSnapshotOwned,
		},
		{
			CheckpointID: "cp-10419-b",
			WIPUnitID:    unit3,
			SessionID:    "sess-10419-b",
			Lane:         "wipinventory-b",
			LeaseID:      "lease-10419-b",
			Registered:   true,
			Scope:        CheckpointSnapshotOwned,
		},
	}
	inputs.Leases = []LiveLaneLease{
		{
			LeaseID:   "lease-10419-a",
			Lane:      "wipinventory-a",
			WIPUnitID: unit2,
			SessionID: "sess-10419-a",
			WorkerID:  "worker-10419-a",
			Live:      true,
		},
		{
			LeaseID:   "lease-10419-b",
			Lane:      "wipinventory-b",
			WIPUnitID: unit3,
			SessionID: "sess-10419-b",
			WorkerID:  "worker-10419-b",
			Live:      true,
		},
	}
	inputs.Worktrees = []ManagedWorktreeBinding{
		{
			WorktreeID: "wt-10419-a",
			WIPUnitID:  string(unit2),
			WorkerID:   "worker-10419-a",
			Lane:       "wipinventory-a",
			LeaseID:    "lease-10419-a",
			Registered: true,
		},
		{
			WorktreeID: "wt-10419-b",
			WIPUnitID:  string(unit3),
			WorkerID:   "worker-10419-b",
			Lane:       "wipinventory-b",
			LeaseID:    "lease-10419-b",
			Registered: true,
		},
	}
	inputs.Now = t0.Add(8 * time.Minute)

	rep6, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 6 ReconcileInventory failed: %v", err)
	}

	// Prove: logical active WIP count becomes exactly 2!
	if rep6.LogicalUnits.Active != 2 {
		t.Errorf("Stage 6: LogicalUnits.Active = %d, want exactly 2", rep6.LogicalUnits.Active)
	}
	if rep6.LogicalUnits.Terminal != 1 {
		t.Errorf("Stage 6: LogicalUnits.Terminal = %d, want 1 (superseded unit-1)", rep6.LogicalUnits.Terminal)
	}
	if rep6.LogicalUnits.Total != 3 {
		t.Errorf("Stage 6: LogicalUnits.Total = %d, want 3", rep6.LogicalUnits.Total)
	}

	// Stage 7: Witnessed landing / retirement transition: lands both units
	trLand2 := tr(TransitionLand, []WIPUnitID{unit2}, []WIPUnitID{unit2})
	trLand2.Timestamp = t0.Add(9 * time.Minute)
	trLand2.References = []SurfaceReference{
		{
			Kind: SurfaceWitnessedRetirement,
			WitnessedRetirement: &WitnessedRetirementReference{
				RetirementID: "retire-10419-a",
				Witness:      "commit:10441a",
			},
		},
	}
	trLand2.Witness = "witness:land-10419-a"

	trLand3 := tr(TransitionLand, []WIPUnitID{unit3}, []WIPUnitID{unit3})
	trLand3.Timestamp = t0.Add(10 * time.Minute)
	trLand3.References = []SurfaceReference{
		{
			Kind: SurfaceWitnessedRetirement,
			WitnessedRetirement: &WitnessedRetirementReference{
				RetirementID: "retire-10419-b",
				Witness:      "commit:10441b",
			},
		},
	}
	trLand3.Witness = "witness:land-10419-b"

	transitions = append(transitions, trLand2, trLand3)
	inputs.Histories[0].Transitions = transitions
	inputs.Now = t0.Add(10 * time.Minute)

	rep7, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 7 ReconcileInventory failed: %v", err)
	}

	// Prove: Active WIP count returns to 0
	if rep7.LogicalUnits.Active != 0 {
		t.Errorf("Stage 7: LogicalUnits.Active = %d, want 0", rep7.LogicalUnits.Active)
	}
	if rep7.LogicalUnits.Total != 3 {
		t.Errorf("Stage 7: LogicalUnits.Total = %d, want 3", rep7.LogicalUnits.Total)
	}
	if rep7.LogicalUnits.Terminal != 3 {
		t.Errorf("Stage 7: LogicalUnits.Terminal = %d, want 3", rep7.LogicalUnits.Terminal)
	}

	// Prove: History is retained in the ledger (no units purged or lost)
	acc := AccountHistory(inputs.Histories[0])
	if acc.ActiveCount != 0 {
		t.Errorf("Stage 7: AccountHistory ActiveCount = %d, want 0", acc.ActiveCount)
	}
	if len(acc.Units) != 3 {
		t.Errorf("Stage 7: AccountHistory Units count = %d, want 3 retained units", len(acc.Units))
	}
	landedCount := 0
	supersededCount := 0
	for _, u := range acc.Units {
		if u.State == AccountedUnitLanded {
			landedCount++
		}
		if u.State == AccountedUnitSuperseded {
			supersededCount++
		}
	}
	if landedCount != 2 {
		t.Errorf("Stage 7: landed units count = %d, want 2", landedCount)
	}
	if supersededCount != 1 {
		t.Errorf("Stage 7: superseded units count = %d, want 1", supersededCount)
	}
	if len(acc.Debt) != 0 {
		t.Errorf("Stage 7: unexpected accounting debt: %+v", acc.Debt)
	}

	// Stage 8: Injects missing join and ambiguous join -> proves typed debt in UnresolvedJoinDebt
	// 1. Missing join: worker session with no issue
	inputs.Bindings.Bindings = append(inputs.Bindings.Bindings, ExecutionBinding{
		RootRegistrationID: "reg-orphan-sess",
		SessionIDs:         []string{"sess-orphan-1"},
		Status:             ExecutionBindingMissing,
		Details:            []string{"worker session lacks matching issue registration"},
	})
	inputs.Sessions = append(inputs.Sessions, "sess-orphan-1")

	// 2. Ambiguous join: managed worktree with conflicting owner
	unitConflict := id(4)
	inputs.Worktrees = append(inputs.Worktrees, ManagedWorktreeBinding{
		WorktreeID: "wt-conflict",
		WIPUnitID:  string(unitConflict),
		WorkerID:   "worker-conflict",
		Lane:       "wipinventory-a", // declares lease-10419-a which is owned by unit2
		LeaseID:    "lease-10419-a",
		Registered: true,
	})

	rep8, err := ReconcileInventory(ctx, "/fak/repo", inputs)
	if err != nil {
		t.Fatalf("Stage 8 ReconcileInventory failed: %v", err)
	}

	if len(rep8.UnresolvedJoinDebt) < 2 {
		t.Fatalf("Stage 8: expected at least 2 debt items, got %d: %+v", len(rep8.UnresolvedJoinDebt), rep8.UnresolvedJoinDebt)
	}

	hasUnjoinedSession := false
	hasConflictingWorktree := false
	for _, debt := range rep8.UnresolvedJoinDebt {
		if debt.Reason == "unjoined_sessions" && debt.Surface == "dispatch_session" {
			hasUnjoinedSession = true
			if debt.Sample == "" || len(debt.Identifiers) == 0 || debt.Details == "" {
				t.Errorf("Stage 8: unjoined_sessions debt item incomplete: %+v", debt)
			}
		}
		if debt.Reason == "conflicting_parentage" && debt.Surface == "managed_worktree" {
			hasConflictingWorktree = true
			if debt.Sample == "" || len(debt.Identifiers) == 0 || debt.Details == "" {
				t.Errorf("Stage 8: conflicting_parentage debt item incomplete: %+v", debt)
			}
		}
	}

	if !hasUnjoinedSession {
		t.Errorf("Stage 8: missing expected unjoined_sessions debt item; got: %+v", rep8.UnresolvedJoinDebt)
	}
	if !hasConflictingWorktree {
		t.Errorf("Stage 8: missing expected conflicting_parentage debt item; got: %+v", rep8.UnresolvedJoinDebt)
	}
}

func TestSpineWitnessArtifactValidity(t *testing.T) {
	// Locate docs/_witnesses/wipinventory/witness.json relative to repository root
	repoRoot := filepath.Join("..", "..")
	witnessPath := filepath.Join(repoRoot, "docs", "_witnesses", "wipinventory", "witness.json")

	data, err := os.ReadFile(witnessPath)
	if err != nil {
		t.Fatalf("read witness artifact %q failed: %v", witnessPath, err)
	}

	var report ReconciliationReport
	if err := json.Unmarshal(data, &report); err != nil {
		t.Fatalf("unmarshal witness JSON failed: %v", err)
	}

	if report.Schema != ReconcileSchema {
		t.Errorf("witness schema = %q, want %q", report.Schema, ReconcileSchema)
	}
	if report.LogicalUnits.Active != 0 {
		t.Errorf("witness logical active count = %d, want 0", report.LogicalUnits.Active)
	}
	if report.LogicalUnits.Total != 3 {
		t.Errorf("witness logical total count = %d, want 3", report.LogicalUnits.Total)
	}
	if report.LogicalUnits.Terminal != 3 {
		t.Errorf("witness logical terminal count = %d, want 3", report.LogicalUnits.Terminal)
	}

	// Verify scrubbed guarantees: no absolute private host filesystem paths or credentials
	contentStr := string(data)
	forbiddenTokens := []string{
		"C:\\",
		"Users\\",
		"anthony",
		"password",
		"token",
		"bearer",
	}
	for _, token := range forbiddenTokens {
		if strings.Contains(strings.ToLower(contentStr), strings.ToLower(token)) {
			t.Errorf("witness artifact contains unscrubbed private token %q", token)
		}
	}
}
