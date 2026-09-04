package agent

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestWorkflowSequentialExecutionPassing(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)

	wf := &Workflow{
		ID:          "test-passing",
		Name:        "Test Passing Workflow",
		Description: "A three-phase workflow with passing gates",
		Phases: []Phase{
			{
				ID:    "phase-0",
				Index: 0,
				Name:  "Setup",
				EntryGates: []EntryGate{
					{
						Name: "CheckReady",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "ready"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name: "CheckSetupComplete",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "setup clean"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["step0"] = "done"
					return Witness{Command: "setup", Passed: true, Description: "Setup executed"}, nil
				},
			},
			{
				ID:    "phase-1",
				Index: 1,
				Name:  "Compute",
				EntryGates: []EntryGate{
					{
						Name: "CheckSetupDone",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							if state.Data["step0"] != "done" {
								return GateVerdict{Kind: GateRefuse, Reason: "step0 not done"}
							}
							return GateVerdict{Kind: GatePass, Reason: "step0 is done"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name: "CheckComputeOutput",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "compute clean"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["step1"] = "computed"
					return Witness{Command: "compute", Passed: true, Description: "Compute executed"}, nil
				},
			},
			{
				ID:    "phase-2",
				Index: 2,
				Name:  "Finalize",
				EntryGates: []EntryGate{
					{
						Name: "CheckComputeDone",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "ready to finalize"}
						},
					},
				},
				ExitGates: []ExitCondition{
					{
						Name: "CheckFinalizeDone",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "finalized"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["step2"] = "finalized"
					return Witness{Command: "finalize", Passed: true, Description: "Finalize executed"}, nil
				},
			},
		},
	}

	engine.Register(wf)

	ctx := context.Background()
	state, err := engine.Execute(ctx, "test-passing", nil)
	if err != nil {
		t.Fatalf("unexpected error executing workflow: %v", err)
	}

	if state.Status != WorkflowCompleted {
		t.Fatalf("expected status %q, got %q", WorkflowCompleted, state.Status)
	}
	if state.CurrentPhase != 3 {
		t.Fatalf("expected current phase 3, got %d", state.CurrentPhase)
	}
	if len(state.Receipts) != 3 {
		t.Fatalf("expected 3 receipts, got %d", len(state.Receipts))
	}
	for i, r := range state.Receipts {
		if r.Verdict.Kind != GatePass {
			t.Errorf("receipt %d verdict kind want %q, got %q", i, GatePass, r.Verdict.Kind)
		}
		if r.FromIndex != i || r.ToIndex != i+1 {
			t.Errorf("receipt %d index transition want %d->%d, got %d->%d", i, i, i+1, r.FromIndex, r.ToIndex)
		}
	}
	if state.Data["step0"] != "done" || state.Data["step1"] != "computed" || state.Data["step2"] != "finalized" {
		t.Fatalf("state data missing expected values: %+v", state.Data)
	}
}

func TestWorkflowGateRefusal(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)

	wf := &Workflow{
		ID:          "test-refusal",
		Name:        "Test Refusal Workflow",
		Description: "EntryGate failure in Phase 1 blocks entry to Phase 2",
		Phases: []Phase{
			{
				ID:    "phase-0",
				Index: 0,
				Name:  "Setup",
				EntryGates: []EntryGate{
					{
						Name: "AlwaysPass",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{Kind: GatePass, Reason: "setup gate pass"}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["phase0"] = true
					return Witness{Command: "init", Passed: true}, nil
				},
			},
			{
				ID:    "phase-1",
				Index: 1,
				Name:  "GatedPhase",
				EntryGates: []EntryGate{
					{
						Name: "CapacityGate",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{
								Kind:        GateRefuse,
								Reason:      "insufficient worker capacity",
								RefusalCode: "REFUSE_CAPACITY",
								Witness:     Witness{Command: "fak capacity", Passed: false, Description: "Capacity check failed"},
							}
						},
					},
				},
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					t.Fatal("Phase 1 Action should not execute on EntryGate refusal")
					return Witness{}, nil
				},
			},
			{
				ID:    "phase-2",
				Index: 2,
				Name:  "UnreachablePhase",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					t.Fatal("Phase 2 Action should never be reached")
					return Witness{}, nil
				},
			},
		},
	}

	engine.Register(wf)
	ctx := context.Background()
	state, err := engine.Execute(ctx, "test-refusal", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if state.Status != WorkflowRefused {
		t.Fatalf("expected status %q, got %q", WorkflowRefused, state.Status)
	}
	if state.CurrentPhase != 1 {
		t.Fatalf("expected current phase to remain 1 (blocked), got %d", state.CurrentPhase)
	}
	if state.LastRefusal == nil {
		t.Fatal("expected LastRefusal to be populated")
	}
	if state.LastRefusal.Kind != GateRefuse {
		t.Errorf("LastRefusal kind want %q, got %q", GateRefuse, state.LastRefusal.Kind)
	}
	if state.LastRefusal.RefusalCode != "REFUSE_CAPACITY" {
		t.Errorf("LastRefusal code want %q, got %q", "REFUSE_CAPACITY", state.LastRefusal.RefusalCode)
	}

	// 2 receipts: Phase 0 pass, Phase 1 refusal
	if len(state.Receipts) != 2 {
		t.Fatalf("expected 2 receipts, got %d", len(state.Receipts))
	}
	refusalReceipt := state.Receipts[1]
	if refusalReceipt.Verdict.Kind != GateRefuse {
		t.Errorf("expected refusal receipt kind %q, got %q", GateRefuse, refusalReceipt.Verdict.Kind)
	}
	if refusalReceipt.FromPhase != "phase-1" {
		t.Errorf("expected from phase %q, got %q", "phase-1", refusalReceipt.FromPhase)
	}

	// Verify checkpoint file exists for phase 1
	checkpointPath := filepath.Join(tempDir, "checkpoint_test-refusal_phase_1.json")
	if _, err := os.Stat(checkpointPath); os.IsNotExist(err) {
		t.Fatalf("expected checkpoint file %q to exist", checkpointPath)
	}

	loaded, err := LoadWorkflowState(checkpointPath)
	if err != nil {
		t.Fatalf("failed to reload refusal checkpoint: %v", err)
	}
	if loaded.Status != WorkflowRefused {
		t.Errorf("reloaded state status want %q, got %q", WorkflowRefused, loaded.Status)
	}
	if loaded.LastRefusal == nil || loaded.LastRefusal.RefusalCode != "REFUSE_CAPACITY" {
		t.Errorf("reloaded state last refusal mismatch: %+v", loaded.LastRefusal)
	}
}

func TestWorkflowExitConditionFailure(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)

	wf := &Workflow{
		ID:          "test-exit-failure",
		Name:        "Test Exit Failure",
		Description: "Exit condition failure halts workflow",
		Phases: []Phase{
			{
				ID:    "phase-0",
				Index: 0,
				Name:  "ProducingPhase",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					return Witness{Command: "generate", Passed: true}, nil
				},
				ExitGates: []ExitCondition{
					{
						Name: "ArtifactVerification",
						Check: func(ctx context.Context, state *WorkflowState) GateVerdict {
							return GateVerdict{
								Kind:        GateRefuse,
								Reason:      "expected artifact missing",
								RefusalCode: "REFUSE_NO_ARTIFACT",
							}
						},
					},
				},
			},
			{
				ID:    "phase-1",
				Index: 1,
				Name:  "SubsequentPhase",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					t.Fatal("Phase 1 should not execute on Phase 0 ExitCondition refusal")
					return Witness{}, nil
				},
			},
		},
	}

	engine.Register(wf)
	ctx := context.Background()
	state, err := engine.Execute(ctx, "test-exit-failure", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if state.Status != WorkflowRefused {
		t.Fatalf("expected status %q, got %q", WorkflowRefused, state.Status)
	}
	if state.CurrentPhase != 0 {
		t.Fatalf("expected CurrentPhase to remain 0, got %d", state.CurrentPhase)
	}
	if state.LastRefusal == nil || state.LastRefusal.RefusalCode != "REFUSE_NO_ARTIFACT" {
		t.Fatalf("expected REFUSE_NO_ARTIFACT, got %+v", state.LastRefusal)
	}
}

func TestWorkflowStep(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)

	wf := &Workflow{
		ID:          "test-step",
		Name:        "Test Step Workflow",
		Description: "Advancing one phase at a time",
		Phases: []Phase{
			{
				ID:    "p0",
				Index: 0,
				Name:  "Phase0",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["p0"] = true
					return Witness{Command: "step0", Passed: true}, nil
				},
			},
			{
				ID:    "p1",
				Index: 1,
				Name:  "Phase1",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["p1"] = true
					return Witness{Command: "step1", Passed: true}, nil
				},
			},
			{
				ID:    "p2",
				Index: 2,
				Name:  "Phase2",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["p2"] = true
					return Witness{Command: "step2", Passed: true}, nil
				},
			},
		},
	}
	engine.Register(wf)
	ctx := context.Background()

	// Step 1
	st, err := engine.Step(ctx, "test-step", nil)
	if err != nil {
		t.Fatalf("step 1 error: %v", err)
	}
	if st.CurrentPhase != 1 {
		t.Fatalf("after step 1, CurrentPhase want 1, got %d", st.CurrentPhase)
	}
	if len(st.Receipts) != 1 {
		t.Fatalf("after step 1, Receipts want 1, got %d", len(st.Receipts))
	}
	if st.Data["p0"] != true {
		t.Fatalf("p0 not executed")
	}

	// Step 2
	st, err = engine.Step(ctx, "test-step", st)
	if err != nil {
		t.Fatalf("step 2 error: %v", err)
	}
	if st.CurrentPhase != 2 {
		t.Fatalf("after step 2, CurrentPhase want 2, got %d", st.CurrentPhase)
	}
	if len(st.Receipts) != 2 {
		t.Fatalf("after step 2, Receipts want 2, got %d", len(st.Receipts))
	}
	if st.Data["p1"] != true {
		t.Fatalf("p1 not executed")
	}

	// Step 3
	st, err = engine.Step(ctx, "test-step", st)
	if err != nil {
		t.Fatalf("step 3 error: %v", err)
	}
	if st.CurrentPhase != 3 {
		t.Fatalf("after step 3, CurrentPhase want 3, got %d", st.CurrentPhase)
	}
	if st.Status != WorkflowCompleted {
		t.Fatalf("after step 3, Status want %q, got %q", WorkflowCompleted, st.Status)
	}
	if len(st.Receipts) != 3 {
		t.Fatalf("after step 3, Receipts want 3, got %d", len(st.Receipts))
	}
	if st.Data["p2"] != true {
		t.Fatalf("p2 not executed")
	}

	// Step 4 (already completed)
	st, err = engine.Step(ctx, "test-step", st)
	if err != nil {
		t.Fatalf("step 4 error: %v", err)
	}
	if st.Status != WorkflowCompleted || st.CurrentPhase != 3 {
		t.Fatalf("step 4 mutated completed state: %+v", st)
	}
}

func TestWorkflowCheckpointPersistenceAndReload(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)

	wf := &Workflow{
		ID:          "test-checkpoint",
		Name:        "Test Checkpoint Workflow",
		Description: "Checkpoints can be persisted and reloaded to continue execution",
		Phases: []Phase{
			{
				ID:    "p0",
				Index: 0,
				Name:  "Phase0",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					state.Data["counter"] = 100
					return Witness{Command: "init-counter", Passed: true}, nil
				},
			},
			{
				ID:    "p1",
				Index: 1,
				Name:  "Phase1",
				Action: func(ctx context.Context, state *WorkflowState) (Witness, error) {
					val := state.Data["counter"].(float64)
					state.Data["counter"] = val + 50
					return Witness{Command: "add-counter", Passed: true}, nil
				},
			},
		},
	}
	engine.Register(wf)
	ctx := context.Background()

	// Run step 1
	st, err := engine.Step(ctx, "test-checkpoint", nil)
	if err != nil {
		t.Fatalf("step 1 failed: %v", err)
	}
	if st.CurrentPhase != 1 {
		t.Fatalf("step 1 CurrentPhase want 1, got %d", st.CurrentPhase)
	}

	// Checkpoint file checkpoint_test-checkpoint_phase_0.json should exist
	expectedPath := filepath.Join(tempDir, "checkpoint_test-checkpoint_phase_0.json")
	if _, err := os.Stat(expectedPath); os.IsNotExist(err) {
		t.Fatalf("checkpoint file %q does not exist", expectedPath)
	}

	// Load latest checkpoint from disk
	loadedState, err := engine.LoadLatestCheckpoint("test-checkpoint")
	if err != nil {
		t.Fatalf("LoadLatestCheckpoint failed: %v", err)
	}
	if loadedState.CurrentPhase != 1 {
		t.Fatalf("loaded CurrentPhase want 1, got %d", loadedState.CurrentPhase)
	}
	if len(loadedState.Receipts) != 1 {
		t.Fatalf("loaded Receipts want 1, got %d", len(loadedState.Receipts))
	}

	// Continue execution from loaded state to complete workflow
	completedState, err := engine.Step(ctx, "test-checkpoint", loadedState)
	if err != nil {
		t.Fatalf("step 2 with loaded state failed: %v", err)
	}
	if completedState.Status != WorkflowCompleted {
		t.Fatalf("status want %q, got %q", WorkflowCompleted, completedState.Status)
	}
	if completedState.CurrentPhase != 2 {
		t.Fatalf("current phase want 2, got %d", completedState.CurrentPhase)
	}
	if completedState.Data["counter"] != float64(150) {
		t.Fatalf("counter want 150, got %v", completedState.Data["counter"])
	}
}

func TestWorkflowBuiltinFleetWave(t *testing.T) {
	tempDir := t.TempDir()
	engine := NewWorkflowEngine(tempDir)
	engine.Register(NewFleetWaveWorkflow())

	wf, ok := engine.Get("fleet-wave")
	if !ok {
		t.Fatal("expected fleet-wave workflow to be registered")
	}

	if len(wf.Phases) != 6 {
		t.Fatalf("expected 6 phases (0..5), got %d", len(wf.Phases))
	}

	expectedNames := []string{"Gate", "Price", "Receipt", "Launch", "Monitor", "Harvest"}
	expectedIDs := []string{"gate", "price", "receipt", "launch", "monitor", "harvest"}

	for i, p := range wf.Phases {
		if p.Index != i {
			t.Errorf("phase %d index mismatch: got %d", i, p.Index)
		}
		if p.Name != expectedNames[i] {
			t.Errorf("phase %d name mismatch: want %q, got %q", i, expectedNames[i], p.Name)
		}
		if p.ID != expectedIDs[i] {
			t.Errorf("phase %d ID mismatch: want %q, got %q", i, expectedIDs[i], p.ID)
		}
	}

	ctx := context.Background()
	state, err := engine.Execute(ctx, "fleet-wave", nil)
	if err != nil {
		t.Fatalf("fleet-wave execution failed: %v", err)
	}

	if state.Status != WorkflowCompleted {
		t.Fatalf("fleet-wave status want %q, got %q", WorkflowCompleted, state.Status)
	}
	if state.CurrentPhase != 6 {
		t.Fatalf("fleet-wave CurrentPhase want 6, got %d", state.CurrentPhase)
	}
	if len(state.Receipts) != 6 {
		t.Fatalf("fleet-wave Receipts want 6, got %d", len(state.Receipts))
	}

	// Verify all data markers set by actions
	for _, marker := range []string{"gate_verified", "priced", "receipt_rendered", "launched", "monitored", "harvested"} {
		if state.Data[marker] != true {
			t.Errorf("expected state data marker %q to be true", marker)
		}
	}

	// Verify checkpoints 0..5 exist
	for i := 0; i < 6; i++ {
		p := filepath.Join(tempDir, filepath.Clean(filepath.FromSlash(filepath.Join("", "checkpoint_fleet-wave_phase_"+string(rune('0'+i))+".json"))))
		if _, err := os.Stat(p); os.IsNotExist(err) {
			t.Errorf("checkpoint for phase %d missing at %q", i, p)
		}
	}
}
