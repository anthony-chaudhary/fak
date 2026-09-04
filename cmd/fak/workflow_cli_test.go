package main

import (
	"bytes"
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestWorkflowCLIAgentFlags(t *testing.T) {
	fs, af := newAgentFlagSet()
	args := []string{
		"--workflow", "fleet-wave",
		"--workflow-step",
		"--workflow-checkpoint-dir", "/custom/checkpoints",
	}
	if err := fs.Parse(args); err != nil {
		t.Fatalf("failed to parse agent flags: %v", err)
	}

	if *af.workflow != "fleet-wave" {
		t.Errorf("af.workflow want %q, got %q", "fleet-wave", *af.workflow)
	}
	if !*af.workflowStep {
		t.Errorf("af.workflowStep want true, got false")
	}
	if *af.workflowCheckpointDir != "/custom/checkpoints" {
		t.Errorf("af.workflowCheckpointDir want %q, got %q", "/custom/checkpoints", *af.workflowCheckpointDir)
	}
}

func TestWorkflowCLIExecutePassing(t *testing.T) {
	tempDir := t.TempDir()
	var outBuf, errBuf bytes.Buffer

	state, err := runWorkflowCore(&outBuf, &errBuf, "fleet-wave", false, tempDir)
	if err != nil {
		t.Fatalf("runWorkflowCore returned error: %v, stderr: %s", err, errBuf.String())
	}

	if state == nil {
		t.Fatal("expected non-nil workflow state")
	}
	if state.Status != agent.WorkflowCompleted {
		t.Fatalf("expected state status %q, got %q", agent.WorkflowCompleted, state.Status)
	}
	if state.CurrentPhase != 6 {
		t.Fatalf("expected CurrentPhase 6, got %d", state.CurrentPhase)
	}
	if len(state.Receipts) != 6 {
		t.Fatalf("expected 6 receipts, got %d", len(state.Receipts))
	}

	outStr := outBuf.String()
	for _, phase := range []string{"gate", "price", "receipt", "launch", "monitor", "harvest"} {
		if !strings.Contains(outStr, phase) {
			t.Errorf("expected stdout output to mention phase %q, got:\n%s", phase, outStr)
		}
	}
}

func TestWorkflowCLIStep(t *testing.T) {
	tempDir := t.TempDir()

	// Step 0: Gate
	var outBuf, errBuf bytes.Buffer
	st0, err := runWorkflowCore(&outBuf, &errBuf, "fleet-wave", true, tempDir)
	if err != nil {
		t.Fatalf("step 0 failed: %v", err)
	}
	if st0.CurrentPhase != 1 {
		t.Fatalf("after step 0, CurrentPhase want 1, got %d", st0.CurrentPhase)
	}

	// Step 1: Price (should reload checkpoint from tempDir)
	outBuf.Reset()
	errBuf.Reset()
	st1, err := runWorkflowCore(&outBuf, &errBuf, "fleet-wave", true, tempDir)
	if err != nil {
		t.Fatalf("step 1 failed: %v", err)
	}
	if st1.CurrentPhase != 2 {
		t.Fatalf("after step 1, CurrentPhase want 2, got %d", st1.CurrentPhase)
	}
	if len(st1.Receipts) != 2 {
		t.Fatalf("after step 1, Receipts want 2, got %d", len(st1.Receipts))
	}
}

func TestWorkflowCLIRefusal(t *testing.T) {
	tempDir := t.TempDir()
	engine := agent.DefaultWorkflowEngine()

	refusalWorkflow := &agent.Workflow{
		ID:          "cli-refusal-test",
		Name:        "CLI Refusal Test",
		Description: "Workflow that refuses at entry gate",
		Phases: []agent.Phase{
			{
				ID:    "blocked-phase",
				Index: 0,
				Name:  "BlockedPhase",
				EntryGates: []agent.EntryGate{
					{
						Name: "AlwaysRefuse",
						Check: func(ctx context.Context, state *agent.WorkflowState) agent.GateVerdict {
							return agent.GateVerdict{
								Kind:        agent.GateRefuse,
								Reason:      "operator quota exhausted",
								RefusalCode: "REFUSE_OPERATOR_QUOTA",
							}
						},
					},
				},
				Action: func(ctx context.Context, state *agent.WorkflowState) (agent.Witness, error) {
					t.Fatal("action should not run on refusal")
					return agent.Witness{}, nil
				},
			},
		},
	}
	engine.Register(refusalWorkflow)

	var outBuf, errBuf bytes.Buffer
	state, err := runWorkflowCore(&outBuf, &errBuf, "cli-refusal-test", false, tempDir)
	if err == nil {
		t.Fatal("expected error on workflow refusal, got nil")
	}

	if state == nil {
		t.Fatal("expected non-nil state on refusal")
	}
	if state.Status != agent.WorkflowRefused {
		t.Fatalf("expected status %q, got %q", agent.WorkflowRefused, state.Status)
	}

	errStr := errBuf.String()
	if !strings.Contains(errStr, "REFUSE_OPERATOR_QUOTA") {
		t.Fatalf("expected stderr to contain refusal code REFUSE_OPERATOR_QUOTA, got:\n%s", errStr)
	}

	// Verify refusal checkpoint exists on disk
	cpPath := filepath.Join(tempDir, "checkpoint_cli-refusal-test_phase_0.json")
	reloaded, loadErr := agent.LoadWorkflowState(cpPath)
	if loadErr != nil {
		t.Fatalf("failed to reload refusal checkpoint: %v", loadErr)
	}
	if reloaded.Status != agent.WorkflowRefused {
		t.Errorf("reloaded state status want %q, got %q", agent.WorkflowRefused, reloaded.Status)
	}
}
