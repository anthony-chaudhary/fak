package tb4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"testing"
)

func TestDeterministicGraderClassification(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "tb4-grader-test-*")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tempDir)

	mockEngine := NewMockContainerEngine()
	defer mockEngine.Close()

	ctx := context.Background()
	config := ContainerConfig{
		ImageDigest: "ghcr.io/fak/tb4@sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
		Name:        "tb4-grader-c",
		NetworkMode: NetworkModeNone,
		WorkingDir:  "/workspace",
	}
	inst, err := mockEngine.CreateContainer(ctx, config)
	if err != nil {
		t.Fatalf("failed to create container: %v", err)
	}

	wsDir := mockEngine.workspaces[inst.ID]
	wsMgr := NewWorkspaceManager(mockEngine, inst.ID, "grade-task-01", wsDir)
	grader := NewGrader()

	// 1. Passing task -> SOLVED
	passScript := "#!/bin/bash\nexit 0\n"
	hPass := sha256.Sum256([]byte(passScript))
	passHash := "sha256:" + hex.EncodeToString(hPass[:])

	taskPass := TaskManifest{
		TaskID:                 "grade-task-01",
		VerificationOracle:     passScript,
		VerificationOracleHash: passHash,
		TimeoutSeconds:         10,
	}

	receiptPass, err := grader.Grade(ctx, "fak_inkernel", taskPass, wsMgr, &ArmExecutionResult{Status: "COMPLETED"})
	if err != nil {
		t.Fatalf("grading failed: %v", err)
	}
	if receiptPass.Verdict != "SOLVED" {
		t.Errorf("expected SOLVED, got %s", receiptPass.Verdict)
	}
	if receiptPass.FailureReason != ReasonSolved {
		t.Errorf("expected ReasonSolved, got %s", receiptPass.FailureReason)
	}
	if receiptPass.ReceiptHash == "" {
		t.Errorf("expected non-empty receipt hash")
	}

	// 2. Failing test -> TEST_FAILED
	failScript := "#!/bin/bash\nexit 1\n"
	hFail := sha256.Sum256([]byte(failScript))
	failHash := "sha256:" + hex.EncodeToString(hFail[:])

	taskFail := TaskManifest{
		TaskID:                 "grade-task-02",
		VerificationOracle:     failScript,
		VerificationOracleHash: failHash,
		TimeoutSeconds:         10,
	}

	receiptFail, err := grader.Grade(ctx, "fak_inkernel", taskFail, wsMgr, &ArmExecutionResult{Status: "COMPLETED"})
	if err != nil {
		t.Fatalf("grading failed: %v", err)
	}
	if receiptFail.Verdict != "FAILED" {
		t.Errorf("expected FAILED, got %s", receiptFail.Verdict)
	}
	if receiptFail.FailureReason != ReasonTestFailed {
		t.Errorf("expected ReasonTestFailed, got %s", receiptFail.FailureReason)
	}

	// 3. Agent timed out -> TIMEOUT_AGENT
	receiptAgentTimeout, err := grader.Grade(ctx, "fak_inkernel", taskFail, wsMgr, &ArmExecutionResult{Status: "TIMEOUT"})
	if err != nil {
		t.Fatalf("grading failed: %v", err)
	}
	if receiptAgentTimeout.FailureReason != ReasonTimeoutAgent {
		t.Errorf("expected ReasonTimeoutAgent, got %s", receiptAgentTimeout.FailureReason)
	}

	// 4. Policy block refusal -> POLICY_BLOCK
	receiptPolicyBlock, err := grader.Grade(ctx, "fak_inkernel", taskFail, wsMgr, &ArmExecutionResult{Status: "RUNNING", PolicyBlocks: 1})
	if err != nil {
		t.Fatalf("grading failed: %v", err)
	}
	if receiptPolicyBlock.FailureReason != ReasonPolicyBlock {
		t.Errorf("expected ReasonPolicyBlock, got %s", receiptPolicyBlock.FailureReason)
	}
}
