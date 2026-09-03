package tb4bench

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// GradingReceipt is an immutable, cryptographically verifiable record of task evaluation.
type GradingReceipt struct {
	TaskID        string        `json:"task_id"`
	Arm           string        `json:"arm"`
	Verdict       string        `json:"verdict"` // SOLVED or FAILED
	FailureReason FailureReason `json:"failure_reason,omitempty"`
	ExitCode      int           `json:"exit_code"`
	DurationMs    int64         `json:"duration_ms"`
	OracleHash    string        `json:"oracle_hash"`
	WorkspaceHash string        `json:"workspace_hash"`
	ReceiptHash   string        `json:"receipt_hash"`
	Timestamp     string        `json:"timestamp"`
}

// ComputeReceiptHash computes the SHA-256 digest of the grading receipt fields.
func (r *GradingReceipt) ComputeReceiptHash() string {
	raw := fmt.Sprintf("%s:%s:%s:%s:%d:%d:%s:%s:%s",
		r.TaskID, r.Arm, r.Verdict, r.FailureReason, r.ExitCode, r.DurationMs,
		r.OracleHash, r.WorkspaceHash, r.Timestamp)
	h := sha256.Sum256([]byte(raw))
	return "sha256:" + hex.EncodeToString(h[:])
}

// Grader manages the execution of task verification oracles and result classification.
type Grader struct{}

// NewGrader creates a new test oracle grader.
func NewGrader() *Grader {
	return &Grader{}
}

// Grade executes the task oracle script against the workspace and generates a GradingReceipt.
func (g *Grader) Grade(ctx context.Context, armID string, task TaskManifest, wsMgr *WorkspaceManager, armExec *ArmExecutionResult) (*GradingReceipt, error) {
	// 1. Verify oracle script hash
	if err := task.VerifyOracleScript([]byte(task.VerificationOracle)); err != nil {
		return nil, fmt.Errorf("oracle script verification failed: %w", err)
	}

	// 2. Inject verification script into container /eval directory
	timeoutSec := task.TimeoutSeconds
	if timeoutSec <= 0 {
		timeoutSec = 60
	}

	// Write oracle script into workspace
	if wsMgr.localDir != "" {
		evalDir := filepath.Join(wsMgr.localDir, "eval")
		_ = os.MkdirAll(evalDir, 0755)
		scriptPath := filepath.Join(evalDir, "verify.sh")
		if err := os.WriteFile(scriptPath, []byte(task.VerificationOracle), 0755); err != nil {
			return nil, err
		}
	} else {
		writeCmd := fmt.Sprintf("mkdir -p /workspace/eval && cat << 'EOF' > /workspace/eval/verify.sh\n%s\nEOF && chmod +x /workspace/eval/verify.sh", task.VerificationOracle)
		_, err := wsMgr.Exec(ctx, []string{"sh", "-c", writeCmd}, 30*time.Second)
		if err != nil {
			return nil, fmt.Errorf("failed to inject oracle script: %w", err)
		}
	}

	// 3. Execute verification script
	start := time.Now()
	timeoutDur := time.Duration(timeoutSec) * time.Second
	execRes, err := wsMgr.Exec(ctx, []string{"sh", "eval/verify.sh"}, timeoutDur)
	durationMs := time.Since(start).Milliseconds()

	// 4. Capture final workspace tree digest
	wsHash, _ := wsMgr.ComputeWorkspaceDigest(ctx)

	// 5. Adjudicate pass/fail and failure taxonomy
	receipt := &GradingReceipt{
		TaskID:        task.TaskID,
		Arm:           armID,
		ExitCode:      0,
		DurationMs:    durationMs,
		OracleHash:    task.VerificationOracleHash,
		WorkspaceHash: wsHash,
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
	}

	oracleTimedOut := false
	if err != nil && (errors.Is(err, context.DeadlineExceeded) || strings.Contains(err.Error(), "timeout")) {
		oracleTimedOut = true
	}

	if execRes != nil {
		receipt.ExitCode = execRes.ExitCode
	} else {
		receipt.ExitCode = 1
	}

	// Automated Failure Classification
	if !oracleTimedOut && receipt.ExitCode == 0 {
		receipt.Verdict = "SOLVED"
		receipt.FailureReason = ReasonSolved
	} else {
		receipt.Verdict = "FAILED"
		if oracleTimedOut {
			receipt.FailureReason = ReasonTimeoutOracle
		} else if armExec != nil && armExec.Status == "TIMEOUT" {
			receipt.FailureReason = ReasonTimeoutAgent
		} else if armExec != nil && armExec.Status == "CRASHED" {
			receipt.FailureReason = ReasonContainerCrash
		} else if armExec != nil && armExec.PolicyBlocks > 0 && armExec.Status != "COMPLETED" {
			receipt.FailureReason = ReasonPolicyBlock
		} else {
			receipt.FailureReason = ReasonTestFailed
		}
	}

	receipt.ReceiptHash = receipt.ComputeReceiptHash()
	return receipt, nil
}

// SaveReceipt persists the grading receipt to a JSON file.
func SaveReceipt(receipt *GradingReceipt, outPath string) error {
	data, err := json.MarshalIndent(receipt, "", "  ")
	if err != nil {
		return err
	}
	_ = os.MkdirAll(filepath.Dir(outPath), 0755)
	return os.WriteFile(outPath, data, 0644)
}

// LoadReceipt reads a grading receipt from disk and verifies its receipt hash.
func LoadReceipt(path string) (*GradingReceipt, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var receipt GradingReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		return nil, err
	}
	if expected := receipt.ComputeReceiptHash(); receipt.ReceiptHash != expected {
		return nil, fmt.Errorf("receipt hash verification failed: declared %s, computed %s", receipt.ReceiptHash, expected)
	}
	return &receipt, nil
}
