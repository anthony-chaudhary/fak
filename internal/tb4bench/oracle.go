package tb4bench

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"
)

// OracleContract defines the test runner script and verification criteria for a task.
type OracleContract struct {
	Script          string   `json:"script"`
	ScriptHash      string   `json:"script_hash"`
	TimeoutSeconds  int      `json:"timeout_seconds"`
	ExpectedOutputs []string `json:"expected_outputs,omitempty"`
}

// Validate ensures the contract has positive timeout and matches its declared SHA-256 hash.
func (c *OracleContract) Validate() error {
	if strings.TrimSpace(c.Script) == "" {
		return errors.New("oracle script cannot be empty")
	}
	if c.TimeoutSeconds <= 0 {
		return fmt.Errorf("timeout_seconds must be positive, got %d", c.TimeoutSeconds)
	}
	if !strings.HasPrefix(c.ScriptHash, "sha256:") {
		return fmt.Errorf("script_hash %q must start with 'sha256:'", c.ScriptHash)
	}

	h := sha256.Sum256([]byte(c.Script))
	expected := "sha256:" + hex.EncodeToString(h[:])
	if expected != c.ScriptHash {
		return fmt.Errorf("oracle script hash mismatch: declared %s, computed %s", c.ScriptHash, expected)
	}
	return nil
}

// OracleResult records the execution result of the test oracle inside the sandbox.
type OracleResult struct {
	TaskID        string        `json:"task_id"`
	ExitCode      int           `json:"exit_code"`
	Stdout        string        `json:"stdout"`
	Stderr        string        `json:"stderr"`
	DurationMs    int64         `json:"duration_ms"`
	Passed        bool          `json:"passed"`
	FailureReason FailureReason `json:"failure_reason,omitempty"`
}

// Validate checks the consistency of an OracleResult.
func (r *OracleResult) Validate() error {
	if r.TaskID == "" {
		return errors.New("task_id cannot be empty")
	}
	if r.Passed {
		if r.ExitCode != 0 {
			return fmt.Errorf("passed=true requires exit_code=0, got %d", r.ExitCode)
		}
		if r.FailureReason != "" && r.FailureReason != ReasonSolved {
			return fmt.Errorf("passed=true cannot have failure reason %q", r.FailureReason)
		}
	} else {
		if r.FailureReason == "" {
			return errors.New("passed=false must specify a closed failure reason")
		}
		if err := ValidateReason(r.FailureReason); err != nil {
			return err
		}
		if r.FailureReason == ReasonSolved {
			return errors.New("passed=false cannot carry reason SOLVED")
		}
	}
	return nil
}
