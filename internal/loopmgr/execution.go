package loopmgr

import (
	"errors"
	"fmt"
	"path/filepath"
	"strings"
)

// MaxExecutionTimeoutSeconds is the upper bound on execution timeout (24 hours).
const MaxExecutionTimeoutSeconds int64 = 86400

// ExecutionSpec defines the durable executable specification for a scheduled job.
// When attached to a Job, it provides the literal argument vector, absolute
// working directory, and execution timeout so scheduled tasks can be executed
// deterministically by the runtime.
type ExecutionSpec struct {
	// Argv is the literal command and arguments to execute.
	// Argv[0] is the binary or script to execute; Argv[1:] are positional arguments.
	Argv []string `json:"argv,omitempty"`

	// WorkDir is the absolute working directory where the command executes.
	WorkDir string `json:"work_dir,omitempty"`

	// TimeoutSeconds is the positive bounded execution timeout in seconds.
	TimeoutSeconds int64 `json:"timeout_seconds,omitempty"`
}

// Validate checks that the execution specification meets all requirements:
// - Argv must be non-empty and its first element must not be empty.
// - WorkDir must be an absolute path (filepath.IsAbs).
// - TimeoutSeconds must be positive and <= MaxExecutionTimeoutSeconds.
func (e *ExecutionSpec) Validate() error {
	if e == nil {
		return nil
	}
	if len(e.Argv) == 0 {
		return errors.New("execution argv must not be empty")
	}
	if strings.TrimSpace(e.Argv[0]) == "" {
		return errors.New("execution argv[0] must not be empty")
	}
	if strings.TrimSpace(e.WorkDir) == "" {
		return errors.New("execution work_dir is required")
	}
	if !filepath.IsAbs(e.WorkDir) {
		return fmt.Errorf("execution work_dir %q must be an absolute path", e.WorkDir)
	}
	if e.TimeoutSeconds <= 0 {
		return fmt.Errorf("execution timeout_seconds %d must be positive", e.TimeoutSeconds)
	}
	if e.TimeoutSeconds > MaxExecutionTimeoutSeconds {
		return fmt.Errorf("execution timeout_seconds %d exceeds maximum allowed %d", e.TimeoutSeconds, MaxExecutionTimeoutSeconds)
	}
	return nil
}

// Clone returns a deep copy of the execution specification.
func (e *ExecutionSpec) Clone() *ExecutionSpec {
	if e == nil {
		return nil
	}
	cp := *e
	if e.Argv != nil {
		cp.Argv = make([]string, len(e.Argv))
		copy(cp.Argv, e.Argv)
	}
	return &cp
}
