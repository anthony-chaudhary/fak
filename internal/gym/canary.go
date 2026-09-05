package gym

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sandbox"
)

// CanaryVerdict represents the structured receipt of a canary test run in an ephemeral gym arena.
type CanaryVerdict struct {
	// Passed indicates whether the execution succeeded and the host trunk was completely unpolluted.
	Passed bool `json:"passed"`
	// ResetDuration records the elapsed time to reset the ephemeral overlay to pristine state (<10ms).
	ResetDuration time.Duration `json:"reset_duration"`
	// ArtifactsModified lists the workspace paths created, modified, or deleted during execution.
	ArtifactsModified []string `json:"artifacts_modified,omitempty"`
	// Error records any execution failure, reset error, or pollution violation.
	Error error `json:"error,omitempty"`
}

// CanaryTask defines an executable task or programmatic validation hook to run inside a canary arena.
type CanaryTask struct {
	Name      string
	Command   string
	Argv      []string
	Env       []string
	Stdin     []byte
	TimeoutMS int64
	Action    func(arena *Arena) error
}

// CanaryRunner orchestrates canary test executions inside ephemeral gym arenas.
type CanaryRunner struct {
	cfg Config
}

// NewCanaryRunner creates a CanaryRunner configured with cfg.
func NewCanaryRunner(cfg Config) *CanaryRunner {
	return &CanaryRunner{cfg: cfg}
}

// Run executes a command request in an ephemeral gym arena, verifies clean teardown without
// polluting the host trunk, and returns the typed CanaryVerdict receipt.
func (r *CanaryRunner) Run(ctx context.Context, req sandbox.ExecutionRequest) (*CanaryVerdict, error) {
	task := CanaryTask{
		Name:      req.Command,
		Command:   req.Command,
		Argv:      req.Argv,
		Env:       req.Env,
		Stdin:     req.Stdin,
		TimeoutMS: req.TimeoutMS,
	}
	return r.RunTask(ctx, task)
}

// RunTask executes a CanaryTask inside an ephemeral gym arena and returns the typed receipt.
func (r *CanaryRunner) RunTask(ctx context.Context, task CanaryTask) (*CanaryVerdict, error) {
	// 1. Snapshot the host trunk / base workspace prior to execution to detect any pollution
	beforeSnapshot, err := snapshotDirectory(r.cfg.BaseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to snapshot base workspace prior to canary run: %w", err)
	}

	// 2. Initialize ephemeral gym arena
	arena, err := Create(ctx, r.cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create ephemeral canary arena: %w", err)
	}
	defer arena.Destroy()

	// 3. Dispatch execution
	var execErr error
	if task.Action != nil {
		execErr = task.Action(arena)
	} else if task.Command != "" {
		req := sandbox.ExecutionRequest{
			Command:    task.Command,
			Argv:       task.Argv,
			Env:        task.Env,
			Stdin:      task.Stdin,
			TimeoutMS:  task.TimeoutMS,
			WorkingDir: arena.Path(),
		}
		res, err := arena.Execute(ctx, req)
		if err != nil {
			execErr = err
		} else if res.ExitCode != 0 {
			execErr = fmt.Errorf("canary execution failed with exit code %d (stderr: %s)", res.ExitCode, string(res.Stderr))
		}
	}

	// 4. Capture modified artifacts
	artifactsModified := arena.ModifiedArtifacts()

	// 5. Measure reset duration
	resetStart := time.Now()
	resetErr := arena.Reset(ctx)
	resetDuration := time.Since(resetStart)

	// 6. Verify base workspace has ZERO residual pollution
	afterSnapshot, err := snapshotDirectory(r.cfg.BaseDir)
	if err != nil {
		return nil, fmt.Errorf("failed to verify base workspace post-canary run: %w", err)
	}

	var pollutionErr error
	if !reflect.DeepEqual(beforeSnapshot, afterSnapshot) {
		pollutionErr = errors.New("host trunk was polluted during canary execution")
	}

	// 7. Assemble verdict
	passed := (execErr == nil && resetErr == nil && pollutionErr == nil)
	var finalErr error
	if execErr != nil {
		finalErr = execErr
	} else if resetErr != nil {
		finalErr = resetErr
	} else if pollutionErr != nil {
		finalErr = pollutionErr
	}

	return &CanaryVerdict{
		Passed:            passed,
		ResetDuration:     resetDuration,
		ArtifactsModified: artifactsModified,
		Error:             finalErr,
	}, nil
}

type fileSnapshot struct {
	RelPath string
	Size    int64
	Mode    os.FileMode
}

func snapshotDirectory(dir string) ([]fileSnapshot, error) {
	var list []fileSnapshot
	err := filepath.Walk(dir, func(p string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil || rel == "." {
			return nil
		}
		list = append(list, fileSnapshot{
			RelPath: filepath.ToSlash(rel),
			Size:    info.Size(),
			Mode:    info.Mode(),
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Slice(list, func(i, j int) bool {
		return list[i].RelPath < list[j].RelPath
	})
	return list, nil
}
