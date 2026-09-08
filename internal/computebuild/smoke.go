package computebuild

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

// SmokeArtifact executes the target binary with "version --json", falling back to
// "--help" or "-h" if version fails or is not supported. It verifies the binary
// starts and exits cleanly (exit code 0) without crashing, serving as shift-left
// proof that the binary links and runs on the host.
func SmokeArtifact(ctx context.Context, path string) SmokeResult {
	if path == "" {
		return SmokeResult{
			Outcome:  "failed",
			ExitCode: 1,
			Error:    "empty artifact path",
		}
	}
	if !FileExists(path) {
		return SmokeResult{
			Command:  []string{path},
			Outcome:  "failed",
			ExitCode: 1,
			Error:    fmt.Sprintf("artifact file not found: %s", path),
		}
	}

	execPath := path
	if abs, err := filepath.Abs(path); err == nil && FileExists(abs) {
		execPath = abs
	}

	candidates := [][]string{
		{"version", "--json"},
		{"--help"},
		{"-h"},
	}

	var lastResult SmokeResult
	for _, cand := range candidates {
		cmd := exec.CommandContext(ctx, execPath, cand...)
		out, err := cmd.CombinedOutput()
		outputStr := strings.TrimSpace(string(out))

		exitCode := 0
		if cmd.ProcessState != nil {
			exitCode = cmd.ProcessState.ExitCode()
		} else if err != nil {
			exitCode = 1
		}

		fullCmd := append([]string{path}, cand...)
		if err == nil && exitCode == 0 {
			return SmokeResult{
				Command:  fullCmd,
				Outcome:  "success",
				Output:   outputStr,
				ExitCode: 0,
			}
		}

		errMsg := ""
		if err != nil {
			errMsg = err.Error()
		}
		lastResult = SmokeResult{
			Command:  fullCmd,
			Outcome:  "failed",
			Output:   outputStr,
			ExitCode: exitCode,
			Error:    errMsg,
		}
	}

	return lastResult
}

// InspectArtifact computes the size and SHA256 checksum of an artifact file.
func InspectArtifact(path string) (*BuildArtifact, error) {
	fi, err := os.Stat(path)
	if err != nil {
		return nil, err
	}
	f, err := os.Open(path)
	if err != nil {
		return &BuildArtifact{
			Path:      path,
			SizeBytes: fi.Size(),
		}, nil
	}
	defer f.Close()

	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return &BuildArtifact{
			Path:      path,
			SizeBytes: fi.Size(),
		}, nil
	}
	return &BuildArtifact{
		Path:      path,
		SizeBytes: fi.Size(),
		SHA256:    hex.EncodeToString(h.Sum(nil)),
	}, nil
}

// WriteReceiptAtomic serializes receipt as formatted JSON and writes it atomically to path.
func WriteReceiptAtomic(path string, receipt *ComputeBuildReceipt) error {
	if path == "" || receipt == nil {
		return nil
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("creating receipt directory: %w", err)
	}

	tmpFile, err := os.CreateTemp(dir, ".compute-build-receipt-*.tmp")
	if err != nil {
		return fmt.Errorf("creating temp receipt: %w", err)
	}
	tmpName := tmpFile.Name()
	cleanTmp := true
	defer func() {
		if cleanTmp {
			_ = tmpFile.Close()
			_ = os.Remove(tmpName)
		}
	}()

	enc := json.NewEncoder(tmpFile)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)
	if err := enc.Encode(receipt); err != nil {
		_ = tmpFile.Close()
		return fmt.Errorf("encoding receipt JSON: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		return fmt.Errorf("closing temp receipt: %w", err)
	}

	// Remove existing destination file before atomic rename to prevent Windows collision.
	_ = os.Remove(path)
	if err := os.Rename(tmpName, path); err != nil {
		return fmt.Errorf("renaming receipt to %s: %w", path, err)
	}
	cleanTmp = false
	return nil
}

type receiptTracker struct {
	receipt *ComputeBuildReceipt
	start   time.Time
}

func newReceiptTracker(backend, command, receiptPath string) *receiptTracker {
	now := time.Now().UTC()
	return &receiptTracker{
		receipt: &ComputeBuildReceipt{
			Schema:      ComputeBuildReceiptSchema,
			Backend:     backend,
			Command:     command,
			Outcome:     "success",
			ExitCode:    0,
			StartedAt:   now.Format(time.RFC3339Nano),
			ReceiptPath: receiptPath,
			Phases:      make([]ComputeBuildPhase, 0),
		},
		start: now,
	}
}

func (t *receiptTracker) recordPhase(name string, fn func() error) error {
	phaseStart := time.Now()
	err := fn()
	elapsed := time.Since(phaseStart).Milliseconds()
	zeroCode := 0
	phase := ComputeBuildPhase{
		Name:      name,
		ElapsedMS: elapsed,
	}
	if err != nil {
		errorCode := 1
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			errorCode = exitErr.ExitCode()
		}
		phase.Outcome = "failed"
		phase.ExitCode = &errorCode
		phase.Error = err.Error()
		t.receipt.Phases = append(t.receipt.Phases, phase)

		t.receipt.Outcome = "failed"
		t.receipt.ExitCode = errorCode
		t.receipt.Error = err.Error()
		return err
	}
	phase.Outcome = "success"
	phase.ExitCode = &zeroCode
	t.receipt.Phases = append(t.receipt.Phases, phase)
	return nil
}

func (t *receiptTracker) recordSmoke(ctx context.Context, binaryPath string) error {
	phaseStart := time.Now()
	smokeResult := SmokeArtifact(ctx, binaryPath)
	elapsed := time.Since(phaseStart).Milliseconds()

	t.receipt.Smoke = &smokeResult
	phase := ComputeBuildPhase{
		Name:      "smoke",
		Outcome:   smokeResult.Outcome,
		ElapsedMS: elapsed,
		ExitCode:  &smokeResult.ExitCode,
		Error:     smokeResult.Error,
	}
	t.receipt.Phases = append(t.receipt.Phases, phase)

	if smokeResult.Outcome != "success" {
		t.receipt.Outcome = "failed"
		t.receipt.ExitCode = smokeResult.ExitCode
		t.receipt.Error = fmt.Sprintf("smoke execution failed: %s", smokeResult.Error)
		return fmt.Errorf("smoke execution failed: %s", smokeResult.Error)
	}
	return nil
}

func (t *receiptTracker) finish(path string) error {
	finish := time.Now().UTC()
	t.receipt.FinishedAt = finish.Format(time.RFC3339Nano)
	t.receipt.ElapsedMS = finish.Sub(t.start).Milliseconds()
	if t.receipt.Outcome != "success" {
		t.receipt.Artifact = nil
		t.receipt.ShaderBundleSHA256 = ""
	}
	if path != "" {
		t.receipt.ReceiptPath = path
		return WriteReceiptAtomic(path, t.receipt)
	}
	return nil
}
