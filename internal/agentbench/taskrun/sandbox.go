// Package taskrun executes agent benchmark fixture tests inside an OS sandbox.
package taskrun

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"
)

const maxTestOutput = 64 << 10

type TestResult struct {
	Output   string
	ExitCode int
	Sandbox  string
	Duration time.Duration
}

// SandboxUnavailableError reports that the required native sandbox cannot be
// used. Callers must not fall back to unsandboxed execution.
type SandboxUnavailableError struct{ Reason string }

func (e *SandboxUnavailableError) Error() string {
	return "agentbench sandbox unavailable: " + e.Reason
}

func SandboxAvailable() error {
	goBinary := filepath.Join(runtime.GOROOT(), "bin", "go")
	if info, err := os.Stat(goBinary); err != nil || !info.Mode().IsRegular() || info.Mode()&0111 == 0 {
		return &SandboxUnavailableError{Reason: "pinned GOROOT go binary is not executable"}
	}
	return platformSandboxAvailable()
}

type sandboxCommandConfig struct {
	Trial         string
	Oracle        string
	Cache         string
	Temp          string
	GoRoot        string
	TrialWritable bool
}

func RunSandboxedTests(ctx context.Context, trialRoot, oracleRoot string) (TestResult, error) {
	return runSandboxedTests(ctx, trialRoot, oracleRoot, true)
}

func runSandboxedTests(ctx context.Context, trialRoot, oracleRoot string, trialWritable bool) (TestResult, error) {
	started := time.Now()
	result := TestResult{ExitCode: -1, Sandbox: platformSandboxName()}
	finish := func() TestResult { result.Duration = time.Since(started); return result }
	if err := SandboxAvailable(); err != nil {
		return finish(), err
	}
	trial, err := canonicalDirectory(trialRoot)
	if err != nil {
		return finish(), fmt.Errorf("trial root: %w", err)
	}
	oracle, err := canonicalDirectory(oracleRoot)
	if err != nil {
		return finish(), fmt.Errorf("oracle root: %w", err)
	}
	if pathsOverlap(trial, oracle) {
		return finish(), errors.New("agentbench sandbox: trial and oracle roots overlap")
	}

	cache, err := privateScratch("agentbench-gocache-")
	if err != nil {
		return finish(), err
	}
	defer os.RemoveAll(cache)
	temp, err := privateScratch("agentbench-gotmp-")
	if err != nil {
		return finish(), err
	}
	defer os.RemoveAll(temp)
	if pathsOverlap(cache, trial) || pathsOverlap(cache, oracle) || pathsOverlap(temp, trial) || pathsOverlap(temp, oracle) {
		return finish(), errors.New("agentbench sandbox: scratch root overlaps protected roots")
	}

	goRoot, err := filepath.EvalSymlinks(runtime.GOROOT())
	if err != nil {
		return finish(), fmt.Errorf("agentbench sandbox: resolve GOROOT: %w", err)
	}
	cmd, err := platformSandboxCommand(ctx, sandboxCommandConfig{
		Trial: trial, Oracle: oracle, Cache: cache, Temp: temp, GoRoot: goRoot, TrialWritable: trialWritable,
	})
	if err != nil {
		return finish(), err
	}
	configureChildProcess(cmd)
	var output boundedBuffer
	cmd.Stdout = &output
	cmd.Stderr = &output
	if err := cmd.Start(); err != nil {
		result.Output = output.String()
		return finish(), fmt.Errorf("agentbench sandbox start: %w", err)
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	var runErr error
	select {
	case runErr = <-done:
	case <-ctx.Done():
		killChildProcess(cmd)
		<-done
		result.Output = output.String()
		return finish(), ctx.Err()
	}
	result.Output = output.String()
	result.ExitCode = 0
	if runErr != nil {
		var exitErr *exec.ExitError
		if errors.As(runErr, &exitErr) {
			result.ExitCode = exitErr.ExitCode()
			return finish(), nil
		}
		return finish(), fmt.Errorf("agentbench sandbox wait: %w", runErr)
	}
	return finish(), nil
}

func canonicalDirectory(path string) (string, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return "", err
	}
	info, err := os.Lstat(abs)
	if err != nil {
		return "", err
	}
	if !info.IsDir() || info.Mode()&os.ModeSymlink != 0 {
		return "", errors.New("must be an existing non-symlink directory")
	}
	real, err := filepath.EvalSymlinks(abs)
	if err != nil {
		return "", err
	}
	return filepath.Clean(real), nil
}

func privateScratch(pattern string) (string, error) {
	dir, err := os.MkdirTemp("", pattern)
	if err != nil {
		return "", fmt.Errorf("agentbench sandbox scratch: %w", err)
	}
	if err := os.Chmod(dir, 0700); err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	real, err := filepath.EvalSymlinks(dir)
	if err != nil {
		os.RemoveAll(dir)
		return "", err
	}
	return real, nil
}

func pathsOverlap(a, b string) bool {
	within := func(path, root string) bool {
		rel, err := filepath.Rel(root, path)
		return err == nil && rel != ".." && !strings.HasPrefix(rel, ".."+string(filepath.Separator))
	}
	return within(a, b) || within(b, a)
}

type boundedBuffer struct {
	mu        sync.Mutex
	buf       bytes.Buffer
	truncated bool
}

func (b *boundedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	n := len(p)
	remaining := maxTestOutput - b.buf.Len()
	if remaining > 0 {
		if len(p) > remaining {
			p = p[:remaining]
		}
		_, _ = b.buf.Write(p)
	}
	if n > remaining {
		b.truncated = true
	}
	return n, nil
}

func (b *boundedBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
