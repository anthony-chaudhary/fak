package taskrun

import (
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// runCandidateTests assembles immutable candidate and visible-test inputs in a
// fresh root. The sandbox may write only compiler caches and temporary files.
func runCandidateTests(ctx context.Context, candidateRoot, targetFile, visibleRoot, oracleRoot string) (TestResult, error) {
	candidate, err := canonicalDirectory(candidateRoot)
	if err != nil {
		return TestResult{ExitCode: -1}, fmt.Errorf("candidate root: %w", err)
	}
	visible, err := canonicalDirectory(visibleRoot)
	if err != nil {
		return TestResult{ExitCode: -1}, fmt.Errorf("visible root: %w", err)
	}
	oracle, err := canonicalDirectory(oracleRoot)
	if err != nil {
		return TestResult{ExitCode: -1}, fmt.Errorf("oracle root: %w", err)
	}
	if pathsOverlap(candidate, visible) || pathsOverlap(candidate, oracle) || pathsOverlap(visible, oracle) {
		return TestResult{ExitCode: -1}, errors.New("candidate, visible, and oracle roots must be disjoint")
	}
	if filepath.Base(targetFile) != targetFile || targetFile == "." || targetFile == ".." {
		return TestResult{ExitCode: -1}, errors.New("target file must be a base name")
	}
	target, err := readRegularBounded(filepath.Join(candidate, targetFile), 1<<20)
	if err != nil {
		return TestResult{ExitCode: -1}, fmt.Errorf("candidate target: %w", err)
	}
	visibleTest, err := readRegularBounded(filepath.Join(visible, "visible_test.go"), 1<<20)
	if err != nil {
		return TestResult{ExitCode: -1}, fmt.Errorf("visible test: %w", err)
	}
	check, err := privateScratch("agentbench-check-")
	if err != nil {
		return TestResult{ExitCode: -1}, err
	}
	defer os.RemoveAll(check)
	for name, body := range map[string][]byte{
		"go.mod":          []byte("module fixture\n\ngo 1.26\n"),
		targetFile:        target,
		"visible_test.go": visibleTest,
	} {
		if err := os.WriteFile(filepath.Join(check, name), body, 0400); err != nil {
			return TestResult{ExitCode: -1}, err
		}
	}
	return runSandboxedTests(ctx, check, oracle, false)
}

func readRegularBounded(path string, limit int64) ([]byte, error) {
	info, err := os.Lstat(path)
	if err != nil {
		return nil, err
	}
	if !info.Mode().IsRegular() || info.Mode()&os.ModeSymlink != 0 || info.Size() > limit {
		return nil, errors.New("must be a bounded regular non-symlink file")
	}
	return os.ReadFile(path)
}
