package main

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"
)

func TestValidatePhaseOrderIncludesSmoke(t *testing.T) {
	withoutSmoke := validatePhaseOrder(false, false, false)
	if slices.Contains(withoutSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder without smoke to omit 'smoke', got %v", withoutSmoke)
	}

	withSmoke := validatePhaseOrder(false, false, true)
	if !slices.Contains(withSmoke, "smoke") {
		t.Fatalf("expected validatePhaseOrder with smoke to contain 'smoke', got %v", withSmoke)
	}
	if withSmoke[len(withSmoke)-1] != "smoke" {
		t.Fatalf("expected 'smoke' phase to be at the end, got %v", withSmoke)
	}
}

func TestValidateSmokePolicyPathUsesForwardSlashes(t *testing.T) {
	policyPath := validateSmokePolicyPath()
	if strings.Contains(policyPath, `\`) {
		t.Fatalf("expected validateSmokePolicyPath to contain no backslashes, got %q", policyPath)
	}
	if !strings.Contains(policyPath, "/") {
		t.Fatalf("expected validateSmokePolicyPath to contain forward slashes, got %q", policyPath)
	}
	expected := "examples/customer-support-readonly-policy.json"
	if policyPath != expected {
		t.Fatalf("expected validateSmokePolicyPath == %q, got %q", expected, policyPath)
	}

	repoRel := filepath.Join("..", "..", filepath.FromSlash(policyPath))
	if info, err := os.Stat(repoRel); err != nil || info.IsDir() {
		t.Fatalf("expected policy path %q to resolve to an existing file in repo, err: %v", repoRel, err)
	}
}

func TestValidateSmokeWSLPathNormalization(t *testing.T) {
	// On Windows, filepath.Join with a WSL Linux path like /tmp/... generates backslashes.
	// Verify that filepath.ToSlash correctly normalizes it for WSL execution.
	wslDir := "/tmp/fak-validate-12345"
	reportPath := filepath.Join(wslDir, "agent-smoke-report.json")
	reportArg := filepath.ToSlash(reportPath)

	if strings.Contains(reportArg, `\`) {
		t.Fatalf("expected WSL smoke report arg to contain no backslashes, got %q", reportArg)
	}
	expected := "/tmp/fak-validate-12345/agent-smoke-report.json"
	if reportArg != expected {
		t.Fatalf("expected WSL smoke report arg == %q, got %q", expected, reportArg)
	}
}
