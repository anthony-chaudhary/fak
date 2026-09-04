package main

import (
	"bytes"
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunCodeDebtFromFile(t *testing.T) {
	tempDir := t.TempDir()
	scorecardJSON := `{
  "workspace": "/test/ws",
  "corpus": {
    "score": 85.0,
    "grade": "B",
    "code_debt": 3,
    "debt_by_category": {
      "modularity": 2,
      "internal_consistency": 1
    }
  },
  "kpis": [
    {
      "kpi": "architecture",
      "score": 88,
      "defects": [
        "god-file cmd/fak/main.go (1600 lines > 1500)",
        "god-function internal/gateway/stream.go:handleStream (250 lines > 200)"
      ]
    },
    {
      "kpi": "format",
      "score": 88,
      "defects": [
        "unformatted (run gofmt -w): internal/foo/foo.go"
      ]
    }
  ]
}`
	filePath := filepath.Join(tempDir, "scorecard.json")
	if err := os.WriteFile(filePath, []byte(scorecardJSON), 0o644); err != nil {
		t.Fatal(err)
	}

	// 1. Query by KPI
	var stdout, stderr bytes.Buffer
	code := runCodeDebt(&stdout, &stderr, []string{"--from", filePath, "--kpi", "architecture"})
	if code != 1 {
		t.Fatalf("runCodeDebt code = %d, want 1 (defects present); stderr: %s", code, stderr.String())
	}
	outStr := stdout.String()
	if !strings.Contains(outStr, "2 matched defect(s)") || !strings.Contains(outStr, "god-file") {
		t.Fatalf("unexpected stdout: %s", outStr)
	}

	// 2. Query clean KPI
	stdout.Reset()
	stderr.Reset()
	code = runCodeDebt(&stdout, &stderr, []string{"--from", filePath, "--kpi", "tests"})
	if code != 0 {
		t.Fatalf("runCodeDebt code = %d, want 0 (no defects); stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "0 matched defect(s)") {
		t.Fatalf("unexpected stdout for clean KPI: %s", stdout.String())
	}

	// 3. Count flag
	stdout.Reset()
	stderr.Reset()
	code = runCodeDebt(&stdout, &stderr, []string{"--from", filePath, "--kpi", "format", "--count"})
	if code != 1 {
		t.Fatalf("count code = %d, want 1", code)
	}
	if strings.TrimSpace(stdout.String()) != "1" {
		t.Fatalf("count = %q, want \"1\"", stdout.String())
	}

	// 4. JSON output
	stdout.Reset()
	stderr.Reset()
	code = runCodeDebt(&stdout, &stderr, []string{"--from", filePath, "--category", "modularity", "--json"})
	if code != 1 {
		t.Fatalf("json code = %d, want 1", code)
	}
	var parsed map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &parsed); err != nil {
		t.Fatalf("json unmarshal failed: %v", err)
	}
	if matched, ok := parsed["matched_debt"].(float64); !ok || int(matched) != 2 {
		t.Fatalf("matched_debt = %v, want 2", parsed["matched_debt"])
	}

	// 5. Summary output
	stdout.Reset()
	stderr.Reset()
	code = runCodeDebt(&stdout, &stderr, []string{"--from", filePath, "--summary"})
	if code != 1 {
		t.Fatalf("summary code = %d, want 1", code)
	}
	if !strings.Contains(stdout.String(), "code debt summary:") || !strings.Contains(stdout.String(), "modularity") {
		t.Fatalf("summary missing expected text: %s", stdout.String())
	}
}

func TestRunCodeDebtNativeWorkspace(t *testing.T) {
	tempDir := t.TempDir()

	// Minimal go.mod
	if err := os.WriteFile(filepath.Join(tempDir, "go.mod"), []byte("module testmod\n\ngo 1.26\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	// File with god-function
	pkgDir := filepath.Join(tempDir, "pkg")
	if err := os.MkdirAll(pkgDir, 0o755); err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	sb.WriteString("package pkg\n\nfunc OversizedFunc() {\n")
	for i := 0; i < 210; i++ {
		sb.WriteString("\t_ = 1\n")
	}
	sb.WriteString("}\n")
	if err := os.WriteFile(filepath.Join(pkgDir, "code.go"), []byte(sb.String()), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runCodeDebt(&stdout, &stderr, []string{"--workspace", tempDir, "--kpi", "architecture"})
	if code != 1 {
		t.Fatalf("code = %d, want 1; stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "god-function") {
		t.Fatalf("missing god-function in stdout: %s", stdout.String())
	}
}

func TestRunCodeQualityScoreQueryFlags(t *testing.T) {
	root := t.TempDir()
	writeCodeQualityCheckers(t, root)

	called := false
	var passedArgv []string
	run := func(_ context.Context, _ string, argv []string) ([]byte, []byte, int, error) {
		called = true
		passedArgv = argv
		return []byte("code-debt query: 1 matched defect(s)"), nil, 1, nil
	}

	var stdout, stderr bytes.Buffer
	code := runCodeQualityScore(&stdout, &stderr, []string{
		"--workspace", root,
		"--deterministic",
		"--kpi", "architecture",
		"--count",
	}, run)

	if !called {
		t.Fatal("expected checker to be called")
	}
	if code != 1 {
		t.Fatalf("code = %d, want 1", code)
	}

	foundDeterministic := false
	foundKPI := false
	foundCount := false
	for _, arg := range passedArgv {
		if arg == "--deterministic" {
			foundDeterministic = true
		}
		if arg == "--kpi" {
			foundKPI = true
		}
		if arg == "--count" {
			foundCount = true
		}
	}
	if !foundDeterministic || !foundKPI || !foundCount {
		t.Fatalf("passedArgv = %v, expected --deterministic, --kpi, --count", passedArgv)
	}
}
