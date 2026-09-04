package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/flowmetrics"
)

func TestSweepCensusTextOutput(t *testing.T) {
	root := t.TempDir()
	sweepGit(t, root, "init")
	sweepGit(t, root, "config", "user.email", "test@example.com")
	sweepGit(t, root, "config", "user.name", "Test User")

	// Commit initial file so HEAD rev exists
	initial := filepath.Join(root, "init.txt")
	if err := os.WriteFile(initial, []byte("initial\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sweepGit(t, root, "add", "init.txt")
	sweepGit(t, root, "commit", "-m", "chore: initial commit")

	// Create files:
	// 1. zz_test.go -> litter
	// 2. .hidden.go -> litter
	// 3. root_throwaway.log -> litter
	// 4. real_code.go -> unlanded
	// 5. worker.go -> unlanded
	if err := os.WriteFile(filepath.Join(root, "zz_test.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, ".hidden.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "scratch.log"), []byte("log\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "real_code.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "worker.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runSweep(&stdout, &stderr, []string{"--dir", root, "--census"})
	if code != 0 {
		t.Fatalf("runSweep --census exit=%d, want 0; stderr=%s", code, stderr.String())
	}

	out := stdout.String()

	// Check ceilings, verdicts, rev, candidates
	expectedSubstrings := []string{
		"Rev-pinned working tree census:",
		"HEAD rev:",
		"Untracked source files:",
		"PASS",
		"Scratch probe files:",
		"Recent writers (last 10m):",
		"Modified source files:",
		"Added/Deleted lines churn:",
		"Oldest untracked file age:",
		"Candidate paths preview:",
		"Litter candidate paths",
		"zz_test.go",
		".hidden.go",
		"scratch.log",
		"Unlanded candidate paths",
		"real_code.go",
		"worker.go",
		"Preview only: removed nothing. Run with --clean-junk or explicit fak commit to actuate.",
	}

	for _, sub := range expectedSubstrings {
		if !strings.Contains(out, sub) {
			t.Errorf("stdout missing expected substring %q;\nOutput:\n%s", sub, out)
		}
	}
}

func TestSweepCensusJSONOutput(t *testing.T) {
	root := t.TempDir()
	sweepGit(t, root, "init")
	sweepGit(t, root, "config", "user.email", "test@example.com")
	sweepGit(t, root, "config", "user.name", "Test User")

	initial := filepath.Join(root, "main.go")
	if err := os.WriteFile(initial, []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sweepGit(t, root, "add", "main.go")
	sweepGit(t, root, "commit", "-m", "chore: initial commit")

	// Scratch probe and real code
	if err := os.WriteFile(filepath.Join(root, "zz_probe.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "feature.go"), []byte("package main\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	var stdout, stderr bytes.Buffer
	code := runSweep(&stdout, &stderr, []string{"--dir", root, "--census", "--json"})
	if code != 0 {
		t.Fatalf("runSweep --census --json exit=%d, want 0; stderr=%s", code, stderr.String())
	}

	var res sweepCensusResult
	if err := json.Unmarshal(stdout.Bytes(), &res); err != nil {
		t.Fatalf("json unmarshal failed: %v\nOutput:\n%s", err, stdout.String())
	}

	if res.Schema != "fak-sweep-census/1" {
		t.Errorf("schema = %q, want fak-sweep-census/1", res.Schema)
	}
	if res.Rev == "" {
		t.Errorf("rev is empty")
	}
	if !res.Census.Measured {
		t.Errorf("census.measured = false, want true")
	}
	if res.RemovedCount != 0 {
		t.Errorf("removed_count = %d, want 0", res.RemovedCount)
	}

	// Litter vs Unlanded paths
	hasProbeInLitter := false
	for _, p := range res.LitterPaths {
		if p == "zz_probe.go" {
			hasProbeInLitter = true
		}
	}
	if !hasProbeInLitter {
		t.Errorf("expected zz_probe.go in litter_paths, got: %v", res.LitterPaths)
	}

	hasFeatureInUnlanded := false
	for _, p := range res.UnlandedPaths {
		if p == "feature.go" {
			hasFeatureInUnlanded = true
		}
	}
	if !hasFeatureInUnlanded {
		t.Errorf("expected feature.go in unlanded_paths, got: %v", res.UnlandedPaths)
	}
}

func TestSweepCensusDeletesZeroFiles(t *testing.T) {
	root := t.TempDir()
	sweepGit(t, root, "init")
	sweepGit(t, root, "config", "user.email", "test@example.com")
	sweepGit(t, root, "config", "user.name", "Test User")

	initial := filepath.Join(root, "tracked.txt")
	if err := os.WriteFile(initial, []byte("tracked\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	sweepGit(t, root, "add", "tracked.txt")
	sweepGit(t, root, "commit", "-m", "chore: init")

	// Create throwaway files, scratch probes, uncommitted code
	paths := []string{
		"zz_test.go",
		".hidden.go",
		"throwaway.tmp",
		"work.go",
	}
	for _, p := range paths {
		if err := os.WriteFile(filepath.Join(root, p), []byte("content\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	var stdout, stderr bytes.Buffer
	code := runSweep(&stdout, &stderr, []string{"--dir", root, "--census"})
	if code != 0 {
		t.Fatalf("runSweep exit=%d; stderr=%s", code, stderr.String())
	}

	// Verify all files still exist on disk
	for _, p := range paths {
		fullPath := filepath.Join(root, p)
		if _, err := os.Stat(fullPath); os.IsNotExist(err) {
			t.Fatalf("file %q was deleted by --census!", p)
		}
	}
}

func TestSweepCensusClassifyLitterVsUnlanded(t *testing.T) {
	entries := []dirtyEntry{
		{Path: "zz_test.go", Status: "??"},
		{Path: "internal/sub/zz_helper.go", Status: "??"},
		{Path: ".hidden.go", Status: "??"},
		{Path: "pkg/.dot.go", Status: "??"},
		{Path: "stray.log", Status: "??", Untracked: true},
		{Path: "coverage.out", Status: "??", Untracked: true},
		{Path: "real_code.go", Status: "M"},
		{Path: "untracked_work.go", Status: "??"},
		{Path: "internal/service.go", Status: "M"},
	}

	litter, unlanded := classifyCandidatePaths(entries)

	wantLitter := map[string]bool{
		"zz_test.go":                true,
		"internal/sub/zz_helper.go": true,
		".hidden.go":                true,
		"pkg/.dot.go":               true,
		"stray.log":                 true,
		"coverage.out":              true,
	}

	wantUnlanded := map[string]bool{
		"real_code.go":        true,
		"untracked_work.go":   true,
		"internal/service.go": true,
	}

	if len(litter) != len(wantLitter) {
		t.Errorf("litter count = %d, want %d: %v", len(litter), len(wantLitter), litter)
	}
	for _, p := range litter {
		if !wantLitter[p] {
			t.Errorf("unexpected litter path: %s", p)
		}
	}

	if len(unlanded) != len(wantUnlanded) {
		t.Errorf("unlanded count = %d, want %d: %v", len(unlanded), len(wantUnlanded), unlanded)
	}
	for _, p := range unlanded {
		if !wantUnlanded[p] {
			t.Errorf("unexpected unlanded path: %s", p)
		}
	}
}

func TestSweepCensusRenderNotMeasuredAndStatFailures(t *testing.T) {
	// 1. Not measured case
	var buf bytes.Buffer
	unmeasuredTree := flowmetrics.TreeWIP{Measured: false}
	renderSweepCensus(&buf, unmeasuredTree, nil, nil)
	if !strings.Contains(buf.String(), "Working-tree WIP: NOT MEASURED") {
		t.Errorf("expected 'Working-tree WIP: NOT MEASURED', got:\n%s", buf.String())
	}

	// 2. StatFailures > 0 case
	buf.Reset()
	statFailTree := flowmetrics.TreeWIP{
		Measured:             true,
		Rev:                  "abc1234",
		UntrackedGo:          25,
		ScratchLitter:        15,
		RecentWriters:        5,
		ModifiedGo:           2,
		AddedLines:           10,
		DeletedLines:         2,
		OldestUntrackedHours: 48.0,
		StatFailures:         3,
	}
	renderSweepCensus(&buf, statFailTree, []string{"zz_fail.go"}, []string{"real.go"})
	out := buf.String()

	if !strings.Contains(out, "Stat failures: 3 file(s) could not be stat'd") {
		t.Errorf("expected stat failures highlighted, got:\n%s", out)
	}
	if !strings.Contains(out, "DEFECT") {
		t.Errorf("expected DEFECT for ceilings exceeded, got:\n%s", out)
	}
	if !strings.Contains(out, "2.0d (48.0h)") {
		t.Errorf("expected formatted days/hours for oldest untracked, got:\n%s", out)
	}
}
