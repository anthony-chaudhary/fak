package factorymigrate

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"testing"
)

func TestParseInventory_EmbeddedFallback(t *testing.T) {
	items, err := ParseInventory("")
	if err != nil {
		t.Fatalf("ParseInventory(\"\") failed: %v", err)
	}
	if len(items) != 100 {
		t.Fatalf("expected 100 items from embedded inventory, got %d", len(items))
	}

	for i, item := range items {
		expectedNum := i + 1
		if item.Number != expectedNum {
			t.Errorf("item[%d] has number %d, want %d", i, item.Number, expectedNum)
		}
		if item.Cohort == "" {
			t.Errorf("item %d has empty cohort", item.Number)
		}
		if item.SourcePath == "" {
			t.Errorf("item %d has empty source path", item.Number)
		}
		if item.TargetPath == "" {
			t.Errorf("item %d has empty target path", item.Number)
		}
		if item.TargetPkg == "" {
			t.Errorf("item %d has empty target pkg", item.Number)
		}
		if item.Status != StatusUnmigrated {
			t.Errorf("item %d initial status = %q, want %q", item.Number, item.Status, StatusUnmigrated)
		}
	}

	// Verify specific boundary items
	if items[0].SourcePath != "tools/crash_audit.py" || items[0].TargetPath != "platform/watchdogs/crash_audit.go" {
		t.Errorf("item 1 mismatch: got %+v", items[0])
	}
	if items[99].SourcePath != "tools/ctxcost.py" || items[99].TargetPath != "platform/memsync/ctxcost.go" {
		t.Errorf("item 100 mismatch: got %+v", items[99])
	}

	// Test fallback on invalid path
	itemsFallback, err := ParseInventory("nonexistent_path_never_exists.md")
	if err != nil {
		t.Fatalf("ParseInventory on nonexistent file failed: %v", err)
	}
	if len(itemsFallback) != 100 {
		t.Fatalf("expected 100 items from fallback, got %d", len(itemsFallback))
	}
}

func TestParseInventory_FromFile(t *testing.T) {
	// Locate repository root inventory file
	path := filepath.Join("..", "..", "docs", "dev-process-top-100-tools-inventory.md")
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Skip("docs/dev-process-top-100-tools-inventory.md not found at expected relative path")
	}

	items, err := ParseInventory(path)
	if err != nil {
		t.Fatalf("failed to parse %s: %v", path, err)
	}
	if len(items) != 100 {
		t.Fatalf("expected 100 items from file, got %d", len(items))
	}
}

func TestAuditStatus(t *testing.T) {
	tmpDir := t.TempDir()
	fakRoot := filepath.Join(tmpDir, "fak")
	privateRoot := filepath.Join(tmpDir, "fak-private")

	if err := os.MkdirAll(filepath.Join(fakRoot, "tools"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(privateRoot, "platform", "watchdogs"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(fakRoot, "internal", "procguard"), 0755); err != nil {
		t.Fatal(err)
	}

	// Item 1: target exists in privateRoot, legacy source retired (does not exist in fakRoot) -> MIGRATED
	if err := os.WriteFile(filepath.Join(privateRoot, "platform", "watchdogs", "crash_audit.go"), []byte("package watchdogs\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Item 2: known migration (watchdog.go in privateRoot), source still present in fakRoot -> PARTIAL
	if err := os.WriteFile(filepath.Join(fakRoot, "tools", "dos_supervisor_watchdog.py"), []byte("#!/usr/bin/env python3\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(privateRoot, "platform", "watchdogs", "watchdog.go"), []byte("package watchdogs\n"), 0644); err != nil {
		t.Fatal(err)
	}

	// Item 12: known migration (internal/procguard/procguard.go in fakRoot), source in fakRoot -> PARTIAL
	if err := os.WriteFile(filepath.Join(fakRoot, "tools", "proc_resource_guard.py"), []byte("#!/usr/bin/env python3\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakRoot, "internal", "procguard", "procguard.go"), []byte("package procguard\n"), 0644); err != nil {
		t.Fatal(err)
	}

	items, err := ParseInventory("")
	if err != nil {
		t.Fatal(err)
	}

	report := AuditStatus(fakRoot, privateRoot, items)
	if report.Total != 100 {
		t.Errorf("report.Total = %d, want 100", report.Total)
	}
	if report.Migrated != 1 {
		t.Errorf("report.Migrated = %d, want 1", report.Migrated)
	}
	if report.Partial < 2 {
		t.Errorf("report.Partial = %d, want >= 2", report.Partial)
	}
	if report.Percent != 1.0 {
		t.Errorf("report.Percent = %f, want 1.0", report.Percent)
	}
	if len(report.Cohorts) == 0 {
		t.Errorf("expected cohort summaries, got 0")
	}

	// Check Item 1 status
	if report.Items[0].Status != StatusMigrated {
		t.Errorf("item 1 status = %q, want %q", report.Items[0].Status, StatusMigrated)
	}
	// Check Item 2 status
	if report.Items[1].Status != StatusPartial {
		t.Errorf("item 2 status = %q, want %q", report.Items[1].Status, StatusPartial)
	}
	// Check Item 12 status
	if report.Items[11].Status != StatusPartial {
		t.Errorf("item 12 status = %q, want %q", report.Items[11].Status, StatusPartial)
	}
}

func TestNextCandidates(t *testing.T) {
	items, err := ParseInventory("")
	if err != nil {
		t.Fatal(err)
	}

	// Mock report where item 1 is migrated
	items[0].Status = StatusMigrated
	report := Report{
		Total:      len(items),
		Migrated:   1,
		Unmigrated: len(items) - 1,
		Items:      items,
	}

	// Default next 5
	next5 := NextCandidates(report, 5, "")
	if len(next5) != 5 {
		t.Fatalf("expected 5 candidates, got %d", len(next5))
	}
	if next5[0].Number != 2 {
		t.Errorf("expected first candidate to be item 2, got %d", next5[0].Number)
	}

	// Filter by cohort
	watchdogs := NextCandidates(report, 10, "watchdogs")
	for _, cand := range watchdogs {
		if cand.TargetPkg != "platform/watchdogs" {
			t.Errorf("candidate %d targetPkg = %q, want platform/watchdogs", cand.Number, cand.TargetPkg)
		}
	}

	// Uncapped count
	allRemaining := NextCandidates(report, 0, "")
	if len(allRemaining) != 99 {
		t.Errorf("expected 99 remaining candidates, got %d", len(allRemaining))
	}
}

func TestAuditBoundary(t *testing.T) {
	tmpDir := t.TempDir()
	privRoot := filepath.Join(tmpDir, "fak-private")
	platformSample := filepath.Join(privRoot, "platform", "sample")
	if err := os.MkdirAll(platformSample, 0755); err != nil {
		t.Fatal(err)
	}

	// 1. Clean file: imports standard library and fak/pkg/*
	cleanCode := `package sample

import (
	"context"
	"fmt"
	"github.com/anthony-chaudhary/fak-private/platform/dispatch"
	"github.com/anthony-chaudhary/fak/pkg/abi"
)

var _ = context.Background
var _ = fmt.Println
var _ = dispatch.Run
var _ = abi.Version
`
	if err := os.WriteFile(filepath.Join(platformSample, "clean.go"), []byte(cleanCode), 0644); err != nil {
		t.Fatal(err)
	}

	violations, err := AuditBoundary(privRoot)
	if err != nil {
		t.Fatalf("AuditBoundary failed on clean tree: %v", err)
	}
	if len(violations) != 0 {
		t.Fatalf("expected 0 violations on clean code, got %d: %+v", len(violations), violations)
	}

	// 2. Add violations:
	// - illegal internal import
	// - illegal root/cmd fak import
	badCode := `package sample

import (
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/cmd/fak"
)

var _ = engine.Run
var _ = fak.Main
`
	if err := os.WriteFile(filepath.Join(platformSample, "bad.go"), []byte(badCode), 0644); err != nil {
		t.Fatal(err)
	}

	violations, err = AuditBoundary(privRoot)
	if err != nil {
		t.Fatalf("AuditBoundary failed on dirty tree: %v", err)
	}
	if len(violations) != 2 {
		t.Fatalf("expected 2 violations, got %d: %+v", len(violations), violations)
	}

	if violations[0].Rule != "PKG_ONLY_IMPORT" && violations[0].Rule != "NO_INTERNAL_IMPORT" {
		t.Errorf("unexpected rule: %s", violations[0].Rule)
	}
}

func TestAuditBoundary_LiveRepoIfPresent(t *testing.T) {
	privRoot := filepath.Join("..", "..", "..", "fak-private")
	if fi, err := os.Stat(privRoot); err != nil || !fi.IsDir() {
		t.Skip("sibling fak-private not present on this host")
	}

	violations, err := AuditBoundary(privRoot)
	if err != nil {
		t.Fatalf("AuditBoundary on real fak-private failed: %v", err)
	}
	if len(violations) > 0 {
		t.Errorf("found %d unexpected boundary violations in live fak-private: %+v", len(violations), violations)
	}
}

func TestScaffold(t *testing.T) {
	tmpDir := t.TempDir()
	fakRoot := filepath.Join(tmpDir, "fak")
	privateRoot := filepath.Join(tmpDir, "fak-private")

	item := Item{
		Number:     1,
		Cohort:     "Cohort 1: Watchdogs & Session Control",
		SourcePath: "tools/crash_audit.py",
		TargetPath: "platform/watchdogs/crash_audit.go",
		TargetPkg:  "platform/watchdogs",
	}

	// Test dry-run: no files created
	files, err := Scaffold(fakRoot, privateRoot, item, true)
	if err != nil {
		t.Fatalf("dry-run Scaffold failed: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 files in dry-run, got %d", len(files))
	}
	for _, f := range files {
		if _, err := os.Stat(f); !os.IsNotExist(err) {
			t.Errorf("file %s should not exist after dry-run", f)
		}
	}

	// Test real scaffolding
	files, err = Scaffold(fakRoot, privateRoot, item, false)
	if err != nil {
		t.Fatalf("real Scaffold failed: %v", err)
	}
	if len(files) != 2 {
		t.Fatalf("expected 2 files created, got %d", len(files))
	}

	for _, f := range files {
		content, err := os.ReadFile(f)
		if err != nil {
			t.Fatalf("failed to read created file %s: %v", f, err)
		}
		// Validate that the generated Go source compiles/parses
		fset := token.NewFileSet()
		if _, err := parser.ParseFile(fset, f, content, parser.AllErrors); err != nil {
			t.Errorf("generated file %s failed Go syntax validation: %v", f, err)
		}
	}

	// Test duplicate scaffolding fails safely
	_, err = Scaffold(fakRoot, privateRoot, item, false)
	if err == nil {
		t.Errorf("expected error when scaffolding over existing file, got nil")
	}
}
