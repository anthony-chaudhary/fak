package factorymigrate

import (
	"fmt"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var (
	benchItemsSink      []Item
	benchReportSink     Report
	benchCandidatesSink []Item
	benchPlanSink       MigrationPlan
	benchViolationsSink []BoundaryViolation
	benchFilesSink      []string
	benchStringSink     string
)

func TestPlanMigration(t *testing.T) {
	items, err := ParseInventory("")
	if err != nil {
		t.Fatalf("ParseInventory failed: %v", err)
	}

	report := Report{
		Total:      len(items),
		Unmigrated: len(items),
		Items:      items,
	}

	// Default batch size (5)
	plan := PlanMigration(report, 0, "")
	if plan.BatchSize != 5 {
		t.Errorf("plan.BatchSize = %d, want 5", plan.BatchSize)
	}
	if plan.TotalPending != 100 {
		t.Errorf("plan.TotalPending = %d, want 100", plan.TotalPending)
	}
	if len(plan.Waves) != 20 {
		t.Errorf("len(plan.Waves) = %d, want 20", len(plan.Waves))
	}
	for i, wave := range plan.Waves {
		if len(wave) != 5 {
			t.Errorf("wave[%d] size = %d, want 5", i, len(wave))
		}
	}

	// Custom batch size (12)
	plan12 := PlanMigration(report, 12, "")
	if plan12.BatchSize != 12 {
		t.Errorf("plan12.BatchSize = %d, want 12", plan12.BatchSize)
	}
	// 100 / 12 = 8 waves of 12 + 1 wave of 4 = 9 waves
	if len(plan12.Waves) != 9 {
		t.Errorf("len(plan12.Waves) = %d, want 9", len(plan12.Waves))
	}
	lastWave := plan12.Waves[len(plan12.Waves)-1]
	if len(lastWave) != 4 {
		t.Errorf("last wave size = %d, want 4", len(lastWave))
	}

	// Filtered by cohort
	planWatchdogs := PlanMigration(report, 5, "watchdogs")
	if planWatchdogs.TotalPending != 20 {
		t.Errorf("planWatchdogs.TotalPending = %d, want 20", planWatchdogs.TotalPending)
	}
	if len(planWatchdogs.Waves) != 4 {
		t.Errorf("len(planWatchdogs.Waves) = %d, want 4", len(planWatchdogs.Waves))
	}
}

func TestValidateItem(t *testing.T) {
	valid := Item{
		Number:     1,
		Cohort:     "Cohort 1: Watchdogs & Session Control",
		SourcePath: "tools/crash_audit.py",
		TargetPath: "platform/watchdogs/crash_audit.go",
		TargetPkg:  "platform/watchdogs",
	}
	if err := ValidateItem(valid); err != nil {
		t.Errorf("ValidateItem(valid) returned unexpected error: %v", err)
	}

	// Invalid number
	invNum := valid
	invNum.Number = 0
	if err := ValidateItem(invNum); err == nil {
		t.Errorf("expected error for Number <= 0, got nil")
	}

	// Empty source
	emptySrc := valid
	emptySrc.SourcePath = ""
	if err := ValidateItem(emptySrc); err == nil {
		t.Errorf("expected error for empty SourcePath, got nil")
	}

	// Empty target path
	emptyTgt := valid
	emptyTgt.TargetPath = ""
	if err := ValidateItem(emptyTgt); err == nil {
		t.Errorf("expected error for empty TargetPath, got nil")
	}

	// Target not .go
	nonGoTgt := valid
	nonGoTgt.TargetPath = "platform/watchdogs/crash_audit.py"
	if err := ValidateItem(nonGoTgt); err == nil {
		t.Errorf("expected error for non-.go TargetPath, got nil")
	}

	// Empty target pkg
	emptyPkg := valid
	emptyPkg.TargetPkg = ""
	if err := ValidateItem(emptyPkg); err == nil {
		t.Errorf("expected error for empty TargetPkg, got nil")
	}
}

func TestValidateReport(t *testing.T) {
	items, err := ParseInventory("")
	if err != nil {
		t.Fatalf("ParseInventory failed: %v", err)
	}

	report := Report{
		Total:      100,
		Migrated:   10,
		Partial:    20,
		Unmigrated: 70,
		Items:      items,
	}
	if err := ValidateReport(report); err != nil {
		t.Errorf("ValidateReport(valid) unexpected error: %v", err)
	}

	// Mismatched total
	badTotal := report
	badTotal.Total = 99
	if err := ValidateReport(badTotal); err == nil {
		t.Errorf("expected error for mismatched total, got nil")
	}

	// Mismatched items length
	badItems := report
	badItems.Items = items[:50]
	if err := ValidateReport(badItems); err == nil {
		t.Errorf("expected error for mismatched items length, got nil")
	}
}

func TestScaffoldBatch(t *testing.T) {
	tmpDir := t.TempDir()
	fakRoot := filepath.Join(tmpDir, "fak")
	privateRoot := filepath.Join(tmpDir, "fak-private")

	items := []Item{
		{
			Number:     1,
			Cohort:     "Cohort 1: Watchdogs & Session Control",
			SourcePath: "tools/crash_audit.py",
			TargetPath: "platform/watchdogs/crash_audit.go",
			TargetPkg:  "platform/watchdogs",
		},
		{
			Number:     2,
			Cohort:     "Cohort 1: Watchdogs & Session Control",
			SourcePath: "tools/dos_supervisor_watchdog.py",
			TargetPath: "platform/watchdogs/supervisor.go",
			TargetPkg:  "platform/watchdogs",
		},
	}

	// Dry run
	files, err := ScaffoldBatch(fakRoot, privateRoot, items, true)
	if err != nil {
		t.Fatalf("ScaffoldBatch dryRun failed: %v", err)
	}
	if len(files) != 4 {
		t.Fatalf("expected 4 files in dry-run, got %d", len(files))
	}

	// Real execution
	files, err = ScaffoldBatch(fakRoot, privateRoot, items, false)
	if err != nil {
		t.Fatalf("ScaffoldBatch real failed: %v", err)
	}
	if len(files) != 4 {
		t.Fatalf("expected 4 files created, got %d", len(files))
	}
	for _, f := range files {
		if _, err := os.Stat(f); err != nil {
			t.Errorf("expected file %s to exist: %v", f, err)
		}
	}
}

func TestPascalCase(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"crash_audit", "CrashAudit"},
		{"dos-supervisor-watchdog", "DosSupervisorWatchdog"},
		{"topup.go", "TopupGo"},
		{"", "Component"},
		{"a", "A"},
		{"foo_bar_baz", "FooBarBaz"},
	}
	for _, tc := range tests {
		got := PascalCase(tc.input)
		if got != tc.want {
			t.Errorf("PascalCase(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

// -----------------------------------------------------------------------------
// BENCHMARKS: Migration Planning
// -----------------------------------------------------------------------------

func BenchmarkPlan_ParseInventoryEmbedded(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items, err := ParseInventory("")
		if err != nil {
			b.Fatalf("ParseInventory failed: %v", err)
		}
		benchItemsSink = items
	}
}

func BenchmarkPlan_ParseInventoryLarge(b *testing.B) {
	// Synthesize a 500-item inventory across 10 cohorts
	var sb strings.Builder
	for c := 1; c <= 10; c++ {
		fmt.Fprintf(&sb, "### Cohort %d: Synthetic Cohort %d (platform/synth%d)\n", c, c, c)
		for item := 1; item <= 50; item++ {
			num := (c-1)*50 + item
			fmt.Fprintf(&sb, "%d. tools/tool_%d.py -> platform/synth%d/tool_%d.go\n", num, num, c, num)
		}
		sb.WriteString("\n")
	}
	content := sb.String()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		items, err := parseInventoryText(content)
		if err != nil {
			b.Fatalf("parseInventoryText failed: %v", err)
		}
		benchItemsSink = items
	}
}

func BenchmarkPlan_AuditStatus(b *testing.B) {
	tmpDir := b.TempDir()
	fakRoot := filepath.Join(tmpDir, "fak")
	privateRoot := filepath.Join(tmpDir, "fak-private")

	_ = os.MkdirAll(filepath.Join(fakRoot, "tools"), 0755)
	_ = os.MkdirAll(filepath.Join(privateRoot, "platform", "watchdogs"), 0755)
	_ = os.MkdirAll(filepath.Join(fakRoot, "internal", "procguard"), 0755)

	_ = os.WriteFile(filepath.Join(privateRoot, "platform", "watchdogs", "crash_audit.go"), []byte("package watchdogs\n"), 0644)
	_ = os.WriteFile(filepath.Join(fakRoot, "tools", "dos_supervisor_watchdog.py"), []byte("#!/usr/bin/env python3\n"), 0644)
	_ = os.WriteFile(filepath.Join(privateRoot, "platform", "watchdogs", "watchdog.go"), []byte("package watchdogs\n"), 0644)
	_ = os.WriteFile(filepath.Join(fakRoot, "tools", "proc_resource_guard.py"), []byte("#!/usr/bin/env python3\n"), 0644)
	_ = os.WriteFile(filepath.Join(fakRoot, "internal", "procguard", "procguard.go"), []byte("package procguard\n"), 0644)

	items, err := ParseInventory("")
	if err != nil {
		b.Fatalf("ParseInventory failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := AuditStatus(fakRoot, privateRoot, items)
		benchReportSink = report
	}
}

func BenchmarkPlan_NextCandidates(b *testing.B) {
	items, err := ParseInventory("")
	if err != nil {
		b.Fatalf("ParseInventory failed: %v", err)
	}
	report := Report{
		Total:      len(items),
		Unmigrated: len(items),
		Items:      items,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		candidates := NextCandidates(report, 10, "watchdogs")
		benchCandidatesSink = candidates
	}
}

func BenchmarkPlan_WavePartitioning(b *testing.B) {
	items, err := ParseInventory("")
	if err != nil {
		b.Fatalf("ParseInventory failed: %v", err)
	}
	report := Report{
		Total:      len(items),
		Unmigrated: len(items),
		Items:      items,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		plan := PlanMigration(report, 5, "")
		benchPlanSink = plan
	}
}

// -----------------------------------------------------------------------------
// BENCHMARKS: Migration Validation
// -----------------------------------------------------------------------------

func BenchmarkValidate_BoundaryAudit(b *testing.B) {
	tmpDir := b.TempDir()
	privRoot := filepath.Join(tmpDir, "fak-private")
	sampleDir := filepath.Join(privRoot, "platform", "benchsample")
	_ = os.MkdirAll(sampleDir, 0755)

	cleanCode := `package benchsample

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
	for i := 0; i < 5; i++ {
		_ = os.WriteFile(filepath.Join(sampleDir, fmt.Sprintf("clean_%d.go", i)), []byte(cleanCode), 0644)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		violations, err := AuditBoundary(privRoot)
		if err != nil {
			b.Fatalf("AuditBoundary failed: %v", err)
		}
		benchViolationsSink = violations
	}
}

func BenchmarkValidate_ItemInvariants(b *testing.B) {
	items, err := ParseInventory("")
	if err != nil {
		b.Fatalf("ParseInventory failed: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for j := range items {
			if err := ValidateItem(items[j]); err != nil {
				b.Fatalf("ValidateItem failed: %v", err)
			}
		}
	}
}

func BenchmarkValidate_ReportConsistency(b *testing.B) {
	items, err := ParseInventory("")
	if err != nil {
		b.Fatalf("ParseInventory failed: %v", err)
	}
	report := Report{
		Total:      100,
		Migrated:   20,
		Partial:    30,
		Unmigrated: 50,
		Items:      items,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := ValidateReport(report); err != nil {
			b.Fatalf("ValidateReport failed: %v", err)
		}
	}
}

func BenchmarkValidate_ScaffoldAST(b *testing.B) {
	tmpl := `// Package sample provides migrated autonomous factory functionality for fak-private.
package sample

import (
	"context"
	"fmt"
)

type WorkerConfig struct {
	Workspace string
	DryRun    bool
}

type Worker struct {
	cfg WorkerConfig
}

func NewWorker(cfg WorkerConfig) *Worker {
	return &Worker{cfg: cfg}
}

func (w *Worker) Run(ctx context.Context) error {
	if w.cfg.Workspace == "" {
		return fmt.Errorf("worker: workspace root cannot be empty")
	}
	return nil
}
`
	src := []byte(tmpl)
	fset := token.NewFileSet()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		node, err := parser.ParseFile(fset, "sample.go", src, parser.AllErrors)
		if err != nil || node == nil {
			b.Fatalf("ParseFile failed: %v", err)
		}
	}
}

// -----------------------------------------------------------------------------
// BENCHMARKS: Migration Execution
// -----------------------------------------------------------------------------

func BenchmarkExecute_ScaffoldDryRun(b *testing.B) {
	item := Item{
		Number:     1,
		Cohort:     "Cohort 1: Watchdogs & Session Control",
		SourcePath: "tools/crash_audit.py",
		TargetPath: "platform/watchdogs/crash_audit.go",
		TargetPkg:  "platform/watchdogs",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		files, err := Scaffold("", "C:/dummy/fak-private", item, true)
		if err != nil {
			b.Fatalf("Scaffold dryRun failed: %v", err)
		}
		benchFilesSink = files
	}
}

func BenchmarkExecute_ScaffoldBatchDryRun(b *testing.B) {
	items := make([]Item, 10)
	for i := 0; i < 10; i++ {
		items[i] = Item{
			Number:     i + 1,
			Cohort:     "Cohort 1: Watchdogs & Session Control",
			SourcePath: fmt.Sprintf("tools/tool_%d.py", i+1),
			TargetPath: fmt.Sprintf("platform/watchdogs/tool_%d.go", i+1),
			TargetPkg:  "platform/watchdogs",
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		files, err := ScaffoldBatch("", "C:/dummy/fak-private", items, true)
		if err != nil {
			b.Fatalf("ScaffoldBatch dryRun failed: %v", err)
		}
		benchFilesSink = files
	}
}

func BenchmarkExecute_ScaffoldPhysical(b *testing.B) {
	tmpDir := b.TempDir()
	privateRoot := filepath.Join(tmpDir, "fak-private")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		item := Item{
			Number:     i + 1,
			Cohort:     "Cohort 1: Watchdogs & Session Control",
			SourcePath: fmt.Sprintf("tools/tool_%d.py", i),
			TargetPath: fmt.Sprintf("platform/watchdogs/tool_%d.go", i),
			TargetPkg:  "platform/watchdogs",
		}
		files, err := Scaffold("", privateRoot, item, false)
		if err != nil {
			b.Fatalf("Scaffold physical failed: %v", err)
		}
		benchFilesSink = files
	}
}

func BenchmarkExecute_PascalCase(b *testing.B) {
	names := []string{
		"dos_supervisor_watchdog",
		"fleet_dos_dispatch_watchdog",
		"concept_disambiguation_scorecard",
		"release_readiness_scorecard",
		"qwen36_standalone_readiness",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, name := range names {
			benchStringSink = PascalCase(name)
		}
	}
}
