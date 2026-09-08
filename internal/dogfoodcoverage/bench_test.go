package dogfoodcoverage

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

var (
	benchGradeSink      string
	benchFormatSink     string
	benchAuditRowsSink  int
	benchJournalsSink   int
	benchDiagnoseSink   string
	benchReportSink     Report
	benchGuardSink      bool
	benchWhichSink      string
	benchResolveBinSink string
)

// BenchmarkGrade measures mapping coverage percent and dogfood debt to letter grades.
func BenchmarkGrade(b *testing.B) {
	testCases := []struct {
		coverage float64
		debt     int
	}{
		{100.0, 0},
		{95.0, 0},
		{85.0, 0},
		{75.0, 1},
		{60.0, 2},
		{30.0, 5},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := testCases[i%len(testCases)]
		benchGradeSink = Grade(tc.coverage, tc.debt)
	}
}

// BenchmarkFormatReport measures formatting a full Report into a human-readable status string.
func BenchmarkFormatReport(b *testing.B) {
	rep := Report{
		Schema:      Schema,
		Coverage:    85.7,
		Met:         6,
		Total:       7,
		DogfoodDebt: 1,
		Grade:       "C",
		AuditRows:   128,
		KPIs: []KPI{
			{Key: "fleet_leaf_guarded", OK: true, Hard: true, Detail: "dispatch_worker fronts claude worker", Evidence: "fak guard --provider"},
			{Key: "fak_bin_resolvable", OK: true, Hard: true, Detail: "fak binary resolves", Evidence: "/usr/local/bin/fak"},
			{Key: "guard_default_on", OK: true, Hard: true, Detail: "FLEET_DOGFOOD_GUARD enabled", Evidence: "FLEET_DOGFOOD_GUARD=<unset=ON>"},
			{Key: "issue_dispatch_wired", OK: false, Hard: true, Detail: "routes through guarded_launch", Evidence: "MISSING"},
			{Key: "guard_verb_present", OK: true, Hard: true, Detail: "fak guard exists", Evidence: "cmd/fak/guard.go"},
			{Key: "guard_documented", OK: true, Hard: false, Detail: "DOGFOOD-CLAUDE.md documentation", Evidence: "DOGFOOD-CLAUDE.md"},
			{Key: "audit_journal_evidence", OK: true, Hard: false, Detail: "decision records in audit journal", Evidence: "128 decision row(s)"},
		},
		WorstFirst: []string{"issue_dispatch_wired"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchFormatSink = FormatReport(rep)
	}
}

// BenchmarkGuardEnabled measures evaluating FLEET_DOGFOOD_GUARD environment toggle semantics.
func BenchmarkGuardEnabled(b *testing.B) {
	envs := []map[string]string{
		nil,
		{"FLEET_DOGFOOD_GUARD": "1"},
		{"FLEET_DOGFOOD_GUARD": "0"},
		{"FLEET_DOGFOOD_GUARD": "false"},
		{"FLEET_DOGFOOD_GUARD": "off"},
		{"FLEET_DOGFOOD_GUARD": "no"},
		{"FLEET_DOGFOOD_GUARD": "disable"},
		{"FLEET_DOGFOOD_GUARD": "true"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e := envs[i%len(envs)]
		benchGuardSink = guardEnabled(e)
	}
}

// BenchmarkWhichOnPath measures executable candidate resolution across PATH entries.
func BenchmarkWhichOnPath(b *testing.B) {
	td := b.TempDir()
	binDir := filepath.Join(td, "target_bin")
	if err := os.MkdirAll(binDir, 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	target := "fak"
	exe := target
	if runtime.GOOS == "windows" {
		exe = target + ".exe"
	}
	if err := os.WriteFile(filepath.Join(binDir, exe), []byte("dummy"), 0755); err != nil {
		b.Fatalf("write binary: %v", err)
	}

	pathVal := strings.Join([]string{
		filepath.Join(td, "nonexistent_dir1"),
		filepath.Join(td, "nonexistent_dir2"),
		binDir,
		filepath.Join(td, "nonexistent_dir3"),
	}, string(os.PathListSeparator))

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchWhichSink = whichOnPath(target, pathVal)
	}
}

// BenchmarkResolveFakBin measures locating the best fak binary candidate across root and path.
func BenchmarkResolveFakBin(b *testing.B) {
	td := b.TempDir()
	toolsBin := filepath.Join(td, "tools", ".bin")
	if err := os.MkdirAll(toolsBin, 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	exe := "fak"
	if runtime.GOOS == "windows" {
		exe = "fak.exe"
	}
	if err := os.WriteFile(filepath.Join(toolsBin, exe), []byte("fak-bin"), 0755); err != nil {
		b.Fatalf("write: %v", err)
	}

	env := map[string]string{
		"PATH": td,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResolveBinSink = resolveFakBin(td, env)
	}
}

// BenchmarkCountAuditRows measures parsing and counting JSONL decision rows across audit journals.
func BenchmarkCountAuditRows(b *testing.B) {
	td := b.TempDir()
	jdir := filepath.Join(td, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(jdir, 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	for idx := 0; idx < 4; idx++ {
		var buf strings.Builder
		for row := 0; row < 50; row++ {
			buf.WriteString(fmt.Sprintf(`{"seq":%d,"lane":"lane-%d","admitted":true}`+"\n", row, idx))
		}
		if err := os.WriteFile(filepath.Join(jdir, fmt.Sprintf("lane-%d.jsonl", idx)), []byte(buf.String()), 0644); err != nil {
			b.Fatalf("write: %v", err)
		}
	}
	env := map[string]string{"XDG_CONFIG_HOME": td}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchAuditRowsSink, benchJournalsSink = CountAuditRows(td, env)
	}
}

// BenchmarkDiagnoseAuditGap measures evaluating why an audit journal directory or files are empty.
func BenchmarkDiagnoseAuditGap(b *testing.B) {
	td := b.TempDir()
	jdir := filepath.Join(td, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(jdir, 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(filepath.Join(jdir, "empty.jsonl"), []byte("\n\n"), 0644); err != nil {
		b.Fatalf("write: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDiagnoseSink = DiagnoseAuditGap(td)
	}
}

// BenchmarkEvaluate measures full scorecard evaluation and KPI calculation.
func BenchmarkEvaluate(b *testing.B) {
	td := b.TempDir()
	if err := os.MkdirAll(filepath.Join(td, "tools"), 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(td, "cmd", "fak"), 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	if err := os.MkdirAll(filepath.Join(td, "docs", "fak"), 0755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	_ = os.WriteFile(filepath.Join(td, "cmd", "fak", "guard.go"), []byte("package main\n"), 0644)
	_ = os.WriteFile(filepath.Join(td, "tools", "issue_dispatch.py"), []byte("guarded_launch_command"), 0644)
	_ = os.WriteFile(filepath.Join(td, "DOGFOOD-CLAUDE.md"), []byte("fak guard doc"), 0644)
	_ = os.WriteFile(filepath.Join(td, "docs", "fak", "always-on-dogfood-server.md"), []byte("server spec"), 0644)
	_ = os.WriteFile(filepath.Join(td, "tools", "com.fak.serve-gateway.plist"), []byte("plist"), 0644)

	env := map[string]string{
		"FLEET_DOGFOOD_GUARD": "1",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReportSink = Evaluate(td, env)
	}
}

// TestBenchmarkSanity verifies that benchmark execution works cleanly.
func TestBenchmarkSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkGrade)
	if res.N <= 0 {
		t.Fatalf("BenchmarkGrade did not iterate: %v", res)
	}

	resFormat := testing.Benchmark(BenchmarkFormatReport)
	if resFormat.N <= 0 {
		t.Fatalf("BenchmarkFormatReport did not iterate: %v", resFormat)
	}
}
