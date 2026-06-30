package uiquality

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTree materializes a minimal render-source tree under a temp root so the
// scorecard can be exercised against controlled fixtures (the source IS the oracle,
// so the test feeds it source, not a data file).
func writeTree(t *testing.T, files map[string]string) string {
	t.Helper()
	root := t.TempDir()
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

// cleanFixtures is a render tree that should grade clean: rune-aware helpers
// present, no width-padded verb over a bare trimTUI, every pane has an empty-state
// line, the info legend covers every term, and every console subcommand is in help.
func cleanFixtures() map[string]string {
	return map[string]string{
		"cmd/fak/tui.go": `package main
func runTUI() {
	switch argv[0] {
	case "loops":
	case "guard":
	case "help":
	}
}
func runTUIIssues() {}
`,
		"cmd/fak/tui_loop_render.go": `package main
func dispWidthTUI(s string) int { return 0 }
func padRightTUI(s string, w int) string { return s }
func takeCellsTUI(s string, n int) string { return s }
func trimTUI(s string, width int) string {
	// old byte-indexed s[:width] is gone; we use takeCellsTUI.
	return takeCellsTUI(s, width)
}
func renderTUILoops() {
	if len(rows) == 0 { print("no loops found") }
	fmt.Fprintf(&b, "%s %s\n", padRightTUI(trimTUI(x, 8), 8), trimTUI(y, 20))
}
func tuiUsage(w io.Writer) {
	fmt.Fprint(w, ` + "`" + `fak console
  fak console loops [--json]
  fak console guard --guard-json FILE
` + "`" + `)
}
`,
		"cmd/fak/tui_guard_report.go": `package main
func renderTUIGuard() {
	if len(report.Rows) == 0 { print("no guard rows") }
}
`,
		"cmd/fak/tui_issues_garden.go": `package main
func renderTUIGarden() {
	if len(report.Rows) == 0 { print("no garden members") }
}
`,
		"cmd/fak/tui_overview_sessions.go": `package main
func renderTUISessions() {
	if len(rows) == 0 { print("no sessions") }
}
`,
		"cmd/fak/info.go": `package main
func guardInfoLegend() string {
	return "cache floor turns inflight up"
}
func runInfo() {
	if term.IsTerminal(0) {
		fmt.Fprintf(stdout, "\033[K %s", line)
	}
}
`,
		"cmd/fak/guard_split.go": `package main
func runGuardSplit() {}
`,
	}
}

func TestBuildCleanTreeScoresZeroDebt(t *testing.T) {
	root := writeTree(t, cleanFixtures())
	p := Build(Options{Root: root})
	if !p.OK {
		t.Fatalf("clean tree should be OK; got verdict=%s debt=%v\nkpis:\n%s",
			p.Verdict, p.Corpus["ui_quality_debt"], Render(p))
	}
	if got := p.Corpus["ui_quality_debt"]; got != 0 {
		t.Fatalf("clean tree ui_quality_debt = %v, want 0\n%s", got, Render(p))
	}
	if p.Corpus["grade"] != "A" {
		t.Fatalf("clean tree grade = %v, want A", p.Corpus["grade"])
	}
}

// TestBuildDetectsByteSliceTruncation is the paired honesty test: a tree carrying
// the original bug (byte-indexed s[:width-3] truncation, no rune-aware helpers)
// MUST be flagged. A scorecard that cannot catch the defect it exists to catch is
// theater.
func TestBuildDetectsByteSliceTruncation(t *testing.T) {
	f := cleanFixtures()
	// Regress trimTUI to the buggy byte-slice form and drop the helpers.
	f["cmd/fak/tui_loop_render.go"] = `package main
func trimTUI(s string, width int) string {
	if len(s) <= width { return s }
	if width <= 3 { return s[:width] }
	return s[:width-3] + "..."
}
func renderTUILoops() {
	if len(rows) == 0 { print("no loops found") }
}
func tuiUsage(w io.Writer) {
	fmt.Fprint(w, ` + "`" + `fak console
  fak console loops
  fak console guard
` + "`" + `)
}
`
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	if p.OK {
		t.Fatalf("buggy tree graded clean — scorecard failed to catch byte-slice truncation\n%s", Render(p))
	}
	rune := kpiByKey(p, "rune_safety")
	if len(rune.Defects) == 0 {
		t.Fatalf("rune_safety reported no defects on the buggy tree\n%s", Render(p))
	}
	joined := strings.Join(rune.Defects, "\n")
	if !strings.Contains(joined, "s[:width-3]") {
		t.Fatalf("rune_safety did not flag the s[:width-3] byte-slice; defects:\n%s", joined)
	}
	if !strings.Contains(joined, "dispWidthTUI") {
		t.Fatalf("rune_safety did not flag the missing dispWidthTUI helper; defects:\n%s", joined)
	}
}

// TestWidthConsistencyNoFalsePositiveOnTrailingTrim guards the FP the detector was
// hardened against: a trimTUI() feeding a PLAIN trailing %s (no width pad) is fine
// and must NOT be flagged, even though the same Fprintf also has a %-Ns padding a
// DIFFERENT column.
func TestWidthConsistencyNoFalsePositiveOnTrailingTrim(t *testing.T) {
	f := cleanFixtures()
	f["cmd/fak/tui_overview_sessions.go"] = `package main
func renderTUISessions() {
	if len(rows) == 0 { print("no sessions") }
	fmt.Fprintf(&b, "%-18s %-12s %s\n", kv.Name, kv.Source, trimTUI(value, 20))
	fmt.Fprintf(&b, "%-10s %s\n", action.Pane, trimTUI(action.Command, 30))
}
`
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	wc := kpiByKey(p, "width_consistency")
	if len(wc.Defects) != 0 {
		t.Fatalf("trailing-%%s trimTUI flagged as a byte-pad (false positive):\n%v", wc.Defects)
	}
}

// TestWidthConsistencyCatchesPaddedTrim is the matching true-positive: a %-Ns whose
// own argument IS a bare trimTUI() is the real shear and must be flagged.
func TestWidthConsistencyCatchesPaddedTrim(t *testing.T) {
	f := cleanFixtures()
	f["cmd/fak/tui_guard_report.go"] = `package main
func renderTUIGuard() {
	if len(report.Rows) == 0 { print("no guard rows") }
	fmt.Fprintf(&b, "%-24s %s\n", trimTUI(row.Artifact, 24), tags)
}
`
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	wc := kpiByKey(p, "width_consistency")
	if len(wc.Defects) == 0 {
		t.Fatalf("a %%-24s consuming a bare trimTUI was NOT flagged (false negative)\n%s", Render(p))
	}
}

// TestHelpCompletenessReadsUsageFromAnyFile guards the bug where tuiUsage lives in
// a different file than the runTUI dispatch — the detector must find it anywhere.
func TestHelpCompletenessReadsUsageFromAnyFile(t *testing.T) {
	root := writeTree(t, cleanFixtures())
	p := Build(Options{Root: root})
	help := kpiByKey(p, "help_completeness")
	if len(help.Defects) != 0 {
		t.Fatalf("help_completeness false-flagged documented subcommands: %v", help.Defects)
	}
}

// TestHelpCompletenessCatchesUndocumented flags a real gap.
func TestHelpCompletenessCatchesUndocumented(t *testing.T) {
	f := cleanFixtures()
	f["cmd/fak/tui_loop_render.go"] = strings.Replace(
		f["cmd/fak/tui_loop_render.go"],
		"  fak console guard --guard-json FILE\n", "", 1)
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	help := kpiByKey(p, "help_completeness")
	joined := strings.Join(help.Defects, " ")
	if !strings.Contains(joined, "guard") {
		t.Fatalf("undocumented 'guard' subcommand not flagged; defects: %v", help.Defects)
	}
}

func TestCompareReportsRetiredDebt(t *testing.T) {
	cur := Build(Options{Root: writeTree(t, cleanFixtures())})
	base := map[string]any{"corpus": map[string]any{"ui_quality_debt": 4}}
	out := Compare(cur, base)
	if !strings.Contains(out, "4 -> 0") || !strings.Contains(out, "retired 4") {
		t.Fatalf("compare did not report the retired delta: %s", out)
	}
	if !strings.Contains(out, "PASS") {
		t.Fatalf("compare should PASS when debt drops: %s", out)
	}
}

func kpiByKey(p Payload, key string) KPI {
	for _, k := range p.KPIs {
		if k.Key == key {
			return k
		}
	}
	return KPI{}
}
