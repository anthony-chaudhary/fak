package uiquality

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/scorecard"
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
			p.Verdict, p.Corpus["ui_quality_debt"], scorecard.Render(p, DebtKey))
	}
	if got := p.Corpus["ui_quality_debt"]; got != 0 {
		t.Fatalf("clean tree ui_quality_debt = %v, want 0\n%s", got, scorecard.Render(p, DebtKey))
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
		t.Fatalf("buggy tree graded clean — scorecard failed to catch byte-slice truncation\n%s", scorecard.Render(p, DebtKey))
	}
	rune := kpiByKey(p, "rune_safety")
	if len(rune.Defects) == 0 {
		t.Fatalf("rune_safety reported no defects on the buggy tree\n%s", scorecard.Render(p, DebtKey))
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
		t.Fatalf("a %%-24s consuming a bare trimTUI was NOT flagged (false negative)\n%s", scorecard.Render(p, DebtKey))
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

// TestHelpCompletenessReadsTUIPaneRegistry pins the fallback path: when runTUI
// dispatches through tuiplugin.Lookup instead of a switch, the pane registry is the
// only oracle for the subcommand set.
//
// The registry fixture must live in a file the card actually loads. An earlier version
// of this test wrote it to cmd/fak/tui_registry.go, which is NOT in renderFiles, so
// loadSources never read it, `subs` stayed empty and the test passed without ever
// exercising the extractor — it was vacuous. It now writes into a corpus file and
// asserts the SIZE of the checked set, which is the assertion that makes it bite.
func TestHelpCompletenessReadsTUIPaneRegistry(t *testing.T) {
	f := cleanFixtures()
	f["cmd/fak/tui.go"] = `package main
func runTUI() {
	pane, ok := tuiplugin.Lookup(argv[0])
	if !ok { tuiUsage(stderr) }
	_ = pane
}
func runTUIIssues() {}
`
	f["cmd/fak/tui_guard_report.go"] += `
func registerBuiltinTUIPanes() {
	tuiplugin.Register(tuiplugin.Pane{ID: "loops"})
	tuiplugin.Register(tuiplugin.Pane{ID: "guard"})
	tuiplugin.Register(tuiplugin.Pane{ID: "panes", Controls: []tuiplugin.Control{{ID: "json"}}})
}
`
	f["cmd/fak/tui_loop_render.go"] = strings.Replace(
		f["cmd/fak/tui_loop_render.go"],
		"  fak console guard --guard-json FILE\n",
		"  fak console guard --guard-json FILE\n  fak console panes [--json]\n",
		1)
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	help := kpiByKey(p, "help_completeness")
	if len(help.Defects) != 0 {
		t.Fatalf("registry help_completeness false-flagged documented panes: %v", help.Defects)
	}
	// Non-vacuity: the clean detail names the size of the checked set, so this fails
	// if the registry went unread and `subs` came back empty.
	if !strings.Contains(help.Detail, "all 3 console subcommands") {
		t.Fatalf("pane registry not read: want a 3-pane checked set, detail = %q", help.Detail)
	}
}

// TestHelpCompletenessBindsPaneIDNotControlID pins the extractor defect this KPI
// shipped with: a Pane literal nests a `Controls: []tuiplugin.Control{{ID: ...}}`
// slice carrying ID fields of its own, and the flat regex (greedy `[^}]*`, which
// cannot cross a `}`) backtracked onto the FIRST control's id instead of the pane's.
// On the live tree that reported `as-of`, `at` and `check` as undocumented
// `fak console` subcommands. None of the three is dispatchable — runTUI looks argv[0]
// up with tuiplugin.Lookup, whose registry is keyed on Pane.ID — so the debt was
// false and "fixing" it by writing those flags into tuiUsage would have been false help.
//
// The fixture asserts BOTH directions, so no repair can pass by merely making findings
// disappear:
//   - pane "issues" is documented while its control "as-of" is not, so a control id
//     leaking into the checked set shows up as a defect; and
//   - pane "sessions" is undocumented while its control "top" IS documented, so an
//     extractor that binds the control would report zero defects and miss a real gap.
func TestHelpCompletenessBindsPaneIDNotControlID(t *testing.T) {
	f := cleanFixtures()
	f["cmd/fak/tui.go"] = `package main
func runTUI() {
	pane, ok := tuiplugin.Lookup(argv[0])
	if !ok { tuiUsage(stderr) }
	_ = pane
}
func runTUIIssues() {}
`
	f["cmd/fak/tui_issues_garden.go"] += `
func init() {
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "issues",
		Summary: "fold GitHub issues into triage lanes",
		Controls: []tuiplugin.Control{
			{ID: "as-of", Label: "As Of", Kind: "input", Flag: "--as-of"},
			{ID: "epic", Label: "Epic", Kind: "input", Flag: "--epic"},
		},
		Run: runTUIIssues,
	})
	tuiplugin.Register(tuiplugin.Pane{
		ID:      "sessions",
		Summary: "render live gateway session state",
		Controls: []tuiplugin.Control{
			{ID: "top", Label: "Top Rows", Kind: "input", Flag: "--top"},
		},
		Run: runTUISessions,
	})
}
`
	// Document the "issues" PANE and the "top" CONTROL; leave the "sessions" pane out.
	f["cmd/fak/tui_loop_render.go"] = strings.Replace(
		f["cmd/fak/tui_loop_render.go"],
		"  fak console guard --guard-json FILE\n",
		"  fak console guard --guard-json FILE\n  fak console issues [--json]\n  fak console top [--json]\n",
		1)
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	help := kpiByKey(p, "help_completeness")
	joined := strings.Join(help.Defects, " ")
	for _, control := range []string{"as-of", "epic", "top"} {
		if strings.Contains(joined, `"`+control+`"`) {
			t.Fatalf("control id %q entered the console subcommand set; defects: %v",
				control, help.Defects)
		}
	}
	if !strings.Contains(joined, `"sessions"`) {
		t.Fatalf("undocumented pane \"sessions\" not flagged; defects: %v", help.Defects)
	}
	if len(help.Defects) != 1 {
		t.Fatalf("want exactly the one real gap (sessions), got: %v", help.Defects)
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
	out := scorecard.Compare(cur, base, DebtKey)
	if !strings.Contains(out, "4 -> 0") || !strings.Contains(out, "improved by 4") {
		t.Fatalf("compare did not report the retired delta: %s", out)
	}
	if !strings.Contains(out, "improved") {
		t.Fatalf("compare should report improved when debt drops: %s", out)
	}
}

// TestHeaderAlignmentPassesWhenPinnedPairPresent confirms the drift tripwire reads
// clean when BOTH the header literal and its matched row format are present.
func TestHeaderAlignmentPassesWhenPinnedPairPresent(t *testing.T) {
	f := cleanFixtures()
	// Inject the exact pinned loop header + row format so the pane matches the pin.
	f["cmd/fak/tui_loop_render.go"] += "\n" + `var _ = "attention loop                         state          age    runs             witness tags"` + "\n" +
		`var _ = "%9d %s %s %-6s %-16s %-7s %s\n"` + "\n"
	f["cmd/fak/tui_guard_report.go"] += "\n" + `var _ = "attention artifact                 kind                 tool             verdict reason         count tags"` + "\n" +
		`var _ = "%9d %s %s %s %s %s %-5s %s\n"` + "\n"
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	ha := kpiByKey(p, "header_alignment")
	if len(ha.Defects) != 0 {
		t.Fatalf("aligned pinned pair flagged as drift: %v", ha.Defects)
	}
	if ha.Score != 100 {
		t.Fatalf("header_alignment score = %v, want 100\n%s", ha.Score, scorecard.Render(p, DebtKey))
	}
}

// TestHeaderAlignmentCatchesOneSidedDrift is the true positive: when the header
// changes but its matched row format does not (or vice versa), the pair is now
// inconsistent and MUST be flagged — that is exactly the silent header-drift this
// KPI exists to catch.
func TestHeaderAlignmentCatchesOneSidedDrift(t *testing.T) {
	f := cleanFixtures()
	// Header present, but the row format is the pinned guard format only — the loop
	// row format is ABSENT, so the loop pane's header is present without its row.
	f["cmd/fak/tui_loop_render.go"] += "\n" +
		`var _ = "attention loop                         state          age    runs             witness tags"` + "\n"
	// (no loop row format literal injected → one-sided)
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	ha := kpiByKey(p, "header_alignment")
	if len(ha.Defects) == 0 {
		t.Fatalf("one-sided header/row drift was NOT flagged (false negative)\n%s", scorecard.Render(p, DebtKey))
	}
	if p.OK {
		t.Fatalf("payload should be DEBT on a header-drift defect\n%s", scorecard.Render(p, DebtKey))
	}
}

func TestLegendCoverageResolvesSplitInfoFormat(t *testing.T) {
	f := cleanFixtures()
	// Move guardInfoLegend out of info.go into info_format.go
	delete(f, "cmd/fak/info.go")
	f["cmd/fak/info.go"] = "package main\nfunc runInfo() {}\n"
	f["cmd/fak/info_format.go"] = `package main
func guardInfoLegend() string {
	return "cache floor turns inflight up"
}
`
	root := writeTree(t, f)
	p := Build(Options{Root: root})
	legend := kpiByKey(p, "legend_coverage")
	if len(legend.Defects) != 0 {
		t.Fatalf("expected 0 defects when legend is in info_format.go, got: %v", legend.Defects)
	}
}

func kpiByKey(p scorecard.Payload, key string) scorecard.KPI {
	for _, k := range p.KPIs {
		if k.Key == key {
			return k
		}
	}
	return scorecard.KPI{}
}
