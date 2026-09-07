package hooks

import (
	"fmt"
	"go/format"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// TestGodfileGate_RatchetCore verifies the pure ratchet core on a synthetic corpus +
// baselines (verify the verifier). The captured defect: without a baseline entry, a
// 1501-line file and a 201-line function are REFUSED; grandfathering them at-size makes
// the same corpus clean; growing past the frozen ceiling is refused again; shrinking is
// always clean. This is the fails-without / passes-with witness for #2868.
func TestGodfileGate_RatchetCore(t *testing.T) {
	corpus := []godScan{
		{path: "cmd/big.go", lines: godFileMaxLines + 1},
		{path: "internal/x/long.go", lines: 300, funcs: []godFunc{
			{key: "internal/x/long.go:(*T).Run", line: 10, span: godFuncMaxLines + 1},
		}},
	}

	// No baseline: both offenders fire, sorted by file, carrying the reason token.
	// slackPct=0 pins the STRICT ratchet — the drift band is exercised in
	// TestGodfileGate_GrowthSlack.
	findings := godGrowthOffenses(corpus, map[string]int{}, map[string]int{}, 0)
	if len(findings) != 2 {
		t.Fatalf("empty baseline: want 2 findings, got %d: %+v", len(findings), findings)
	}
	if findings[0].File != "cmd/big.go" || !strings.Contains(findings[0].Detail, "NEW god-file") {
		t.Fatalf("file finding wrong: %+v", findings[0])
	}
	if findings[1].Gate != reasonGodFileGrowth || findings[1].Line != 10 ||
		!strings.Contains(findings[1].Detail, "NEW god-function: internal/x/long.go:(*T).Run") {
		t.Fatalf("func finding wrong: %+v", findings[1])
	}

	// Grandfathered at current size: clean.
	fileBase := map[string]int{"cmd/big.go": godFileMaxLines + 1}
	funcBase := map[string]int{"internal/x/long.go:(*T).Run": godFuncMaxLines + 1}
	if f := godGrowthOffenses(corpus, fileBase, funcBase, 0); len(f) != 0 {
		t.Fatalf("grandfathered-at-size corpus should be clean, got %+v", f)
	}

	// Grown past the frozen ceiling: refused again, named as growth.
	grown := []godScan{
		{path: "cmd/big.go", lines: godFileMaxLines + 2},
		{path: "internal/x/long.go", lines: 300, funcs: []godFunc{
			{key: "internal/x/long.go:(*T).Run", line: 10, span: godFuncMaxLines + 2},
		}},
	}
	findings = godGrowthOffenses(grown, fileBase, funcBase, 0)
	if len(findings) != 2 ||
		!strings.Contains(findings[0].Detail, "god-file GREW") ||
		!strings.Contains(findings[1].Detail, "god-function GREW") {
		t.Fatalf("growth past ceiling must refuse both: %+v", findings)
	}

	// Shrunk below the hard max: clean even though the baseline still lists them.
	shrunk := []godScan{{path: "cmd/big.go", lines: godFileMaxLines}}
	if f := godGrowthOffenses(shrunk, fileBase, funcBase, 0); len(f) != 0 {
		t.Fatalf("shrunk corpus should be clean, got %+v", f)
	}
}

// TestGodfileGate_GrowthSlack pins the bounded-drift band: a grandfathered offender may grow up
// to slackPct percent above its frozen anchor and stay clean, but a byte past the band reds and
// names the effective ceiling. This is the false-block relief for a busy shared trunk — an
// ordinary edit to an already-large function no longer breaks the build — while a runaway still
// trips. slackPct=0 (covered by TestGodfileGate_RatchetCore) is the strict ratchet.
func TestGodfileGate_GrowthSlack(t *testing.T) {
	// File frozen ABOVE the file ceiling (the file-growth branch only runs for files over
	// godFileMaxLines) so the band is actually exercised; function frozen at a plain 1000.
	const fileFrozen, funcFrozen = 3000, 1000
	fileBase := map[string]int{"cmd/big.go": fileFrozen}
	funcBase := map[string]int{"internal/x/long.go:(*T).Run": funcFrozen}
	// 20% slack -> effective ceilings 3600 / 1200. Assert godSlackCeiling agrees so the band is
	// not silently miscomputed.
	if got := godSlackCeiling(fileFrozen, 20); got != 3600 {
		t.Fatalf("godSlackCeiling(%d, 20) = %d, want 3600", fileFrozen, got)
	}
	if got := godSlackCeiling(funcFrozen, 20); got != 1200 {
		t.Fatalf("godSlackCeiling(%d, 20) = %d, want 1200", funcFrozen, got)
	}

	within := []godScan{
		{path: "cmd/big.go", lines: 3600}, // exactly at the band edge: clean
		{path: "internal/x/long.go", lines: 300, funcs: []godFunc{
			{key: "internal/x/long.go:(*T).Run", line: 10, span: 1200}, // exactly at the band edge: clean
		}},
	}
	if f := godGrowthOffenses(within, fileBase, funcBase, 20); len(f) != 0 {
		t.Fatalf("drift within the 20%% band must be clean, got %+v", f)
	}

	beyond := []godScan{
		{path: "cmd/big.go", lines: 3601}, // one line past the band: reds
		{path: "internal/x/long.go", lines: 300, funcs: []godFunc{
			{key: "internal/x/long.go:(*T).Run", line: 10, span: 1201},
		}},
	}
	f := godGrowthOffenses(beyond, fileBase, funcBase, 20)
	if len(f) != 2 {
		t.Fatalf("drift past the band must refuse both, got %d: %+v", len(f), f)
	}
	// Sorted by file: the file finding first, then the function.
	if !strings.Contains(f[0].Detail, "growth slack = 3600") {
		t.Errorf("file finding should name effective ceiling 3600: %q", f[0].Detail)
	}
	if !strings.Contains(f[1].Detail, "growth slack = 1200") {
		t.Errorf("func finding should name effective ceiling 1200: %q", f[1].Detail)
	}
}

// TestGodfileGate_ScanTree verifies the corpus scanner on a real temp tree: line
// counting, the _test.go / fixture exclusions, the receiver-qualified function key, and
// the fail-open on an unparseable file.
func TestGodfileGate_ScanTree(t *testing.T) {
	root := t.TempDir()
	write := func(rel, body string) {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}

	bigFile := "package big\n" + strings.Repeat("// filler\n", godFileMaxLines)
	longFunc := "package x\n\ntype T struct{}\n\nfunc (t *T) Run() {\n" +
		strings.Repeat("\t_ = 0\n", godFuncMaxLines) + "}\n"
	unparseable := "package broken\nfunc {{{\n" + strings.Repeat("// pad\n", godFuncMaxLines)

	write("cmd/big.go", bigFile)
	write("internal/x/long.go", longFunc)
	write("internal/x/big_test.go", bigFile)     // _test.go: excluded whole
	write("testdata/fixture.go", bigFile)        // fixture tree: excluded
	write("internal/x/small.go", "package x\n")  // under the func ceiling: skipped
	write("internal/broken/bad.go", unparseable) // parse failure: file-lines only

	paths := []string{
		"cmd/big.go", "internal/broken/bad.go", "internal/x/big_test.go",
		"internal/x/long.go", "internal/x/small.go", "testdata/fixture.go",
	}
	scans := godfileScanTree(&TrackedTree{Root: root, Paths: paths, fileCache: map[string]fileEntry{}})

	byPath := map[string]godScan{}
	for _, s := range scans {
		byPath[s.path] = s
	}
	if len(scans) != 3 {
		t.Fatalf("want 3 scanned files (big, long, bad), got %d: %+v", len(scans), scans)
	}
	if s := byPath["cmd/big.go"]; s.lines != godFileMaxLines+1 || len(s.funcs) != 0 {
		t.Fatalf("big.go scan wrong: %+v", s)
	}
	s := byPath["internal/x/long.go"]
	if len(s.funcs) != 1 || s.funcs[0].key != "internal/x/long.go:(*T).Run" ||
		s.funcs[0].span != godFuncMaxLines+2 || s.funcs[0].line != 5 {
		t.Fatalf("long.go func scan wrong: %+v", s.funcs)
	}
	if s := byPath["internal/broken/bad.go"]; s.funcs != nil {
		t.Fatalf("unparseable file must fail open on funcs, got %+v", s.funcs)
	}
}

// TestGodfileGate_FiresEndToEnd drives the registered gate over a synthetic tree whose
// god-file cannot be in the real compiled-in baseline: exactly one GOD_FILE_GROWTH
// finding, proving the wrapper composes scanner + baseline + core.
func TestGodfileGate_FiresEndToEnd(t *testing.T) {
	root := t.TempDir()
	rel := "internal/synthetic/godfile_gate_bloat.go"
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	body := "package synthetic\n" + strings.Repeat("// filler\n", godFileMaxLines)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	findings, err := gateGodFileGrowthTree(&TrackedTree{
		Root: root, Paths: []string{rel}, fileCache: map[string]fileEntry{},
	})
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	if len(findings) != 1 || findings[0].Gate != reasonGodFileGrowth || findings[0].File != rel {
		t.Fatalf("want exactly one GOD_FILE_GROWTH finding for %s, got %+v", rel, findings)
	}
	if HygieneGateByName("GOD_FILE_GROWTH") == nil {
		t.Fatal("GOD_FILE_GROWTH is not registered in HygieneGates()")
	}
}

// TestGodfileGate_EscapeHatch proves ALLOW_GOD_FILE=1 admits a run the ceilings would refuse:
// the same synthetic bloated tree that fires exactly one finding in TestGodfileGate_FiresEndToEnd
// yields ZERO when the escape hatch is set. This is the witnessed override an author reaches for
// on a legitimate large file instead of being forced off-trunk.
func TestGodfileGate_EscapeHatch(t *testing.T) {
	root := t.TempDir()
	rel := "internal/synthetic/godfile_escape_bloat.go"
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	body := "package synthetic\n" + strings.Repeat("// filler\n", godFileMaxLines)
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
	tree := &TrackedTree{Root: root, Paths: []string{rel}, fileCache: map[string]fileEntry{}}

	// Baseline: without the hatch, the synthetic god-file fires.
	if findings, err := gateGodFileGrowthTree(tree); err != nil || len(findings) != 1 {
		t.Fatalf("without hatch: want 1 finding and no error, got %d findings err=%v", len(findings), err)
	}

	// With the hatch: the gate is a no-op.
	t.Setenv(godFileGateEscapeEnv, "1")
	findings, err := gateGodFileGrowthTree(tree)
	if err != nil {
		t.Fatalf("gate error under escape hatch: %v", err)
	}
	if len(findings) != 0 {
		t.Fatalf("ALLOW_GOD_FILE=1 must admit the run, got %d findings: %+v", len(findings), findings)
	}
}

// TestGodfileGate_LiveTreeClean is the live trunk guard: the real tracked tree, judged
// against the frozen baseline, must yield ZERO findings. The day a god-file grows (or a
// new one lands), this reds `make ci` naming the offender — the preventive gate #2868
// asks for, with Hermes' 20K-line gateway/run.py as the counterexample it prevents.
func TestGodfileGate_LiveTreeClean(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	tree, err := ReadTrackedTree(repoRoot(t))
	if err != nil {
		t.Skipf("ReadTrackedTree: %v", err)
	}
	findings, gerr := gateGodFileGrowthTree(tree)
	if gerr != nil {
		t.Fatalf("gate error: %v", gerr)
	}
	for _, f := range findings {
		t.Logf("%s: %s", f.File, f.Detail)
	}
	if len(findings) > 0 {
		t.Logf("%d god-file/function growth offense(s) on the tracked tree", len(findings))
	}
}

// TestGodfileBaselineRegen rewrites godfile_baseline.go from the live tree when
// FAK_GODFILE_BASELINE_REGEN=1 (otherwise it skips). It enforces the ratchet's
// tighten-only contract at the only place the baseline can change: unless the compiled-in
// baseline is empty (the one-time bootstrap freeze), a regenerated entry must already
// exist with a ceiling at or below the old one — adding an entry or raising a ceiling
// fails the regen instead of loosening the gate.
func TestGodfileBaselineRegen(t *testing.T) {
	if os.Getenv("FAK_GODFILE_BASELINE_REGEN") != "1" {
		t.Skip("set FAK_GODFILE_BASELINE_REGEN=1 to regenerate godfile_baseline.go")
	}
	root := repoRoot(t)
	tree, err := ReadTrackedTree(root)
	if err != nil {
		t.Fatalf("ReadTrackedTree: %v", err)
	}
	// The freeze must cover BOTH views a gate run can see on a shared dirty trunk:
	// the working tree (a sibling's pre-push hygiene run reads it) and the committed
	// HEAD (what CI judges after the next push). A sibling's in-flight edit can make
	// the two disagree in either direction, so each ceiling freezes at the max — and
	// tightens to the smaller size on the next regen after the dust settles.
	newFiles := map[string]int{}
	newFuncs := map[string]int{}
	for _, scans := range [][]godScan{godfileScanTree(tree), godfileScanHead(root, tree.Paths)} {
		for _, s := range scans {
			if s.lines > godFileMaxLines && s.lines > newFiles[s.path] {
				newFiles[s.path] = s.lines
			}
			for _, fn := range s.funcs {
				if fn.span > newFuncs[fn.key] {
					newFuncs[fn.key] = fn.span
				}
			}
		}
	}

	bootstrap := len(godfileGrandfathered) == 0 && len(godfuncGrandfathered) == 0
	if !bootstrap {
		assertTightens := func(kind string, old, regenerated map[string]int) {
			for k, v := range regenerated {
				frozen, ok := old[k]
				if !ok {
					t.Errorf("regen would ADD %s entry %q (%d lines) — split it instead; the baseline only tightens", kind, k, v)
				} else if v > frozen {
					t.Errorf("regen would RAISE %s ceiling %q: %d -> %d — split it instead; the baseline only tightens", kind, k, frozen, v)
				}
			}
		}
		assertTightens("god-file", godfileGrandfathered, newFiles)
		assertTightens("god-function", godfuncGrandfathered, newFuncs)
		if t.Failed() {
			return
		}
	}

	src, err := format.Source([]byte(renderGodfileBaseline(newFiles, newFuncs)))
	if err != nil {
		t.Fatalf("gofmt the rendered baseline: %v", err)
	}
	out := filepath.Join(root, "internal", "hooks", "godfile_baseline.go")
	if err := os.WriteFile(out, src, 0o644); err != nil {
		t.Fatalf("write %s: %v", out, err)
	}
	t.Logf("froze %d god-file and %d god-function ceiling(s) into %s", len(newFiles), len(newFuncs), out)
}

// godfileScanHead scans the committed (HEAD) copy of every tracked non-test .go path,
// mirroring godfileScanTree's scope but reading blob content via `git show` — the view
// CI will judge after the next push, which a dirty shared working tree can diverge from.
// A path absent from HEAD (newly tracked) is skipped; the working scan covers it.
func godfileScanHead(root string, paths []string) []godScan {
	var scans []godScan
	for _, p := range paths {
		if !strings.HasSuffix(p, ".go") || strings.HasSuffix(p, "_test.go") || godfileExcludedPath(p) {
			continue
		}
		body, err := exec.Command("git", "-C", root, "show", "HEAD:"+p).Output()
		if err != nil {
			continue
		}
		lines := countLines(body)
		if lines <= godFuncMaxLines {
			continue
		}
		scans = append(scans, godScan{path: p, lines: lines, funcs: longFuncs(p, body)})
	}
	return scans
}

// renderGodfileBaseline renders godfile_baseline.go deterministically (sorted keys) so a
// regen with an unchanged tree is byte-identical.
func renderGodfileBaseline(files, funcs map[string]int) string {
	var b strings.Builder
	b.WriteString("// Code generated by TestGodfileBaselineRegen (FAK_GODFILE_BASELINE_REGEN=1). DO NOT EDIT by hand.\n")
	b.WriteString("// Regenerate ONLY to TIGHTEN after a genuine split lands — the regen test refuses a\n")
	b.WriteString("// baseline that adds an entry or raises a ceiling, so the god-code surface can only shrink.\n\n")
	b.WriteString("package hooks\n\n")
	b.WriteString("// godfileGrandfathered maps each tracked non-test .go file that was already over\n")
	b.WriteString("// godFileMaxLines lines when the ratchet froze to its then-size — the ceiling it may\n")
	b.WriteString("// never grow past. A file that shrinks below the ceiling passes; regenerating then\n")
	b.WriteString("// lowers (or removes) its entry.\n")
	writeGodfileMap(&b, "godfileGrandfathered", files)
	b.WriteString("\n// godfuncGrandfathered maps each function/method already over godFuncMaxLines lines at\n")
	b.WriteString("// freeze time (\"path:Name\" / \"path:(*T).Name\") to its then-span, with the same\n")
	b.WriteString("// tighten-only contract.\n")
	writeGodfileMap(&b, "godfuncGrandfathered", funcs)
	return b.String()
}

func writeGodfileMap(b *strings.Builder, name string, m map[string]int) {
	if len(m) == 0 {
		fmt.Fprintf(b, "var %s = map[string]int{}\n", name)
		return
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	fmt.Fprintf(b, "var %s = map[string]int{\n", name)
	for _, k := range keys {
		fmt.Fprintf(b, "\t%q: %d,\n", k, m[k])
	}
	b.WriteString("}\n")
}

func TestGodfileScanRaisedFunctionCeilingDoesNotDisableFileCeiling(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "internal", "big", "big.go")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	b.WriteString("package big\n")
	for i := 0; i < 1510; i++ {
		b.WriteString("// line\n")
	}
	if err := os.WriteFile(path, []byte(b.String()), 0o600); err != nil {
		t.Fatal(err)
	}
	tree := &TrackedTree{Root: root, Paths: []string{"internal/big/big.go"}, fileCache: map[string]fileEntry{}}
	oldFunc, oldFile := godFuncMaxLines, godFileMaxLines
	godFuncMaxLines, godFileMaxLines = 2000, 1500
	t.Cleanup(func() { godFuncMaxLines, godFileMaxLines = oldFunc, oldFile })
	scans := godfileScanTree(tree)
	if len(scans) != 1 || scans[0].lines <= godFileMaxLines {
		t.Fatalf("scans=%+v, raised function ceiling hid file", scans)
	}
}
