package hooks

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// gate_deadcode_test.go — unit tests for the DEAD_CODE hygiene gate plus a differential parity
// harness that asserts the Go gate agrees with the Python oracle (tools/code_slop_scorecard.py's
// kpi_dead_code) on canned trees AND on the live tracked tree. The unit tests pin the exact
// verdict on the corner cases the port has to get right (exported skip, //slop:keep opt-out,
// string/comment references not counted, asm-referenced symbol live, per-file cap). The parity
// test proves the whole thing is behavior-identical to the scorecard it fronts.

// treeFromFiles builds a TrackedTree in-memory from a path->content map, with no git or disk —
// the gate reads FileBytes, which the fileCache serves directly.
func treeFromFiles(files map[string]string) *TrackedTree {
	paths := make([]string, 0, len(files))
	cache := map[string]fileEntry{}
	for p, body := range files {
		paths = append(paths, p)
		cache[p] = fileEntry{data: []byte(body), exists: true}
	}
	sort.Strings(paths)
	return &TrackedTree{Root: "", Paths: paths, fileCache: cache}
}

// deadNames returns the sorted "file :: name" keys of a gate's findings.
func deadNames(findings []Finding) []string {
	out := make([]string, 0, len(findings))
	for _, f := range findings {
		// The Detail leads with "<file> :: <name> is a dead ...".
		key := f.File + " :: " + strings.TrimSpace(strings.SplitN(strings.TrimPrefix(f.Detail, f.File+" :: "), " is a dead", 2)[0])
		out = append(out, key)
	}
	sort.Strings(out)
	return out
}

func TestGateDeadCodeTree_FlagsUnreferencedUnexported(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc used() int { return 1 }\n\nfunc dead() int { return 2 }\n\nfunc Caller() int { return used() }\n",
	}
	got, err := gateDeadCodeTree(treeFromFiles(files))
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	names := deadNames(got)
	want := []string{"internal/x/a.go :: dead"}
	if strings.Join(names, "|") != strings.Join(want, "|") {
		t.Fatalf("dead symbols: got %v want %v", names, want)
	}
}

func TestGateDeadCodeTree_ExportedNeverGraded(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc Exported() int { return 1 }\n\ntype AlsoExported struct{}\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("exported symbols must not be graded, got %v", deadNames(got))
	}
}

func TestGateDeadCodeTree_InitMainBlankSkipped(t *testing.T) {
	files := map[string]string{
		"cmd/y/main.go": "package main\n\nfunc init() {}\n\nfunc main() {}\n\nvar _ = 1\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("init/main/_ must not be graded, got %v", deadNames(got))
	}
}

func TestGateDeadCodeTree_SlopKeepOptOut(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go": "package x\n\n//slop:keep symbol-table provenance marker\nfunc intentionallyDead() int { return 1 }\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("//slop:keep must suppress the finding, got %v", deadNames(got))
	}
}

func TestGateDeadCodeTree_ReferenceInStringNotCounted(t *testing.T) {
	// `dead` appears again only inside a string literal and a comment — neither is a real
	// reference, so the symbol is still dead. This is the code-only-scan contract: the string
	// husk is blanked before the identifier scan, so the mention inside it is not a phantom
	// reference. `Keep` is exported (used by an external caller a static scan can't see) so it
	// anchors the string without itself being graded, keeping the fixture to a single expected
	// dead symbol.
	files := map[string]string{
		"internal/x/a.go": "package x\n\nfunc dead() int { return 1 }\n\n// dead is mentioned here\nfunc Keep() string { return \"call dead now\" }\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	names := deadNames(got)
	if strings.Join(names, "|") != "internal/x/a.go :: dead" {
		t.Fatalf("string/comment mention must not count as a reference, got %v", names)
	}
}

func TestGateDeadCodeTree_TestFileReferenceCountsLive(t *testing.T) {
	files := map[string]string{
		"internal/x/a.go":      "package x\n\nfunc helper() int { return 1 }\n",
		"internal/x/a_test.go": "package x\n\nimport \"testing\"\n\nfunc TestH(t *testing.T) { _ = helper() }\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a symbol referenced from a _test.go is live, got %v", deadNames(got))
	}
}

func TestGateDeadCodeTree_AsmReferenceCountsLive(t *testing.T) {
	files := map[string]string{
		"internal/x/simd.go": "package x\n\nfunc kernelDot(a, b []float32) float32\n",
		"internal/x/simd.s":  "// +build amd64\nTEXT ·kernelDot(SB), 0, $0-52\n\tRET\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("a symbol whose body is asm (referenced via ·name) is live, got %v", deadNames(got))
	}
}

func TestGateDeadCodeTree_PerFileCap(t *testing.T) {
	// Nine dead symbols in one file; the cap is deadCapPerFile (5).
	var b strings.Builder
	b.WriteString("package x\n\n")
	for i := 0; i < 9; i++ {
		b.WriteString("func dead")
		b.WriteByte(byte('A' + i))
		b.WriteString("() {}\n")
	}
	files := map[string]string{"internal/x/a.go": b.String()}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != deadCapPerFile {
		t.Fatalf("per-file cap: got %d findings, want %d", len(got), deadCapPerFile)
	}
}

func TestGateDeadCodeTree_ExcludedDirsNotGraded(t *testing.T) {
	files := map[string]string{
		"internal/x/testdata/fixture.go": "package fixture\n\nfunc deadFixture() int { return 1 }\n",
		"vendor/dep/dep.go":              "package dep\n\nfunc deadVendor() int { return 1 }\n",
	}
	got, _ := gateDeadCodeTree(treeFromFiles(files))
	if len(got) != 0 {
		t.Fatalf("testdata/vendor must be excluded, got %v", deadNames(got))
	}
}

// --- parity against the Python oracle -------------------------------------

// scorecardDeadDefects runs tools/code_slop_scorecard.py --json over root and returns the
// sorted "rel :: name" keys from the dead_code KPI's defect list — the oracle verdict.
func scorecardDeadDefects(t *testing.T, clone, root string) []string {
	t.Helper()
	py, pyArgs := pyExe()
	if py == "" {
		t.Skip("python not on PATH")
	}
	script := filepath.Join(clone, "tools", "code_slop_scorecard.py")
	if _, err := os.Stat(script); err != nil {
		t.Skipf("scorecard not found: %v", err)
	}
	args := append(append([]string{}, pyArgs...), script, "--json", "--workspace", root)
	cmd := exec.Command(py, args...)
	// The scorecard is a gate: it may exit nonzero when slop-debt > 0. We only care about its
	// stdout JSON, so capture stdout and tolerate a nonzero exit — but a run that emitted NO
	// JSON at all (interpreter missing a module, wrong python) is a genuine skip, not a mismatch.
	// This keeps the harness green in pure-Linux CI (one OS for both tools) and skips cleanly on
	// the dev box's mixed WSL/Windows python interop.
	var stdout, stderr strings.Builder
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	_ = cmd.Run()
	out := []byte(strings.TrimSpace(stdout.String()))
	if len(out) == 0 || out[0] != '{' {
		t.Skipf("scorecard produced no JSON (python env): %s", strings.TrimSpace(stderr.String()))
	}
	var payload struct {
		Kpis []struct {
			Kpi     string   `json:"kpi"`
			Defects []string `json:"defects"`
		} `json:"kpis"`
	}
	if err := json.Unmarshal(out, &payload); err != nil {
		t.Fatalf("scorecard json parse: %v", err)
	}
	var keys []string
	for _, k := range payload.Kpis {
		if k.Kpi != "dead_code" {
			continue
		}
		for _, d := range k.Defects {
			// "dead unexported symbol (defined, never referenced): <rel> :: <name>" — split on
			// the FIRST "): " so the "<rel> :: <name>" tail (which itself contains " :: ") stays
			// intact and matches the Go gate's file-plus-name key exactly.
			if i := strings.Index(d, "): "); i >= 0 {
				keys = append(keys, strings.TrimSpace(d[i+3:]))
			}
		}
	}
	sort.Strings(keys)
	return keys
}

// initTempRepo creates a git repo under a temp dir, writes files, and stages them so the
// scorecard's git-ls-files corpus selection sees them (matching the live-tree code path).
func initTempRepo(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "t@t")
	run("config", "user.name", "t")
	for rel, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	run("add", "-A")
	return dir
}

func TestGateDeadCodeTree_ParityWithScorecard(t *testing.T) {
	if testing.Short() {
		t.Skip("parity harness skipped under -short")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	clone := repoRoot(t)
	cases := map[string]map[string]string{
		"mixed": {
			"internal/x/a.go":      "package x\n\nfunc used() int { return 1 }\n\nfunc dead() int { return 2 }\n\nfunc Caller() int { return used() }\n",
			"internal/x/a_test.go": "package x\n\nimport \"testing\"\n\nfunc TestC(t *testing.T) { _ = Caller() }\n",
		},
		"slopkeep_and_string": {
			"internal/x/b.go": "package x\n\n//slop:keep marker\nfunc keptDead() {}\n\nfunc reallyDead() {}\n\nvar s = \"reallyDead keptDead\"\n",
		},
		"clean": {
			"internal/x/c.go": "package x\n\nfunc a() int { return b() }\n\nfunc b() int { return 1 }\n\nvar _ = a\n",
		},
	}
	for name, files := range cases {
		t.Run(name, func(t *testing.T) {
			dir := initTempRepo(t, files)
			want := scorecardDeadDefects(t, clone, dir)
			tree, err := ReadTrackedTree(dir)
			if err != nil {
				t.Fatalf("ReadTrackedTree: %v", err)
			}
			got := deadNames(mustGate(t, tree))
			if strings.Join(got, "|") != strings.Join(want, "|") {
				t.Fatalf("parity mismatch:\n go: %v\n py: %v", got, want)
			}
		})
	}
}

// TestGateDeadCodeTree_ParityLiveTree runs both the Go gate and the Python oracle over the REAL
// tracked tree and asserts the exact same dead-symbol set — the strongest parity witness, and
// the check that must pass before DEAD_CODE flips DefaultOff:false.
func TestGateDeadCodeTree_ParityLiveTree(t *testing.T) {
	if testing.Short() || os.Getenv("CI") == "" {
		t.Skip("live-tree parity skipped under local test")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	clone := repoRoot(t)
	tree, err := ReadTrackedTree(clone)
	if err != nil {
		t.Fatalf("ReadTrackedTree: %v", err)
	}
	got := deadNames(mustGate(t, tree))
	want := scorecardDeadDefects(t, clone, clone)
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("live-tree parity mismatch:\n go (%d): %v\n py (%d): %v", len(got), got, len(want), want)
	}
}

func mustGate(t *testing.T, tree *TrackedTree) []Finding {
	t.Helper()
	f, err := gateDeadCodeTree(tree)
	if err != nil {
		t.Fatalf("gate error: %v", err)
	}
	return f
}
