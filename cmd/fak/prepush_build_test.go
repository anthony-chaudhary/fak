package main

import (
	"archive/tar"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// --- seam-injected unit tests (no git/go needed) ---------------------------------------

// prepushSeamSnapshot captures every overridable seam so a test can restore them.
type prepushSeamSnapshot struct {
	revParse     func(string, string) (string, error)
	resolveBase  func(string) string
	changedFiles func(string, string, string) ([]string, error)
	extractTip   func(string, string) (string, error)
	listGraph    func(string) (map[string]string, map[string][]string, int, error)
	listTestOnly func(string, []string) map[string]bool
	build        func(string, []string) (string, bool)
	now          func() time.Time
}

func snapshotPrepushSeams() prepushSeamSnapshot {
	return prepushSeamSnapshot{
		revParse:     prepushRevParse,
		resolveBase:  prepushResolveBase,
		changedFiles: prepushChangedFiles,
		extractTip:   prepushExtractTip,
		listGraph:    prepushListGraph,
		listTestOnly: prepushListTestOnly,
		build:        prepushBuild,
		now:          prepushNow,
	}
}

func (s prepushSeamSnapshot) restore() {
	prepushRevParse = s.revParse
	prepushResolveBase = s.resolveBase
	prepushChangedFiles = s.changedFiles
	prepushExtractTip = s.extractTip
	prepushListGraph = s.listGraph
	prepushListTestOnly = s.listTestOnly
	prepushBuild = s.build
	prepushNow = s.now
}

// setupHappyPrepushSeams wires a green, one-package-change happy path. Individual tests
// override a single seam to exercise one branch.
func setupHappyPrepushSeams(t *testing.T) {
	t.Helper()
	snap := snapshotPrepushSeams()
	t.Cleanup(snap.restore)
	prepushRevParse = func(string, string) (string, error) { return "deadbeefcafef00dfeed", nil }
	prepushResolveBase = func(string) string { return "origin/main" }
	prepushChangedFiles = func(string, string, string) ([]string, error) { return []string{"internal/q/q.go"}, nil }
	prepushExtractTip = func(string, string) (string, error) { return t.TempDir(), nil }
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		// pkg q changed; pkg p imports q — so an edit to q must select p too.
		return map[string]string{"internal/q/q.go": "mod/q"},
			map[string][]string{"mod/p": {"mod/q"}}, 2, nil
	}
	prepushListTestOnly = func(string, []string) map[string]bool { return nil }
	prepushBuild = func(string, []string) (string, bool) { return "", true }
	prepushNow = time.Now
}

func TestEvaluatePrePushBuildOK(t *testing.T) {
	setupHappyPrepushSeams(t)
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "OK" || !res.OK {
		t.Fatalf("want OK/exit0, got verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	// pkg p (importer of the changed q) must be in the built set — the closure that catches
	// the cross-package break.
	if !contains(res.SelectedPackages, "mod/p") || !contains(res.SelectedPackages, "mod/q") {
		t.Fatalf("selected packages missing importer closure: %v", res.SelectedPackages)
	}
}

func TestEvaluatePrePushBuildWouldNotCompile(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushBuild = func(string, []string) (string, bool) {
		return "internal/p/p.go:7:9: undefined: q.X", false
	}
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" || res.OK {
		t.Fatalf("want TRUNK_WOULD_NOT_COMPILE/exit1, got verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if res.Reason != "TRUNK_WOULD_NOT_COMPILE" || !strings.Contains(res.Detail, "undefined: q.X") {
		t.Fatalf("reason/detail not propagated: reason=%q detail=%q", res.Reason, res.Detail)
	}
}

func TestEvaluatePrePushBuildNoChangeIsNoop(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushChangedFiles = func(string, string, string) ([]string, error) { return nil, nil }
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "NOOP" || !res.OK {
		t.Fatalf("want NOOP/exit0, got verdict=%s code=%d", res.Verdict, code)
	}
}

func TestEvaluatePrePushBuildEmptySelectionIsNoop(t *testing.T) {
	setupHappyPrepushSeams(t)
	// A changed .go file that maps to no package in the pushed tip (e.g. only-deletion).
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{}, map[string][]string{}, 1, nil
	}
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "NOOP" {
		t.Fatalf("want NOOP/exit0 for empty selection, got verdict=%s code=%d", res.Verdict, code)
	}
}

func TestEvaluatePrePushBuildOmitsProvenTestOnlyPackage(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		return map[string]string{"tools/probe/probe_test.go": "mod/tools/probe"},
			map[string][]string{}, 1, nil
	}
	prepushChangedFiles = func(string, string, string) ([]string, error) {
		return []string{"tools/probe/probe_test.go"}, nil
	}
	prepushListTestOnly = func(_ string, packages []string) map[string]bool {
		if len(packages) != 1 || packages[0] != "mod/tools/probe" {
			t.Fatalf("classifier received %v, want the selected test-only package", packages)
		}
		return map[string]bool{"mod/tools/probe": true}
	}
	prepushBuild = func(string, []string) (string, bool) {
		t.Fatal("go build must not receive a package proven to contain only tests")
		return "", false
	}

	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "NOOP" || !res.OK {
		t.Fatalf("test-only delta must be a compile NOOP: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if !contains(res.ChangedPackages, "mod/tools/probe") || len(res.SelectedPackages) != 0 {
		t.Fatalf("changed package must remain visible while build selection is empty: changed=%v selected=%v", res.ChangedPackages, res.SelectedPackages)
	}
}

func TestEvaluatePrePushBuildKeepsProductionImporter(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushListTestOnly = func(string, []string) map[string]bool {
		return map[string]bool{"mod/q": true}
	}
	prepushBuild = func(_ string, packages []string) (string, bool) {
		if len(packages) != 1 || packages[0] != "mod/p" {
			t.Fatalf("build received %v, want only production importer mod/p", packages)
		}
		return "", true
	}

	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "OK" || !res.OK {
		t.Fatalf("production importer must still build: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if len(res.SelectedPackages) != 1 || res.SelectedPackages[0] != "mod/p" {
		t.Fatalf("selected packages = %v, want [mod/p]", res.SelectedPackages)
	}
}

func TestEvaluatePrePushBuildUnknownPackageStaysFailSafe(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushListTestOnly = func(string, []string) map[string]bool { return nil }
	prepushBuild = func(_ string, packages []string) (string, bool) {
		if !contains(packages, "mod/q") {
			t.Fatalf("unclassified package was removed from compile admission: %v", packages)
		}
		return "no required module provides package mod/q", false
	}

	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" || res.OK {
		t.Fatalf("unclassified load failure must block: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
}

func TestListTestOnlyPackagesUsesExactGoMetadata(t *testing.T) {
	root := t.TempDir()
	writePrepushFixtureFile(t, filepath.Join(root, "go.mod"), "module testonlyfixture\n\ngo 1.26\n")
	writePrepushFixtureFile(t, filepath.Join(root, "only", "only_test.go"), "package only\n\nimport \"testing\"\n\nfunc TestOnly(t *testing.T) {}\n")
	writePrepushFixtureFile(t, filepath.Join(root, "real", "real.go"), "package real\n")

	got := listTestOnlyPackages(root, []string{"testonlyfixture/only", "testonlyfixture/real", "testonlyfixture/missing"})
	if !got["testonlyfixture/only"] {
		t.Fatalf("test-only package not classified: %v", got)
	}
	if got["testonlyfixture/real"] || got["testonlyfixture/missing"] {
		t.Fatalf("production or unreported package classified test-only: %v", got)
	}
}

func TestParseTestOnlyPackagesRetainsEveryUnprovenShape(t *testing.T) {
	tests := []struct {
		name string
		json string
	}{
		{name: "unknown", json: `{"ImportPath":"mod/unknown"}`},
		{name: "production go", json: `{"ImportPath":"mod/unknown","GoFiles":["real.go"],"TestGoFiles":["real_test.go"]}`},
		{name: "production cgo", json: `{"ImportPath":"mod/unknown","CgoFiles":["real.go"],"XTestGoFiles":["real_test.go"]}`},
		{name: "ignored only", json: `{"ImportPath":"mod/unknown","IgnoredGoFiles":["platform.go"]}`},
		{name: "malformed", json: `{"ImportPath":"mod/unknown","GoFiles":"not-an-array","TestGoFiles":["real_test.go"]}`},
		{name: "different package", json: `{"ImportPath":"mod/not-selected","TestGoFiles":["only_test.go"]}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := parseTestOnlyPackages(strings.NewReader(tt.json), []string{"mod/unknown"})
			if got["mod/unknown"] {
				t.Fatalf("unproven package was classified test-only: %v", got)
			}
		})
	}

	got := parseTestOnlyPackages(strings.NewReader(
		`{"ImportPath":"mod/only","TestGoFiles":["only_test.go"]}`),
		[]string{"mod/only", "mod/unlisted"})
	if !got["mod/only"] {
		t.Fatalf("exact positively proven test-only package not classified: %v", got)
	}
	if got["mod/unlisted"] {
		t.Fatalf("package absent from metadata must stay selected: %v", got)
	}
}

func TestGoBuildPackageArgsAddsExactlyOneTrimpathBeforePackages(t *testing.T) {
	packages := []string{"example.com/mod/a", "example.com/mod/b", "./cmd/..."}
	got := goBuildPackageArgs(packages)
	want := []string{"build", "-trimpath", "example.com/mod/a", "example.com/mod/b", "./cmd/..."}

	if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("go build argv = %q, want %q", got, want)
	}
	var trimpaths int
	for _, arg := range got {
		if arg == "-trimpath" {
			trimpaths++
		}
	}
	if trimpaths != 1 {
		t.Fatalf("go build argv contains %d -trimpath arguments, want exactly one: %q", trimpaths, got)
	}
}

func writePrepushFixtureFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestEvaluatePrePushBuildLatencyIsAdvisoryNotBlock(t *testing.T) {
	setupHappyPrepushSeams(t)
	// Advance the clock across the build so elapsed exceeds the budget, but the build is GREEN.
	var calls int
	base := time.Unix(1_700_000_000, 0)
	prepushNow = func() time.Time {
		calls++
		switch calls {
		case 1, 2, 3, 4, 5:
			return base
		default:
			return base.Add(90 * time.Second)
		}
	}
	res, code := evaluatePrePushBuild("/repo", "", 60*time.Second, false)
	if code != 0 || res.Verdict != "GATE_LATENCY_REGRESSION" || !res.OK {
		t.Fatalf("a slow-but-green build must advise not block: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
}

func TestEvaluatePrePushBuildCouldNotRunFailsOpen(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		return nil, nil, 0, os.ErrNotExist
	}
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 2 || res.Verdict != "COULD_NOT_RUN" {
		t.Fatalf("a could-not-run gate must fail open (exit 2): verdict=%s code=%d", res.Verdict, code)
	}
}

func TestEvaluatePrePushBuildRevParseFailureFailsOpen(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushRevParse = func(string, string) (string, error) { return "", os.ErrNotExist }
	_, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 2 {
		t.Fatalf("HEAD-unresolvable must fail open (exit 2), got code=%d", code)
	}
}

// TestEvaluatePrePushBuildArchiveStallFailsOpen pins the #3432 contract at the gate level: when
// materializing the tip cannot complete (a timed-out / wedged `git archive`), the gate must FAIL
// OPEN (COULD_NOT_RUN, exit 2 → push allowed), never block and never hang. The archive itself is
// bounded by prepushArchiveTimeout (see TestExtractArchivePipeHonorsDeadline); here we assert the
// verdict the shell sees once that bound trips.
func TestEvaluatePrePushBuildArchiveStallFailsOpen(t *testing.T) {
	setupHappyPrepushSeams(t)
	prepushExtractTip = func(string, string) (string, error) {
		return "", fmt.Errorf("archive timed out after 2m0s (failing open): %w", context.DeadlineExceeded)
	}
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 2 || res.Verdict != "COULD_NOT_RUN" || res.OK {
		t.Fatalf("a wedged/timed-out archive must fail open (exit 2, push allowed): verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
}

// --- #3618 pre-existing-red attribution -------------------------------------------------

func TestFailingPackagesFromBuild(t *testing.T) {
	out := "# mod/p\ninternal/p/p.go:7:9: undefined: q.X\ninternal/p/p.go:8:1: too many errors\n" +
		"# mod/r\nr.go:1:1: undefined: Z\n" +
		"no required module provides package mod/missing\n" // load error: no `# ` header → not a compile failure
	got := failingPackagesFromBuild(out)
	if len(got) != 2 || got[0] != "mod/p" || got[1] != "mod/r" {
		t.Fatalf("failingPackagesFromBuild = %v, want [mod/p mod/r] (load-error line excluded)", got)
	}
	if len(failingPackagesFromBuild("")) != 0 {
		t.Errorf("empty build output must yield no failing packages")
	}
}

// setTipFailureSeams wires the happy seams to a tip cone build that fails with a `# pkg` compile
// header for each package in tipFail, and a per-package BASE build where baseGreen[pkg]==true means
// the package builds cleanly at the base (a regression this push introduced) and otherwise it fails
// at the base WITH a compile header (a peer's pre-existing red). The tip cone build is the multi-
// package call; base builds are single-package.
func setTipFailureSeams(t *testing.T, tipFail []string, baseGreen map[string]bool) {
	t.Helper()
	setupHappyPrepushSeams(t)
	var tipDetail strings.Builder
	for _, p := range tipFail {
		fmt.Fprintf(&tipDetail, "# %s\nx.go:1:1: undefined: Y\n", p)
	}
	prepushBuild = func(dir string, pkgs []string) (string, bool) {
		if len(pkgs) != 1 { // the tip cone build (SelectedPackages: mod/p + mod/q)
			return tipDetail.String(), false
		}
		if baseGreen[pkgs[0]] {
			return "", true // green at base → this push regressed it
		}
		return fmt.Sprintf("# %s\nx.go:1:1: undefined: Y\n", pkgs[0]), false // compile-red at base → pre-existing
	}
}

func TestEvaluatePrePushBuildPeerRedIsAlreadyRed(t *testing.T) {
	// Acceptance #3618: the tip is red purely from a peer package (mod/p, red at BOTH tip and
	// base) and this push's delta compiles atop the base → PASS, not TRUNK_WOULD_NOT_COMPILE.
	setTipFailureSeams(t, []string{"mod/p"}, map[string]bool{})
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 0 || res.Verdict != "TRUNK_ALREADY_RED" || !res.OK {
		t.Fatalf("a peer-red tip with a clean delta must PASS: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if !contains(res.PreExistingRed, "mod/p") || len(res.Regressions) != 0 {
		t.Fatalf("attribution wrong: preExisting=%v regressions=%v", res.PreExistingRed, res.Regressions)
	}
	if res.BaseSha == "" { // Acceptance #3618: --json reports the baseline sha it built against.
		t.Fatalf("the baseline sha built against must be reported")
	}
}

func TestEvaluatePrePushBuildDeltaRegressionStillBlocks(t *testing.T) {
	// Acceptance #3618: a delta that genuinely reds the base (mod/p builds at base, fails at tip)
	// still returns TRUNK_WOULD_NOT_COMPILE.
	setTipFailureSeams(t, []string{"mod/p"}, map[string]bool{"mod/p": true})
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" || res.OK {
		t.Fatalf("a delta that reds the base must still block: verdict=%s code=%d ok=%v", res.Verdict, code, res.OK)
	}
	if !contains(res.Regressions, "mod/p") {
		t.Fatalf("mod/p should be attributed as this push's regression: %v", res.Regressions)
	}
}

func TestEvaluatePrePushBuildMixedRegressionAndPreExistingBlocks(t *testing.T) {
	// A real regression (mod/q, green at base) mixed with a pre-existing peer red (mod/p) still
	// blocks — any regression wins — but both are attributed for the operator.
	setTipFailureSeams(t, []string{"mod/p", "mod/q"}, map[string]bool{"mod/q": true})
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" {
		t.Fatalf("a mixed failure with a real regression must block: verdict=%s code=%d", res.Verdict, code)
	}
	if !contains(res.Regressions, "mod/q") || !contains(res.PreExistingRed, "mod/p") {
		t.Fatalf("mixed attribution wrong: regressions=%v preExisting=%v", res.Regressions, res.PreExistingRed)
	}
}

func TestEvaluatePrePushBuildBaseLoadErrorIsFailSafeRegression(t *testing.T) {
	// mod/p fails at the tip; at the base it fails WITHOUT a compile header (an absent/new package
	// or a load error). Triviality of the pre-existing claim cannot be PROVEN → fail-safe to a block.
	setupHappyPrepushSeams(t)
	prepushBuild = func(dir string, pkgs []string) (string, bool) {
		if len(pkgs) != 1 {
			return "# mod/p\np.go:1:1: undefined: Y\n", false
		}
		return "no required module provides package mod/p", false // no `# ` header
	}
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" {
		t.Fatalf("an unprovable base red must fail safe to a block: verdict=%s code=%d", res.Verdict, code)
	}
	if !contains(res.Regressions, "mod/p") || len(res.PreExistingRed) != 0 {
		t.Fatalf("fail-safe: an unprovable base red is this push's regression: reg=%v pre=%v", res.Regressions, res.PreExistingRed)
	}
}

func TestEvaluatePrePushBuildBaselineToleranceOffIsLegacy(t *testing.T) {
	// With the tolerance disarmed, the pre-#3618 whole-tip verdict stands and no base attribution runs.
	setTipFailureSeams(t, []string{"mod/p"}, map[string]bool{}) // would be TRUNK_ALREADY_RED if armed
	old := prepushBaselineTolerance
	prepushBaselineTolerance = false
	t.Cleanup(func() { prepushBaselineTolerance = old })
	res, code := evaluatePrePushBuild("/repo", "", time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" {
		t.Fatalf("with tolerance off the legacy whole-tip verdict must stand: verdict=%s code=%d", res.Verdict, code)
	}
	if res.BaseSha != "" || len(res.PreExistingRed) != 0 {
		t.Fatalf("tolerance off must not run the baseline attribution: base=%q pre=%v", res.BaseSha, res.PreExistingRed)
	}
}

func contains(xs []string, want string) bool {
	for _, x := range xs {
		if x == want {
			return true
		}
	}
	return false
}

// --- fixture-repo integration test: a real commit that breaks an importer ---------------

// TestPrePushBuildFixtureCatchesImporterBreak builds a throwaway two-package git repo where a
// commit removes an exported symbol its importer still calls (the #1338 shape), and asserts the
// gate — run end-to-end through the REAL seams against the archived committed tip — refuses it.
func TestPrePushBuildFixtureCatchesImporterBreak(t *testing.T) {
	if testing.Short() {
		t.Skip("integration: needs git+go")
	}
	// No `tar` here: the gate now untars in-process (the #3432 fix), so the fixture needs only
	// git + go — and is no longer hostage to which tar flavor is first on PATH.
	for _, tool := range []string{"git", "go"} {
		if _, err := exec.LookPath(tool); err != nil {
			t.Skipf("%s not on PATH: %v", tool, err)
		}
	}
	repo := t.TempDir()
	writeFile(t, repo, "go.mod", "module fakgatefix\n\ngo 1.21\n")
	writeFile(t, repo, "a/a.go", "package a\n\n// X is called by package b.\nfunc X() int { return 1 }\n")
	writeFile(t, repo, "b/b.go", "package b\n\nimport \"fakgatefix/a\"\n\n// Y depends on a.X.\nfunc Y() int { return a.X() }\n")
	gitFixture(t, repo, "init", "-q")
	gitFixture(t, repo, "config", "user.email", "t@t")
	gitFixture(t, repo, "config", "user.name", "t")
	gitFixture(t, repo, "add", ".")
	gitFixture(t, repo, "commit", "-q", "-m", "base: green two-package module")
	baseSha := strings.TrimSpace(mustGitOut(t, repo, "rev-parse", "HEAD"))

	// Green control: an additive commit to a (new symbol) must still build.
	writeFile(t, repo, "a/a.go", "package a\n\nfunc X() int { return 1 }\n\n// W is additive; harmless.\nfunc W() int { return 2 }\n")
	gitFixture(t, repo, "add", "a/a.go")
	gitFixture(t, repo, "commit", "-q", "-m", "feat(a): additive W")
	res, code := evaluatePrePushBuild(repo, baseSha, time.Minute, false)
	if code != 0 || res.Verdict != "OK" {
		t.Fatalf("additive green change must pass: verdict=%s code=%d detail=%q", res.Verdict, code, res.Detail)
	}
	if !contains(res.SelectedPackages, "fakgatefix/b") {
		t.Fatalf("importer fakgatefix/b should be in the built closure, got %v", res.SelectedPackages)
	}

	// The break: remove X from a WITHOUT touching b (which still calls a.X).
	base2 := strings.TrimSpace(mustGitOut(t, repo, "rev-parse", "HEAD"))
	writeFile(t, repo, "a/a.go", "package a\n\n// X removed — b still calls it.\nfunc W() int { return 2 }\n")
	gitFixture(t, repo, "add", "a/a.go")
	gitFixture(t, repo, "commit", "-q", "-m", "refactor(a): drop X")
	res, code = evaluatePrePushBuild(repo, base2, time.Minute, false)
	if code != 1 || res.Verdict != "TRUNK_WOULD_NOT_COMPILE" {
		t.Fatalf("removing an imported symbol must be caught: verdict=%s code=%d", res.Verdict, code)
	}
	if !strings.Contains(res.Detail, "undefined") || !strings.Contains(res.Detail, "X") {
		t.Fatalf("compiler detail should name the undefined symbol, got %q", res.Detail)
	}

	// NOOP control: a docs-only commit adds no Go delta.
	base3 := strings.TrimSpace(mustGitOut(t, repo, "rev-parse", "HEAD"))
	writeFile(t, repo, "a/a.go", "package a\n\nfunc W() int { return 2 }\n") // restore green first
	gitFixture(t, repo, "add", "a/a.go")
	gitFixture(t, repo, "commit", "-q", "-m", "fix(a): green")
	base4 := strings.TrimSpace(mustGitOut(t, repo, "rev-parse", "HEAD"))
	writeFile(t, repo, "README.md", "# fixture\n")
	gitFixture(t, repo, "add", "README.md")
	gitFixture(t, repo, "commit", "-q", "-m", "docs: readme")
	_ = base3
	res, code = evaluatePrePushBuild(repo, base4, time.Minute, false)
	if code != 0 || res.Verdict != "NOOP" {
		t.Fatalf("docs-only push must be NOOP, got verdict=%s code=%d", res.Verdict, code)
	}
}

// TestExtractArchiveHonorsDeadline witnesses two #3432 properties end-to-end with a real `git
// archive`: (1) a live context untars the committed tip in-process to disk (a nested dir proves
// recursion); (2) an ALREADY-EXPIRED deadline makes extract RETURN promptly rather than hang — in
// this half exec.CommandContext's Start() refuses to launch git at all, so NO producer is spawned
// (that Start-refusal is exactly what this half asserts, not a kill). The complementary and
// load-bearing case — the deadline killing a producer that is already RUNNING and stalled mid-stream
// — is witnessed deterministically by TestExtractArchiveDeadlineKillsRunningProducer. Needs only git
// now (extraction is in-process, no external tar), so it no longer depends on which tar flavor is
// first on PATH — itself part of the #3432 fix.
func TestExtractArchiveHonorsDeadline(t *testing.T) {
	if testing.Short() {
		t.Skip("integration: needs git")
	}
	if _, err := exec.LookPath("git"); err != nil {
		t.Skipf("git not on PATH: %v", err)
	}
	repo := t.TempDir()
	writeFile(t, repo, "f.txt", "hello\n")
	writeFile(t, repo, "sub/dir/g.txt", "nested\n")
	gitFixture(t, repo, "init", "-q")
	gitFixture(t, repo, "config", "user.email", "t@t")
	gitFixture(t, repo, "config", "user.name", "t")
	gitFixture(t, repo, "add", ".")
	gitFixture(t, repo, "commit", "-q", "-m", "seed")
	sha := strings.TrimSpace(mustGitOut(t, repo, "rev-parse", "HEAD"))

	// Happy path: a live context extracts the committed tip (top-level + nested) to disk.
	dir := t.TempDir()
	if err := extractArchive(context.Background(), repo, sha, dir); err != nil {
		t.Fatalf("live extract failed: %v", err)
	}
	for _, rel := range []string{"f.txt", "sub/dir/g.txt"} {
		if _, err := os.Stat(filepath.Join(dir, filepath.FromSlash(rel))); err != nil {
			t.Fatalf("archived file %s missing after extract: %v", rel, err)
		}
	}

	// No-wedge (Start-refusal half): an already-expired deadline makes exec.CommandContext.Start()
	// refuse to launch git, so extract RETURNS promptly without ever spawning a producer. A generous
	// watchdog fails loudly if it ever hangs. The running-producer kill is covered separately by
	// TestExtractArchiveDeadlineKillsRunningProducer.
	ctx, cancel := context.WithDeadline(context.Background(), time.Unix(1, 0))
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- extractArchive(ctx, repo, sha, t.TempDir()) }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("expired-deadline extract must return an error, got nil")
		}
	case <-time.After(30 * time.Second):
		t.Fatal("extractArchive hung past an expired deadline — the #3432 wedge is not fixed")
	}
}

// TestUntarIntoRoundTripAndTraversal exercises the in-process untar (no git/tar needed): a
// hand-built tar stream with a dir + regular file round-trips to disk, and a `../` entry is
// refused so the build gate can never write outside its throwaway dir.
func TestUntarIntoRoundTripAndTraversal(t *testing.T) {
	dir := t.TempDir()
	var buf strings.Builder
	tw := tar.NewWriter(&buf)
	if err := tw.WriteHeader(&tar.Header{Name: "pkg/", Typeflag: tar.TypeDir, Mode: 0o755}); err != nil {
		t.Fatal(err)
	}
	body := []byte("package pkg\n")
	if err := tw.WriteHeader(&tar.Header{Name: "pkg/a.go", Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(body))}); err != nil {
		t.Fatal(err)
	}
	if _, err := tw.Write(body); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	if err := untarInto(strings.NewReader(buf.String()), dir); err != nil {
		t.Fatalf("untarInto round-trip failed: %v", err)
	}
	got, err := os.ReadFile(filepath.Join(dir, "pkg", "a.go"))
	if err != nil || string(got) != string(body) {
		t.Fatalf("extracted file mismatch: err=%v got=%q", err, got)
	}

	// A traversal entry must be refused, not written above the root.
	var evil strings.Builder
	ew := tar.NewWriter(&evil)
	payload := []byte("x")
	if err := ew.WriteHeader(&tar.Header{Name: "../escape.txt", Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(payload))}); err != nil {
		t.Fatal(err)
	}
	_, _ = ew.Write(payload)
	_ = ew.Close()
	if err := untarInto(strings.NewReader(evil.String()), dir); err == nil {
		t.Fatal("untarInto must refuse a ../ traversal entry")
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "escape.txt")); err == nil {
		t.Fatal("traversal entry escaped the extraction root")
	}
}

// TestHelperProcess is the controllable producer stand-in for `git archive`, injected via the
// prepushArchiveCommand seam. Mode selects the stream shape, then it BLOCKS (never closes stdout,
// never exits) so the consumer is left with a LIVE producer — the exact #3432 condition the shipped
// TestExtractArchiveHonorsDeadline could not reach. It is a no-op unless run as a helper subprocess.
func TestHelperProcess(t *testing.T) {
	if os.Getenv("GO_WANT_HELPER_PROCESS") != "1" {
		return
	}
	switch os.Getenv("GO_HELPER_MODE") {
	case "garbage":
		// A block of non-zero bytes: tar.Reader.Next() reads it as a header and fails its checksum
		// → untarInto returns an error while THIS producer is still alive (exercises the kill branch).
		junk := make([]byte, 4096)
		for i := range junk {
			junk[i] = 'A'
		}
		_, _ = os.Stdout.Write(junk)
	case "valid-partial":
		// A valid dir + file entry, then NO end-of-archive marker: untarInto consumes both and then
		// blocks on the next header read, with this producer still alive (exercises the ctx branch).
		tw := tar.NewWriter(os.Stdout)
		_ = tw.WriteHeader(&tar.Header{Name: "pkg/", Typeflag: tar.TypeDir, Mode: 0o755})
		body := []byte("package pkg\n")
		_ = tw.WriteHeader(&tar.Header{Name: "pkg/a.go", Typeflag: tar.TypeReg, Mode: 0o644, Size: int64(len(body))})
		_, _ = tw.Write(body)
		_ = tw.Flush()
	}
	_ = os.Stdout.Sync()
	time.Sleep(10 * time.Minute) // block until the parent kills us
	os.Exit(0)
}

// withHelperProducer swaps the git-archive seam for a helper-process producer of the given mode.
func withHelperProducer(t *testing.T, mode string) {
	t.Helper()
	old := prepushArchiveCommand
	prepushArchiveCommand = func(ctx context.Context, r, sha string) *exec.Cmd {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestHelperProcess$")
		cmd.Env = append(os.Environ(), "GO_WANT_HELPER_PROCESS=1", "GO_HELPER_MODE="+mode)
		return cmd
	}
	t.Cleanup(func() { prepushArchiveCommand = old })
}

// TestExtractArchiveKillsProducerOnUntarError witnesses the kill-on-untar-error branch
// (ar.Process.Kill() at prepush_build.go): a LIVE producer emits an unparseable stream, so untarInto
// errors while the producer is still running; extractArchive must KILL it and return bounded, never
// block in ar.Wait() on a producer that will never exit. Long deadline, so ctx.Err()==nil isolates
// the untar-error branch. The watchdog fails loudly if the kill regresses (Wait would hang on the
// helper's 10-minute sleep).
func TestExtractArchiveKillsProducerOnUntarError(t *testing.T) {
	withHelperProducer(t, "garbage")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()
	done := make(chan error, 1)
	go func() { done <- extractArchive(ctx, "x", "y", t.TempDir()) }()
	select {
	case err := <-done:
		if err == nil {
			t.Fatal("a garbage stream must yield an error, got nil")
		}
		if errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected an untar error, not a timeout: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("extractArchive hung — a live producer was NOT killed after an untar error (#3432 regression)")
	}
}

// TestExtractArchiveDeadlineKillsRunningProducer witnesses the ctx.Err() timeout branch with a
// genuinely RUNNING producer — the property TestExtractArchiveHonorsDeadline cannot reach (an
// already-expired ctx makes Start refuse before launch). The producer streams a valid partial
// archive then stalls; the deadline fires while untarInto is blocked on the next header;
// CommandContext kills the producer and extractArchive returns a bounded timeout error.
func TestExtractArchiveDeadlineKillsRunningProducer(t *testing.T) {
	withHelperProducer(t, "valid-partial")
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	done := make(chan error, 1)
	start := time.Now()
	go func() { done <- extractArchive(ctx, "x", "y", t.TempDir()) }()
	select {
	case err := <-done:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("expected a deadline error from a stalled running producer, got: %v", err)
		}
		if el := time.Since(start); el > 20*time.Second {
			t.Fatalf("returned but took %s — too slow to have been the deadline", el)
		}
	case <-time.After(30 * time.Second):
		t.Fatal("extractArchive hung past its deadline with a running producer — the #3432 wedge is NOT fixed")
	}
}

// TestSafeArchiveJoinRefusesEscapes pins the traversal guard: absolute, bare `..`, `..`-climbing,
// and (on Windows) volume-relative names are refused, while clean repo-relative names resolve under
// root. Covers the IsAbs, VolumeName, and `..` arms that the round-trip test's single `../` case did
// not.
func TestSafeArchiveJoinRefusesEscapes(t *testing.T) {
	root := filepath.Join(t.TempDir(), "root")
	refuse := []string{"../escape.txt", "..", "a/../../b", "pkg/../../../etc/passwd"}
	if os.PathSeparator == '\\' { // Windows volume-relative names only exist on Windows
		refuse = append(refuse, `C:..\..\Windows\System32\evil.go`, `C:foo`, `\\server\share\x`)
	}
	for _, name := range refuse {
		if _, err := safeArchiveJoin(root, name); err == nil {
			t.Errorf("safeArchiveJoin must refuse %q, but it was accepted", name)
		}
	}
	for _, name := range []string{"pkg/a.go", "go.mod", "a/b/c.go", "pkg/./a.go"} {
		got, err := safeArchiveJoin(root, name)
		if err != nil {
			t.Errorf("safeArchiveJoin must accept clean name %q: %v", name, err)
			continue
		}
		if rel, rerr := filepath.Rel(root, got); rerr != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(os.PathSeparator)) {
			t.Errorf("accepted name %q resolved outside root: %q", name, got)
		}
	}
}

func writeFile(t *testing.T, root, rel, body string) {
	t.Helper()
	p := filepath.Join(root, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func gitFixture(t *testing.T, repo string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-C", repo}, args...)...)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
}

func mustGitOut(t *testing.T, repo string, args ...string) string {
	t.Helper()
	out, err := gitOut(repo, args...)
	if err != nil {
		t.Fatalf("git %s: %v", strings.Join(args, " "), err)
	}
	return out
}

func TestPrepushBuildAuditsExplicitTipInsteadOfHead(t *testing.T) {
	saved := snapshotPrepushSeams()
	defer saved.restore()
	var resolved []string
	prepushRevParse = func(_ string, ref string) (string, error) {
		resolved = append(resolved, ref)
		return ref, nil
	}
	prepushResolveBase = func(string) string { return "origin/main" }
	var diffTip string
	prepushChangedFiles = func(_, _, tip string) ([]string, error) {
		diffTip = tip
		return nil, nil
	}
	res, code := evaluatePrePushBuildAt("repo", "", "pushed-sha", time.Minute, false)
	if code != 0 || res.Ref != "pushed-sha" || diffTip != "pushed-sha" {
		t.Fatalf("explicit tip result=%+v code=%d diffTip=%q resolved=%v", res, code, diffTip, resolved)
	}
	if len(resolved) == 0 || resolved[0] != "pushed-sha" {
		t.Fatalf("rev-parse refs=%v, want pushed-sha first", resolved)
	}
}

func TestPrepushSuccessReceiptReusesOnlySameFreshTip(t *testing.T) {
	commonDir := t.TempDir()
	oldCommonDir := prepushSuccessCommonDir
	prepushSuccessCommonDir = func(string) string { return commonDir }
	t.Cleanup(func() { prepushSuccessCommonDir = oldCommonDir })

	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	if prepushSuccessReusable("repo", "tip-a", now) {
		t.Fatal("missing receipt reused")
	}
	recordPrepushSuccess("repo", "tip-a", now)
	if !prepushSuccessReusable("repo", "tip-a", now.Add(3*time.Minute)) {
		t.Fatal("same-tip success older than the former two-minute window not reused")
	}
	if prepushSuccessReusable("repo", "tip-b", now.Add(time.Second)) {
		t.Fatal("changed tip reused")
	}
	path := prepushSuccessReceiptPath("repo", "tip-a")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var receipt prepushSuccessReceipt
	if err := json.Unmarshal(raw, &receipt); err != nil {
		t.Fatal(err)
	}
	receipt.GateContract = "old-contract"
	raw, _ = json.Marshal(receipt)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	if prepushSuccessReusable("repo", "tip-a", now.Add(time.Second)) {
		t.Fatal("changed gate contract reused")
	}
	recordPrepushSuccess("repo", "tip-a", now)
	if prepushSuccessReusable("repo", "tip-a", now.Add(prepushSuccessReuseTTL+time.Second)) {
		t.Fatal("expired success reused")
	}
}

func TestPrepushFailureDoesNotCreateSuccessReceipt(t *testing.T) {
	commonDir := t.TempDir()
	oldCommonDir := prepushSuccessCommonDir
	prepushSuccessCommonDir = func(string) string { return commonDir }
	t.Cleanup(func() { prepushSuccessCommonDir = oldCommonDir })

	if prepushSuccessReusable("repo", "failed-tip", time.Now()) {
		t.Fatal("failure path unexpectedly reusable")
	}
	if _, err := os.Stat(prepushSuccessReceiptPath("repo", "tip-a")); !os.IsNotExist(err) {
		t.Fatalf("failure created receipt: %v", err)
	}
}

func TestClaimPrepushTipCoalescesOnIndependentSuccess(t *testing.T) {
	commonDir := t.TempDir()
	oldCommonDir, oldSleep := prepushSuccessCommonDir, prepushSuccessSleep
	prepushSuccessCommonDir = func(string) string { return commonDir }
	prepushSuccessSleep = func(time.Duration) {
		recordPrepushSuccess("repo", "same-tip", time.Now())
	}
	t.Cleanup(func() {
		prepushSuccessCommonDir, prepushSuccessSleep = oldCommonDir, oldSleep
	})

	owner, release := claimPrepushTip("repo", "same-tip", time.Now)
	if !owner {
		t.Fatal("first claimant was not owner")
	}
	defer release()
	coalescedOwner, coalescedRelease := claimPrepushTip("repo", "same-tip", time.Now)
	coalescedRelease()
	if coalescedOwner {
		t.Fatal("same-tip waiter reran gate after witnessed success")
	}
}

func TestPrepushClaimHelper(t *testing.T) {
	t.Helper()
	mode := os.Getenv("GO_PREPUSH_CLAIM_HELPER")
	if mode == "" {
		t.Skip("helper process")
		return
	}
	root, tip := os.Getenv("GO_PREPUSH_CLAIM_ROOT"), os.Getenv("GO_PREPUSH_CLAIM_TIP")
	owner, release := claimPrepushTip(root, tip, time.Now)
	defer release()
	switch mode {
	case "owner":
		if !owner {
			os.Exit(3)
		}
		if err := os.WriteFile(os.Getenv("GO_PREPUSH_CLAIM_READY"), []byte("ready"), 0o644); err != nil {
			os.Exit(4)
		}
		time.Sleep(300 * time.Millisecond)
		recordPrepushSuccess(root, tip, time.Now())
		_, _ = fmt.Fprintln(os.Stdout, "OWNER")
	case "waiter":
		if owner {
			os.Exit(5)
		}
		_, _ = fmt.Fprintln(os.Stdout, "COALESCED")
	default:
		os.Exit(6)
	}
}

func TestClaimPrepushTipCoalescesAcrossProcesses(t *testing.T) {
	root := t.TempDir()
	cmd := exec.Command("git", "init", "-q", root)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("git init unavailable: %v: %s", err, out)
	}
	tip := strings.Repeat("a", 40)
	ready := filepath.Join(root, "owner.ready")
	helper := func(mode string) *exec.Cmd {
		cmd := exec.Command(os.Args[0], "-test.run=^TestPrepushClaimHelper$")
		cmd.Env = append(os.Environ(),
			"GO_PREPUSH_CLAIM_HELPER="+mode,
			"GO_PREPUSH_CLAIM_ROOT="+root,
			"GO_PREPUSH_CLAIM_TIP="+tip,
			"GO_PREPUSH_CLAIM_READY="+ready,
		)
		return cmd
	}
	owner := helper("owner")
	var ownerOut bytes.Buffer
	owner.Stdout, owner.Stderr = &ownerOut, &ownerOut
	if err := owner.Start(); err != nil {
		t.Fatal(err)
	}
	deadline := time.Now().Add(5 * time.Second)
	for {
		if _, err := os.Stat(ready); err == nil {
			break
		}
		if time.Now().After(deadline) {
			_ = owner.Process.Kill()
			t.Fatal("owner never published claim readiness")
		}
		time.Sleep(10 * time.Millisecond)
	}
	waiterOut, waiterErr := helper("waiter").CombinedOutput()
	if err := owner.Wait(); err != nil {
		t.Fatalf("owner: %v: %s", err, ownerOut.String())
	}
	if waiterErr != nil {
		t.Fatalf("waiter: %v: %s", waiterErr, waiterOut)
	}
	if !strings.Contains(ownerOut.String(), "OWNER") || !strings.Contains(string(waiterOut), "COALESCED") {
		t.Fatalf("owner=%q waiter=%q", ownerOut.String(), waiterOut)
	}
	// A caller arriving after the owner removes its claim must still reuse the success witness.
	lateOut, lateErr := helper("waiter").CombinedOutput()
	if lateErr != nil || !strings.Contains(string(lateOut), "COALESCED") {
		t.Fatalf("late waiter: %v: %s", lateErr, lateOut)
	}
}

func TestPrepushSuccessReceiptsSurviveInterleavedTips(t *testing.T) {
	commonDir := t.TempDir()
	oldCommonDir := prepushSuccessCommonDir
	prepushSuccessCommonDir = func(string) string { return commonDir }
	t.Cleanup(func() { prepushSuccessCommonDir = oldCommonDir })
	now := time.Date(2026, 8, 14, 12, 0, 0, 0, time.UTC)
	recordPrepushSuccess("repo", "tip-a", now)
	recordPrepushSuccess("repo", "tip-b", now.Add(time.Minute))
	if !prepushSuccessReusable("repo", "tip-a", now.Add(2*time.Minute)) || !prepushSuccessReusable("repo", "tip-b", now.Add(2*time.Minute)) {
		t.Fatal("interleaved success evicted a still-reusable tip")
	}
}

func TestPrunePrepushSuccessReceiptsBoundsFiles(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	for i := 0; i < prepushSuccessMaxFiles+5; i++ {
		path := filepath.Join(dir, fmt.Sprintf("%03d.json", i))
		if err := os.WriteFile(path, []byte(`{}`), 0o600); err != nil {
			t.Fatal(err)
		}
		stamp := now.Add(time.Duration(i) * time.Second)
		if err := os.Chtimes(path, stamp, stamp); err != nil {
			t.Fatal(err)
		}
	}
	prunePrepushSuccessReceipts(dir, now.Add(time.Minute))
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != prepushSuccessMaxFiles {
		t.Fatalf("receipt files=%d want=%d", len(entries), prepushSuccessMaxFiles)
	}
}

func TestPrepushTreeReceiptReusesCommitBuildCheck(t *testing.T) {
	oldCommon := prepushSuccessCommonDir
	dir := t.TempDir()
	prepushSuccessCommonDir = func(string) string { return dir }
	t.Cleanup(func() { prepushSuccessCommonDir = oldCommon })
	now := time.Unix(1_700_000_000, 0)
	recordPrepushSuccessForTree("repo", "tree-a", now)
	if !prepushTreeSuccessReusable("repo", "tree-a", now.Add(time.Minute)) {
		t.Fatal("green prospective-tree receipt was not reusable by pre-push")
	}
	if prepushTreeSuccessReusable("repo", "tree-b", now.Add(time.Minute)) {
		t.Fatal("receipt for a different immutable tree was reused")
	}
}

func TestRunPrepushReusesCommitBuildReceiptOnlyForCoveredCommit(t *testing.T) {
	oldCommon, oldRev, oldCovered, oldNow := prepushSuccessCommonDir, prepushTreeResolveFn, prepushCommitPathsCoveredFn, prepushNow
	dir := t.TempDir()
	prepushSuccessCommonDir = func(string) string { return dir }
	prepushTreeResolveFn = func(string, string) (string, error) { return "tree-a", nil }
	prepushCommitPathsCoveredFn = func(string, string) bool { return true }
	now := time.Unix(1_700_000_000, 0)
	prepushNow = func() time.Time { return now }
	t.Cleanup(func() {
		prepushSuccessCommonDir, prepushTreeResolveFn, prepushCommitPathsCoveredFn, prepushNow = oldCommon, oldRev, oldCovered, oldNow
	})
	recordPrepushSuccessForTree("repo", "tree-a", now)
	var out, errOut bytes.Buffer
	if code := runHooksPrePush(&out, &errOut, []string{"--root", "repo", "--tip", "tip-a"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if !strings.Contains(out.String(), "source=commit-build-check") {
		t.Fatalf("output did not name receipt source: %q", out.String())
	}
}
