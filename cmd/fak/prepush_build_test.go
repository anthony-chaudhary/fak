package main

import (
	"archive/tar"
	"context"
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
	changedFiles func(string, string) ([]string, error)
	extractTip   func(string, string) (string, error)
	listGraph    func(string) (map[string]string, map[string][]string, int, error)
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
	prepushChangedFiles = func(string, string) ([]string, error) { return []string{"internal/q/q.go"}, nil }
	prepushExtractTip = func(string, string) (string, error) { return t.TempDir(), nil }
	prepushListGraph = func(string) (map[string]string, map[string][]string, int, error) {
		// pkg q changed; pkg p imports q — so an edit to q must select p too.
		return map[string]string{"internal/q/q.go": "mod/q"},
			map[string][]string{"mod/p": {"mod/q"}}, 2, nil
	}
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
	prepushChangedFiles = func(string, string) ([]string, error) { return nil, nil }
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

func TestEvaluatePrePushBuildLatencyIsAdvisoryNotBlock(t *testing.T) {
	setupHappyPrepushSeams(t)
	// Advance the clock across the build so elapsed exceeds the budget, but the build is GREEN.
	var calls int
	base := time.Unix(1_700_000_000, 0)
	prepushNow = func() time.Time {
		calls++
		if calls == 1 {
			return base
		}
		return base.Add(90 * time.Second)
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

// TestExtractArchiveHonorsDeadline is the direct end-to-end witness for the #3432 fix: with a live
// context `git archive` is untarred in-process to disk (a nested dir proves recursion); with an
// ALREADY-EXPIRED deadline the extract must RETURN an error (the producer killed) rather than
// hang — the exact property whose absence wedged every agent's trunk push for 14+ hours. Needs
// only git now (extraction is in-process, no external tar), so it no longer depends on which
// tar flavor is first on PATH — itself part of the #3432 fix.
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

	// No-wedge: an already-expired deadline must make the extract RETURN (producer killed), not
	// hang. A generous watchdog fails loudly if the #3432 wedge ever regresses.
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
