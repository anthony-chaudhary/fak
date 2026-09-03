package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestCheckFreshDetectsHeadlineFamilyAndNoop(t *testing.T) {
	root := fixtureRepo(t)
	res, err := CheckFresh(root)
	if err != nil || !res.Fresh {
		t.Fatalf("initial freshness = %+v, %v", res, err)
	}
	readme := filepath.Join(root, filepath.FromSlash(GeneratedReadme))
	original, _ := os.ReadFile(readme)
	cases := []struct{ name, old, new string }{
		{"score headline", "Legacy bounded score (saturates; not the driver) |", "Legacy bounded score (saturates; not the driver) stale |"},
		{"family count", "**Crystal-clear concepts (and climbing)** |", "**Crystal-clear concepts (and climbing)** stale |"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			mutated := strings.Replace(string(original), tc.old, tc.new, 1)
			if mutated == string(original) {
				t.Fatalf("fixture marker %q missing", tc.old)
			}
			if err := os.WriteFile(readme, []byte(mutated), 0644); err != nil {
				t.Fatal(err)
			}
			got, err := CheckFresh(root)
			if err != nil {
				t.Fatal(err)
			}
			if got.Fresh || len(got.StalePaths) != 1 || got.StalePaths[0] != GeneratedReadme {
				t.Fatalf("got %+v", got)
			}
			_ = os.WriteFile(readme, original, 0644)
		})
	}
}

func TestGeneratedBytesEqualNormalizesOnlyCheckoutLineEndings(t *testing.T) {
	lf := []byte("# score\n\nvalue\n")
	crlf := []byte("# score\r\n\r\nvalue\r\n")
	if !generatedBytesEqual(lf, crlf) || !generatedBytesEqual(crlf, lf) {
		t.Fatal("LF and CRLF forms of the same generated text must compare fresh")
	}
	if generatedBytesEqual(lf, []byte("# score\n\nchanged\n")) {
		t.Fatal("content drift must remain stale")
	}
	if generatedBytesEqual([]byte("a\rb"), []byte("a\nb")) {
		t.Fatal("bare carriage returns are content, not checkout line endings")
	}
}

func TestGeneratedSnapshotIsByteStableAndPortable(t *testing.T) {
	root := fixtureRepo(t)
	a := filepath.Join(t.TempDir(), "a")
	b := filepath.Join(t.TempDir(), "b")
	if err := generate(root, a); err != nil {
		t.Fatal(err)
	}
	if err := generate(root, b); err != nil {
		t.Fatal(err)
	}
	aa, _ := os.ReadFile(filepath.Join(a, "README.md"))
	bb, _ := os.ReadFile(filepath.Join(b, "README.md"))
	if string(aa) != string(bb) {
		t.Fatal("generation is not byte stable")
	}
	if strings.Contains(string(aa), root) {
		t.Fatal("generated output leaks machine path")
	}
}

func TestCheckInvariantUsesOneGreenScorecardSnapshot(t *testing.T) {
	root := fixtureRepo(t)
	calls := installInvariantSnapshotHelper(t, "green")

	got, err := CheckInvariant(root)
	if err != nil {
		t.Fatal(err)
	}
	if *calls != 1 {
		t.Fatalf("scorecard snapshot processes = %d, want exactly 1", *calls)
	}
	if !got.Freshness.Fresh || !got.SemanticValid || !got.CriticalClean {
		t.Fatalf("green invariant flags = %+v", got)
	}
	if got.Coverage != 98.25 || got.CoverageDebt != 7 || got.ClarityDebt != 0 {
		t.Fatalf("green invariant totals = %+v", got)
	}
	if got.FamilyCoverage["cache"] != 87.5 {
		t.Fatalf("cache family coverage = %v, want 87.5", got.FamilyCoverage["cache"])
	}
}

func TestCheckInvariantParsesStructuredRedExit(t *testing.T) {
	root := fixtureRepo(t)
	installInvariantSnapshotHelper(t, "red")

	got, err := CheckInvariant(root)
	if err != nil {
		t.Fatalf("structured ACTION exit must remain a result, got %v", err)
	}
	if !got.Freshness.Fresh || !got.SemanticValid {
		t.Fatalf("structured red snapshot lost valid companion results: %+v", got)
	}
	if got.CriticalClean || got.ClarityDebt != 3 || got.CoverageDebt != 9 {
		t.Fatalf("structured red payload fold = %+v", got)
	}
	if got.Detail != "clarity debt remains" {
		t.Fatalf("structured red detail = %q", got.Detail)
	}
}

func TestExecuteInvariantSnapshotReturnsProcessStartFailure(t *testing.T) {
	old := invariantSnapshotCommand
	t.Cleanup(func() { invariantSnapshotCommand = old })
	invariantSnapshotCommand = func(_, _ string) *exec.Cmd {
		return exec.Command(filepath.Join(t.TempDir(), "missing-scorecard-executable"))
	}

	_, err := executeInvariantSnapshot(t.TempDir(), filepath.Join(t.TempDir(), "generated"))
	if err == nil {
		t.Fatal("missing executable must return its process-start error")
	}
	if _, ok := err.(*exec.ExitError); ok {
		t.Fatalf("process-start failure was misclassified as a scorecard verdict: %v", err)
	}
}

func TestExecuteInvariantSnapshotReturnsMissingArtifactError(t *testing.T) {
	root := t.TempDir()
	generated := filepath.Join(t.TempDir(), "generated")
	installInvariantSnapshotHelper(t, "missing-artifact")

	_, err := executeInvariantSnapshot(root, generated)
	if err == nil {
		t.Fatal("partial snapshot must return the missing artifact error")
	}
	if !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("missing artifact error must preserve os.ErrNotExist, got %T: %v", err, err)
	}
	if !strings.Contains(err.Error(), "INDEX.md") {
		t.Fatalf("missing artifact error does not name the absent file: %v", err)
	}
}

func installInvariantSnapshotHelper(t *testing.T, mode string) *int {
	t.Helper()
	old := invariantSnapshotCommand
	t.Cleanup(func() { invariantSnapshotCommand = old })
	calls := 0
	invariantSnapshotCommand = func(root, generated string) *exec.Cmd {
		calls++
		cmd := exec.Command(os.Args[0], "-test.run=^TestInvariantSnapshotHelperProcess$")
		cmd.Env = append(os.Environ(),
			"FAK_TEST_INVARIANT_HELPER=1",
			"FAK_TEST_INVARIANT_MODE="+mode,
			"FAK_TEST_INVARIANT_ROOT="+root,
			"FAK_TEST_INVARIANT_GENERATED="+generated,
		)
		return cmd
	}
	return &calls
}

func TestInvariantSnapshotHelperProcess(t *testing.T) {
	if os.Getenv("FAK_TEST_INVARIANT_HELPER") != "1" {
		return
	}
	root := os.Getenv("FAK_TEST_INVARIANT_ROOT")
	generated := os.Getenv("FAK_TEST_INVARIANT_GENERATED")
	mode := os.Getenv("FAK_TEST_INVARIANT_MODE")
	if err := os.MkdirAll(generated, 0755); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	for i, art := range generatedArtifacts {
		if mode == "missing-artifact" && i == len(generatedArtifacts)-1 {
			continue
		}
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(art.Tracked)))
		if err != nil {
			b = []byte("synthetic generated artifact\n")
		}
		if err := os.WriteFile(filepath.Join(generated, art.Name), b, 0644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(2)
		}
	}
	if mode == "red" || mode == "missing-artifact" {
		fmt.Print(`{"ok":false,"reason":"clarity debt remains","corpus":{"coverage_debt":9,"clarity_defects":3,"coverage":{"coverage_pct":97.75,"per_family":[{"family":"cache","discovered":8,"covered":6}]}}}`)
		os.Exit(1)
	}
	fmt.Print(`{"ok":true,"reason":"","corpus":{"coverage_debt":7,"clarity_defects":0,"coverage":{"coverage_pct":98.25,"per_family":[{"family":"cache","discovered":8,"covered":7}]}}}`)
	os.Exit(0)
}

// TestGeneratedReadmeRoundTripsThroughGitStaging is the #5136 regression: the
// generator must emit bytes that git staging leaves unchanged, even under the
// Windows-default core.autocrlf=true normalization. A CRLF write is staged as
// LF, so the committed artifact can never byte-match a fresh regeneration and
// CONCEPT_FRESHNESS becomes structurally unsatisfiable on Windows.
func TestGeneratedReadmeRoundTripsThroughGitStaging(t *testing.T) {
	root := fixtureRepo(t)
	disk, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(GeneratedReadme)))
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(disk, []byte("\r\n")) {
		t.Fatal("generator wrote CRLF; git normalizes the staged blob to LF so freshness can never match")
	}
	repo := t.TempDir()
	if _, err := git(repo, "init", "-q"); err != nil {
		t.Fatal(err)
	}
	if _, err := git(repo, "config", "core.autocrlf", "true"); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "README.md"), disk, 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := git(repo, "add", "README.md"); err != nil {
		t.Fatal(err)
	}
	staged, err := gitStdout(repo, "show", ":README.md")
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(staged, disk) {
		t.Fatalf("generated README does not round-trip through git staging: disk %d bytes, staged %d bytes", len(disk), len(staged))
	}
}

// gitStdout runs git capturing stdout only, so blob content is never mixed
// with normalization warnings the way the CombinedOutput helper would.
func gitStdout(root string, args ...string) ([]byte, error) {
	cmd := exec.Command("git", args...)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	return cmd.Output()
}

func fixtureRepo(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	// Tests use committed inputs so peer WIP cannot affect the fixture.
	tmp := t.TempDir()
	out, err := git(root, "archive", "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	// CheckGitTree owns safe archive extraction; seed by cloning the tracked tree cheaply.
	cmdRoot := filepath.Join(tmp, "repo")
	if err := os.MkdirAll(cmdRoot, 0755); err != nil {
		t.Fatal(err)
	}
	_ = out
	// Use git archive through a temporary worktree-like extraction command for test brevity.
	// tar is part of the repository's supported build environment.
	// On Windows bsdtar is available as tar.exe.
	tr := tar.NewReader(bytes.NewReader(out))
	for {
		h, e := tr.Next()
		if e == io.EOF {
			break
		}
		if e != nil {
			t.Fatal(e)
		}
		dst := filepath.Join(cmdRoot, filepath.FromSlash(h.Name))
		if h.FileInfo().IsDir() {
			_ = os.MkdirAll(dst, 0755)
			continue
		}
		if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
			t.Fatal(err)
		}
		f, e := os.Create(dst)
		if e != nil {
			t.Fatal(e)
		}
		if _, e = io.Copy(f, tr); e != nil {
			f.Close()
			t.Fatal(e)
		}
		if e = f.Close(); e != nil {
			t.Fatal(e)
		}
	}
	// Exercise the generator under test, while all corpus/data inputs remain the committed tree.
	script, err := os.ReadFile(filepath.Join(root, "tools", "concept_disambiguation_scorecard.py"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(cmdRoot, "tools", "concept_disambiguation_scorecard.py"), script, 0644); err != nil {
		t.Fatal(err)
	}
	if err := generate(cmdRoot, filepath.Join(cmdRoot, "docs", "concept-disambiguation-scorecard")); err != nil {
		t.Fatal(err)
	}
	return cmdRoot
}

// TestRegenerateFromGitTreeClearsTheFreshnessGate is the #5829 witness: the cure the
// CONCEPT_FRESHNESS refusal names has to clear the refusal.
//
// The gate scores a git tree, so the fixture is built the way a pathspec commit builds
// one - a staged artifact change on top of HEAD - and an UNSTAGED peer package sits
// beside it whose directory name carries a family root, so the generator discovers one
// more confusable token from the worktree than from the tree. That single unstaged
// directory is the whole gap: it makes the worktree regeneration (RegenerateCommand)
// answer a tree the gate never scored, which is why running it left the refusal
// standing. The two modes are asserted to disagree at the end, so a future change that
// quietly points the tree-scoped path at the worktree fails here instead of shipping.
func TestRegenerateFromGitTreeClearsTheFreshnessGate(t *testing.T) {
	root := fixtureGitRepo(t)
	readme := filepath.Join(root, filepath.FromSlash(GeneratedReadme))
	original, err := os.ReadFile(readme)
	if err != nil {
		t.Fatal(err)
	}
	const marker = "Legacy bounded score (saturates; not the driver) |"
	mutated := strings.Replace(string(original), marker, marker+" stale", 1)
	if mutated == string(original) {
		t.Fatalf("fixture marker %q missing", marker)
	}
	if err := os.WriteFile(readme, []byte(mutated), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := git(root, "add", "--", GeneratedReadme); err != nil {
		t.Fatal(err)
	}
	// The peer's in-flight package: worktree only, never staged. A directory name is a
	// structural concept, and this one carries the guard-gate family root, so it shifts
	// the discovered-token count the generator reports.
	peer := filepath.Join(root, "internal", "guardpeerwip")
	if err := os.MkdirAll(peer, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(peer, "guardpeerwip.go"), []byte("package guardpeerwip\n\n// Probe is a peer's unsaved work.\nfunc Probe() bool { return true }\n"), 0644); err != nil {
		t.Fatal(err)
	}

	before, err := CheckGitTree(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if before.Fresh {
		t.Fatal("fixture must start stale for the staged tree")
	}
	if !strings.HasPrefix(before.Regenerate, RegenerateStagedCommand+" --tree ") {
		t.Fatalf("a git-tree refusal must pin its cure to the tree it scored, got %q", before.Regenerate)
	}
	pinnedTree := strings.TrimPrefix(before.Regenerate, RegenerateStagedCommand+" --tree ")

	// The operator reads the refusal and runs the printed command in their OWN shell, where
	// "the current index" is not the index the gate scored - a pre-commit hook gets git's
	// temporary partial-commit index, the shell gets the shared .git/index that every peer
	// stages into. Moving the index here is what makes this test able to tell a tree-pinned
	// cure from a bare one: with a bare `--staged` the regeneration below would answer the
	// polluted index and the retry would refuse again, which is the #5829 defect surviving
	// one level down.
	if _, err := git(root, "add", "--", filepath.Join("internal", "guardpeerwip")); err != nil {
		t.Fatal(err)
	}
	moved, err := git(root, "write-tree")
	if err != nil {
		t.Fatal(err)
	}
	if strings.TrimSpace(string(moved)) == pinnedTree {
		t.Fatal("fixture did not move the index off the scored tree, so it cannot distinguish a pinned cure from a bare one")
	}

	// Verbatim: the tree named in the refusal, not whatever the shell's index happens to be.
	written, err := RegenerateFromGitTree(root, pinnedTree, root)
	if err != nil {
		t.Fatal(err)
	}
	if len(written) != len(generatedArtifacts) {
		t.Fatalf("wrote %v, want every tracked artifact", written)
	}
	// The retry is a fresh pathspec commit, so rebuild the index the way `git commit -- <paths>`
	// does: HEAD plus exactly the committer's pathspec. This also discards the pollution above,
	// which is the point - the peer's staged work was never in the tree being committed.
	if _, err := git(root, "read-tree", "HEAD"); err != nil {
		t.Fatal(err)
	}
	for _, p := range written {
		if _, err := git(root, "add", "--", p); err != nil {
			t.Fatal(err)
		}
	}
	after, err := CheckGitTree(root, "")
	if err != nil {
		t.Fatal(err)
	}
	if !after.Fresh {
		t.Fatalf("the printed cure %q did not clear the check; still stale: %v", before.Regenerate, after.StalePaths)
	}

	// The modes must genuinely differ, or this test would pass for the wrong reason.
	worktree, err := CheckFresh(root)
	if err != nil {
		t.Fatal(err)
	}
	if worktree.Fresh {
		t.Fatal("the unstaged peer package did not move the generator's answer, so this fixture cannot tell tree mode from worktree mode")
	}
	if worktree.Regenerate != RegenerateCommand {
		t.Fatalf("worktree mode must keep the worktree cure, got %q", worktree.Regenerate)
	}
}

// fixtureGitRepo is fixtureRepo committed into a hermetic repo, so CheckGitTree has an
// index to resolve. Hooks and signing are disabled: the fixture is an input, not a
// commit under test.
func fixtureGitRepo(t *testing.T) string {
	t.Helper()
	root := fixtureRepo(t)
	if _, err := git(root, "init", "-q"); err != nil {
		t.Fatal(err)
	}
	// Exact bytes on both sides of staging; the CRLF path has its own #5136 test.
	if _, err := git(root, "config", "core.autocrlf", "false"); err != nil {
		t.Fatal(err)
	}
	if _, err := git(root, "add", "-A"); err != nil {
		t.Fatal(err)
	}
	if _, err := git(root, "-c", "user.name=concept-test", "-c", "user.email=concept-test@example.invalid",
		"-c", "commit.gpgsign=false", "-c", "core.hooksPath=", "commit", "-q", "-m", "fixture"); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestGenerateFailsLoudWhenPythonMissing(t *testing.T) {
	root := t.TempDir()
	out := filepath.Join(root, "docs", "concept-disambiguation-scorecard")
	if err := os.MkdirAll(out, 0755); err != nil {
		t.Fatal(err)
	}
	for _, art := range generatedArtifacts {
		if err := os.WriteFile(filepath.Join(out, art.Name), []byte("stale content\n"), 0644); err != nil {
			t.Fatal(err)
		}
	}
	emptyPath := t.TempDir()
	t.Setenv("PATH", emptyPath)

	err := generate(root, out)
	if err == nil {
		t.Fatal("generate must fail loudly when Python interpreter is missing, even if artifacts pre-exist on disk")
	}
	if !strings.Contains(err.Error(), "Python interpreter not found") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestResolvePythonHermetic(t *testing.T) {
	tmp := t.TempDir()
	t.Setenv("PATH", tmp)

	if _, err := ResolvePython(); err == nil {
		t.Fatal("want error when no python on PATH")
	}

	py3Name := "python3"
	if runtime.GOOS == "windows" {
		py3Name = "python3.bat"
	}
	py3Path := filepath.Join(tmp, py3Name)
	if err := os.WriteFile(py3Path, []byte("#!/bin/sh\nexit 0\n"), 0755); err != nil {
		t.Fatal(err)
	}
	got, err := ResolvePython()
	if err != nil {
		t.Fatalf("ResolvePython with python3 on PATH failed: %v", err)
	}
	if !strings.Contains(strings.ToLower(got), "python3") {
		t.Fatalf("ResolvePython=%q, want python3", got)
	}
}
