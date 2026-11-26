package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/committedtree"
	"github.com/anthony-chaudhary/fak/internal/conceptcatalog/scorecard"
	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// fak-test:runtime fast est=1s
func TestExtractGitTreeStreamsAndReapsArchive(t *testing.T) {
	if mode := os.Getenv("FAK_TEST_GIT_ARCHIVE_MODE"); mode != "" {
		writeGitArchiveHelper(t, mode)
		return
	}
	old := gitArchiveCommand
	t.Cleanup(func() { gitArchiveCommand = old })
	for _, mode := range []string{"stream", "producer-failure", "truncated-failure", "extraction-failure", "start-failure"} {
		t.Run(mode, func(t *testing.T) {
			dst := t.TempDir()
			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			var cmd *exec.Cmd
			gitArchiveCommand = func(string) *exec.Cmd {
				if mode == "start-failure" {
					cmd = exec.CommandContext(ctx, filepath.Join(t.TempDir(), "missing-git"))
				} else {
					cmd = exec.CommandContext(ctx, os.Args[0], "-test.run=^TestExtractGitTreeStreamsAndReapsArchive$")
					cmd.Env = append(os.Environ(), "FAK_TEST_GIT_ARCHIVE_MODE="+mode, "FAK_TEST_GIT_ARCHIVE_DEST="+dst)
				}
				return cmd
			}
			err := extractGitTree(t.TempDir(), "fixture-tree", dst)
			if ctx.Err() != nil {
				t.Fatalf("archive producer did not finish or get reaped promptly: %v", err)
			}
			if mode != "start-failure" && cmd.ProcessState == nil {
				t.Fatal("archive producer was not reaped before return")
			}
			switch mode {
			case "stream":
				if err != nil {
					t.Fatal(err)
				}
				if body, err := os.ReadFile(filepath.Join(dst, "first.txt")); err != nil || string(body) != "first body\n" {
					t.Fatalf("extracted file = %q, %v", body, err)
				}
			case "producer-failure", "truncated-failure":
				var exitErr *exec.ExitError
				if !errors.As(err, &exitErr) || exitErr.ExitCode() != 7 || !strings.Contains(err.Error(), "git archive fixture-tree") {
					t.Fatalf("producer failure = %v, want wrapped exit 7", err)
				}
				wantStderr := "archive producer failed\n"
				if mode == "producer-failure" {
					wantStderr = strings.Repeat("a", 32<<10) + "\n... omitting 131072 bytes ...\n" + strings.Repeat("c", 32<<10)
				} else if !errors.Is(err, io.ErrUnexpectedEOF) {
					t.Fatalf("partial archive lost its extraction error: %v", err)
				}
				if string(exitErr.Stderr) != wantStderr {
					t.Fatalf("producer stderr did not preserve its bounded prefix/suffix: got %d bytes, want %d", len(exitErr.Stderr), len(wantStderr))
				}
			case "extraction-failure":
				if err == nil || !strings.HasPrefix(err.Error(), `unsafe archive path "../escaped.txt"`) {
					t.Fatalf("extraction failure was lost to producer termination: %v", err)
				}
			case "start-failure":
				if (!errors.Is(err, os.ErrNotExist) && !errors.Is(err, exec.ErrNotFound)) || !strings.Contains(err.Error(), "git archive fixture-tree") {
					t.Fatalf("start failure = %v, want wrapped missing executable", err)
				}
			}
		})
	}
}

func writeGitArchiveHelper(t *testing.T, mode string) {
	t.Helper()
	name := "first.txt"
	if mode == "extraction-failure" {
		name = "../escaped.txt"
	}
	const body = "first body\n"
	tw := tar.NewWriter(os.Stdout)
	size := int64(len(body))
	if mode == "truncated-failure" {
		size++
		fmt.Fprintln(os.Stderr, "archive producer failed")
	}
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0644, Size: size}); err != nil {
		t.Fatal(err)
	}
	if _, err := io.WriteString(tw, body); err != nil {
		t.Fatal(err)
	}
	if mode == "truncated-failure" {
		os.Exit(7)
	}
	if err := tw.Flush(); err != nil {
		t.Fatal(err)
	}
	if mode == "extraction-failure" {
		// A failed consumer must stop and reap a producer that has not exited.
		time.Sleep(10 * time.Second)
		os.Exit(8)
	}
	if mode == "stream" {
		deadline := time.Now().Add(3 * time.Second)
		for {
			got, err := os.ReadFile(filepath.Join(os.Getenv("FAK_TEST_GIT_ARCHIVE_DEST"), name))
			if err == nil && string(got) == body {
				break
			}
			if time.Now().After(deadline) {
				fmt.Fprintln(os.Stderr, "file was not extracted before producer completion")
				os.Exit(9)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}
	// Output beyond the tar end marker must be drained before waiting for git.
	if _, err := os.Stdout.Write(make([]byte, 128<<10)); err != nil {
		t.Fatal(err)
	}
	if mode == "producer-failure" {
		fmt.Fprint(os.Stderr, strings.Repeat("a", 32<<10), strings.Repeat("b", 128<<10), strings.Repeat("c", 32<<10))
		os.Exit(7)
	}
	os.Exit(0)
}

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

func TestCheckInvariantReturnsSnapshotFailure(t *testing.T) {
	old := runInvariantSnapshot
	t.Cleanup(func() { runInvariantSnapshot = old })
	want := errors.New("scorecard fold failed")
	runInvariantSnapshot = func(_, _ string) (scorecard.Summary, error) { return scorecard.Summary{}, want }

	if _, err := CheckInvariant(t.TempDir()); !errors.Is(err, want) {
		t.Fatalf("snapshot failure = %v, want %v", err, want)
	}
}

func TestCheckInvariantReturnsMissingArtifactError(t *testing.T) {
	installInvariantSnapshotHelper(t, "missing-artifact")

	_, err := CheckInvariant(t.TempDir())
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

// installInvariantSnapshotHelper replaces the in-process snapshot with a fixed
// verdict that copies the tracked artifacts as its generated output.
func installInvariantSnapshotHelper(t *testing.T, mode string) *int {
	t.Helper()
	old := runInvariantSnapshot
	t.Cleanup(func() { runInvariantSnapshot = old })
	calls := 0
	runInvariantSnapshot = func(root, generated string) (scorecard.Summary, error) {
		calls++
		if err := os.MkdirAll(generated, 0755); err != nil {
			return scorecard.Summary{}, err
		}
		for i, art := range generatedArtifacts {
			if mode == "missing-artifact" && i == len(generatedArtifacts)-1 {
				continue
			}
			b, readErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(art.Tracked)))
			if readErr != nil {
				b = []byte("synthetic generated artifact\n")
			}
			if writeErr := os.WriteFile(filepath.Join(generated, art.Name), b, 0644); writeErr != nil {
				return scorecard.Summary{}, writeErr
			}
		}
		var s scorecard.Summary
		if mode == "red" || mode == "missing-artifact" {
			s.Reason = "clarity debt remains"
			s.Corpus.CoverageDebt, s.Corpus.ClarityDefects = 9, 3
			s.Corpus.Coverage.CoveragePct = 97.75
			s.Corpus.Coverage.PerFamily = []scorecard.FamilyCount{{Family: "cache", Discovered: 8, Covered: 6}}
			return s, nil
		}
		s.OK = true
		s.Corpus.CoverageDebt = 7
		s.Corpus.Coverage.CoveragePct = 98.25
		s.Corpus.Coverage.PerFamily = []scorecard.FamilyCount{{Family: "cache", Discovered: 8, Covered: 7}}
		return s, nil
	}
	return &calls
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
	// Tests use committed inputs so peer WIP cannot affect the fixture. committedtree keeps
	// export-ignored paths such as docs/ and tools/, which the generator reads.
	tmp := t.TempDir()
	var snapshot bytes.Buffer
	if err := committedtree.WriteTar(context.Background(), root, "HEAD", &snapshot); err != nil {
		t.Fatal(err)
	}
	cmdRoot := filepath.Join(tmp, "repo")
	if err := os.MkdirAll(cmdRoot, 0755); err != nil {
		t.Fatal(err)
	}
	tr := tar.NewReader(&snapshot)
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

func TestGenerateFailsLoudWithoutACatalog(t *testing.T) {
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

	err := generate(root, out)
	if err == nil {
		t.Fatal("generate must fail loudly without a catalog, even if artifacts pre-exist on disk")
	}
	if !strings.Contains(err.Error(), "refusing to render an undated scorecard") {
		t.Fatalf("unexpected error message: %v", err)
	}
	for _, art := range generatedArtifacts {
		if b, _ := os.ReadFile(filepath.Join(out, art.Name)); string(b) != "stale content\n" {
			t.Fatalf("%s was rewritten by a failed generation: %q", art.Name, b)
		}
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

// TestRelevantPathCoversEveryGeneratorInput pins the staleness gap: a commit touching only
// a Go source the generator scans (or a doc it reads) must trigger the freshness gate,
// while test files, which the generator skips, must not.
func TestRelevantPathCoversEveryGeneratorInput(t *testing.T) {
	relevant := []string{
		"internal/hooks/gate_conceptfreshness.go",
		"cmd/fak/main.go",
		"docs/cli-reference.md",
		"docs/fak/glossary-extra.md",
		"README.md",
		"CLAIMS.md",
		"tools/concept_disambiguation_scorecard.py",
		"tools/concept_disambiguation_scorecard.data/rows.json",
		GeneratedIndex,
	}
	for _, p := range relevant {
		if !RelevantPath(p) {
			t.Errorf("RelevantPath(%q) = false; the generator reads this input, so a change must trigger the gate", p)
		}
	}
	irrelevant := []string{
		"internal/hooks/gate_conceptfreshness_test.go",
		"platform/dispatch/runner.go",
		"docs/images/logo.png",
		"scripts/build.sh",
	}
	for _, p := range irrelevant {
		if RelevantPath(p) {
			t.Errorf("RelevantPath(%q) = true; the generator does not read this path", p)
		}
	}
}

// Test-only oracle: git archive is the one producer that applies export-ignore, so tests
// compare committedtree snapshots against it to prove the export rules are in effect. It is
// never a snapshot path.
var gitArchiveCommand = func(treeish string) *exec.Cmd {
	return exec.Command("git", "archive", "--format=tar", treeish)
}

func extractGitTree(root, treeish, tmp string) error {
	return extractGitTreeUsing(root, treeish, tmp, extractGitTar)
}

func extractGitTreeUsing(root, treeish, tmp string, extract func(io.Reader, string) error) error {
	cmd := gitArchiveCommand(treeish)
	cmd.Dir = root
	windowgate.ConfigureBackgroundCommand(cmd)
	var stderr gitArchiveStderr
	cmd.Stderr = &stderr
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("git archive %s: %w", treeish, err)
	}
	if err := cmd.Start(); err != nil {
		_ = stdout.Close()
		return fmt.Errorf("git archive %s: %w", treeish, err)
	}

	extractErr := extract(stdout, tmp)
	var drainErr error
	if extractErr == nil {
		// tar.Reader stops at its end marker; consume the remaining padding so
		// the producer can finish even when its output exceeds the pipe buffer.
		_, drainErr = io.Copy(io.Discard, stdout)
	}
	_ = stdout.Close()
	if extractErr != nil || drainErr != nil {
		// Extraction may fail before git has finished writing. Reap it without
		// waiting for the rest of an archive that can no longer be consumed.
		_ = cmd.Process.Kill()
	}
	waitErr := cmd.Wait()
	if waitErr != nil {
		if exitErr, ok := waitErr.(*exec.ExitError); ok {
			exitErr.Stderr = stderr.Bytes()
		}
		waitErr = fmt.Errorf("git archive %s: %w", treeish, waitErr)
	}
	// Keep a consumer failure first, including when stopping git induced a
	// second error, without hiding a genuine producer failure on a partial tar.
	return errors.Join(extractErr, drainErr, waitErr)
}

// gitArchiveStderr retains the same bounded prefix/suffix diagnostics as
// exec.Cmd.Output without retaining the archive itself.
type gitArchiveStderr struct {
	prefix, suffix []byte
	total          int64
}

func (w *gitArchiveStderr) Write(p []byte) (int, error) {
	const half = 32 << 10
	n := len(p)
	w.total += int64(n)
	if remaining := half - len(w.prefix); remaining > 0 {
		keep := min(remaining, len(p))
		w.prefix = append(w.prefix, p[:keep]...)
		p = p[keep:]
	}
	if len(p) >= half {
		w.suffix = append(w.suffix[:0], p[len(p)-half:]...)
	} else {
		if overflow := len(w.suffix) + len(p) - half; overflow > 0 {
			w.suffix = w.suffix[:copy(w.suffix, w.suffix[overflow:])]
		}
		w.suffix = append(w.suffix, p...)
	}
	return n, nil
}

func (w *gitArchiveStderr) Bytes() []byte {
	var out bytes.Buffer
	out.Write(w.prefix)
	if omitted := w.total - int64(len(w.prefix)+len(w.suffix)); omitted > 0 {
		fmt.Fprintf(&out, "\n... omitting %d bytes ...\n", omitted)
	}
	out.Write(w.suffix)
	return out.Bytes()
}

// fak-test:runtime fast est=50ms lane=default
func TestExtractTarStreamUsingCarriesFailuresWithoutDeadlock(t *testing.T) {
	sentinel := errors.New("producer sentinel")
	writeUntilRejected := func(w io.Writer) error {
		chunk := make([]byte, 64<<10)
		for {
			if _, err := w.Write(chunk); err != nil {
				return err
			}
		}
	}
	for _, tc := range []struct {
		name    string
		produce func(io.Writer) error
		extract func(io.Reader, string) error
		check   func(t *testing.T, dst string, err error)
	}{
		{
			name: "stream",
			produce: func(w io.Writer) error {
				return writeTarFixture(w, "first.txt", "first body\n")
			},
			extract: extractGitTar,
			check: func(t *testing.T, dst string, err error) {
				if err != nil {
					t.Fatal(err)
				}
				if body, err := os.ReadFile(filepath.Join(dst, "first.txt")); err != nil || string(body) != "first body\n" {
					t.Fatalf("extracted file = %q, %v", body, err)
				}
			},
		},
		{
			name: "producer-error",
			produce: func(w io.Writer) error {
				if err := writeTarFixture(w, "first.txt", "first body\n"); err != nil {
					return err
				}
				return sentinel
			},
			extract: extractGitTar,
			check: func(t *testing.T, dst string, err error) {
				if !errors.Is(err, sentinel) || !strings.Contains(err.Error(), "committed tree fixture-tree") {
					t.Fatalf("producer failure = %v, want wrapped sentinel", err)
				}
			},
		},
		{
			name: "consumer-error-stops-blocked-producer",
			produce: func(w io.Writer) error {
				tw := tar.NewWriter(w)
				if err := tw.WriteHeader(&tar.Header{Name: "../escaped.txt", Mode: 0644, Size: 4 << 20}); err != nil {
					return err
				}
				return writeUntilRejected(w)
			},
			extract: extractGitTar,
			check: func(t *testing.T, dst string, err error) {
				if err == nil || !strings.HasPrefix(err.Error(), `unsafe archive path "../escaped.txt"`) {
					t.Fatalf("consumer failure was lost: %v", err)
				}
			},
		},
		{
			name:    "early-consumer-drains-padding",
			produce: func(w io.Writer) error { return writeTarFixture(w, "first.txt", "first body\n") },
			extract: func(io.Reader, string) error { return nil },
			check: func(t *testing.T, _ string, err error) {
				if err != nil {
					t.Fatalf("drained stream reported %v", err)
				}
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dst := t.TempDir()
			done := make(chan error, 1)
			go func() {
				done <- extractTarStreamUsing("fixture-tree", dst, tc.extract, tc.produce)
			}()
			select {
			case err := <-done:
				tc.check(t, dst, err)
			case <-time.After(10 * time.Second):
				t.Fatal("tar stream pipeline deadlocked")
			}
		})
	}
}

func writeTarFixture(w io.Writer, name, body string) error {
	tw := tar.NewWriter(w)
	if err := tw.WriteHeader(&tar.Header{Name: name, Mode: 0644, Size: int64(len(body))}); err != nil {
		return err
	}
	if _, err := io.WriteString(tw, body); err != nil {
		return err
	}
	return tw.Close()
}

// fak-test:runtime fast est=500ms lane=default
func TestCommittedTreeSnapshotKeepsExportIgnoredPaths(t *testing.T) {
	t.Setenv(conceptTreeProfileEnv, "")
	root := t.TempDir()
	files := map[string]string{
		".gitattributes":                            "docs/** export-ignore\ntools/** export-ignore\n",
		"docs/fak/concept-glossary.md":              "# glossary\n",
		"tools/concept_disambiguation_scorecard.py": "print('scorecard')\n",
		"internal/keep/keep.go":                     "package keep\n",
	}
	for rel, body := range files {
		path := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(body), 0644); err != nil {
			t.Fatal(err)
		}
	}
	gitInitFixture(t, root)

	tree, cleanup, err := materializeGitTree(root, "HEAD")
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()
	for _, rel := range []string{"docs/fak/concept-glossary.md", "tools/concept_disambiguation_scorecard.py", "internal/keep/keep.go"} {
		if _, err := os.Stat(filepath.Join(tree.root, filepath.FromSlash(rel))); err != nil {
			t.Fatalf("snapshot dropped %s: %v", rel, err)
		}
	}

	// Without this, the assertions above could pass on a fixture that export-ignore never touches.
	oracle := t.TempDir()
	if err := extractGitTreeUsing(root, "HEAD", oracle, extractGitTar); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(oracle, "docs", "fak", "concept-glossary.md")); !os.IsNotExist(err) {
		t.Fatalf("git archive kept an export-ignored path (%v); the fixture does not exercise export-ignore", err)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestExpandExportSubstMatchesGitPlaceholders(t *testing.T) {
	id := commitIdentity{
		hash: "aaaa1111bbbb2222cccc3333dddd4444eeee5555", abbrevHash: "aaaa111",
		tree: "tttt1111", abbrevTree: "tttt111",
		parents: "pppp1111 pppp2222", abbrevParents: "pppp111 pppp222",
	}
	cases := []struct {
		name, in, want string
	}{
		{"no placeholder", "plain\n", "plain\n"},
		{"full hash", "$Format:%H$\n", id.hash + "\n"},
		{"abbrev and tree", "$Format:%h %T %t$", "aaaa111 tttt1111 tttt111"},
		{"parents", "$Format:%P|%p$", "pppp1111 pppp2222|pppp111 pppp222"},
		{"literal and percent escape", "v=$Format:100%% %H$!", "v=100% " + id.hash + "!"},
		{"newline code", "$Format:a%nb$", "a\nb"},
		{"empty expansion keeps the rest", "$Format:$x", "x"},
		{"unterminated span stays verbatim", "$Format:%H", "$Format:%H"},
		{"two spans", "$Format:%h$-$Format:%h$", "aaaa111-aaaa111"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := expandExportSubst([]byte(tc.in), id)
			if err != nil || string(got) != tc.want {
				t.Fatalf("expand %q = %q, %v; want %q", tc.in, got, err, tc.want)
			}
		})
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestExpandExportSubstRejectsUnsupportedPlaceholder(t *testing.T) {
	for _, in := range []string{"$Format:%d$", "$Format:%ad$", "$Format:trailing %$"} {
		if got, err := expandExportSubst([]byte(in), commitIdentity{}); err == nil {
			t.Fatalf("expand %q passed through as %q; want a loud error", in, got)
		}
	}
}
