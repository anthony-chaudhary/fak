package conceptcatalog

import (
	"archive/tar"
	"bytes"
	"io"
	"os"
	"os/exec"
	"path/filepath"
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
