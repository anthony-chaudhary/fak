package conceptcatalog

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// gitInitFixture turns a scorecard fixture root into a git repository whose HEAD commit
// holds every fixture file. The commit is load-bearing, not ceremony: classify renders
// from HEAD, so a repository with only an index would have no tree to render from.
func gitInitFixture(t *testing.T, root string) {
	t.Helper()
	for _, args := range [][]string{
		{"init"},
		// Pin the transport details a shared checkout would otherwise vary: an autocrlf
		// rewrite would make the exported tree differ from the worktree for reasons that
		// have nothing to do with what this test measures.
		{"config", "core.autocrlf", "false"},
		{"config", "user.email", "fixture@example.com"},
		{"config", "user.name", "fixture"},
		// Do not inherit the developer's global hooks or signing key: this throwaway
		// repository must commit identically on every machine, and it is not the tree any
		// gate is guarding.
		{"config", "core.hooksPath", ".git/fixture-hooks-none"},
		{"config", "commit.gpgsign", "false"},
		{"add", "-A"},
		{"commit", "-q", "-m", "fixture"},
	} {
		cmd := exec.Command("git", args...)
		cmd.Dir = root
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, out)
		}
	}
}

func gitAddFixture(t *testing.T, root string, pathspec ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"add", "--"}, pathspec...)...)
	cmd.Dir = root
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git add %v: %v: %s", pathspec, err, out)
	}
}

func gitWriteTree(t *testing.T, root string) string {
	t.Helper()
	cmd := exec.Command("git", "write-tree")
	cmd.Dir = root
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git write-tree: %v: %s", err, out)
	}
	return strings.TrimSpace(string(out))
}

// TestClassifyRendersScorecardFromTheGitTreeNotThePeerDirtyWorktree is the witness for
// #6521. `fak concept classify` does not only write the corpus row - it regenerates the
// two COMMITTED scorecard docs, and it used to regenerate them by walking the repo root.
// On a permanently peer-dirty shared trunk that root carries every other live session's
// uncommitted symbols, so the numbers the classifying agent was told to stage were
// computed over a tree that exists in no commit and would move again the moment those
// peers landed or abandoned their WIP.
//
// The fixture makes that dirt concrete: CachePeerNoise exists only as an UNTRACKED file,
// exactly like a peer's work in progress. The assertion is the one the issue names - after
// classify, the worktree README must be byte-identical to a `fak concept generate --staged`
// render of the same classification, which is the tree CONCEPT_FRESHNESS actually scores.
func TestClassifyRendersScorecardFromTheGitTreeNotThePeerDirtyWorktree(t *testing.T) {
	c, root := scorecardE2EFixture(t, "CacheNoise")
	gitInitFixture(t, root)

	// A peer session's uncommitted symbol. Untracked, so it is in no git tree, but the
	// generator walks internal/ and cmd/ and would count it from the worktree.
	peer := filepath.Join(root, "internal", "peer")
	if err := os.MkdirAll(peer, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(peer, "peer.go"), []byte("package peer\nconst CachePeerNoise=1\n"), 0600); err != nil {
		t.Fatal(err)
	}

	// Guard against a vacuous pass: the peer file must actually move the render, or this
	// test would hold for a classify that still walked the worktree.
	worktreeRender := filepath.Join(t.TempDir(), "worktree")
	if err := generate(root, worktreeRender); err != nil {
		t.Fatal(err)
	}
	treeRender := filepath.Join(t.TempDir(), "tree")
	if _, err := RegenerateFromGitTree(root, gitWriteTree(t, root), treeRender); err != nil {
		t.Fatal(err)
	}
	dirty, err := os.ReadFile(filepath.Join(worktreeRender, "README.md"))
	if err != nil {
		t.Fatal(err)
	}
	clean, err := os.ReadFile(filepath.Join(treeRender, GeneratedReadme))
	if err != nil {
		t.Fatal(err)
	}
	if generatedBytesEqual(dirty, clean) {
		t.Fatal("fixture cannot witness the bug: the untracked peer symbol moves no scorecard number")
	}

	plan, err := PlanClassify(c, ClassifyRequest{Family: "cache", Token: "CacheNoise", Category: "incidental", Reason: "fixture-only incidental name"})
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Files) != 3 {
		t.Fatalf("classify must still plan the corpus row and both generated docs, got %v", plan.Files)
	}
	if err := Apply(plan); err != nil {
		t.Fatal(err)
	}

	// What the agent is instructed to stage: the corpus row plus the two docs. The docs
	// must equal what the tree-scoped cure renders from that same staged tree.
	gitAddFixture(t, root, DataRel)
	want := filepath.Join(t.TempDir(), "staged")
	if _, err := RegenerateFromGitTree(root, gitWriteTree(t, root), want); err != nil {
		t.Fatal(err)
	}
	for _, art := range generatedArtifacts {
		expected, readErr := os.ReadFile(filepath.Join(want, filepath.FromSlash(art.Tracked)))
		if readErr != nil {
			t.Fatal(readErr)
		}
		actual, readErr := os.ReadFile(filepath.Join(root, filepath.FromSlash(art.Tracked)))
		if readErr != nil {
			t.Fatal(readErr)
		}
		if !generatedBytesEqual(actual, expected) {
			t.Errorf("classify wrote %s from the peer-dirty worktree, not from the staged tree", art.Tracked)
		}
	}
}
