package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// commit_buildcheck_test.go — proves the COMMITTED_RED commit-boundary gate (#4152) reads the
// PROSPECTIVE COMMITTED TREE (HEAD + exactly the commit's paths), refuses only a red THIS commit
// introduces, and fails open on a pre-existing HEAD red. The integration cases run against a real
// hermetic temp git repo built with plumbing commits (update-index → write-tree → commit-tree →
// update-ref: no hooks, no signing, no porcelain surprises).

func TestCommitBuildCheckPackages(t *testing.T) {
	cases := []struct {
		name string
		in   []string
		want []string
	}{
		{"single go file adds its dir plus cmd/fak", []string{"internal/zoo/a.go"}, []string{"./cmd/fak", "./internal/zoo"}},
		{"cmd/fak path dedups against the always-included target", []string{"cmd/fak/commit.go"}, []string{"./cmd/fak"}},
		{"dedup and sort across dirs", []string{"internal/zoo/a.go", "internal/aaa/b.go", "internal/zoo/c.go"}, []string{"./cmd/fak", "./internal/aaa", "./internal/zoo"}},
		{"root-level go file maps to dot", []string{"main.go"}, []string{".", "./cmd/fak"}},
		{"non-go commit returns nil", []string{"README.md", "docs/a.txt"}, nil},
		{"mixed commit keeps only go dirs", []string{"README.md", "internal/x/y.go"}, []string{"./cmd/fak", "./internal/x"}},
		{"empty returns nil", nil, nil},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := commitBuildCheckPackages(tc.in); !reflect.DeepEqual(got, tc.want) {
				t.Fatalf("commitBuildCheckPackages(%v) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

func TestExtractUndefinedSymbol(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want string
	}{
		{"plain compiler line", "p/caller.go:4:9: undefined: missingDef\n", "missingDef"},
		{"first of several", "# mod/p\np/a.go:1:1: undefined: First\np/b.go:2:2: undefined: Second\n", "First"},
		{"qualified symbol", "x.go:3:5: undefined: pkg.Symbol\n", "pkg.Symbol"},
		{"symbol at end of output without newline", "x.go:1:1: undefined: Tail", "Tail"},
		{"no undefined occurrence", "x.go:1:1: syntax error: unexpected }", ""},
		{"empty output", "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := extractUndefinedSymbol(tc.in); got != tc.want {
				t.Fatalf("extractUndefinedSymbol(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestCommitBuildCheckGate_nonGoCommitSkipsGate(t *testing.T) {
	// No .go path → no packages → the gate admits without touching git at all (the root need not
	// even be a repo).
	var stderr bytes.Buffer
	ok, reason, detail := commitBuildCheckGate(&stderr, t.TempDir(), []string{"README.md", "docs/x.txt"})
	if !ok || reason != "" || detail != "" {
		t.Fatalf("non-Go commit must skip the gate; got ok=%v reason=%q detail=%q", ok, reason, detail)
	}
	if stderr.Len() != 0 {
		t.Fatalf("expected no warnings; got %q", stderr.String())
	}
}

// seedBuildCheckRepo initializes a hermetic temp git repo and returns it with a git runner that
// isolates global/system config (no hooks, no signing, no template interference) and fails the
// test on any git error. Skips when git or the go toolchain is unavailable.
func seedBuildCheckRepo(t *testing.T) (repo string, git func(args ...string) string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go toolchain not on PATH")
	}
	repo = t.TempDir()
	noCfg := filepath.Join(t.TempDir(), "empty-gitconfig")
	git = func(args ...string) string {
		t.Helper()
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(),
			"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
			"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t",
			"GIT_CONFIG_GLOBAL="+noCfg,
			"GIT_CONFIG_SYSTEM="+noCfg,
			"GIT_CONFIG_NOSYSTEM=1",
		)
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	return repo, git
}

func writeBuildCheckFile(t *testing.T, repo, rel, content string) {
	t.Helper()
	full := filepath.Join(repo, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// commitBuildCheckPlumbing commits already-written worktree files via plumbing only.
func commitBuildCheckPlumbing(t *testing.T, repo string, git func(args ...string) string, msg string, rels ...string) {
	t.Helper()
	git(append([]string{"update-index", "--add", "--"}, rels...)...)
	tree := git("write-tree")
	args := []string{"commit-tree", "-m", msg}
	if buildCheckHeadExists(repo) {
		args = append(args, "-p", "HEAD")
	}
	sha := git(append(args, tree)...)
	git("update-ref", "HEAD", sha)
}

func buildCheckHeadExists(repo string) bool {
	c := exec.Command("git", "-C", repo, "rev-parse", "--verify", "-q", "HEAD")
	return c.Run() == nil
}

const buildCheckGoMod = "module buildcheck.test\n\ngo 1.21\n"

func TestCommitBuildCheckGate_introducedRedRefused(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Base() int { return 1 }\n")
	commitBuildCheckPlumbing(t, repo, git, "seed green", "go.mod", "p/p.go")

	// The caller references a symbol whose definition is NOT among the commit's paths: the
	// prospective committed tree is red although the author's working tree may look fine.
	writeBuildCheckFile(t, repo, "p/caller.go", "package p\n\nfunc Caller() int {\n\treturn missingDef()\n}\n")

	var stderr bytes.Buffer
	ok, reason, detail := commitBuildCheckGate(&stderr, repo, []string{"p/caller.go"})
	if ok || reason != "COMMITTED_RED" {
		t.Fatalf("expected COMMITTED_RED refusal; got ok=%v reason=%q detail=%q stderr=%s", ok, reason, detail, stderr.String())
	}
	if !strings.HasPrefix(detail, "undefined: missingDef") {
		t.Fatalf("detail should headline the undefined symbol; got %q", detail)
	}

	// Including the definition among the commit's paths heals the prospective tree: same working
	// tree, one more path, green verdict.
	writeBuildCheckFile(t, repo, "p/def.go", "package p\n\nfunc missingDef() int { return 2 }\n")
	stderr.Reset()
	ok, reason, detail = commitBuildCheckGate(&stderr, repo, []string{"p/caller.go", "p/def.go"})
	if !ok {
		t.Fatalf("definition included: expected admit; got reason=%q detail=%q stderr=%s", reason, detail, stderr.String())
	}

	// Committing only an unchanged path reproduces HEAD's tree exactly — the untracked red
	// caller.go still on disk must be MASKED, and the gate short-circuits green without building.
	stderr.Reset()
	ok, reason, detail = commitBuildCheckGate(&stderr, repo, []string{"p/p.go"})
	if !ok {
		t.Fatalf("no-effective-change commit: expected admit; got reason=%q detail=%q stderr=%s", reason, detail, stderr.String())
	}
}

func TestCommitBuildCheckGate_preexistingHeadRedFailsOpen(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedBuildCheckRepo(t)
	// HEAD itself is red: the committed p.go references a symbol that never existed.
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Broken() int {\n\treturn neverDefined()\n}\n")
	commitBuildCheckPlumbing(t, repo, git, "seed red", "go.mod", "p/p.go")

	// An innocent addition to the already-red package: the red is PRE-EXISTING at HEAD, not
	// introduced by this commit, so the gate must warn and admit (fail open), never block.
	writeBuildCheckFile(t, repo, "p/extra.go", "package p\n\nfunc Extra() int { return 3 }\n")

	var stderr bytes.Buffer
	ok, reason, detail := commitBuildCheckGate(&stderr, repo, []string{"p/extra.go"})
	if !ok {
		t.Fatalf("pre-existing HEAD red must fail open; got reason=%q detail=%q stderr=%s", reason, detail, stderr.String())
	}
	if !strings.Contains(stderr.String(), "ALREADY red") {
		t.Fatalf("expected the pre-existing-red advisory on stderr; got %q", stderr.String())
	}
}
