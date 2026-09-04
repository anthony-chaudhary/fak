package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safecommit"
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
		{"objective-c-only file triggers its package", []string{"internal/native/bridge.m"}, []string{"./cmd/fak", "./internal/native"}},
		{"header-only file triggers its package", []string{"internal/native/bridge.h"}, []string{"./cmd/fak", "./internal/native"}},
		{"assembly-only file triggers its package", []string{"internal/native/bridge.s"}, []string{"./cmd/fak", "./internal/native"}},
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
	outcome, detail := commitBuildCheckGate(&stderr, t.TempDir(), []string{"README.md", "docs/x.txt"})
	if outcome != safecommit.BuildCheckNotApplicable || detail != "" {
		t.Fatalf("non-Go commit must skip the gate; got outcome=%q detail=%q", outcome, detail)
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
	outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/caller.go"})
	if _, admit, reason := safecommit.DecideBuildCheck(outcome, detail, false); admit || reason != safecommit.ReasonCommittedRed {
		t.Fatalf("expected COMMITTED_RED refusal; got outcome=%q admit=%v reason=%q detail=%q stderr=%s", outcome, admit, reason, detail, stderr.String())
	}
	if !strings.HasPrefix(detail, "undefined: missingDef") {
		t.Fatalf("detail should headline the undefined symbol; got %q", detail)
	}

	// Including the definition among the commit's paths heals the prospective tree: same working
	// tree, one more path, green verdict.
	writeBuildCheckFile(t, repo, "p/def.go", "package p\n\nfunc missingDef() int { return 2 }\n")
	stderr.Reset()
	outcome, detail = commitBuildCheckGate(&stderr, repo, []string{"p/caller.go", "p/def.go"})
	if outcome != safecommit.BuildCheckPassed {
		t.Fatalf("definition included: expected a PASSED gate; got outcome=%q detail=%q stderr=%s", outcome, detail, stderr.String())
	}

	// Committing only an unchanged path reproduces HEAD's tree exactly — the untracked red
	// caller.go still on disk must be MASKED, and the gate short-circuits green without building.
	stderr.Reset()
	outcome, detail = commitBuildCheckGate(&stderr, repo, []string{"p/p.go"})
	if outcome != safecommit.BuildCheckNotApplicable {
		t.Fatalf("no-effective-change commit: expected a not-applicable gate; got outcome=%q detail=%q stderr=%s", outcome, detail, stderr.String())
	}
}

func buildCheckGitState(t *testing.T, git func(args ...string) string) (head, index string) {
	t.Helper()
	return git("rev-parse", "HEAD"), git("write-tree")
}

func requireBuildCheckGitState(t *testing.T, git func(args ...string) string, head, index string) {
	t.Helper()
	if gotHead, gotIndex := buildCheckGitState(t, git); gotHead != head || gotIndex != index {
		t.Fatalf("prospective refusal mutated git state: HEAD %s -> %s, index %s -> %s", head, gotHead, index, gotIndex)
	}
}

func requireProspectiveRefusal(t *testing.T, outcome safecommit.BuildCheckOutcome, detail string, wants ...string) {
	t.Helper()
	if _, admit, reason := safecommit.DecideBuildCheck(outcome, detail, false); admit || reason != safecommit.ReasonCommittedRed {
		t.Fatalf("outcome=%q admit=%v reason=%q detail=%q; want typed COMMITTED_RED refusal", outcome, admit, reason, detail)
	}
	for _, want := range append(wants, "next: fak validate --ref HEAD") {
		if !strings.Contains(detail, want) {
			t.Fatalf("detail=%q; want actionable evidence %q", detail, want)
		}
	}
}

func TestCommitBuildCheckGateNativeOnlyMissingSymbolRefusesBeforeGitMutation(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	if out, err := exec.Command("go", "env", "CGO_ENABLED").Output(); err != nil || strings.TrimSpace(string(out)) != "1" {
		t.Skip("native-only link witness requires cgo")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", `package p

/*
int native_value(void);
*/
import "C"

func Value() int { return int(C.native_value()) }
`)
	writeBuildCheckFile(t, repo, "p/native.c", "int native_value(void) { return 1; }\n")
	writeBuildCheckFile(t, repo, "cmd/fak/main.go", `package main

import "buildcheck.test/p"

func main() { _ = p.Value() }
`)
	commitBuildCheckPlumbing(t, repo, git, "seed green native package", "go.mod", "p/p.go", "p/native.c", "cmd/fak/main.go")
	t.Setenv("GOCACHE", t.TempDir())
	head, index := buildCheckGitState(t, git)

	writeBuildCheckFile(t, repo, "p/native.c", "extern int missing_native(void);\nint native_value(void) { return missing_native(); }\n")
	var stderr bytes.Buffer
	outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/native.c"})
	requireProspectiveRefusal(t, outcome, detail, "missing_native")
	requireBuildCheckGitState(t, git, head, index)

	writeBuildCheckFile(t, repo, "p/native.c", "int native_value(void) { return 2; }\n")
	if outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/native.c"}); outcome != safecommit.BuildCheckPassed {
		t.Fatalf("restored native fixture outcome=%q detail=%q stderr=%s", outcome, detail, stderr.String())
	}
}

func TestCommitBuildCheckGateFailingChangedPackageTestRefusesBeforeGitMutation(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
	writeBuildCheckFile(t, repo, "p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 { t.Fatal(\"bad\") }\n}\n")
	commitBuildCheckPlumbing(t, repo, git, "seed passing test", "go.mod", "p/p.go", "p/p_test.go")
	t.Setenv("GOCACHE", t.TempDir())
	head, index := buildCheckGitState(t, git)

	writeBuildCheckFile(t, repo, "p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tt.Fatal(\"deliberate prospective failure\")\n}\n")
	var stderr bytes.Buffer
	outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/p_test.go"})
	requireProspectiveRefusal(t, outcome, detail, "test", "deliberate prospective failure")
	requireBuildCheckGitState(t, git, head, index)

	writeBuildCheckFile(t, repo, "p/p_test.go", "package p\n\nimport \"testing\"\n\nfunc TestValue(t *testing.T) {\n\tif Value() != 1 { t.Fatal(\"bad\") }\n}\n")
	if outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/p_test.go"}); outcome != safecommit.BuildCheckNotApplicable {
		t.Fatalf("restored identical fixture outcome=%q detail=%q", outcome, detail)
	}
}

func TestCommitBuildCheckGateUnformattedOwnedPathRefusesBeforeGitMutation(t *testing.T) {
	if testing.Short() {
		t.Skip("integration test; skipped under -short")
	}
	repo, git := seedBuildCheckRepo(t)
	writeBuildCheckFile(t, repo, "go.mod", buildCheckGoMod)
	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 1 }\n")
	commitBuildCheckPlumbing(t, repo, git, "seed formatted package", "go.mod", "p/p.go")
	t.Setenv("GOCACHE", t.TempDir())
	head, index := buildCheckGitState(t, git)

	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value( ) int { return 2 }\n")
	var stderr bytes.Buffer
	outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/p.go"})
	requireProspectiveRefusal(t, outcome, detail, "gofmt", "p/p.go")
	requireBuildCheckGitState(t, git, head, index)

	writeBuildCheckFile(t, repo, "p/p.go", "package p\n\nfunc Value() int { return 2 }\n")
	if outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/p.go"}); outcome != safecommit.BuildCheckPassed {
		t.Fatalf("formatted fixture outcome=%q detail=%q stderr=%s", outcome, detail, stderr.String())
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
	outcome, detail := commitBuildCheckGate(&stderr, repo, []string{"p/extra.go"})
	if _, admit, reason := safecommit.DecideBuildCheck(outcome, detail, false); !admit {
		t.Fatalf("pre-existing HEAD red must fail open; got outcome=%q reason=%q detail=%q stderr=%s", outcome, reason, detail, stderr.String())
	}
	if outcome != safecommit.BuildCheckHeadRed {
		t.Fatalf("pre-existing HEAD red must be reported as %q, not hidden behind a bare admit; got %q", safecommit.BuildCheckHeadRed, outcome)
	}
	// The advisory must not stay a silent shrug: it names the shared break (the undefined symbol)
	// and frames it as needing a fix at its source, aligned with the pre-push gate's TRUNK_ALREADY_RED
	// render. Keeps the "ALREADY red" phrase the pre-existing case is recognized by.
	for _, want := range []string{"ALREADY red", "neverDefined", "fix at its source"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("advisory missing %q; got %q", want, stderr.String())
		}
	}
	// The fleet witness this advisory emits must land in the repo being GATED — the
	// throwaway one seeded above — never in the developer's real checkout. This test
	// once filed its synthetic `buildcheck.test/p undefined: neverDefined` break into
	// the real ledger on every run, where the temp repo's base sha resolves against
	// nothing and the row can never fold out: `fak trunk-red` grew to 28 fleet-wide
	// "shared breaks" that were all THIS fixture, burying the one real break.
	if _, err := os.Stat(filepath.Join(repo, ".fak", "trunk-red.jsonl")); err != nil {
		t.Fatalf("witness should have landed in the gated repo's ledger: %v", err)
	}
	if outside := trunkRedLedgerDefault(); outside != "" && !strings.HasPrefix(outside, repo) {
		before := trunkRedLedgerLineCount(outside)
		stderr.Reset()
		if again, _ := commitBuildCheckGate(&stderr, repo, []string{"p/extra.go"}); again != safecommit.BuildCheckHeadRed {
			t.Fatalf("second pre-existing-red pass must still fail open; got %q", again)
		}
		if after := trunkRedLedgerLineCount(outside); after != before {
			t.Fatalf("gating %s polluted this repo's ledger %s: %d -> %d row(s)", repo, outside, before, after)
		}
	}
}

// trunkRedLedgerLineCount counts rows in a trunk-red ledger; a missing ledger is 0.
func trunkRedLedgerLineCount(path string) int {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0
	}
	n := 0
	for _, line := range strings.Split(string(b), "\n") {
		if strings.TrimSpace(line) != "" {
			n++
		}
	}
	return n
}

// TestFormatPreexistingRedAdvisory pins the pure advisory formatter without git/go: it must keep the
// "ALREADY red" recognition phrase AND name the failing package + first undefined symbol + the
// fix-at-source framing (the "work together" upgrade over the old silent shrug).
func TestFormatPreexistingRedAdvisory(t *testing.T) {
	out := formatPreexistingRedAdvisory("# buildcheck.test/p\np/p.go:3:9: undefined: neverDefined\n")
	for _, want := range []string{"ALREADY red", "buildcheck.test/p", "undefined: neverDefined", "fix at its source"} {
		if !strings.Contains(out, want) {
			t.Fatalf("advisory missing %q:\n%s", want, out)
		}
	}
	// A build output with no parseable package header / symbol still yields the base advisory —
	// never empty, never a panic.
	if base := formatPreexistingRedAdvisory(""); !strings.Contains(base, "ALREADY red") {
		t.Fatalf("empty build output should still carry the base advisory; got %q", base)
	}
}

func TestDeriveValidateEvidence(t *testing.T) {
	cases := []struct {
		name        string
		res         validateResult
		wantCompile safecommit.EvidenceOutcome
		wantTest    safecommit.EvidenceOutcome
	}{
		{
			name: "build vet and tests passed",
			res: validateResult{
				OK:     true,
				Tested: []string{"github.com/anthony-chaudhary/fak/internal/foo"},
				Phases: []validatePhase{
					{Name: "build", Status: "ok"},
					{Name: "vet", Status: "ok"},
					{Name: "test", Status: "ok"},
				},
			},
			wantCompile: safecommit.EvidencePassed,
			wantTest:    safecommit.EvidencePassed,
		},
		{
			name: "tests not required when no tested packages",
			res: validateResult{
				OK:     true,
				Tested: []string{},
				Phases: []validatePhase{
					{Name: "build", Status: "ok"},
					{Name: "vet", Status: "ok"},
				},
			},
			wantCompile: safecommit.EvidencePassed,
			wantTest:    safecommit.EvidenceNotRequired,
		},
		{
			name: "build failed",
			res: validateResult{
				OK: false,
				Phases: []validatePhase{
					{Name: "build", Status: "failed"},
				},
			},
			wantCompile: safecommit.EvidenceFailed,
			wantTest:    safecommit.EvidenceUnrun,
		},
		{
			name: "tests failed",
			res: validateResult{
				OK:     false,
				Tested: []string{"pkg/x"},
				Phases: []validatePhase{
					{Name: "build", Status: "ok"},
					{Name: "vet", Status: "ok"},
					{Name: "test", Status: "failed"},
				},
			},
			wantCompile: safecommit.EvidencePassed,
			wantTest:    safecommit.EvidenceFailed,
		},
		{
			name: "tests timed out",
			res: validateResult{
				OK:       false,
				TimedOut: true,
				Tested:   []string{"pkg/x"},
				Phases: []validatePhase{
					{Name: "build", Status: "ok"},
					{Name: "vet", Status: "ok"},
					{Name: "test", Status: "timeout"},
				},
			},
			wantCompile: safecommit.EvidencePassed,
			wantTest:    safecommit.EvidenceSkipped,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			gotCompile, gotTest := deriveValidateEvidence(tc.res)
			if gotCompile != tc.wantCompile || gotTest != tc.wantTest {
				t.Fatalf("deriveValidateEvidence() = (%s, %s), want (%s, %s)", gotCompile, gotTest, tc.wantCompile, tc.wantTest)
			}
		})
	}
}
