package witness

import (
	"context"
	"os"
	"os/exec"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// fakeGit is an injected Runner mapping a git invocation to a scripted (stdout,
// code, err). Keyed by the first two args so "merge-base --is-ancestor" and
// "cat-file -e" are distinguishable.
type fakeGit struct {
	out  string
	code int
	err  error
	seen [][]string
}

func (f *fakeGit) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	f.seen = append(f.seen, args)
	return f.out, f.code, f.err
}

func TestAncestorClaim(t *testing.T) {
	ctx := context.Background()
	// confirmed: merge-base --is-ancestor exits 0.
	if got := NewWithRunner((&fakeGit{code: 0}).run, "").Resolve(ctx, nil, "ancestor:abc123"); got != abi.WitnessConfirmed {
		t.Fatalf("ancestor present => Confirmed, got %v", got)
	}
	// refuted: exits 1 (ref is NOT an ancestor — the claimed phase did not ship).
	if got := NewWithRunner((&fakeGit{code: 1}).run, "").Resolve(ctx, nil, "ancestor:abc123"); got != abi.WitnessRefuted {
		t.Fatalf("ancestor absent => Refuted, got %v", got)
	}
	// abstain: a bad/unknown ref (git exit 128) is not evidence of absence.
	if got := NewWithRunner((&fakeGit{code: 128}).run, "").Resolve(ctx, nil, "ancestor:bogus"); got != abi.WitnessAbstain {
		t.Fatalf("bad ref => Abstain, got %v", got)
	}
}

func TestGitMissingAbstains(t *testing.T) {
	ctx := context.Background()
	r := NewWithRunner((&fakeGit{err: exec.ErrNotFound}).run, "")
	if got := r.Resolve(ctx, nil, "ancestor:abc"); got != abi.WitnessAbstain {
		t.Fatalf("git unavailable must Abstain (fail-to-abstain), got %v", got)
	}
}

func TestUnparseableClaimAbstains(t *testing.T) {
	ctx := context.Background()
	r := NewWithRunner((&fakeGit{code: 0}).run, "")
	for _, claim := range []string{"", "no-colon", "unknownkind:x", ":empty", "trailing:"} {
		if got := r.Resolve(ctx, nil, claim); got != abi.WitnessAbstain {
			t.Errorf("claim %q must Abstain, got %v", claim, got)
		}
	}
}

func TestCommittedAndGrep(t *testing.T) {
	ctx := context.Background()
	// committed: ls-files --error-unmatch exits 0 => tracked.
	if got := NewWithRunner((&fakeGit{code: 0}).run, "").Resolve(ctx, nil, "committed:go.mod"); got != abi.WitnessConfirmed {
		t.Fatalf("tracked path => Confirmed")
	}
	if got := NewWithRunner((&fakeGit{code: 1}).run, "").Resolve(ctx, nil, "committed:nope"); got != abi.WitnessRefuted {
		t.Fatalf("untracked path => Refuted")
	}
	// grep: a non-empty %H means a matching commit exists.
	if got := NewWithRunner((&fakeGit{out: "deadbeef\n", code: 0}).run, "").Resolve(ctx, nil, "grep:fix"); got != abi.WitnessConfirmed {
		t.Fatalf("grep match => Confirmed")
	}
	if got := NewWithRunner((&fakeGit{out: "", code: 0}).run, "").Resolve(ctx, nil, "grep:nope"); got != abi.WitnessRefuted {
		t.Fatalf("grep no-match => Refuted")
	}
}

func TestNotestsClaim(t *testing.T) {
	ctx := context.Background()
	// A commit that touched only non-test source => CONFIRMED (clean ship).
	clean := "internal/witness/witness.go\ninternal/abi/abi.go\n"
	if got := NewWithRunner((&fakeGit{out: clean, code: 0}).run, "").Resolve(ctx, nil, "notests:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("commit touching no test => Confirmed, got %v", got)
	}
	// A commit that edited a gating test => REFUTED (reward-hack flagged).
	dirty := "internal/witness/witness.go\ninternal/witness/witness_test.go\n"
	if got := NewWithRunner((&fakeGit{out: dirty, code: 0}).run, "").Resolve(ctx, nil, "notests:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("commit editing its own gating test => Refuted, got %v", got)
	}
	// Windows-style separators in the file list are matched too.
	winDirty := "internal\\witness\\witness_test.go\n"
	if got := NewWithRunner((&fakeGit{out: winDirty, code: 0}).run, "").Resolve(ctx, nil, "notests:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("backslash-separated test path => Refuted, got %v", got)
	}
	// A bad/unknown ref (git exit non-zero) abstains — never a false Confirm that
	// would let a test-rewriting commit through unflagged.
	if got := NewWithRunner((&fakeGit{code: 128}).run, "").Resolve(ctx, nil, "notests:bogus"); got != abi.WitnessAbstain {
		t.Fatalf("bad ref => Abstain, got %v", got)
	}
}

func TestIsGatingTestPath(t *testing.T) {
	cases := map[string]bool{
		"internal/witness/witness_test.go": true,
		"witness_test.go":                  true,
		"a\\b\\foo_test.go":                true,
		"internal/witness/witness.go":      false,
		"testdata/golden.json":             false, // a fixture is not a gating test
		"docs/test-plan.md":                false,
		"":                                 false,
	}
	for p, want := range cases {
		if got := isGatingTestPath(p); got != want {
			t.Errorf("isGatingTestPath(%q)=%v, want %v", p, got, want)
		}
	}
}

// TestRealGitAncestor exercises the DEFAULT real-git runner against this actual
// repository: HEAD is trivially its own ancestor (Confirmed), and an all-zero sha
// is not a valid commit (Abstain), proving the resolver works end-to-end with the
// real git binary — not just the fake.
func TestRealGitAncestor(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	ctx := context.Background()
	r := New()
	if got := r.Resolve(ctx, nil, "ancestor:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("HEAD must be its own ancestor (Confirmed), got %v", got)
	}
	// committed: paths are REPO-ROOT-relative (git's ":/" anchor), so this is
	// cwd-independent. go.mod is a tracked file at the repo root; .git lives there too.
	if got := r.Resolve(ctx, nil, "committed:go.mod"); got != abi.WitnessConfirmed {
		t.Fatalf("go.mod is tracked => Confirmed, got %v", got)
	}
	if got := r.Resolve(ctx, nil, "committed:this-path-does-not-exist.xyz"); got != abi.WitnessRefuted {
		t.Fatalf("an untracked path => Refuted, got %v", got)
	}
	// a 40-zero sha is well-formed but not a real commit => not an ancestor.
	if got := r.Resolve(ctx, nil, "ancestor:0000000000000000000000000000000000000000"); got == abi.WitnessConfirmed {
		t.Fatalf("the null sha must not Confirm")
	}
}

// TestRealGitNotests proves the reward-hack guard against the real git binary on
// this repo: a synthesized commit that edits a _test.go file is REFUTED, and one
// that edits only a non-test file is CONFIRMED — built in a throwaway temp repo so
// the test owns its evidence and does not depend on this repo's commit history.
func TestRealGitNotests(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	ctx := context.Background()
	git := func(args ...string) {
		t.Helper()
		cmd := exec.CommandContext(ctx, "git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
	}
	write := func(name, body string) {
		t.Helper()
		if err := writeFile(dir, name, body); err != nil {
			t.Fatal(err)
		}
	}
	git("init", "-q")
	git("config", "user.email", "t@t")
	git("config", "user.name", "t")
	write("foo.go", "package foo\n")
	git("add", "foo.go")
	git("commit", "-q", "-m", "src only")
	r := New() // dir defaults to git discovery; drive via r.dir
	r.dir = dir
	if got := r.Resolve(ctx, nil, "notests:HEAD"); got != abi.WitnessConfirmed {
		t.Fatalf("src-only commit => Confirmed, got %v", got)
	}
	write("foo_test.go", "package foo\n")
	git("add", "foo_test.go")
	git("commit", "-q", "-m", "added a test")
	if got := r.Resolve(ctx, nil, "notests:HEAD"); got != abi.WitnessRefuted {
		t.Fatalf("commit editing a _test.go => Refuted, got %v", got)
	}
}

func writeFile(dir, name, body string) error {
	return os.WriteFile(dir+string(os.PathSeparator)+name, []byte(body), 0o644)
}
