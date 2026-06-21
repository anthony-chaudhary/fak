package witness

import (
	"context"
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
