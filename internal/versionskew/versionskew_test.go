package versionskew

import (
	"context"
	"os/exec"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/binstamp"
)

// stamped is a convenience for a clean, VCS-stamped running binary at rev.
func stamped(rev string) binstamp.Stamp {
	return binstamp.Stamp{Revision: rev, HasVCS: true}
}

// TestClassify_pure pins the pure kernel: every (stamp, tip, relation) input lands on exactly the
// intended token, and — the load-bearing property — the un-attestable and behind cases NEVER fall
// through to Unknown.
func TestClassify_pure(t *testing.T) {
	const tip = "1111111111111111111111111111111111111111"
	cases := []struct {
		name    string
		running binstamp.Stamp
		tip     string
		rel     Relation
		want    Verdict
	}{
		{"unstamped dominates even with a tip and a relation", binstamp.Stamp{}, tip, RelEqual, Unstamped},
		{"unstamped: HasVCS but empty rev", binstamp.Stamp{HasVCS: true}, tip, RelBehind, Unstamped},
		{"dirty is its own token, not Unknown", binstamp.Stamp{Revision: "abc1234", HasVCS: true, Dirty: true}, tip, RelBehind, Dirty},
		{"no trunk tip -> Unknown residual", stamped("abc1234"), "", RelUndetermined, Unknown},
		{"undetermined ancestry with a tip -> Unknown", stamped("abc1234"), tip, RelUndetermined, Unknown},
		{"equal -> Fresh", stamped(tip), tip, RelEqual, Fresh},
		{"strict ancestor -> Skewed", stamped("abc1234"), tip, RelBehind, Skewed},
		{"newer than tip -> Ahead", stamped("abc1234"), tip, RelAhead, Ahead},
		{"off-trunk -> Diverged", stamped("abc1234"), tip, RelDiverged, Diverged},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := Classify(c.running, c.tip, c.rel); got != c.want {
				t.Fatalf("Classify(%+v, tip=%q, rel=%d) = %v, want %v", c.running, c.tip, c.rel, got, c.want)
			}
		})
	}
}

// TestRefusable_closedSet locks the actionable half of the contract: the behind and un-attestable
// tokens are refusable (a gate can act on them); Fresh, Ahead, and the honest Unknown are not. If
// a future edit lets Unstamped or Skewed silently become non-refusable, this reds.
func TestRefusable_closedSet(t *testing.T) {
	refusable := map[Verdict]bool{Skewed: true, Diverged: true, Unstamped: true, Dirty: true}
	for _, v := range []Verdict{Unknown, Fresh, Skewed, Ahead, Diverged, Unstamped, Dirty} {
		if got, want := v.Refusable(), refusable[v]; got != want {
			t.Errorf("%v.Refusable() = %v, want %v", v, got, want)
		}
	}
	// The key regression guard: an unstamped binary must NEVER read like a benign Unknown.
	if Unstamped == Unknown || !Unstamped.Refusable() {
		t.Fatalf("UNSTAMPED must be a distinct, refusable token, not conflated with UNKNOWN")
	}
}

func TestVerdict_String(t *testing.T) {
	for v, want := range map[Verdict]string{
		Unknown: "UNKNOWN", Fresh: "FRESH", Skewed: "SKEWED", Ahead: "AHEAD",
		Diverged: "DIVERGED", Unstamped: "UNSTAMPED", Dirty: "DIRTY",
	} {
		if got := v.String(); got != want {
			t.Errorf("Verdict(%d).String() = %q, want %q", int(v), got, want)
		}
	}
}

// TestAssessStamp_gitAncestry drives the FULL impure path (resolveRev + ancestryOf over real git)
// against a deterministic two-commit temp repo, proving the DoD end to end:
//   - a stamped running commit that is an ANCESTOR of the trunk tip -> SKEWED;
//   - the tip itself -> FRESH; a commit newer than the ref -> AHEAD;
//   - an ABSENT/DIRTY/UNSTAMPED stamp -> its own token, never a silent Unknown.

func TestAssessStampExactFullSHAStopsBeforeAncestry(t *testing.T) {
	sha := "0123456789abcdef0123456789abcdef01234567"
	calls := make([]string, 0, 2)
	runner := func(_ context.Context, _ string, command string, args ...string) (string, bool) {
		calls = append(calls, command+" "+strings.Join(args, " "))
		if command == "git" && len(args) >= 2 && args[0] == "rev-parse" {
			return sha + "\n", true
		}
		t.Fatalf("unexpected ancestry subprocess: %s %s", command, strings.Join(args, " "))
		return "", false
	}
	got := AssessStamp(context.Background(), runner, ".", "origin/main", stamped(sha))
	if got.Verdict != Fresh || got.Relation != RelEqual {
		t.Fatalf("assessment=%+v, want fresh/equal", got)
	}
	if len(calls) != 1 || strings.Contains(calls[0], "merge-base") {
		t.Fatalf("exact SHA used ancestry subprocesses: %v", calls)
	}
}

func TestAssessStamp_gitAncestry(t *testing.T) {
	dir := t.TempDir()
	git := func(args ...string) string {
		t.Helper()
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		out, err := cmd.CombinedOutput()
		if err != nil {
			t.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
		}
		return strings.TrimSpace(string(out))
	}

	git("init", "-q")
	git("config", "user.email", "test@example.com")
	git("config", "user.name", "versionskew test")
	git("config", "commit.gpgsign", "false")

	git("commit", "--allow-empty", "-q", "-m", "first")
	older := git("rev-parse", "HEAD")
	git("commit", "--allow-empty", "-q", "-m", "second")
	tip := git("rev-parse", "HEAD")

	if older == tip {
		t.Fatalf("temp repo did not advance: older == tip == %s", older)
	}

	ctx := context.Background()
	assess := func(running binstamp.Stamp, ref string) Assessment {
		return AssessStamp(ctx, RealRunner, dir, ref, running)
	}

	// (a) ancestor of the tip -> SKEWED; the tip itself -> FRESH.
	if a := assess(stamped(older), tip); a.Verdict != Skewed {
		t.Errorf("older-vs-tip: got %v, want SKEWED (%+v)", a.Verdict, a)
	}
	if a := assess(stamped(tip), tip); a.Verdict != Fresh {
		t.Errorf("tip-vs-tip: got %v, want FRESH (%+v)", a.Verdict, a)
	}
	// Resolving the ref by NAME (not raw SHA) must classify identically — origin/main is a name.
	if a := assess(stamped(older), "HEAD"); a.Verdict != Skewed {
		t.Errorf("older-vs-HEAD(name): got %v, want SKEWED (%+v)", a.Verdict, a)
	}
	// Newer than the ref -> AHEAD (running=tip, ref=older).
	if a := assess(stamped(tip), older); a.Verdict != Ahead {
		t.Errorf("tip-vs-older: got %v, want AHEAD (%+v)", a.Verdict, a)
	}

	// (b) the un-attestable / undeterminable cases each get their OWN token, NOT Unknown-that-
	// reads-like-success.
	if a := assess(binstamp.Stamp{}, tip); a.Verdict != Unstamped {
		t.Errorf("absent stamp: got %v, want UNSTAMPED (%+v)", a.Verdict, a)
	}
	if a := assess(binstamp.Stamp{Revision: older, HasVCS: true, Dirty: true}, tip); a.Verdict != Dirty {
		t.Errorf("dirty stamp: got %v, want DIRTY (%+v)", a.Verdict, a)
	}
	// A commit that is not in this repo -> the honest, narrow Unknown (ancestry uncomputable),
	// distinct from the un-attestable tokens above.
	absent := "0123456789012345678901234567890123456789"
	if a := assess(stamped(absent), tip); a.Verdict != Unknown {
		t.Errorf("absent commit: got %v, want UNKNOWN (%+v)", a.Verdict, a)
	}
	// An unresolvable trunk ref -> Unknown too (no tip to compare against).
	if a := assess(stamped(older), "no-such-ref"); a.Verdict != Unknown {
		t.Errorf("bad ref: got %v, want UNKNOWN (%+v)", a.Verdict, a)
	}
}
