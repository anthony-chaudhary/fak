package relay

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// Rung C6 (issue #3878) witnesses. The pure fold (RematerializeWip / ResumeRematerialize) is
// tested hermetically over a fake WipRestorer; the wire round-trip proves wip_tree is a
// back-compatible optional cursor field; the real-git test proves GitWipRestorer applies a
// clean delta and, above all, defers a conflicting one WITHOUT mutating the tree (fail-closed).

// fakeRestorer is a hermetic WipRestorer: it returns a fixed verdict and counts calls so a
// test can assert the fold short-circuited (never touched the store) on a stale anchor.
type fakeRestorer struct {
	verdict WipApplyVerdict
	detail  string
	calls   *int
}

func (f fakeRestorer) Restore(string) WipApplyResult {
	if f.calls != nil {
		*f.calls++
	}
	return WipApplyResult{Verdict: f.verdict, Detail: f.detail}
}

// TestRematerializeWipFold pins the store-verdict -> resume-outcome fold: an empty WipTree is
// absent (nothing to do), a clean apply is rematerialized, and BOTH a conflict and an
// unavailable object fail closed to a WipDeferred carrying ReasonWipStale (never a clobber).
func TestRematerializeWipFold(t *testing.T) {
	cases := []struct {
		name        string
		wipTree     string
		verdict     WipApplyVerdict
		wantVerdict RematerializeVerdict
		wantReason  string
	}{
		{"empty wip_tree is absent", "", WipApplied /*unused*/, WipAbsent, ""},
		{"clean apply rematerializes", "deadbeef", WipApplied, WipRematerialized, ""},
		{"conflict defers stale", "deadbeef", WipConflict, WipDeferred, ReasonWipStale},
		{"unavailable defers stale", "deadbeef", WipUnavailable, WipDeferred, ReasonWipStale},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := 0
			out := RematerializeWip(
				ProgressCursor{StartSHA: "abc", WipTree: tc.wipTree},
				fakeRestorer{verdict: tc.verdict, calls: &calls},
			)
			if out.Verdict != tc.wantVerdict {
				t.Errorf("verdict = %q, want %q (detail=%s)", out.Verdict, tc.wantVerdict, out.Detail)
			}
			if out.Reason != tc.wantReason {
				t.Errorf("reason = %q, want %q", out.Reason, tc.wantReason)
			}
			// An absent wip_tree must not call the store; a present one must.
			wantCalls := 1
			if tc.wipTree == "" {
				wantCalls = 0
			}
			if calls != wantCalls {
				t.Errorf("restorer calls = %d, want %d", calls, wantCalls)
			}
			// A deferral always names the object it held back, so an operator can find it.
			if tc.wantVerdict == WipDeferred && out.WipTree != tc.wipTree {
				t.Errorf("deferred outcome must carry the wip_tree object, got %q", out.WipTree)
			}
		})
	}
}

// TestResumeRematerializeGatesOnStartSHA pins the resume ordering issue #3878 requires: the
// delta is re-materialized ONLY after VerifyReload confirms start_sha is fresh. A stale anchor
// defers WITHOUT ever calling the restorer (re-applying over a diverged base is unsafe); a
// fresh anchor applies; an absent wip_tree short-circuits to absent regardless of the anchor.
func TestResumeRematerializeGatesOnStartSHA(t *testing.T) {
	const anchor = "0123456789abcdef0123456789abcdef01234567"
	fresh := fakeResolver{verified: map[string]bool{anchor: true}}
	diverged := fakeResolver{verified: map[string]bool{}} // anchor no longer resolves

	// Fresh anchor + clean apply -> rematerialized, store consulted once.
	calls := 0
	b := Baton{ProgressCursor: ProgressCursor{StartSHA: anchor, WipTree: "obj-a"}}
	out := ResumeRematerialize(b, fresh, fakeRestorer{verdict: WipApplied, calls: &calls})
	if out.Verdict != WipRematerialized {
		t.Errorf("fresh anchor: verdict = %q, want rematerialized (detail=%s)", out.Verdict, out.Detail)
	}
	if calls != 1 {
		t.Errorf("fresh anchor must consult the store exactly once, got %d calls", calls)
	}

	// Stale anchor -> deferred with RELAY_WIP_STALE, and the restorer is NEVER called.
	calls = 0
	out = ResumeRematerialize(b, diverged, fakeRestorer{verdict: WipApplied, calls: &calls})
	if out.Verdict != WipDeferred || out.Reason != ReasonWipStale {
		t.Errorf("stale anchor: got verdict=%q reason=%q, want deferred/%s", out.Verdict, out.Reason, ReasonWipStale)
	}
	if calls != 0 {
		t.Errorf("a stale anchor must defer WITHOUT touching the store; got %d restorer calls", calls)
	}
	if out.WipTree != "obj-a" {
		t.Errorf("deferred outcome must carry the wip_tree object, got %q", out.WipTree)
	}

	// Absent wip_tree -> absent even with a stale anchor (an old baton resumes as today).
	calls = 0
	b0 := Baton{ProgressCursor: ProgressCursor{StartSHA: anchor}}
	out = ResumeRematerialize(b0, diverged, fakeRestorer{verdict: WipApplied, calls: &calls})
	if out.Verdict != WipAbsent {
		t.Errorf("absent wip_tree: verdict = %q, want absent", out.Verdict)
	}
	if calls != 0 {
		t.Errorf("absent wip_tree must not consult the store; got %d calls", calls)
	}
}

// TestWipTreeWireRoundTrip proves wip_tree is a back-compatible optional cursor field: it
// survives Marshal->Parse when present, and is entirely absent from the wire (and byte-stable)
// when unset — so a pre-#3878 baton serializes exactly as before.
func TestWipTreeWireRoundTrip(t *testing.T) {
	obj := "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
	b := Baton{
		Schema:         Schema,
		RelayID:        "RLY-wip-1",
		ProgressCursor: ProgressCursor{StartSHA: "abcabcabc", WipTree: obj},
	}
	raw, err := Marshal(b)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"wip_tree":"`+obj+`"`) {
		t.Errorf("encoded baton is missing the wip_tree pointer; json=%s", raw)
	}
	got, err := Parse(raw)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if got.ProgressCursor.WipTree != obj {
		t.Errorf("wip_tree round-trip = %q, want %q", got.ProgressCursor.WipTree, obj)
	}
	// Marshal(Parse(x)) is byte-identical — the field does not perturb determinism.
	if raw2, _ := Marshal(got); string(raw2) != string(raw) {
		t.Errorf("wip_tree round-trip not byte-stable:\n a=%s\n b=%s", raw, raw2)
	}

	// Back-compat: a baton with no wip_tree omits the key entirely (omitempty).
	b0 := Baton{Schema: Schema, RelayID: "RLY-wip-0", ProgressCursor: ProgressCursor{StartSHA: "abc"}}
	raw0, err := Marshal(b0)
	if err != nil {
		t.Fatalf("marshal b0: %v", err)
	}
	if strings.Contains(string(raw0), "wip_tree") {
		t.Errorf("an unset wip_tree must not appear on the wire; json=%s", raw0)
	}
}

// gitT runs a git subcommand in dir and fails the test on error, returning trimmed stdout.
func gitT(t *testing.T, dir string, args ...string) string {
	t.Helper()
	full := append([]string{"-C", dir, "-c", "commit.gpgsign=false"}, args...)
	out, err := exec.Command("git", full...).CombinedOutput()
	if err != nil {
		t.Fatalf("git %s: %v: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return strings.TrimSpace(string(out))
}

// TestGitWipRestorerAgainstRepo is the production-path witness: build a throwaway repo, mint a
// checkpoint object from an uncommitted delta (git stash create, the spine's form), wipe the
// tree, and prove GitWipRestorer (a) re-materializes the delta byte-for-byte on a clean tree,
// (b) DEFERS a conflicting tree as WipConflict while leaving it untouched (fail-closed), and
// (c) reports a bogus object id as WipUnavailable. Skips where git is unavailable.
func TestGitWipRestorerAgainstRepo(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skipf("git unavailable in this environment: %v", err)
	}
	dir := t.TempDir()
	gitT(t, dir, "init", "-q")
	gitT(t, dir, "config", "user.email", "wip@test")
	gitT(t, dir, "config", "user.name", "wip test")

	file := filepath.Join(dir, "f.txt")
	if err := os.WriteFile(file, []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	gitT(t, dir, "add", "f.txt")
	gitT(t, dir, "commit", "-q", "-m", "base")

	// Dirty the tracked file (do not stage) and mint the checkpoint object.
	want := "base\nan uncommitted line\n"
	if err := os.WriteFile(file, []byte(want), 0o644); err != nil {
		t.Fatal(err)
	}
	obj := gitT(t, dir, "stash", "create")
	if obj == "" {
		t.Skip("git stash create produced no object (nothing to stash); environment cannot mint a checkpoint")
	}

	restorer := NewGitWipRestorer(dir)

	// (a) clean tree -> the delta re-materializes byte-for-byte.
	gitT(t, dir, "checkout", "--", "f.txt") // wipe the delta, as a peer sweep / reset would
	if res := restorer.Restore(obj); res.Verdict != WipApplied {
		t.Fatalf("clean re-materialize: verdict = %q, want applied (detail=%s)", res.Verdict, res.Detail)
	}
	if got, _ := os.ReadFile(file); string(got) != want {
		t.Errorf("re-materialized content = %q, want %q", got, want)
	}

	// (b) diverged tree -> WipConflict, and the tree is left EXACTLY as it was (fail-closed).
	gitT(t, dir, "checkout", "--", "f.txt")
	diverged := "totally different content\nno base line here\n"
	if err := os.WriteFile(file, []byte(diverged), 0o644); err != nil {
		t.Fatal(err)
	}
	if res := restorer.Restore(obj); res.Verdict != WipConflict {
		t.Fatalf("diverged tree: verdict = %q, want conflict (detail=%s)", res.Verdict, res.Detail)
	}
	if got, _ := os.ReadFile(file); string(got) != diverged {
		t.Errorf("a deferred conflict must not mutate the tree; content = %q, want %q", got, diverged)
	}

	// (c) a bogus object id -> WipUnavailable, never a false apply.
	if res := restorer.Restore("0000000000000000000000000000000000000000"); res.Verdict != WipUnavailable {
		t.Errorf("bogus object: verdict = %q, want unavailable (detail=%s)", res.Verdict, res.Detail)
	}
}
