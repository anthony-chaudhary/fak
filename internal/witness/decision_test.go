package witness

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// recordingGit is an injected Runner that scripts a (stdout, code, err) per
// git-subcommand and remembers every argv it received, so a test can assert BOTH
// the round-trip behavior and the exact refs the recorder touched. It keys its
// scripted output on the notes verb (append vs show) so one fake serves both
// halves of a round-trip.
type recordingGit struct {
	store    string // accumulated note body the fake echoes back on `show`
	showCode int    // forced exit for `show` (1 => no note for object)
	appendC  int
	err      error
	seen     [][]string
}

func (g *recordingGit) run(ctx context.Context, dir string, args ...string) (string, int, error) {
	// copy the argv so a later mutation by the caller can't rewrite history.
	cp := append([]string(nil), args...)
	g.seen = append(g.seen, cp)
	if g.err != nil {
		return "", -1, g.err
	}
	switch notesVerb(args) {
	case "append":
		// emulate `git notes append`: read the -F payload the recorder wrote and
		// concatenate it onto the stored note body BEFORE the recorder deletes the
		// temp file, so a later `show` round-trips exactly what was appended.
		if p := payloadFile(args); p != "" {
			if b, err := os.ReadFile(p); err == nil {
				g.store += string(b)
			}
		}
		return "", g.appendC, nil
	case "show":
		if g.showCode != 0 {
			return "", g.showCode, nil
		}
		return g.store, 0, nil
	}
	return "", 0, nil
}

// payloadFile returns the path passed to `-F` in a git argv, or "".
func payloadFile(args []string) string {
	for i, a := range args {
		if a == "-F" && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// notesVerb finds the `git notes --ref=... <verb> ...` verb in an argv.
func notesVerb(args []string) string {
	for i, a := range args {
		if a == "notes" && i+2 < len(args) {
			return args[i+2] // skip the --ref= flag that follows "notes"
		}
	}
	return ""
}

// assertNeverTouchesTrunk fails if any recorded argv targets main / HEAD /
// refs/heads — the recorder must write ONLY to refs/fak/decisions.
func assertNeverTouchesTrunk(t *testing.T, seen [][]string) {
	t.Helper()
	for _, argv := range seen {
		joined := strings.Join(argv, " ")
		for _, forbidden := range []string{"refs/heads", "HEAD", "--force", "push"} {
			if strings.Contains(joined, forbidden) {
				t.Fatalf("recorder argv must never reference %q, got: %v", forbidden, argv)
			}
		}
		// every notes argv must name the dedicated side ref, never a bare ref.
		if contains(argv, "notes") && !contains(argv, "--ref="+decisionsRef) {
			t.Fatalf("notes argv must target %s, got: %v", decisionsRef, argv)
		}
	}
}

func contains(s []string, want string) bool {
	for _, v := range s {
		if v == want {
			return true
		}
	}
	return false
}

// TestAllowDecisionRoundTrips records an ALLOW and reads it back. The fake echoes
// the appended payload as the next `show` output, proving append→read round-trips.
func TestAllowDecisionRoundTrips(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{}
	rec := NewRecorderWithRunner(g.run, "")

	d := Decision{
		Op:      "commit",
		Verdict: VerdictAllow,
		Lane:    "gateway",
		Tree:    []string{"internal/gateway/**"},
	}
	if err := rec.AppendDecision(ctx, "abc123", d); err != nil {
		t.Fatalf("append allow: %v", err)
	}

	got, err := rec.ReadDecisions(ctx, "abc123")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 1 || got[0].Verdict != VerdictAllow || got[0].Op != "commit" {
		t.Fatalf("allow did not round-trip: %+v", got)
	}
	if len(got[0].Tree) != 1 || got[0].Tree[0] != "internal/gateway/**" {
		t.Fatalf("tree did not round-trip: %+v", got[0])
	}
	assertNeverTouchesTrunk(t, g.seen)
}

// TestRefuseDecisionRoundTripsWithReasonClass records a REFUSE carrying its reason
// class + refused argv and reads them back.
func TestRefuseDecisionRoundTripsWithReasonClass(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{}
	rec := NewRecorderWithRunner(g.run, "")

	d := Decision{
		Op:          "commit",
		Verdict:     VerdictRefuse,
		ReasonClass: "OFF_TRUNK",
		Lane:        "witness",
		RefusedArgv: []string{"git", "commit", "-m", "wip"},
	}
	if err := rec.AppendDecision(ctx, "deadbeef", d); err != nil {
		t.Fatalf("append refuse: %v", err)
	}

	got, err := rec.ReadDecisions(ctx, "deadbeef")
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected 1 decision, got %d: %+v", len(got), got)
	}
	if got[0].Verdict != VerdictRefuse || got[0].ReasonClass != "OFF_TRUNK" {
		t.Fatalf("refuse reason class did not round-trip: %+v", got[0])
	}
	if strings.Join(got[0].RefusedArgv, " ") != "git commit -m wip" {
		t.Fatalf("refused argv did not round-trip: %+v", got[0])
	}
	assertNeverTouchesTrunk(t, g.seen)
}

// TestTargetsOnlyDecisionsRef asserts the append argv names refs/fak/decisions and
// nothing on the trunk — the core safety contract.
func TestTargetsOnlyDecisionsRef(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{}
	rec := NewRecorderWithRunner(g.run, "")
	if err := rec.AppendDecision(ctx, "abc", Decision{Op: "exec", Verdict: VerdictAllow}); err != nil {
		t.Fatalf("append: %v", err)
	}
	if len(g.seen) != 1 {
		t.Fatalf("expected exactly one git invocation, got %d", len(g.seen))
	}
	argv := g.seen[0]
	if !contains(argv, "notes") || !contains(argv, "--ref="+decisionsRef) || !contains(argv, "append") {
		t.Fatalf("append must be `git notes --ref=%s append ...`, got: %v", decisionsRef, argv)
	}
	assertNeverTouchesTrunk(t, g.seen)
}

// TestEmptySHAAnchorsToSentinel proves a pre-commit refusal (empty SHA) anchors to
// the empty-tree sentinel and does NOT error.
func TestEmptySHAAnchorsToSentinel(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{}
	rec := NewRecorderWithRunner(g.run, "")

	d := Decision{Op: "commit", Verdict: VerdictRefuse, ReasonClass: "PATHSPEC_RACE"}
	for _, sha := range []string{"", "   "} {
		before := len(g.seen)
		if err := rec.AppendDecision(ctx, sha, d); err != nil {
			t.Fatalf("pre-commit refusal (sha=%q) must not error: %v", sha, err)
		}
		argv := g.seen[before]
		if argv[len(argv)-1] != EmptyTreeSHA {
			t.Fatalf("empty/blank sha must anchor to %s, got target %q", EmptyTreeSHA, argv[len(argv)-1])
		}
	}
	// read-back resolves the empty sha to the same sentinel; both appends landed.
	got, err := rec.ReadDecisions(ctx, "")
	if err != nil {
		t.Fatalf("read pre-commit anchor: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("both blank-sha appends must anchor to the same sentinel, got %d: %+v", len(got), got)
	}
	if got[0].ReasonClass != "PATHSPEC_RACE" {
		t.Fatalf("pre-commit decision did not round-trip: %+v", got)
	}
}

// TestReadMissingNoteIsEmptyNotError proves the "no note for this object" case
// (git notes show non-zero) returns no decisions and no error.
func TestReadMissingNoteIsEmptyNotError(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{showCode: 1} // git notes show: no note for object
	rec := NewRecorderWithRunner(g.run, "")
	got, err := rec.ReadDecisions(ctx, "abc")
	if err != nil {
		t.Fatalf("missing note must not error, got: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("missing note must be empty, got: %+v", got)
	}
}

// TestGitMissingErrorsOnAppend proves a failure to RUN git (git binary missing) is
// surfaced as an error, not swallowed — the recorder must not silently drop a
// decision it could not durably write.
func TestGitMissingErrorsOnAppend(t *testing.T) {
	ctx := context.Background()
	g := &recordingGit{err: exec.ErrNotFound}
	rec := NewRecorderWithRunner(g.run, "")
	if err := rec.AppendDecision(ctx, "abc", Decision{Verdict: VerdictAllow}); err == nil {
		t.Fatalf("git-missing append must error")
	}
}

// TestRealGitDecisionRoundTrip exercises the DEFAULT real-git recorder against a
// throwaway temp repository: it appends an allow + a refuse to refs/fak/decisions
// on a real commit, reads them back, and asserts the trunk ref was never moved.
// This is the end-to-end proof that the append/read pair works with the real git
// binary — not just the fake — mirroring TestRealGitAncestor.
func TestRealGitDecisionRoundTrip(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	ctx := context.Background()
	dir := t.TempDir()

	mustGit := func(args ...string) string {
		t.Helper()
		out, code, err := gitRunner(ctx, dir, args...)
		if err != nil {
			t.Fatalf("git %v: run error: %v", args, err)
		}
		if code != 0 {
			t.Fatalf("git %v: exit %d: %s", args, code, out)
		}
		return out
	}

	mustGit("init", "-q")
	mustGit("config", "user.email", "witness@test")
	mustGit("config", "user.name", "witness test")
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("hi\n"), 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	mustGit("add", "f.txt")
	mustGit("commit", "-q", "-m", "seed")
	headBefore := strings.TrimSpace(mustGit("rev-parse", "HEAD"))

	rec := NewRecorderWithRunner(gitRunner, dir)

	if err := rec.AppendDecision(ctx, headBefore, Decision{
		Op: "commit", Verdict: VerdictAllow, Lane: "witness", Tree: []string{"internal/witness/**"},
	}); err != nil {
		t.Fatalf("append allow: %v", err)
	}
	if err := rec.AppendDecision(ctx, headBefore, Decision{
		Op: "commit", Verdict: VerdictRefuse, ReasonClass: "OFF_TRUNK", RefusedArgv: []string{"git", "commit", "--amend"},
	}); err != nil {
		t.Fatalf("append refuse: %v", err)
	}

	got, err := rec.ReadDecisions(ctx, headBefore)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 appended decisions, got %d: %+v", len(got), got)
	}
	if got[0].Verdict != VerdictAllow || got[1].Verdict != VerdictRefuse || got[1].ReasonClass != "OFF_TRUNK" {
		t.Fatalf("decisions did not round-trip in order: %+v", got)
	}

	// the trunk must be untouched — HEAD unmoved, and the side ref exists.
	if headAfter := strings.TrimSpace(mustGit("rev-parse", "HEAD")); headAfter != headBefore {
		t.Fatalf("HEAD moved: %s -> %s", headBefore, headAfter)
	}
	if _, code, _ := gitRunner(ctx, dir, "show-ref", "--verify", "--quiet", decisionsRef); code != 0 {
		t.Fatalf("%s should exist after append", decisionsRef)
	}

	// the pre-commit (empty-SHA) anchor is independent of the commit anchor.
	if err := rec.AppendDecision(ctx, "", Decision{Op: "commit", Verdict: VerdictRefuse, ReasonClass: "PATHSPEC_RACE"}); err != nil {
		t.Fatalf("append pre-commit refusal: %v", err)
	}
	pre, err := rec.ReadDecisions(ctx, "")
	if err != nil {
		t.Fatalf("read pre-commit: %v", err)
	}
	if len(pre) != 1 || pre[0].ReasonClass != "PATHSPEC_RACE" {
		t.Fatalf("pre-commit anchor did not round-trip: %+v", pre)
	}
	// reading the commit anchor still returns its 2 — the anchors do not bleed.
	if again, _ := rec.ReadDecisions(ctx, headBefore); len(again) != 2 {
		t.Fatalf("commit anchor should still have 2 after a pre-commit append, got %d", len(again))
	}
}

// TestCompactSentinelNoteBoundsSentinelNote proves the unbounded-sentinel fold
// (census #5345): the empty-tree sentinel note — the ONE object every pre-commit
// refusal appends to — is bounded to a keep-last-N tail, while a commit-anchored note
// (bounded forensic evidence) is NEVER touched, HEAD never moves, and a second fold is
// a no-op. This is the collector for the only unbounded producer on
// refs/notes/fak/decisions.
func TestCompactSentinelNoteBoundsSentinelNote(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	ctx := context.Background()
	dir := t.TempDir()

	mustGit := func(args ...string) string {
		t.Helper()
		out, code, err := gitRunner(ctx, dir, args...)
		if err != nil {
			t.Fatalf("git %v: run error: %v", args, err)
		}
		if code != 0 {
			t.Fatalf("git %v: exit %d: %s", args, code, out)
		}
		return out
	}

	mustGit("init", "-q")
	mustGit("config", "user.email", "witness@test")
	mustGit("config", "user.name", "witness test")
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("hi\n"), 0o644); err != nil {
		t.Fatalf("seed file: %v", err)
	}
	mustGit("add", "f.txt")
	mustGit("commit", "-q", "-m", "seed")
	head := strings.TrimSpace(mustGit("rev-parse", "HEAD"))

	rec := NewRecorderWithRunner(gitRunner, dir)

	// A missing sentinel note is a no-op (nothing appended yet).
	if dropped, err := rec.CompactSentinelNote(ctx, 2); err != nil || dropped != 0 {
		t.Fatalf("compact of a missing sentinel note = (%d,%v), want (0,nil)", dropped, err)
	}

	// One commit-anchored decision (bounded forensic evidence — must survive the fold).
	if err := rec.AppendDecision(ctx, head, Decision{Op: "commit", Verdict: VerdictAllow, Lane: "witness"}); err != nil {
		t.Fatalf("append commit-anchored: %v", err)
	}
	// The pre-commit refusals all pile onto the single empty-tree sentinel note. The
	// assertions below relate to THIS seeded count, not to a frozen literal.
	const seededRefusals = 5
	for i := 0; i < seededRefusals; i++ {
		if err := rec.AppendDecision(ctx, "", Decision{Op: "commit", Verdict: VerdictRefuse, ReasonClass: "OFF_TRUNK"}); err != nil {
			t.Fatalf("append pre-commit refusal %d: %v", i, err)
		}
	}
	if pre, _ := rec.ReadDecisions(ctx, ""); len(pre) != seededRefusals {
		t.Fatalf("sentinel should hold %d refusals before the fold, got %d", seededRefusals, len(pre))
	}

	// maxLines <= 0 is a fail-safe no-op: it must NOT truncate the record.
	if dropped, err := rec.CompactSentinelNote(ctx, 0); err != nil || dropped != 0 {
		t.Fatalf("compact(0) = (%d,%v), want (0,nil) fail-safe", dropped, err)
	}
	if pre, _ := rec.ReadDecisions(ctx, ""); len(pre) != seededRefusals {
		t.Fatalf("compact(0) must keep all %d, got %d", seededRefusals, len(pre))
	}

	// Fold to the last 2 refusals: 3 dropped.
	if dropped, err := rec.CompactSentinelNote(ctx, 2); err != nil || dropped != 3 {
		t.Fatalf("compact(2) = (%d,%v), want (3,nil)", dropped, err)
	}
	if pre, err := rec.ReadDecisions(ctx, ""); err != nil || len(pre) != 2 {
		t.Fatalf("sentinel should hold 2 refusals after the fold, got %d (err=%v)", len(pre), err)
	}
	// The commit-anchored note is untouched — the fold never bleeds across anchors.
	if again, _ := rec.ReadDecisions(ctx, head); len(again) != 1 || again[0].Verdict != VerdictAllow {
		t.Fatalf("commit-anchored note must survive the sentinel fold, got %+v", again)
	}
	// HEAD never moved and the side ref still exists.
	if now := strings.TrimSpace(mustGit("rev-parse", "HEAD")); now != head {
		t.Fatalf("HEAD moved during fold: %s -> %s", head, now)
	}
	if _, code, _ := gitRunner(ctx, dir, "show-ref", "--verify", "--quiet", decisionsRef); code != 0 {
		t.Fatalf("%s should still exist after the fold", decisionsRef)
	}

	// Idempotent: a second fold to the same bound drops nothing.
	if dropped, err := rec.CompactSentinelNote(ctx, 2); err != nil || dropped != 0 {
		t.Fatalf("second compact(2) = (%d,%v), want (0,nil) idempotent", dropped, err)
	}
}
