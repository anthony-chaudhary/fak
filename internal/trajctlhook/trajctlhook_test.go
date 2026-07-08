package trajctlhook

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// fixtureRepo makes a temp git repo with one real commit and returns (root, sha)
// using the exact plumbing the leaseref/safecommit tests use. Skips the whole
// test when git is unavailable (the native-Windows path); it runs under WSL/CI.
func fixtureRepo(t *testing.T) (string, string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	run := func(args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = dir
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	run("init", "-q")
	run("config", "user.email", "t@example.com")
	run("config", "user.name", "t")
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("hi\n"), 0o644); err != nil {
		t.Fatalf("write fixture file: %v", err)
	}
	run("add", "f.txt")
	run("commit", "-q", "-m", "seed")
	sha := run("rev-parse", "HEAD")
	if len(sha) < 7 {
		t.Fatalf("unexpected sha %q", sha)
	}
	return dir, sha
}

func TestGitEvidenceResolver_Verified(t *testing.T) {
	root, sha := fixtureRepo(t)
	resolve := GitEvidenceResolver(root)
	if got := resolve(trajctl.EvidenceRef{Kind: "commit", Ref: sha}); got != trajctl.EvidenceVerified {
		t.Fatalf("live commit resolved %q, want verified", got)
	}
}

func TestGitEvidenceResolver_Dangling(t *testing.T) {
	root, _ := fixtureRepo(t)
	resolve := GitEvidenceResolver(root)
	// A well-formed but non-existent SHA is dangling, not unknown.
	const ghost = "0123456789abcdef0123456789abcdef01234567"
	if got := resolve(trajctl.EvidenceRef{Kind: "commit", Ref: ghost}); got != trajctl.EvidenceDangling {
		t.Fatalf("ghost commit resolved %q, want dangling", got)
	}
}

func TestGitEvidenceResolver_UnknownKindAndEmpty(t *testing.T) {
	root, sha := fixtureRepo(t)
	resolve := GitEvidenceResolver(root)
	// A non-commit kind is never judged by the commit resolver.
	if got := resolve(trajctl.EvidenceRef{Kind: "transcript-span", Ref: sha}); got != trajctl.EvidenceUnknown {
		t.Fatalf("non-commit kind resolved %q, want unknown", got)
	}
	// An empty ref is unknown, not dangling — nothing to ask git about.
	if got := resolve(trajctl.EvidenceRef{Kind: "commit", Ref: "  "}); got != trajctl.EvidenceUnknown {
		t.Fatalf("empty commit ref resolved %q, want unknown", got)
	}
}

func TestLoadSessions_SkipsBlankAndFolds(t *testing.T) {
	dir := t.TempDir()
	// One real (empty) transcript file; Analyze is fail-soft on content.
	p := filepath.Join(dir, "sess-a.jsonl")
	if err := os.WriteFile(p, []byte(""), 0o644); err != nil {
		t.Fatalf("write transcript: %v", err)
	}
	got := LoadSessions([]string{p, "", "   "})
	if len(got) != 1 {
		t.Fatalf("LoadSessions returned %d sessions, want 1 (blanks skipped)", len(got))
	}
	if got[0].Session != "sess-a" {
		t.Fatalf("session id = %q, want sess-a", got[0].Session)
	}
	if LoadSessions(nil) != nil {
		t.Fatalf("LoadSessions(nil) must be nil")
	}
}

func TestBuildWindow_CarriesStateAndInput(t *testing.T) {
	prior := trajctl.ScoreRow{ObjectiveID: "o1", Value: 0.5, Method: trajctl.CommitScorerMethod, Version: "1", Witness: trajctl.W3}
	state := trajctl.State{Scores: []trajctl.ScoreRow{prior}}
	in := WindowInput{
		PhaseCommits: map[string][]string{"p1": {"deadbeef"}},
		UnixMillis:   777,
	}
	win := BuildWindow(state, in)
	if len(win.PriorScores) != 1 || win.PriorScores[0].Value != 0.5 {
		t.Fatalf("PriorScores not carried from state: %+v", win.PriorScores)
	}
	if win.PhaseCommits["p1"][0] != "deadbeef" {
		t.Fatalf("PhaseCommits not carried: %+v", win.PhaseCommits)
	}
	if win.UnixMillis != 777 {
		t.Fatalf("UnixMillis = %d, want 777", win.UnixMillis)
	}
}

// TestRunTurnEnd_WitnessedProgress is the end-to-end wiring proof: a planned,
// active objective whose one phase is bound to a REAL commit scores a W3 progress
// row of 1.0, and the row lands in the ledger with the caller's session stamp.
func TestRunTurnEnd_WitnessedProgress(t *testing.T) {
	root, sha := fixtureRepo(t)
	ledger := filepath.Join(root, "trajctl.jsonl")

	obj := trajctl.Objective{
		ID:        "o1",
		Statement: "ship the wiring",
		Plan:      []trajctl.PlanPhase{{ID: "p1", Title: "phase one"}},
		Status:    trajctl.StatusActive,
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("seed objective: %v", err)
	}

	res := RunTurnEnd(ledger, WindowInput{
		PhaseCommits: map[string][]string{"p1": {sha}},
		Resolve:      GitEvidenceResolver(root),
		UnixMillis:   1000,
	}, trajctl.Stamp{SessionID: "guard-abc", RunID: "run-9"})

	if res.Err != nil {
		t.Fatalf("RunTurnEnd err: %v", res.Err)
	}
	if res.Appended != 1 {
		t.Fatalf("appended %d rows, want 1", res.Appended)
	}
	if len(res.Sample.Rows) != 1 {
		t.Fatalf("sample has %d rows, want 1", len(res.Sample.Rows))
	}
	row := res.Sample.Rows[0]
	if row.Method != trajctl.CommitScorerMethod || row.Value != 1.0 || row.Witness != trajctl.W3 {
		t.Fatalf("row = method %q value %v witness %v, want witnessed-commit 1.0 W3", row.Method, row.Value, row.Witness)
	}
	if row.SessionID != "guard-abc" || row.RunID != "run-9" {
		t.Fatalf("row stamp = session %q run %q, want guard-abc/run-9", row.SessionID, row.RunID)
	}

	// The row is durable: re-folding the ledger sees the W3 progress point.
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	got := st.ScoresFor("o1")
	if len(got) != 1 || got[0].Value != 1.0 {
		t.Fatalf("re-folded scores = %+v, want one 1.0 progress row", got)
	}
}

// TestRunTurnEnd_NilResolverScoresZero proves the fail-closed rung: with no
// resolver, a planned objective still scores (a point per turn) but at value 0 —
// unverified work is never credited.
func TestRunTurnEnd_NilResolverScoresZero(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "trajctl.jsonl")
	obj := trajctl.Objective{
		ID:        "o1",
		Statement: "unverified",
		Plan:      []trajctl.PlanPhase{{ID: "p1"}},
		Status:    trajctl.StatusActive,
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("seed objective: %v", err)
	}
	res := RunTurnEnd(ledger, WindowInput{
		PhaseCommits: map[string][]string{"p1": {"anything"}},
		Resolve:      nil, // no resolver → nothing verifies
		UnixMillis:   1,
	}, trajctl.Stamp{})
	if res.Err != nil {
		t.Fatalf("RunTurnEnd err: %v", res.Err)
	}
	if len(res.Sample.Rows) != 1 || res.Sample.Rows[0].Value != 0 {
		t.Fatalf("sample = %+v, want one value-0 row", res.Sample.Rows)
	}
}

// TestRunTurnEnd_EmptyLedgerIsFailOpen proves an absent ledger folds to an empty
// state and yields nothing — no objectives, no rows, no error.
func TestRunTurnEnd_EmptyLedgerIsFailOpen(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "does-not-exist.jsonl")
	res := RunTurnEnd(ledger, WindowInput{}, trajctl.Stamp{})
	if res.Err != nil || res.Appended != 0 || len(res.Sample.Rows) != 0 {
		t.Fatalf("empty-ledger turn = %+v, want no rows/no error", res)
	}
}

// TestRunCompaction_BoundaryPerOpenObjective proves the PreCompact twin appends
// one W0 boundary marker per open objective and skips the closed ones.
func TestRunCompaction_BoundaryPerOpenObjective(t *testing.T) {
	dir := t.TempDir()
	ledger := filepath.Join(dir, "trajctl.jsonl")
	for _, o := range []trajctl.Objective{
		{ID: "open1", Statement: "a", Status: trajctl.StatusActive},
		{ID: "paused1", Statement: "b", Status: trajctl.StatusPaused},
		{ID: "done1", Statement: "c", Status: trajctl.StatusMet},
	} {
		if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(o)); err != nil {
			t.Fatalf("seed %s: %v", o.ID, err)
		}
	}
	res := RunCompaction(ledger, 5000, trajctl.Stamp{SessionID: "guard-xyz"})
	if res.Err != nil {
		t.Fatalf("RunCompaction err: %v", res.Err)
	}
	if res.Appended != 2 {
		t.Fatalf("appended %d markers, want 2 (open+paused, not met)", res.Appended)
	}
	seen := map[string]bool{}
	for _, r := range res.Sample.Rows {
		if r.Method != trajctl.CompactionBoundaryMethod || r.Witness != trajctl.W0 || r.Value != 0 {
			t.Fatalf("marker = %+v, want W0 value-0 compaction-boundary", r)
		}
		if r.UnixMillis != 5000 {
			t.Fatalf("marker stamp = %d, want 5000", r.UnixMillis)
		}
		seen[r.ObjectiveID] = true
	}
	if !seen["open1"] || !seen["paused1"] || seen["done1"] {
		t.Fatalf("boundary objectives = %v, want open1+paused1 only", seen)
	}
}

func TestCheapScorers_MethodSet(t *testing.T) {
	got := CheapScorers()
	if len(got) != 2 {
		t.Fatalf("CheapScorers len = %d, want 2", len(got))
	}
	methods := map[string]bool{}
	for _, s := range got {
		methods[s.Method()] = true
	}
	if !methods[trajctl.CommitScorerMethod] || !methods[trajctl.ActivityDivergenceScorerMethod] {
		t.Fatalf("cheap set methods = %v, want commit+divergence", methods)
	}
}
