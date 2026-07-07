package trajctlhook

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// phaseFixtureRepo makes a temp git repo and returns a runner that commits a file
// with the given message and returns the new HEAD SHA. It skips the whole test
// when git is unavailable (the native-Windows path); it runs under WSL/CI.
func phaseFixtureRepo(t *testing.T) (string, func(file, contents, message string) string) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	git := func(args ...string) string {
		c := exec.Command("git", args...)
		c.Dir = dir
		out, err := c.CombinedOutput()
		if err != nil {
			t.Fatalf("git %v: %v\n%s", args, err, out)
		}
		return strings.TrimSpace(string(out))
	}
	git("init", "-q")
	git("config", "user.email", "t@example.com")
	git("config", "user.name", "t")
	commit := func(file, contents, message string) string {
		if err := os.WriteFile(filepath.Join(dir, file), []byte(contents), 0o644); err != nil {
			t.Fatalf("write %s: %v", file, err)
		}
		git("add", file)
		git("commit", "-q", "-m", message)
		return git("rev-parse", "HEAD")
	}
	return dir, commit
}

// TestGitPhaseCommits_ReadsTrailers proves the impure git-log walk reads the
// Trajctl-Phase trailers out of REAL commits and binds each phase to its commit
// SHA. This is the second dogfood gap closed: bindings assembled from git, not
// hand-built.
func TestGitPhaseCommits_ReadsTrailers(t *testing.T) {
	root, commit := phaseFixtureRepo(t)
	sha1 := commit("a.txt", "a", "feat(trajctl): phase one (fak trajctl)\n\nTrajctl-Phase: phase-1")
	sha2 := commit("b.txt", "b", "feat(trajctl): phase two (fak trajctl)\n\nbody line\n\nTrajctl-Phase: phase-2")
	commit("c.txt", "c", "docs: unrelated commit, no phase trailer")

	got := GitPhaseCommits(root, 0)
	if len(got["phase-1"]) != 1 || got["phase-1"][0] != sha1 {
		t.Fatalf("phase-1 = %v, want [%s]", got["phase-1"], sha1)
	}
	if len(got["phase-2"]) != 1 || got["phase-2"][0] != sha2 {
		t.Fatalf("phase-2 = %v, want [%s]", got["phase-2"], sha2)
	}
	if len(got) != 2 {
		t.Fatalf("bound %d phases, want 2 (the no-trailer commit binds nothing): %+v", len(got), got)
	}
}

// TestGitPhaseCommits_NoGitIsFailSoft proves a non-repo directory yields nil, not
// a panic or error — the fail-soft contract the turn-boundary producer relies on.
func TestGitPhaseCommits_NoGitIsFailSoft(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	if got := GitPhaseCommits(t.TempDir(), 0); got != nil {
		t.Fatalf("GitPhaseCommits on a non-repo = %+v, want nil", got)
	}
}

// TestLiveTurnEnd_DrivesRunTurnEndFromTrailers is the headline #3129 witness: a
// LIVE session (a real git repo whose commits carry Trajctl-Phase trailers, plus
// a declared objective in the ledger) drives RunTurnEnd end to end. It proves the
// full producer path the dogfood said did not exist — trailer → GitPhaseCommits →
// EvidenceWindow → RunTurnEnd → a durable W3 witnessed-progress row — with NOTHING
// hand-built: the phase→commit bindings and the resolver both come from git.
func TestLiveTurnEnd_DrivesRunTurnEndFromTrailers(t *testing.T) {
	root, commit := phaseFixtureRepo(t)
	// The live session lands one of the objective's two phases as a real commit.
	commit("phase1.go", "package p", "feat(trajctl): land phase one (fak trajctl)\n\nTrajctl-Phase: phase-1")

	ledger := filepath.Join(root, "docs", "nightrun", "trajctl.jsonl")
	obj := trajctl.Objective{
		ID:        "trajctl-live",
		Statement: "score a live session by its commit trailers",
		Plan:      []trajctl.PlanPhase{{ID: "phase-1"}, {ID: "phase-2"}},
		Status:    trajctl.StatusActive,
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("seed objective: %v", err)
	}

	// The single call a running session's turn-end hook makes: assemble the window
	// from git (trailers + resolver) and score. No hand-built PhaseCommits.
	res := LiveTurnEnd(ledger, root, nil, 1234, trajctl.Stamp{SessionID: "live-sess", RunID: "run-1"})
	if res.Err != nil {
		t.Fatalf("LiveTurnEnd err: %v", res.Err)
	}

	// The W3 witnessed-commit row is present, at 0.5 (one of two phases witnessed
	// from a real, resolvable commit), stamped with the session attribution.
	var w3 *trajctl.ScoreRow
	for i := range res.Sample.Rows {
		if res.Sample.Rows[i].Method == trajctl.CommitScorerMethod {
			w3 = &res.Sample.Rows[i]
		}
	}
	if w3 == nil {
		t.Fatalf("no witnessed-commit row produced; sample = %+v", res.Sample.Rows)
	}
	if w3.Value != 0.5 || w3.Witness != trajctl.W3 {
		t.Fatalf("W3 row = value %v witness %v, want 0.5 W3", w3.Value, w3.Witness)
	}
	if w3.SessionID != "live-sess" || w3.RunID != "run-1" {
		t.Fatalf("W3 row stamp = session %q run %q, want live-sess/run-1", w3.SessionID, w3.RunID)
	}
	if len(w3.Evidence) != 1 || w3.Evidence[0].Detail != "phase-1" {
		t.Fatalf("W3 evidence = %+v, want one pointer for phase-1", w3.Evidence)
	}

	// The row is durable: re-folding the ledger sees the witnessed-progress point,
	// so a curve reader gets a real (non-zero) W3 point from a live session.
	st := trajctl.Fold(trajctl.ReadLedgerFile(ledger))
	var found bool
	for _, s := range st.ScoresFor("trajctl-live") {
		if s.Method == trajctl.CommitScorerMethod && s.Value == 0.5 {
			found = true
		}
	}
	if !found {
		t.Fatalf("re-folded ledger has no durable 0.5 W3 row: %+v", st.ScoresFor("trajctl-live"))
	}
}

// TestLiveTurnEnd_ZeroWithoutTrailers proves the fail-closed rung end to end: a
// live session that has NOT yet landed a phase-bearing commit still scores (a W3
// point per turn) but at value 0 — unverified work is never credited, exactly the
// dogfood's blind spot #2 made safe rather than fabricated.
func TestLiveTurnEnd_ZeroWithoutTrailers(t *testing.T) {
	root, commit := phaseFixtureRepo(t)
	commit("seed.txt", "x", "chore: seed with no phase trailer")

	ledger := filepath.Join(root, "trajctl.jsonl")
	obj := trajctl.Objective{
		ID:        "no-progress-yet",
		Statement: "declared but unwitnessed",
		Plan:      []trajctl.PlanPhase{{ID: "phase-1"}},
		Status:    trajctl.StatusActive,
	}
	if err := trajctl.Append(ledger, trajctl.ObjectiveRecord(obj)); err != nil {
		t.Fatalf("seed objective: %v", err)
	}
	res := LiveTurnEnd(ledger, root, nil, 1, trajctl.Stamp{})
	if res.Err != nil {
		t.Fatalf("LiveTurnEnd err: %v", res.Err)
	}
	var w3 *trajctl.ScoreRow
	for i := range res.Sample.Rows {
		if res.Sample.Rows[i].Method == trajctl.CommitScorerMethod {
			w3 = &res.Sample.Rows[i]
		}
	}
	if w3 == nil || w3.Value != 0 {
		t.Fatalf("W3 row = %+v, want one value-0 row (no witnessed phase)", w3)
	}
}
