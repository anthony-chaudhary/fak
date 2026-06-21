package shipgate

import (
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

// unit 94: keep only on a strict, witness-confirmed gain.
func TestEvaluateKeepsStrictGain(t *testing.T) {
	d, w := Evaluate(Witness{Metric: "p50_ns", Before: 1000, After: 800, LowerBetter: true, SuiteGreen: true, TruthClean: true})
	if d != KEEP || !w.Kept() {
		t.Fatalf("strict gain + green witness must KEEP, got %v kept=%v", d, w.Kept())
	}
}

// unit 94: a non-improving change is REVERTED.
func TestEvaluateRevertsNoGain(t *testing.T) {
	for _, w := range []Witness{
		{Before: 1000, After: 1000, LowerBetter: true, SuiteGreen: true, TruthClean: true}, // no change
		{Before: 1000, After: 1200, LowerBetter: true, SuiteGreen: true, TruthClean: true}, // worse
	} {
		if d, got := Evaluate(w); d != REVERT || got.Kept() {
			t.Fatalf("non-improving change must REVERT, got %v kept=%v", d, got.Kept())
		}
	}
}

// the non-author witness gates: a strict gain with a RED suite still reverts.
func TestEvaluateRevertsIfSuiteRed(t *testing.T) {
	if d, _ := Evaluate(Witness{Before: 1000, After: 500, LowerBetter: true, SuiteGreen: false, TruthClean: true}); d != REVERT {
		t.Fatalf("a red suite must block KEEP even with a metric gain, got %v", d)
	}
	if d, _ := Evaluate(Witness{Before: 1000, After: 500, LowerBetter: true, SuiteGreen: true, TruthClean: false}); d != REVERT {
		t.Fatalf("a dirty truth syscall must block KEEP, got %v", d)
	}
}

// unit 94 one-shot: a non-improving cache-size tweak is provably blocked.
func TestTuneCacheSizeRevertsNonImproving(t *testing.T) {
	if d, _ := TuneCacheSize(0.50, 0.50, true, true); d != REVERT {
		t.Fatalf("equal hit-rate must REVERT (gate blocks a non-improving change), got %v", d)
	}
	if d, _ := TuneCacheSize(0.50, 0.62, true, true); d != KEEP {
		t.Fatalf("a strictly higher hit-rate must KEEP, got %v", d)
	}
}

// the keep-bit is non-forgeable: a zero Witness is never "kept".
func TestKeepBitNonForgeable(t *testing.T) {
	var w Witness
	if w.Kept() {
		t.Fatal("an unevaluated witness must never report Kept()")
	}
}

// unit 95: K consecutive non-keeps escalate; a KEEP resets the breaker.
func TestGateEscalates(t *testing.T) {
	g := NewGate(3)
	if d := g.Record(REVERT); d != REVERT {
		t.Fatalf("1st non-keep => REVERT, got %v", d)
	}
	if d := g.Record(REVERT); d != REVERT {
		t.Fatalf("2nd non-keep => REVERT, got %v", d)
	}
	if d := g.Record(REVERT); d != ESCALATE {
		t.Fatalf("3rd consecutive non-keep => ESCALATE, got %v", d)
	}
	// a KEEP resets the breaker.
	g2 := NewGate(2)
	g2.Record(REVERT)
	g2.Record(KEEP)
	if g2.ConsecutiveNonKeeps() != 0 {
		t.Fatalf("KEEP must reset the breaker, got %d", g2.ConsecutiveNonKeeps())
	}
}

// unit 93: a candidate is applied in an isolated worktree; main is untouched.
func TestApplyInWorktreeIsolatesMain(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	repo := t.TempDir()
	run := func(args ...string) error {
		c := exec.Command("git", append([]string{"-C", repo}, args...)...)
		c.Env = append(os.Environ(), "GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t", "GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
		out, err := c.CombinedOutput()
		if err != nil {
			t.Logf("git %v: %s", args, out)
		}
		return err
	}
	if run("init", "-q") != nil {
		t.Skip("git init failed")
	}
	mainFile := filepath.Join(repo, "main.txt")
	if err := os.WriteFile(mainFile, []byte("original"), 0o644); err != nil {
		t.Fatal(err)
	}
	_ = run("add", "-A")
	if run("commit", "-qm", "init") != nil {
		t.Skip("git commit failed")
	}

	wt := filepath.Join(t.TempDir(), "wt")
	applied := false
	err := ApplyInWorktree(repo, wt, func(worktree string) error {
		applied = true
		// mutate the file IN THE WORKTREE only
		return os.WriteFile(filepath.Join(worktree, "main.txt"), []byte("candidate"), 0o644)
	})
	if err != nil {
		t.Fatalf("ApplyInWorktree: %v", err)
	}
	if !applied {
		t.Fatal("apply was not invoked")
	}
	// main is untouched.
	b, _ := os.ReadFile(mainFile)
	if string(b) != "original" {
		t.Fatalf("main worktree was mutated: %q", b)
	}
	_ = RemoveWorktree(repo, wt)
}
