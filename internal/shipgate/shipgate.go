// Package shipgate is RSI-as-ship-gate: the propose -> measure -> keep-or-revert
// loop with the one rule no prior auto-improver enforced — a candidate is KEPT
// only if a witness the candidate's author did NOT write confirms a STRICT metric
// gain; otherwise it is REVERTED. The keep/revert/escalate decision is a typed
// verdict derived from the measurement, never from the candidate's own claim (the
// "non-forgeable keep-bit"). A run of consecutive non-keeps trips a breaker that
// ESCALATEs to a human (surfaced, in the real loop, via `dos decisions`).
//
// Candidates are evaluated in an ISOLATED git worktree (ApplyInWorktree) so the
// kernel adjudicating a change is never the kernel being rewritten. The worktree
// path uses os/exec — this is the RSI harness, NOT the dispatch hot path, so the
// os/exec-absence proof (unit 72) does not apply here.
package shipgate

import (
	"fmt"
	"os/exec"
)

// Decision is the typed improve-verdict.
type Decision uint8

const (
	REVERT   Decision = iota // candidate did not strictly improve — reverted
	KEEP                     // a non-author witness confirmed a strict gain
	ESCALATE                 // too many consecutive non-keeps — hand to a human
)

func (d Decision) String() string {
	switch d {
	case KEEP:
		return "KEEP"
	case REVERT:
		return "REVERT"
	case ESCALATE:
		return "ESCALATE"
	}
	return "?"
}

// Witness is the measured evidence the loop did not author: a before/after metric
// plus the suite-green and truth-clean bits a real run would also require.
type Witness struct {
	Metric      string
	Before      float64
	After       float64
	LowerBetter bool // true: a smaller metric (e.g. p50 latency) is better
	SuiteGreen  bool // the test suite passed on a clean worktree
	TruthClean  bool // the truth syscall (dos verify) was clean
	improvedBit bool // set ONLY by Evaluate — the non-forgeable keep-bit
}

// improved reports a STRICT gain under the witness's direction.
func (w Witness) improved() bool {
	if w.LowerBetter {
		return w.After < w.Before
	}
	return w.After > w.Before
}

// Evaluate is the keep-or-revert rule. It KEEPs only if (1) a strict metric gain,
// (2) the suite is green, and (3) the truth syscall is clean — all three from the
// witness, none from the candidate's say-so. It sets the non-forgeable keep-bit.
func Evaluate(w Witness) (Decision, Witness) {
	w.improvedBit = w.improved() && w.SuiteGreen && w.TruthClean
	if w.improvedBit {
		return KEEP, w
	}
	return REVERT, w
}

// Kept reports the non-forgeable keep-bit. A caller cannot fabricate a KEEP: the
// bit is only ever set inside Evaluate from a measured witness.
func (w Witness) Kept() bool { return w.improvedBit }

// Gate tracks consecutive non-keeps and trips a breaker after K of them.
type Gate struct {
	K        int
	nonKeeps int
}

// NewGate builds a breaker that escalates after k consecutive non-keeps.
func NewGate(k int) *Gate {
	if k <= 0 {
		k = 3
	}
	return &Gate{K: k}
}

// Record folds a candidate decision into the breaker. A KEEP resets the counter;
// a REVERT advances it; the K-th consecutive non-keep upgrades the decision to
// ESCALATE.
func (g *Gate) Record(d Decision) Decision {
	if d == KEEP {
		g.nonKeeps = 0
		return KEEP
	}
	g.nonKeeps++
	if g.nonKeeps >= g.K {
		return ESCALATE
	}
	return REVERT
}

// ConsecutiveNonKeeps is the current breaker count.
func (g *Gate) ConsecutiveNonKeeps() int { return g.nonKeeps }

// ----------------------------------------------------------------------------
// Worktree isolation (unit 93) — the candidate is applied to an isolated copy so
// main is never touched while it is adjudicated.
// ----------------------------------------------------------------------------

// ApplyInWorktree creates a detached git worktree at dir off HEAD, runs apply()
// against it (the candidate change), and returns any error. The caller removes
// the worktree with RemoveWorktree. main is never modified. Best-effort: if git
// worktrees are unavailable the error is surfaced (the loop degrades, it does not
// silently mutate main).
func ApplyInWorktree(repo, dir string, apply func(worktree string) error) error {
	add := exec.Command("git", "-C", repo, "worktree", "add", "--detach", dir)
	if out, err := add.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree add: %v: %s", err, out)
	}
	if err := apply(dir); err != nil {
		_ = RemoveWorktree(repo, dir)
		return err
	}
	return nil
}

// RemoveWorktree tears down an isolated worktree.
func RemoveWorktree(repo, dir string) error {
	rm := exec.Command("git", "-C", repo, "worktree", "remove", "--force", dir)
	if out, err := rm.CombinedOutput(); err != nil {
		return fmt.Errorf("worktree remove: %v: %s", err, out)
	}
	return nil
}

// ----------------------------------------------------------------------------
// The scripted one-shot (cut-order fallback): tune the vDSO cache-size constant.
// ----------------------------------------------------------------------------

// TuneCacheSize is the v0.1 RSI one-shot: a candidate proposes a new cache size;
// we measure a KPI (e.g. hit-rate, lower-is-worse) under both and keep only on a
// strict gain. A non-improving tweak is REVERTED — the demonstrated property
// (unit 94): the gate provably BLOCKS a non-improving change.
func TuneCacheSize(baselineKPI, candidateKPI float64, suiteGreen, truthClean bool) (Decision, Witness) {
	return Evaluate(Witness{
		Metric:      "vdso_hit_rate",
		Before:      baselineKPI,
		After:       candidateKPI,
		LowerBetter: false, // a higher hit-rate is better
		SuiteGreen:  suiteGreen,
		TruthClean:  truthClean,
	})
}
