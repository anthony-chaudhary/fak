// Command rsicycle drives ONE recursive-self-improvement keep-or-revert decision
// through fak's own non-forgeable keep-bit (internal/shipgate.Evaluate). It takes
// the three environment-authored witnesses as flags — the loop author cannot move
// the decision by narrating; only a measured strict gain + green suite + clean
// truth syscall yields KEEP. This is the assembly of the shipped shipgate
// primitives into a runnable one-shot (the audit's "ASSEMBLE-WHAT-EXISTS" path).
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

// decide runs the candidate's measured witnesses through fak's non-forgeable
// keep-bit and returns the decision, the keep-bit, and the process exit code
// (0 KEEP, 3 REVERT — mirrors the dos improve verdict-as-exit-code contract).
func decide(w shipgate.Witness) (shipgate.Decision, bool, int) {
	d, ev := shipgate.Evaluate(w)
	code := 3
	if d == shipgate.KEEP {
		code = 0
	}
	return d, ev.Kept(), code
}

func main() {
	metric := flag.String("metric", "", "the measured metric name")
	before := flag.Float64("before", 0, "baseline metric (lower-better=false: higher is better)")
	after := flag.Float64("after", 0, "candidate metric")
	lowerBetter := flag.Bool("lower-better", false, "true if a smaller metric is better")
	suiteGreen := flag.Bool("suite-green", false, "the test suite passed on a clean worktree")
	truthClean := flag.Bool("truth-clean", false, "the truth syscall (dos commit-audit) was clean")
	flag.Parse()

	w := shipgate.Witness{
		Metric:      *metric,
		Before:      *before,
		After:       *after,
		LowerBetter: *lowerBetter,
		SuiteGreen:  *suiteGreen,
		TruthClean:  *truthClean,
	}
	decision, kept, code := decide(w)
	fmt.Printf("metric=%s before=%.3f after=%.3f lower_better=%v suite_green=%v truth_clean=%v\n",
		w.Metric, w.Before, w.After, w.LowerBetter, w.SuiteGreen, w.TruthClean)
	fmt.Println(scoreLine(w, kept))
	fmt.Printf("DECISION=%s kept=%v\n", decision, kept)
	// Exit code carries the verdict: 0 KEEP, 3 REVERT (mirrors dos improve).
	os.Exit(code)
}

func scoreLine(w shipgate.Witness, kept bool) string {
	return fmt.Sprintf("score=rsicycle value=%.3f grade=%s metric_delta=%.3f suite_green=%v truth_clean=%v",
		w.After, scoreGrade(w, kept), directionalDelta(w), w.SuiteGreen, w.TruthClean)
}

func scoreGrade(w shipgate.Witness, kept bool) string {
	switch {
	case kept:
		return "kept"
	case !w.TruthClean:
		return "truth-dirty"
	case !w.SuiteGreen:
		return "suite-red"
	case directionalDelta(w) <= 0:
		return "no-gain"
	default:
		return "reverted"
	}
}

func directionalDelta(w shipgate.Witness) float64 {
	if w.LowerBetter {
		return w.Before - w.After
	}
	return w.After - w.Before
}
