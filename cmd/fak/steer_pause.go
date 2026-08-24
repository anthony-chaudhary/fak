package main

// `fak steer pause` / `fak steer resume` (#5031): the strongest steer rung
// short of escalation — stop the fleet spending on an intent while the
// operator decides, and release it when they have.
//
// Pause holds the unit's BOUND ISSUE out of future dispatch by appending a row
// to the overlay pause ledger; the dispatch route fold
// (holdSteerPausedForRoute) reads that ledger every tick and moves the
// bound issue into the skipped set under the dispatcher's EXISTING
// BLOCKED_BY_HUMAN token — the same seam the guard's HUMAN_RESIDUAL
// escalations already ride. No new backpressure mechanism, no second reason
// vocabulary.
//
// Pause is NOT a kill: an in-flight worker on a paused intent finishes and
// lands cleanly — the hold only keeps the intent from being picked up AGAIN.
// And a pause with no release path is a leak, so both verbs ship together:
// resume releases the hold even when the unit no longer appears in the forming
// view (a unit that dissolved while paused must still be releasable).

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// runSteerPause records an operator hold against a forming unit's bound
// intent. It writes ONLY the pause ledger — never a Verdict, a Band, git, or a
// live worker — and the dispatch loop enacts the hold on its next tick.
func runSteerPause(stdout, stderr io.Writer, argv []string) int {
	// The unit name may come before the flags or after them; accept both.
	unitArg, argv := splitSteerUnitArg(argv)
	fs := flag.NewFlagSet("fak steer pause", flag.ContinueOnError)
	fs.SetOutput(stderr)
	note := fs.String("m", "", "optional reason recorded with the hold (why the spend should stop)")
	by := fs.String("by", "", "who is pausing (default: git config user.name; the row must be attributable)")
	base := fs.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := fs.String("head", "", "range head ref (default: <release_source> tip)")
	if !parseFlags(fs, argv) {
		return 2
	}
	const usage = `usage: fak steer pause <unit> [-m "<reason>"] [--by WHO] [--base REF] [--head REF]`
	var ok bool
	if unitArg, ok = finishSteerUnitArg(fs, unitArg, usage, stderr); !ok {
		return 2
	}

	root := steerRoot()
	unit, view, err := resolveSteerUnit(root, *base, *head, unitArg)
	if err != nil {
		fmt.Fprintf(stderr, "fak steer pause: %v\n", err)
		return 1
	}
	if unit == nil {
		fmt.Fprintf(stderr, "fak steer pause: no forming unit %q in %s — see `fak steer prs` for the units forming now\n",
			unitArg, releaseStatusString(view["range"]))
		return 1
	}
	if len(unit.Resolves) == 0 {
		// The dispatch loop holds work by issue number; a unit that binds no
		// issue leaves the hold nothing to land on. Refusing (rather than
		// ledgering a no-op) keeps every ledgered pause a real hold.
		fmt.Fprintf(stderr, "fak steer pause: unit %q binds no issue (#N), so the dispatch loop would have nothing to hold — `fak steer redirect` can still re-aim it\n", unitArg)
		return 1
	}

	who := steerActor(root, *by)
	rec, err := steerpr.NewPause(unit.Leaf, who, *note, unit.Resolves[0], time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak steer pause: %v\n", err)
		return 2
	}
	if p, ok := steerpr.PausedFor(steerpr.LoadPauses(steerpr.PauseLedgerPath(root)), unit.Leaf); ok {
		fmt.Fprintf(stderr, "fak steer pause: %s is already paused by %s since %s — `fak steer resume %s` releases it\n",
			unit.Leaf, p.By, p.At, unit.Leaf)
		return 1
	}
	if code, done := appendAndWriteSteerRecord(stdout, stderr, "fak steer pause", rec, func() error {
		return steerpr.AppendPause(steerpr.PauseLedgerPath(root), rec)
	}); done {
		return code
	}
	fmt.Fprintf(stdout, "paused %s (bound %s) as %s — dispatch skips %s with BLOCKED_BY_HUMAN from its next tick; an in-flight worker still lands cleanly (pause is not a kill); release with `fak steer resume %s`\n",
		unit.Leaf, rec.Issue, who, rec.Issue, unit.Leaf)
	return 0
}

// runSteerResume releases a ledgered hold. It deliberately does NOT resolve
// the unit through the forming view: a unit that dissolved (or whose range
// moved) while paused must still be releasable, or the pause becomes the
// silent starvation leak the verb pair exists to prevent.
func runSteerResume(stdout, stderr io.Writer, argv []string) int {
	unitArg, argv := splitSteerUnitArg(argv)
	fs := flag.NewFlagSet("fak steer resume", flag.ContinueOnError)
	fs.SetOutput(stderr)
	by := fs.String("by", "", "who is releasing (default: git config user.name; the row must be attributable)")
	if !parseFlags(fs, argv) {
		return 2
	}
	const usage = `usage: fak steer resume <unit> [--by WHO]`
	var ok bool
	if unitArg, ok = finishSteerUnitArg(fs, unitArg, usage, stderr); !ok {
		return 2
	}

	root := steerRoot()
	held, ok := steerpr.PausedFor(steerpr.LoadPauses(steerpr.PauseLedgerPath(root)), unitArg)
	if !ok {
		fmt.Fprintf(stderr, "fak steer resume: %q is not paused — nothing to release\n", unitArg)
		return 1
	}
	who := steerActor(root, *by)
	rec, err := steerpr.NewResume(unitArg, who, held.Issue, time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak steer resume: %v\n", err)
		return 2
	}
	if code, done := appendAndWriteSteerRecord(stdout, stderr, "fak steer resume", rec, func() error {
		return steerpr.AppendPause(steerpr.PauseLedgerPath(root), rec)
	}); done {
		return code
	}
	fmt.Fprintf(stdout, "resumed %s (was paused by %s since %s) as %s — %s dispatches again from the next tick\n",
		unitArg, held.By, held.At, who, held.Issue)
	return 0
}
