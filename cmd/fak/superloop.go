package main

// superloop.go — the impure shell over internal/superloop: the operator-intent
// META-LOOP. `fak superloop walk improve-quality` is the operator command the concept
// is built for: it WALKS the curated member loops/scorecards/gardens FIRST to read
// their status, then folds a worst-first worklist of what to enter — the layer ABOVE
// a normal loop (see the package doc for the full normal-vs-super differentiation).
//
//	fak superloop list                  the named super loops + their members
//	fak superloop explain <name>        why <name> is a super loop, not a normal loop
//	fak superloop walk <name> [--json]  walk its members, fold the worst-first plan
//
// The walk reads CHEAP, committed status surfaces so it stays fast and honest: a
// scorecard member's debt from the pinned control-pane baseline (the last measured,
// committed value); a loop member's live/dark state from the cross-ledger loop-health
// fold (internal/loopfleet). Container members (a garden or a sub-super-loop) are
// surfaced as DESCEND pointers, not weighed — their status is only knowable by
// descending, which is the named drive/recursion follow-on. It mutates nothing.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/anthony-chaudhary/fak/internal/loopfleet"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/internal/superloop"
)

func cmdSuperloop(argv []string) { os.Exit(runSuperloop(os.Stdout, os.Stderr, argv)) }

func runSuperloop(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		superloopUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "list":
		return runSuperloopList(stdout, stderr, argv[1:])
	case "explain":
		return runSuperloopExplain(stdout, stderr, argv[1:])
	case "walk":
		return runSuperloopWalk(stdout, stderr, argv[1:])
	case "-h", "--help", "help":
		superloopUsage(stdout)
		return 0
	default:
		fmt.Fprintf(stderr, "fak superloop: unknown subcommand %q\n", argv[0])
		superloopUsage(stderr)
		return 2
	}
}

func runSuperloopList(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop list", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the registry as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	reg := superloop.Registry()
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, reg, "fak superloop list")
	}
	fmt.Fprintf(stdout, "fak super loops — operator intents that walk a set of member loops\n\n")
	for _, s := range reg {
		fmt.Fprintf(stdout, "%-18s %s\n", s.Name, s.Title)
		fmt.Fprintf(stdout, "    %s\n", s.About)
		for _, m := range s.Members {
			fmt.Fprintf(stdout, "      - %-10s %-16s %s\n", m.Kind, m.Ref, m.Why)
		}
		fmt.Fprintln(stdout)
	}
	fmt.Fprintf(stdout, "walk one: fak superloop walk %s\n", firstName(reg))
	return 0
}

func runSuperloopExplain(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop explain", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the classification as JSON")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, nil)); err != nil {
		return 2
	}
	name := fs.Arg(0)
	s, ok := superloop.Lookup(name)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop explain: unknown super loop %q (try `fak superloop list`)\n", name)
		return 2
	}
	super := superloop.Classify(superloop.FactsFor(s))
	normal := superloop.Classify(superloop.LeafFacts("a normal task loop (e.g. the dispatch tick)"))
	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, map[string]any{
			"super":  super,
			"normal": normal,
			"loop":   s,
		}, "fak superloop explain")
	}

	fmt.Fprintf(stdout, "%s — %s\n\n", s.Name, s.Title)
	fmt.Fprintf(stdout, "Is it a super loop? %s\n", superYesNo(super.IsSuper))
	fmt.Fprintf(stdout, "  %s\n\n", super.Reason)

	fmt.Fprintln(stdout, "The five properties that separate a super loop from a normal loop:")
	tw := tabwriter.NewWriter(stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(tw, "  PROPERTY\tSUPER LOOP\tNORMAL LOOP\tWHAT IT MEANS")
	for i, p := range super.Properties {
		np := normal.Properties[i]
		fmt.Fprintf(tw, "  %s\t%s\t%s\t%s\n", p.Name, holdMark(p.Holds), holdMark(np.Holds), p.Detail)
	}
	_ = tw.Flush()

	fmt.Fprintf(stdout, "\nA super loop is an INTERIOR node: its unit of work is another LOOP, not a task.\n")
	fmt.Fprintf(stdout, "A normal loop is a LEAF: it ticks and does one concrete thing.\n\n")
	fmt.Fprintf(stdout, "walk it: fak superloop walk %s\n", s.Name)
	return 0
}

func runSuperloopWalk(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("superloop walk", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the walk report as JSON")
	workspace := fs.String("workspace", "", "workspace root (default: repo root)")
	if err := fs.Parse(superloopInterspersedFlagArgs(argv, map[string]bool{"workspace": true})); err != nil {
		return 2
	}
	name := fs.Arg(0)
	s, ok := superloop.Lookup(name)
	if !ok {
		fmt.Fprintf(stderr, "fak superloop walk: unknown super loop %q (try `fak superloop list`)\n", name)
		return 2
	}
	root := *workspace
	if root == "" {
		root = repoRoot()
	} else if abs, err := filepath.Abs(root); err == nil {
		root = abs
	}

	statuses := collectSuperloopStatuses(root, s)
	rep := superloop.Walk(s, statuses)

	if *asJSON {
		return encodeJSONOrFail(stdout, stderr, rep, "fak superloop walk")
	}
	renderSuperloopWalk(stdout, rep)
	if rep.Satisfied {
		return 0
	}
	return 1
}

// collectSuperloopStatuses reads each member's status from the cheap committed
// surfaces, off the hot path. Scorecard debt comes from the pinned control-pane
// baseline; loop liveness from the cross-ledger loop-health fold; container members
// (garden / super loop) are descend pointers, not measured here.
func collectSuperloopStatuses(root string, s superloop.Super) []superloop.MemberStatus {
	baseline, baseErr := loadScorecardBaseline(root)
	fleet := loopfleet.Fold(root, time.Now(), loopmgr.HealthThresholds{})
	loopByKind := map[string]loopfleet.LoopHealth{}
	for _, l := range fleet.Loops {
		loopByKind[l.Kind] = l
	}
	skippedLedger := map[string]string{}
	for _, sk := range fleet.Skipped {
		skippedLedger[sk.Ledger] = sk.Reason
	}

	out := make([]superloop.MemberStatus, 0, len(s.Members))
	for _, m := range s.Members {
		st := superloop.MemberStatus{Member: m}
		switch m.Kind {
		case superloop.KindScorecard:
			if baseErr != nil {
				st.Measured = false
				st.Detail = "no pinned baseline (run `fak scorecard --pin`): " + baseErr.Error()
				break
			}
			debt, present := baseline.Metrics[m.Ref]
			if !present {
				st.Measured = false
				st.Detail = fmt.Sprintf("key %q not in pinned baseline @%s", m.Ref, baseline.Commit)
				break
			}
			st.Measured = true
			st.Debt = debt
			st.Detail = fmt.Sprintf("baseline debt %d (pinned @%s)", debt, baseline.Commit)
		case superloop.KindLoop:
			lh, found := loopByKind[m.Ref]
			if found {
				st.Measured = true
				st.Dark = lh.Dark
				st.Debt = loopDebt(lh)
				st.Detail = fmt.Sprintf("state %s, %d run(s), keep %s", lh.State, lh.Runs, keepRateStr(lh.KeepRate))
				break
			}
			st.Measured = false
			if reason, sk := skippedLedger[m.Ref]; sk {
				st.Detail = "ledger " + reason
			} else {
				st.Detail = "no ledger on this host (loop has not run here)"
			}
		case superloop.KindGarden, superloop.KindSuperloop, superloop.KindSurface:
			st.Container = true
			st.Measured = false
			st.Detail = m.Why
		default:
			st.Measured = false
			st.Detail = "unknown member kind"
		}
		out = append(out, st)
	}
	return out
}

// loopDebt maps a loop-health row to a debt integer for the worst-first fold: a dark
// loop carries its urgency in the Dark flag (debt 0), a stale loop is one unit of
// debt (slipping past its cadence), a live loop is clean.
func loopDebt(lh loopfleet.LoopHealth) int {
	if lh.Dark {
		return 0 // urgency is carried by Dark; tier() ranks it ahead of debt anyway
	}
	if lh.State == loopmgr.HealthStale {
		return 1
	}
	return 0
}

func keepRateStr(r float64) string {
	if r < 0 {
		return "-"
	}
	return fmt.Sprintf("%.2f", r)
}

func loadScorecardBaseline(root string) (scorecardpane.Baseline, error) {
	var b scorecardpane.Baseline
	data, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(scorecardpane.BaselineRel)))
	if err != nil {
		return b, err
	}
	if err := json.Unmarshal(data, &b); err != nil {
		return b, err
	}
	return b, nil
}

func superloopInterspersedFlagArgs(argv []string, takesValue map[string]bool) []string {
	flags := make([]string, 0, len(argv))
	positionals := make([]string, 0, len(argv))
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if !strings.HasPrefix(arg, "-") || arg == "-" {
			positionals = append(positionals, arg)
			continue
		}
		flags = append(flags, arg)
		name := superloopFlagName(arg)
		if takesValue[name] && !strings.Contains(arg, "=") && i+1 < len(argv) && !strings.HasPrefix(argv[i+1], "-") {
			i++
			flags = append(flags, argv[i])
		}
	}
	return append(flags, positionals...)
}

func superloopFlagName(arg string) string {
	arg = strings.TrimLeft(arg, "-")
	if before, _, ok := strings.Cut(arg, "="); ok {
		return before
	}
	return arg
}

func renderSuperloopWalk(w io.Writer, rep superloop.WalkReport) {
	fmt.Fprintf(w, "superloop walk: %s — %s (%s)\n", rep.Name, rep.Verdict, rep.Finding)
	fmt.Fprintf(w, "  aggregate debt %d (floor %d)  members %d  walked %d  unmeasured %d  dark %d\n",
		rep.TotalDebt, rep.Floor, rep.Members, rep.Walked, rep.Unmeasured, rep.Dark)
	fmt.Fprintf(w, "  %s\n\n", rep.Reason)

	if len(rep.Worklist) == 0 {
		fmt.Fprintf(w, "  nothing to enter — every measured member is clean and live\n\n")
	} else {
		fmt.Fprintln(w, "  worst-first — enter these in order:")
		tw := tabwriter.NewWriter(w, 0, 2, 2, ' ', 0)
		fmt.Fprintln(tw, "  #\tMEMBER\tDEBT\tACTION\tWHY")
		for _, it := range rep.Worklist {
			debt := fmt.Sprintf("%d", it.Debt)
			if it.Dark {
				debt = "DARK"
			} else if it.Member.Kind == superloop.KindGarden || it.Member.Kind == superloop.KindSuperloop || it.Member.Kind == superloop.KindSurface {
				debt = "→"
			}
			fmt.Fprintf(tw, "  %d\t%s %s\t%s\t%s\t%s\n",
				it.Rank, it.Member.Kind, it.Member.Ref, debt, it.Action, it.Detail)
		}
		_ = tw.Flush()
		fmt.Fprintln(w)
	}
	fmt.Fprintf(w, "  → %s\n", rep.NextAction)
}

func firstName(reg []superloop.Super) string {
	if len(reg) == 0 {
		return "<name>"
	}
	return reg[0].Name
}

func superYesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func holdMark(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func superloopUsage(w io.Writer) {
	fmt.Fprint(w, `fak superloop - the operator-intent meta-loop: walk a set of member loops, worst-first

  fak superloop list                  the named super loops + their members
  fak superloop explain <name>        why <name> is a super loop, not a normal loop
  fak superloop walk <name> [--json]  walk its members' status, fold a worst-first plan

A SUPER LOOP is keyed on an operator intent ("improve quality"), not a task. Its tick
WALKS its member loops/scorecards/gardens to read their status FIRST, SELECTS the
worst-first member to enter, and exits on the AGGREGATE clearing — the layer above a
normal (task + cadence) loop, which just ticks and does one concrete thing. walk reads
cheap committed surfaces (the pinned scorecard baseline, the loop-health ledger fold)
and mutates nothing; entering/driving a member is the named follow-on.
`)
}
