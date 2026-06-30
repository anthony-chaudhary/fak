package main

// fak milestone-scorecard -- the milestone report's RSI-scorecard surface
// (issue #1444). It folds the SAME two milestone dimensions the report folds (the
// maturity CLIMB + the epic ROADMAP) into the shared pkg/scorecard control-pane
// Payload: a deterministic milestone_debt integer + a grade + a worst-first
// retire worklist, so the RSI control pane folds it and `/score-2x` retires it
// worst-first like every other surface. The existing `fak milestone report|post`
// behavior is unchanged; this ADDS the scorecard verb.
//
//	fak milestone-scorecard            # render the scorecard snapshot
//	fak milestone-scorecard --json     # the control-pane JSON (corpus.milestone_debt)
//	fak milestone-scorecard --check    # advisory gate: exit 0 clean / 1 with debt
//	fak milestone-scorecard --compare prior.json  # before/after debt delta

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func cmdMilestoneScorecard(argv []string) {
	os.Exit(runMilestoneScorecard(os.Stdout, os.Stderr, argv))
}

func runMilestoneScorecard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak milestone-scorecard", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "owner/name for the `gh` roadmap queries (default: the current checkout's gh context)")
	epicsFrom := fs.String("epics-from", "", "load the tracked-epic set from a JSON data file (default: the in-code TrackedEpics). A file carrying a pre-resolved `counts` block folds offline (no gh).")
	asJSON := fs.Bool("json", false, "emit control-pane JSON")
	check := fs.Bool("check", false, "advisory gate: exit non-zero when milestone_debt > 0")
	comparePath := fs.String("compare", "", "compare against a prior --json payload")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak milestone-scorecard: unexpected argument %q\n", fs.Arg(0))
		return 2
	}

	// Collect the SAME two dimensions the report folds. The maturity climb is the
	// pure in-process grid; the roadmap resolves each tracked epic's children live,
	// or hermetically from an `--epics-from` data file carrying pre-resolved counts.
	maturity, epics := milestonereport.Collect(*repo, nil)
	if *epicsFrom != "" {
		f, err := milestonereport.LoadEpicsFile(*epicsFrom)
		if err != nil {
			fmt.Fprintf(stderr, "fak milestone-scorecard: %v\n", err)
			return 2
		}
		if f.HasCounts() {
			epics = f.FoldOffline()
		} else {
			maturity, epics = milestonereport.CollectSpecs(*repo, nil, f.Specs)
		}
	}

	payload := milestonereport.BuildScorecard(maturity, epics)

	if *comparePath != "" {
		base, ok := readCompareBase(stderr, "fak milestone-scorecard", *comparePath)
		if !ok {
			return 2
		}
		fmt.Fprintln(stdout, scorecard.Compare(payload, base, milestonereport.DebtKey))
		return okExit(payload.OK)
	}
	if *asJSON {
		if err := writeIndentedJSON(stdout, payload); err != nil {
			fmt.Fprintf(stderr, "fak milestone-scorecard: encode json: %v\n", err)
			return 1
		}
		return okExit(payload.OK)
	}
	if *check {
		fmt.Fprintln(stdout, scorecard.Render(payload, milestonereport.DebtKey))
		return okExit(payload.OK)
	}
	fmt.Fprintln(stdout, scorecard.Render(payload, milestonereport.DebtKey))
	return okExit(payload.OK)
}
