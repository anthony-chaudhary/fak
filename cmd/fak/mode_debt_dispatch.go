package main

// mode_debt_dispatch.go is the fan-out half of the mode-debt scorer/dispatcher
// pair (#4416, epic #4397, under harness-native #2387 / permission regimes #2389):
// one deduped GitHub issue per HARD un-lifted permission dial, routed to the
// permission-regime backlog (#2389 / #2405). It CONSUMES the sibling scorer leaf's
// scorecard JSON (--scorecard FILE) and maps each HARD un-lifted dial onto the
// internal/dogfoodissues backlog bridge, exactly the same dedup/cap/existing-json/
// fetch-existing/live seam the propagation-/unwired-/qa-process-debt siblings use.
// Dry-run by default; --live files and dedups. A CLEAN (fully lifted) scorecard
// yields an empty candidate set.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/dogfoodissues"
	"github.com/anthony-chaudhary/fak/internal/modedebt"
)

func cmdModeDebtDispatch(argv []string) {
	os.Exit(runModeDebtDispatch(os.Stdout, os.Stderr, argv))
}

func runModeDebtDispatch(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak mode-debt-dispatch", flag.ContinueOnError)
	fs.SetOutput(stderr)
	project := addDogfoodProjectFlags(fs)
	scorecardPath := fs.String("scorecard", "", "mode-debt scorecard JSON emitted by the scorer leaf")
	capN := fs.Int("cap", 10, "maximum HARD un-lifted dials to fan out into issues in one run")
	repo := fs.String("repo", "", "owner/repo for gh; default is the current repo")
	limit := fs.Int("limit", 300, "existing-issue scan limit for live/fetch modes")
	existingJSON := fs.String("existing-json", "", "fixture list of existing gh issues for dry-run dedup tests")
	fetchExisting := fs.Bool("fetch-existing", false, "dry-run but query gh to dedup against existing issue bodies")
	live := fs.Bool("live", false, "create/update GitHub issues with gh")
	asJSON := fs.Bool("json", false, "emit machine-readable plan/result")
	var labels stringList
	fs.Var(&labels, "label", "label to add to newly-created issues; repeatable")
	if !parseFlags(fs, argv) {
		return 2
	}
	// Accept the scorecard as a bare positional too, mirroring learning-debt-dispatch.
	if *scorecardPath == "" && fs.NArg() == 1 {
		*scorecardPath = fs.Arg(0)
	} else if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak mode-debt-dispatch: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *capN < 0 {
		fmt.Fprintln(stderr, "fak mode-debt-dispatch: --cap must be >= 0")
		return 2
	}
	if *scorecardPath == "" {
		fmt.Fprintln(stderr, "fak mode-debt-dispatch: --scorecard is required")
		return 2
	}
	scorecardAbs, err := filepath.Abs(*scorecardPath)
	if err != nil {
		fmt.Fprintf(stderr, "fak mode-debt-dispatch: %v\n", err)
		return 2
	}

	// Fail closed on an absent/unparseable scorecard rather than inventing dials.
	sc, err := modedebt.LoadScorecard(scorecardAbs)
	if err != nil {
		fmt.Fprintf(stderr, "fak mode-debt-dispatch: %v\n", err)
		return 2
	}

	dials := modedebt.SelectHardUnlifted(sc)
	// Cap bounds how many dials become issues in one run (creates + updates), so a
	// first live run over a noisy scorecard can't open a flood; dedup makes re-runs
	// converge on the same set.
	if *capN >= 0 && len(dials) > *capN {
		dials = dials[:*capN]
	}
	items := modedebt.ActionItems(dials, scorecardAbs)

	var existing []dogfoodissues.Issue
	switch {
	case *existingJSON != "":
		b, err := os.ReadFile(*existingJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak mode-debt-dispatch: %v\n", err)
			return 2
		}
		if err := json.Unmarshal(b, &existing); err != nil {
			fmt.Fprintf(stderr, "fak mode-debt-dispatch: --existing-json must be a JSON list: %v\n", err)
			return 2
		}
	case *live || *fetchExisting:
		existing, err = dogfoodissues.FetchExistingIssues(*repo, *limit)
		if err != nil {
			fmt.Fprintf(stderr, "fak mode-debt-dispatch: %v\n", err)
			return 2
		}
	}

	buildOpt := dogfoodissues.BuildOptions{
		Live:           *live,
		DedupeChecked:  *existingJSON != "" || *fetchExisting || *live,
		DedupeCap:      *capN,
		ParentIssue:    *project.parent,
		ParentBaseline: project.baseline(), CompletionStandard: *project.standard, TargetEnvelope: *project.target, WitnessedEnvelope: *project.witnessed,
	}
	plan, skipped := dogfoodissues.BuildPlanWithOptions(items, existing, buildOpt)
	cohort := dogfoodissues.CohortPlan(items, buildOpt)

	mode := "dry-run"
	if *live {
		mode = "live"
	}
	result := dogfoodissues.Result{
		Schema:  dogfoodissues.Schema,
		Mode:    mode,
		Report:  "fak mode-debt-dispatch --scorecard " + scorecardAbs,
		Planned: plan,
		Synced:  []dogfoodissues.SyncRow{},
		Skipped: skipped,
		Cohort:  &cohort,
	}

	exit := 0
	if *live && len(plan) > 0 {
		result.Synced = dogfoodissues.Sync(plan, *repo, []string(labels), nil)
		for _, row := range result.Synced {
			if !row.OK {
				exit = 1
			}
		}
	}

	if code := emitJSONOrPrintln(stdout, stderr, "fak mode-debt-dispatch", *asJSON, result, dogfoodissues.Render(result)); code != 0 {
		return code
	}
	return exit
}
