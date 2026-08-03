package main

// trajctl_backtest.go — issue #2573, epic #2533: `fak trajctl backtest --scorer X --corpus Y`,
// the operator door to the replay backtest. It qualifies a scoring method against a RECORDED
// corpus before that method is allowed to steer anything live, and emits the schema-pinned
// calibration report internal/trajctl folds.
//
// The whole pass is offline by construction: the corpus is a previously recorded trajctl
// ledger, the fold reads no clock and no network, and nothing is appended. That is what makes
// the qualification gate cheap enough to be non-negotiable — a scorer ships with its backtest.
//
// The exit code carries the distinction the issue names, because a shell script cannot read
// prose: 0 QUALIFIED, 3 the backtest RAN and the scorer did not qualify (NOT_QUALIFIED or
// INCONCLUSIVE), 1 the backtest could NOT RUN (BACKTEST_ERROR), 2 a usage error. A tool fault
// therefore can never be mistaken for a refusal, and a refusal can never be mistaken for a
// pass — including by `--json` readers, which get a report in every case but the usage error.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

// Backtest exit codes. 3 is deliberately distinct from 1: "your scorer lost" and "the tool
// could not run" are different facts and a CI step should be able to branch on which.
const (
	trajctlBacktestExitRefused = 3
	trajctlBacktestExitErrored = 1
)

// trajctlReplayable is the set of scorers a backtest can replay offline, keyed by both the
// stable registry method id and a short operator alias. A scorer belongs here only if its
// Score is pure — the judge scorer needs a live gateway, so recorded history cannot qualify
// it and naming it earns a typed SCORER_NOT_REPLAYABLE refusal rather than a silent skip.
func trajctlReplayable() map[string]trajctl.Scorer {
	commit := trajctl.CommitProgressScorer{}
	stall := trajctl.ActivityDivergenceScorer{}
	bench := trajctl.BenchProgressScorer{}
	return map[string]trajctl.Scorer{
		trajctl.CommitScorerMethod:             commit,
		"commit":                               commit,
		trajctl.ActivityDivergenceScorerMethod: stall,
		"stall":                                stall,
		trajctl.BenchScorerMethod:              bench,
		"bench":                                bench,
	}
}

// trajctlReplayableNames lists the accepted spellings for the usage and refusal text.
func trajctlReplayableNames() []string {
	out := make([]string, 0, len(trajctlReplayable()))
	for name := range trajctlReplayable() {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// trajctlResolveReplayable maps an operator-supplied scorer name to a replayable scorer, or
// to the closed reason code explaining why it cannot be replayed here.
func trajctlResolveReplayable(name string) (trajctl.Scorer, trajctl.BacktestReason, string) {
	if sc, ok := trajctlReplayable()[name]; ok {
		return sc, "", ""
	}
	if name == trajctlJudgeMethodAlias || name == trajctl.JudgeScorerMethod {
		return nil, trajctl.BacktestNotReplayable, fmt.Sprintf(
			"%q needs a live gateway per scored objective, so recorded history cannot replay it offline; qualify it with a live calibration run (fak trajctl scorers) instead",
			name)
	}
	return nil, trajctl.BacktestUnknownScorer, fmt.Sprintf(
		"no replayable scorer named %q (replayable: %s)", name, strings.Join(trajctlReplayableNames(), ", "))
}

func runTrajctlBacktest(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak trajctl backtest", flag.ContinueOnError)
	fs.SetOutput(stderr)
	scorer := fs.String("scorer", "", "candidate scorer to qualify: "+strings.Join(trajctlReplayableNames(), "|"))
	corpus := fs.String("corpus", "", "recorded corpus to replay: a trajctl ledger JSONL path (required)")
	incumbent := fs.String("incumbent", "", "incumbent scorer to calibrate the candidate against (empty: calibration only, no regression check)")
	asJSON := fs.Bool("json", false, "emit the pinned "+trajctl.BacktestSchema+" report as JSON")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if *scorer == "" {
		fmt.Fprintln(stderr, "fak trajctl backtest: --scorer is required (replayable: "+strings.Join(trajctlReplayableNames(), ", ")+")")
		return 2
	}
	// --corpus is required rather than defaulting to the live ledger: a backtest run by
	// accident against the ledger it is meant to protect grades a scorer on its own output.
	if *corpus == "" {
		fmt.Fprintln(stderr, "fak trajctl backtest: --corpus is required (a recorded ledger to replay; the live ledger is never the default)")
		return 2
	}

	candidate, reason, detail := trajctlResolveReplayable(*scorer)
	if candidate == nil {
		return trajctlBacktestEmit(stdout, stderr, trajctl.BacktestRefusal(*scorer, reason, detail), *asJSON)
	}
	var against trajctl.Scorer
	if *incumbent != "" {
		sc, ireason, idetail := trajctlResolveReplayable(*incumbent)
		if sc == nil {
			return trajctlBacktestEmit(stdout, stderr, trajctl.BacktestRefusal(*scorer, ireason, "incumbent: "+idetail), *asJSON)
		}
		against = sc
	}

	// A missing corpus is a tool fault, not a scorer verdict — ReadLedgerFile treats an
	// unreadable path as an empty first-run ledger, which here would silently become "no
	// recorded outcome". Stat first so the report names the real cause.
	if _, err := os.Stat(*corpus); err != nil {
		return trajctlBacktestEmit(stdout, stderr, trajctl.BacktestRefusal(*scorer, trajctl.BacktestNoCorpus,
			fmt.Sprintf("cannot read corpus %s: %v", *corpus, err)), *asJSON)
	}

	rep := trajctl.Fold(trajctl.ReadLedgerFile(*corpus)).Backtest(candidate, against)
	return trajctlBacktestEmit(stdout, stderr, rep, *asJSON)
}

// trajctlBacktestEmit renders a report and returns the verdict's exit code, so every exit
// from the subcommand carries both the report and the machine-readable outcome.
func trajctlBacktestEmit(stdout, stderr io.Writer, rep trajctl.BacktestReport, asJSON bool) int {
	if asJSON {
		if code := trajctlEmitJSON(stdout, stderr, rep); code != 0 {
			return code
		}
		return trajctlBacktestExit(rep)
	}
	trajctlRenderBacktest(stdout, rep)
	return trajctlBacktestExit(rep)
}

// trajctlBacktestExit maps the closed verdict vocabulary onto exit codes.
func trajctlBacktestExit(rep trajctl.BacktestReport) int {
	switch rep.Verdict {
	case trajctl.BacktestQualified:
		return 0
	case trajctl.BacktestErrored:
		return trajctlBacktestExitErrored
	default:
		return trajctlBacktestExitRefused
	}
}

// trajctlRenderBacktest prints the calibration report: the candidate's reading, the
// incumbent's when one was named, the per-objective replay, and the verdict with its reason
// code on the last line so the outcome is the thing an operator's eye lands on.
func trajctlRenderBacktest(stdout io.Writer, rep trajctl.BacktestReport) {
	if rep.Verdict == trajctl.BacktestErrored {
		fmt.Fprintf(stdout, "backtest %s: %s (%s)\n%s\n", rep.Method, rep.Verdict, rep.Reason, rep.Detail)
		return
	}
	fmt.Fprintf(stdout, "backtest %s v%s vs the %s outcome over %d replayed objective(s)\n",
		rep.Method, rep.Version, rep.GroundTruth, rep.Outcomes)
	fmt.Fprintf(stdout, "  candidate  %-28s r=%s  %s\n", rep.Method, trajctlBacktestCoeff(rep.Candidate), rep.Candidate.Verdict)
	if rep.Incumbent != nil {
		fmt.Fprintf(stdout, "  incumbent  %-28s r=%s  %-18s delta=%+.2f\n",
			rep.Incumbent.Method, trajctlBacktestCoeff(*rep.Incumbent), rep.Incumbent.Verdict, rep.Delta)
	}
	for _, c := range rep.Cases {
		reading := fmt.Sprintf("%.2f", c.Candidate)
		if !c.Scored {
			reading = "  --" // the scorer declined this objective; shown, never dropped
		}
		fmt.Fprintf(stdout, "    %-24s outcome=%.2f candidate=%s\n", c.ObjectiveID, c.Outcome, reading)
	}
	fmt.Fprintf(stdout, "%s (%s) %s\n", rep.Verdict, rep.Reason, rep.Detail)
}

// trajctlBacktestCoeff renders a coefficient, or n/a when the corpus could not measure one.
func trajctlBacktestCoeff(c trajctl.ScorerCalibration) string {
	if !c.Measured {
		return "  n/a"
	}
	return fmt.Sprintf("%+.2f", c.Coefficient)
}
