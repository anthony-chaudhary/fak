package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/issuecost"
)

// ---------------------------------------------------------------------------
// C9 TIER-OUTCOME CALIBRATION COMMAND (#3046) — the operator surface for the
// pure calibration fold in internal/issuecost (issuecost.Calibrate).
// ---------------------------------------------------------------------------
//
// The fold already joins tier DECISIONS to WITNESSED OUTCOMES and proposes —
// never auto-applies — threshold changes. This verb is the missing acceptance
// arm the issue names ("captured JSON report from the calibration COMMAND or
// test fixture"): a runnable operator readout an agent can capture. It reads
// nothing live and mutates nothing; it is a thin shell over the pure fold, so
// the same rows always render the same report.
//
// It stays ADVISORY/SHADOW by construction: every recommendation the fold emits
// carries auto_apply=false, and this command has no --apply. Automatic policy
// retuning is out of scope and needs a separate keep-bit + human-visible diff.

//fak:ctxplan verb=tier-calibrate enters="nothing live — an offline fold over recorded tier decisions and witnessed-outcome rows" pages="nothing — an advisory calibration report surface, it proposes threshold moves and mutates no live window" warms="nothing — it reads finished-issue records, it touches no prompt cache or KV"
func cmdTierCalibrate(argv []string) { os.Exit(runTierCalibrate(os.Stdout, os.Stderr, argv)) }

// runTierCalibrate is the testable shell for `fak tier-calibrate`.
func runTierCalibrate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("tier-calibrate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the calibration report as JSON (the captured acceptance artifact)")
	decPath := fs.String("decisions", "", "path to a JSON array of tier-decision rows (issuecost.TierDecision)")
	outPath := fs.String("outcomes", "", "path to a JSON array of witnessed-outcome rows (issuecost.WitnessedOutcome)")
	demo := fs.Bool("demo", false, "fold an embedded demo fixture instead of reading files (a runnable, no-input spine)")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}

	var (
		decisions []issuecost.TierDecision
		outcomes  []issuecost.WitnessedOutcome
		err       error
	)
	switch {
	case *demo:
		if *decPath != "" || *outPath != "" {
			fmt.Fprintln(stderr, "fak tier-calibrate: --demo takes no --decisions/--outcomes")
			return 2
		}
		decisions, outcomes = tierCalibrateDemo()
	case *decPath != "" || *outPath != "":
		if decisions, err = readTierDecisions(*decPath); err != nil {
			fmt.Fprintf(stderr, "fak tier-calibrate: %v\n", err)
			return 1
		}
		if outcomes, err = readWitnessedOutcomes(*outPath); err != nil {
			fmt.Fprintf(stderr, "fak tier-calibrate: %v\n", err)
			return 1
		}
	default:
		tierCalibrateUsage(stderr)
		return 2
	}

	rep := issuecost.Calibrate(decisions, outcomes)
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	fmt.Fprint(stdout, rep.Render())
	return 0
}

func tierCalibrateUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak tier-calibrate --demo [--json]")
	fmt.Fprintln(w, "       fak tier-calibrate --decisions decisions.json [--outcomes outcomes.json] [--json]")
	fmt.Fprintln(w, "  Folds tier decisions against witnessed outcomes into an ADVISORY calibration report:")
	fmt.Fprintln(w, "  per-tier buckets, over-tier waste, and threshold proposals (auto_apply=false — it never")
	fmt.Fprintln(w, "  retunes a live policy). A decision with no witnessed outcome is counted but never bucketed.")
}

// readTierCalibRows loads a JSON array of calibration input rows. An empty path is not
// an error — it yields no rows, so an operator can fold outcomes-only or decisions-only
// input. `noun` names the input in the read/parse error text.
func readTierCalibRows[T any](path, noun string) ([]T, error) {
	if path == "" {
		return nil, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read %s %s: %w", noun, path, err)
	}
	var rows []T
	if err := json.Unmarshal(data, &rows); err != nil {
		return nil, fmt.Errorf("parse %s %s: %w", noun, path, err)
	}
	return rows, nil
}

// readTierDecisions loads a JSON array of tier-decision rows.
func readTierDecisions(path string) ([]issuecost.TierDecision, error) {
	return readTierCalibRows[issuecost.TierDecision](path, "decisions")
}

// readWitnessedOutcomes loads a JSON array of witnessed-outcome rows.
func readWitnessedOutcomes(path string) ([]issuecost.WitnessedOutcome, error) {
	return readTierCalibRows[issuecost.WitnessedOutcome](path, "outcomes")
}

// tierCalibrateDemo returns a small, hand-verified fixture that exercises all
// three recommendation branches so `--demo` renders a meaningful readout:
//
//	T0 — two over-tier successes, no rework  -> expand-cheaper
//	T1 — an escalation                       -> raise-floor
//	T2 — a single witnessed success          -> hold
func tierCalibrateDemo() ([]issuecost.TierDecision, []issuecost.WitnessedOutcome) {
	decisions := []issuecost.TierDecision{
		{Issue: 10, Chosen: issuecost.TierT0, Required: issuecost.TierT2, Optimal: issuecost.TierT2},
		{Issue: 11, Chosen: issuecost.TierT0, Required: issuecost.TierT2, Optimal: issuecost.TierT2},
		{Issue: 20, Chosen: issuecost.TierT1, Required: issuecost.TierT1, Optimal: issuecost.TierT1},
		{Issue: 30, Chosen: issuecost.TierT2, Required: issuecost.TierT2, Optimal: issuecost.TierT2},
	}
	green := func(issue int) issuecost.WitnessedOutcome {
		return issuecost.WitnessedOutcome{Issue: issue, CommitWitnessed: true, TestsGreen: true, Closed: true, Turns: 3}
	}
	outcomes := []issuecost.WitnessedOutcome{
		green(10), green(11), // T0 over-tier successes -> expand-cheaper
		{Issue: 20, Escalated: true, Turns: 9}, // T1 escalation          -> raise-floor
		green(30),                              // T2 single success      -> hold
	}
	return decisions, outcomes
}
