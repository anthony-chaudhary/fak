package armtracking

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"text/tabwriter"
	"time"
)

const Usage = `usage: fak baseline-arms <record|compare|leaderboard|audit> [flags]

Tracking and auditing shifting baselines, ablation arms, and sweep configurations.
Compares to next-best latest result by default.

subcommands:
  record       ingest a run, updating latest best and audit history
  compare      compare an arm to the next-best latest result in its workload
  leaderboard  list ranked latest best results for a workload (or list workloads)
  audit        print chronological audit trail of baseline and arm shifts
`

// RunCLI is the testable entrypoint for the `fak baseline-arms` command.
func RunCLI(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		fmt.Fprint(stderr, Usage)
		return 2
	}

	sub, rest := argv[0], argv[1:]
	switch sub {
	case "record":
		return runRecord(stdout, stderr, rest)
	case "compare", "cmp":
		return runCompare(stdout, stderr, rest)
	case "leaderboard", "list", "ls":
		return runLeaderboard(stdout, stderr, rest)
	case "audit", "history":
		return runAudit(stdout, stderr, rest)
	case "-h", "--help", "help":
		fmt.Fprint(stdout, Usage)
		return 0
	default:
		fmt.Fprintf(stderr, "fak baseline-arms: unknown subcommand %q\n%s", sub, Usage)
		return 2
	}
}

func runRecord(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("baseline-arms record", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workload := fs.String("workload", "", "workload identifier (required)")
	armID := fs.String("arm", "", "arm identifier (required)")
	kindStr := fs.String("kind", string(ArmKindBaseline), "arm kind: baseline, ablation, sweep, candidate")
	metric := fs.String("metric", "", "primary metric name (required, e.g. throughput_tok_s, latency_ms)")
	val := fs.Float64("value", 0.0, "measured primary value (required)")
	unit := fs.String("unit", "", "metric unit (e.g. tok/s, ms, µs, MB)")
	dirStr := fs.String("direction", string(HigherIsBetter), "higher_is_better or lower_is_better")
	dimension := fs.String("dimension", "", "ablation/sweep dimension (e.g. target, topology, quantization)")
	feature := fs.String("feature", "", "feature/lever name (e.g. vdso, radix, q4k_gemv)")
	hardware := fs.String("hardware", "", "hardware/environment (e.g. m3pro, strix_halo, h100)")
	commit := fs.String("commit", "", "commit SHA provenance")
	witness := fs.String("witness", "", "reproducible artifact path or witness command")
	notes := fs.String("notes", "", "audit note explaining the run or shift reason")
	storePath := fs.String("store", DefaultPath, "ledger storage file path")
	asJSON := fs.Bool("json", false, "output audit event as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *workload == "" || *armID == "" || *metric == "" {
		fmt.Fprintln(stderr, "fak baseline-arms record: --workload, --arm, and --metric are all required")
		return 2
	}

	kind := ArmKind(*kindStr)
	if err := kind.Validate(); err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms record:", err)
		return 2
	}

	dir := OptimizationDirection(*dirStr)
	if err := dir.Validate(); err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms record:", err)
		return 2
	}

	reg, err := LoadRegistry(*storePath)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms record: load registry:", err)
		return 1
	}

	res := ArmResult{
		ArmID:         *armID,
		Workload:      *workload,
		ArmKind:       kind,
		PrimaryMetric: *metric,
		PrimaryValue:  *val,
		PrimaryUnit:   *unit,
		Direction:     dir,
		Metadata: ArmMetadata{
			Dimension: *dimension,
			Feature:   *feature,
			Hardware:  *hardware,
			CommitSHA: *commit,
			Witness:   *witness,
		},
		Timestamp: time.Now().UTC(),
	}

	evt, err := reg.Record(res, *notes)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms record:", err)
		return 1
	}

	if err := SaveRegistry(reg, *storePath); err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms record: save registry:", err)
		return 1
	}

	if *asJSON {
		b, _ := json.MarshalIndent(evt, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	fmt.Fprintf(stdout, "Recorded %s arm %q on workload %q: %g %s [%s]\n",
		kind, *armID, *workload, *val, *unit, evt.Action)
	if evt.PriorBestValue != nil {
		fmt.Fprintf(stdout, "  Prior best: %g -> New: %g (delta: %+g)\n",
			*evt.PriorBestValue, evt.NewValue, *evt.DeltaFromPrior)
	}
	if *notes != "" {
		fmt.Fprintf(stdout, "  Notes: %s\n", *notes)
	}

	// Print comparison to next-best latest result immediately after recording
	if cmp, err := reg.CompareToNextBest(*workload, *armID); err == nil && cmp.TotalArms > 1 {
		fmt.Fprintf(stdout, "  Next-best comparison: %s\n", cmp.Summary)
	}

	return 0
}

func runCompare(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("baseline-arms compare", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workload := fs.String("workload", "", "workload identifier (required)")
	armID := fs.String("arm", "", "arm identifier (required)")
	storePath := fs.String("store", DefaultPath, "ledger storage file path")
	asJSON := fs.Bool("json", false, "output comparison result as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *workload == "" || *armID == "" {
		fmt.Fprintln(stderr, "fak baseline-arms compare: --workload and --arm are both required")
		return 2
	}

	reg, err := LoadRegistry(*storePath)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms compare: load registry:", err)
		return 1
	}

	cmp, err := reg.CompareToNextBest(*workload, *armID)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms compare:", err)
		return 1
	}

	if *asJSON {
		b, _ := json.MarshalIndent(cmp, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	fmt.Fprintf(stdout, "=== Next-Best Latest Result Comparison: %s (Workload: %s) ===\n", *armID, *workload)
	fmt.Fprintf(stdout, "Target Arm:      %s (Rank %d/%d) [%s]\n", cmp.TargetArm.ArmID, cmp.TargetRank, cmp.TotalArms, cmp.TargetArm.ArmKind)
	fmt.Fprintf(stdout, "Target Value:    %g %s\n", cmp.TargetArm.PrimaryValue, cmp.TargetArm.PrimaryUnit)
	fmt.Fprintf(stdout, "Next Best Arm:   %s (Rank %d/%d) [%s]\n", cmp.NextBestArm.ArmID, cmp.NextBestRank, cmp.TotalArms, cmp.NextBestArm.ArmKind)
	fmt.Fprintf(stdout, "Next Best Value: %g %s\n", cmp.NextBestArm.PrimaryValue, cmp.NextBestArm.PrimaryUnit)
	fmt.Fprintf(stdout, "Delta:           %+g %s\n", cmp.Delta, cmp.TargetArm.PrimaryUnit)
	fmt.Fprintf(stdout, "Speedup / Lift:  %.2f× (%+.1f%%)\n", cmp.SpeedupOrLift, cmp.PercentageDelta)
	fmt.Fprintf(stdout, "Verdict:         %s\n", cmp.Summary)

	return 0
}

func runLeaderboard(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("baseline-arms leaderboard", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workload := fs.String("workload", "", "workload identifier (optional; lists all workloads if omitted)")
	storePath := fs.String("store", DefaultPath, "ledger storage file path")
	asJSON := fs.Bool("json", false, "output leaderboard as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	reg, err := LoadRegistry(*storePath)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms leaderboard: load registry:", err)
		return 1
	}

	if *workload == "" {
		workloads := reg.WorkloadsList()
		if *asJSON {
			b, _ := json.MarshalIndent(workloads, "", "  ")
			fmt.Fprintln(stdout, string(b))
			return 0
		}
		if len(workloads) == 0 {
			fmt.Fprintln(stdout, "No workloads tracked yet in ledger.")
			return 0
		}
		fmt.Fprintln(stdout, "Tracked Workloads:")
		for _, w := range workloads {
			fmt.Fprintf(stdout, "  - %s\n", w)
		}
		fmt.Fprintln(stdout, "\nPass --workload <name> to view detailed leaderboard.")
		return 0
	}

	rows, err := reg.Leaderboard(*workload)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms leaderboard:", err)
		return 1
	}

	if *asJSON {
		b, _ := json.MarshalIndent(rows, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	if len(rows) == 0 {
		fmt.Fprintf(stdout, "No arms recorded for workload %q\n", *workload)
		return 0
	}

	fmt.Fprintf(stdout, "Leaderboard: %s (%d latest best arms)\n\n", *workload, len(rows))
	tw := tabwriter.NewWriter(stdout, 2, 8, 2, ' ', 0)
	fmt.Fprintln(tw, "RANK\tARM ID\tKIND\tLATEST BEST\tUNIT\tVS BEST\tNEXT BEST MARGIN\tRUNS\tCOMMIT")
	for _, r := range rows {
		unit := r.PrimaryUnit
		if unit == "" {
			unit = "—"
		}
		commit := r.CommitSHA
		if commit == "" {
			commit = "—"
		}
		margin := r.NextBestMargin
		if margin == "" {
			margin = "—"
		}
		fmt.Fprintf(tw, "%d\t%s\t%s\t%g\t%s\t%.2f×\t%s\t%d\t%s\n",
			r.Rank, r.ArmID, r.ArmKind, r.PrimaryValue, unit, r.SpeedupVsBest, margin, r.RunCount, commit)
	}
	tw.Flush()

	return 0
}

func runAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("baseline-arms audit", flag.ContinueOnError)
	fs.SetOutput(stderr)

	workload := fs.String("workload", "", "workload identifier (required)")
	armID := fs.String("arm", "", "filter by specific arm identifier (optional)")
	storePath := fs.String("store", DefaultPath, "ledger storage file path")
	asJSON := fs.Bool("json", false, "output audit trail as JSON")

	if err := fs.Parse(argv); err != nil {
		return 2
	}

	if *workload == "" {
		fmt.Fprintln(stderr, "fak baseline-arms audit: --workload is required")
		return 2
	}

	reg, err := LoadRegistry(*storePath)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms audit: load registry:", err)
		return 1
	}

	history, err := reg.AuditHistory(*workload, *armID)
	if err != nil {
		fmt.Fprintln(stderr, "fak baseline-arms audit:", err)
		return 1
	}

	if *asJSON {
		b, _ := json.MarshalIndent(history, "", "  ")
		fmt.Fprintln(stdout, string(b))
		return 0
	}

	if len(history) == 0 {
		fmt.Fprintf(stdout, "No audit events recorded for workload %q\n", *workload)
		return 0
	}

	fmt.Fprintf(stdout, "Audit History: %s (total events: %d)\n\n", *workload, len(history))
	tw := tabwriter.NewWriter(stdout, 2, 8, 2, ' ', 0)
	fmt.Fprintln(tw, "TIMESTAMP\tARM ID\tACTION\tVALUE\tPRIOR\tDELTA\tCOMMIT\tNOTES")
	for _, e := range history {
		prior := "—"
		if e.PriorBestValue != nil {
			prior = fmt.Sprintf("%g", *e.PriorBestValue)
		}
		delta := "—"
		if e.DeltaFromPrior != nil {
			delta = fmt.Sprintf("%+g", *e.DeltaFromPrior)
		}
		commit := e.CommitSHA
		if commit == "" {
			commit = "—"
		}
		notes := e.Notes
		if notes == "" {
			notes = "—"
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%g\t%s\t%s\t%s\t%s\n",
			e.Timestamp.Format("2006-01-02 15:04:05"), e.ArmID, e.Action, e.NewValue, prior, delta, commit, notes)
	}
	tw.Flush()

	return 0
}
