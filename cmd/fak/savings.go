package main

import (
	"flag"
	"fmt"
	"io"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
)

// cmdSavings is the front door for the Track-2 OBSERVED-$ cache-savings AUDIT trio
// (#2780/#2781/#2782): it promotes the throwaway Python fold that produced the
// "net API cost reduction = 80.8%" headline into versioned, tested commands over the
// durable savings ledger (docs/nightrun/cache-savings.jsonl):
//
//	fak savings audit --since 2026-07-01            # per-date + cumulative $ reconciliation
//	fak savings gate  --slo lossless-or-better       # fidelity SLO gate (exits non-zero on breach)
//	fak savings lint                                 # flag dollar-blind rows (token evidence, no $)
func cmdSavings(argv []string) {
	dispatchSubcommands("savings", "audit | gate | lint", argv,
		subcommand{"audit", runSavingsAudit},
		subcommand{"gate", runSavingsGate},
		subcommand{"lint", runSavingsLint},
	)
}

// savingsLedgerRows reads and since-floors the Track-2 ledger, sharing the report
// command's filter so both front doors agree on what "--since" means.
func savingsLedgerRows(ledger, since string) []cachevaluereport.SavingsRow {
	return filterTrack2Since(cachevaluereport.ReadSavingsLedgerFile(ledger), since)
}

// checkSince validates a --since value against the standard YYYY-MM-DD convention,
// rendering the same error shape the sibling cachevalue commands use.
func checkSince(group, since string, stderr io.Writer) bool {
	if since == "" {
		return true
	}
	if _, err := time.Parse("2006-01-02", since); err != nil {
		fmt.Fprintf(stderr, "fak %s: --since must be YYYY-MM-DD: %v\n", group, err)
		return false
	}
	return true
}

// runSavingsAudit folds the ledger into the per-date + cumulative dollar reconciliation
// (#2780): cache-read fraction, rebate vs write-premium, no-cache counterfactual, net
// dollar reduction (dollar-blind rows excluded from the denominator, #2782), and the
// fidelity/mechanism splits (#2781). Always exits 0 — an audit reports, it does not gate.
func runSavingsAudit(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak savings audit", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevaluereport.DefaultSavingsLedgerRel, "the Track-2 OBSERVED-$ savings ledger to reconcile")
	since := fs.String("since", "", "reconcile only rows on or after this date (YYYY-MM-DD)")
	asJSON := fs.Bool("json", false, "emit the reconciliation as JSON instead of the terminal table")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if !checkSince("savings audit", *since, stderr) {
		return 2
	}
	rep := cachevaluereport.FoldAudit(savingsLedgerRows(*ledger, *since), time.Now().UTC())
	rep.Since = *since
	if *asJSON {
		return writeJSON(stdout, rep)
	}
	fmt.Fprint(stdout, cachevaluereport.RenderAudit(rep))
	return 0
}

// runSavingsGate enforces the fidelity SLO (#2781): every saving must be lossless
// (byte-identical provider hit) or bounded-lossy compaction shedding within
// --compaction-budget tokens. It exits 1 on breach so CI can wire it as a gate.
func runSavingsGate(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak savings gate", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevaluereport.DefaultSavingsLedgerRel, "the Track-2 OBSERVED-$ savings ledger to gate")
	since := fs.String("since", "", "gate only rows on or after this date (YYYY-MM-DD)")
	slo := fs.String("slo", cachevaluereport.SLOLosslessOrBetter, "the fidelity SLO to enforce")
	budget := fs.Uint64("compaction-budget", 0, "tokens of bounded-lossy compaction shedding tolerated in the window (0 = strict lossless)")
	asJSON := fs.Bool("json", false, "emit the gate verdict as JSON instead of the terminal table")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if !checkSince("savings gate", *since, stderr) {
		return 2
	}
	rep := cachevaluereport.GateSavings(savingsLedgerRows(*ledger, *since), *slo, *budget, time.Now().UTC())
	rep.Since = *since
	if *asJSON {
		_ = writeJSON(stdout, rep)
	} else {
		fmt.Fprint(stdout, cachevaluereport.RenderGate(rep))
	}
	if !rep.Pass {
		return 1
	}
	return 0
}

// runSavingsLint flags dollar-blind rows (#2782): rows with real token evidence but no
// trusted price, whose dollar axes are placeholders at zero, not a priced no-savings
// result. Reports by default (exit 0); --strict makes it a CI guard (exit 1 on any).
func runSavingsLint(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak savings lint", flag.ContinueOnError)
	fs.SetOutput(stderr)
	ledger := fs.String("ledger", cachevaluereport.DefaultSavingsLedgerRel, "the Track-2 OBSERVED-$ savings ledger to lint")
	since := fs.String("since", "", "lint only rows on or after this date (YYYY-MM-DD)")
	strict := fs.Bool("strict", false, "exit non-zero when any dollar-blind row is found (CI guard)")
	asJSON := fs.Bool("json", false, "emit the lint findings as JSON instead of the terminal table")
	if code, done := parseFlagsRejectArgs(fs, argv, stderr); done {
		return code
	}
	if !checkSince("savings lint", *since, stderr) {
		return 2
	}
	rep := cachevaluereport.LintSavings(savingsLedgerRows(*ledger, *since), time.Now().UTC())
	rep.Since = *since
	if *asJSON {
		_ = writeJSON(stdout, rep)
	} else {
		fmt.Fprint(stdout, cachevaluereport.RenderLint(rep))
	}
	if *strict && rep.DollarBlindRows > 0 {
		return 1
	}
	return 0
}
