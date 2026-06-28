package main

// fak horizon-recovery -- ground the budget-recovery term r of the horizon
// multiplier (docs/explainers/compounding-benefits-of-a-saved-call.md) from a REAL
// ctxplanbench replay. It reads a `fak ctxplanbench --out report.json` artifact and
// emits, per real session and (above a session-count floor) in aggregate, the
// MEASURED recovery operands (linear vs bounded resident tokens, their ratio, the
// reclaimed budget) co-located with their fault-rate FENCE -- and STRUCTURALLY
// refuses to multiply them into a printed r. The doc cites this command, never a
// number for r.

import (
	"flag"
	"fmt"
	"io"
	"os"

	"github.com/anthony-chaudhary/fak/internal/horizonrecovery"
)

func cmdHorizonRecovery(argv []string) { os.Exit(runHorizonRecovery(os.Stdout, os.Stderr, argv)) }

func runHorizonRecovery(stdout, stderr io.Writer, argv []string) int {
	if len(argv) > 0 && argv[0] == "selfcheck" {
		return runReportSelfcheck(stdout, stderr, argv[1:], "horizon-recovery", horizonrecovery.Selfcheck,
			"SELFCHECK OK -- band emits no r/horizon_multiplier (r stays structural); "+
				"recovery operand and its fault-rate fence co-occur; every field measured; "+
				"a single session never yields a population band (floor refuses it).")
	}

	fs := flag.NewFlagSet("fak horizon-recovery", flag.ContinueOnError)
	fs.SetOutput(stderr)
	reportPath := fs.String("report", "", "path to a ctxplanbench report JSON (fak ctxplanbench --out); or pass it positionally")
	scope := fs.String("scope", "aggregate", "which band to emit: aggregate|sessions")
	asJSON := fs.Bool("json", false, "emit the band(s) as JSON")
	if err := fs.Parse(reorderLeadingPositional(argv)); err != nil {
		if err == flag.ErrHelp {
			return 0
		}
		return 2
	}

	path, ok := resolveReportPath(fs, stderr, "horizon-recovery",
		"fak horizon-recovery: give a ctxplanbench report JSON path "+
			"(positional or --report), or run 'fak horizon-recovery selfcheck'",
		*reportPath)
	if !ok {
		return 2
	}

	if *scope != "aggregate" && *scope != "sessions" {
		fmt.Fprintf(stderr, "fak horizon-recovery: unknown scope %q; pick aggregate|sessions\n", *scope)
		return 2
	}

	report, ok := readReportJSON[horizonrecovery.Report](path, "horizon-recovery", stderr)
	if !ok {
		return 2
	}

	if *scope == "sessions" {
		bands := horizonrecovery.BandsFromReport(report)
		if len(bands) == 0 {
			fmt.Fprintln(stderr, "fak horizon-recovery: report has no sessions")
			return 2
		}
		if *asJSON {
			return encodeJSON(stdout, stderr, bands)
		}
		for _, b := range bands {
			writeHorizonBandHuman(stdout, b)
			fmt.Fprintln(stdout, "")
		}
		return 0
	}

	band, err := horizonrecovery.AggregateBand(report)
	if err != nil {
		// the population-floor refusal is a structured honesty result, not a crash.
		fmt.Fprintf(stderr, "fak horizon-recovery: %v\n", err)
		return 1
	}
	if *asJSON {
		return encodeJSON(stdout, stderr, band)
	}
	writeHorizonBandHuman(stdout, band)
	return 0
}

func encodeJSON(stdout, stderr io.Writer, v any) int {
	return encodeJSONOrFail(stdout, stderr, v, "fak horizon-recovery")
}

func writeHorizonBandHuman(w io.Writer, b horizonrecovery.RecoveryBand) {
	fmt.Fprintf(w, "horizon budget-recovery band (%s) -- %d turns, all fields MEASURED on real transcripts\n",
		b.Scope, b.Turns)
	fmt.Fprintln(w, "  (this is r's measured INPUTS; r itself is never printed -- it needs a task-success eval)")
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "  RECOVERY operand (resident-token budget returned to the pool):")
	fmt.Fprintf(w, "    linear resident tokens  = %d\n", b.LinearResidentTok)
	fmt.Fprintf(w, "    bounded resident tokens = %d\n", b.BoundedResidentTok)
	fmt.Fprintf(w, "    recovery ratio          = %.3f  (>=1 means budget came back; ==1 is the no-recovery floor)\n", b.RecoveryRatio)
	fmt.Fprintf(w, "    reclaimed tokens        = %d\n", b.ReclaimedTok)
	fmt.Fprintln(w, "")
	fmt.Fprintln(w, "  FENCE (the recovery's correctness price -- never shown apart from the operand):")
	fmt.Fprintf(w, "    fault rate              = %.4f  (forecast misses / real references)\n", b.FaultRate)
	fmt.Fprintf(w, "    faults served          = %d  (recovered via DemandPage)\n", b.FaultsServed)
	fmt.Fprintf(w, "    faults refused         = %d  (gate held -- the floor, not a loss)\n", b.FaultsRefused)
	fmt.Fprintf(w, "    fault tax tokens       = %d  (re-prefill the misses cost)\n", b.FaultTaxTokens)
	fmt.Fprintf(w, "    compaction loss turns  = %d  (turns naive compaction destroyed a fact)\n", b.CompactionLoss)
	fmt.Fprintf(w, "    facts recovered        = %d  (planned view recovered what compaction lost)\n", b.FactsRecovered)
}
