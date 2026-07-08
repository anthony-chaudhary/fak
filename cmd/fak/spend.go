package main

// spend.go — `fak spend`: the cross-account spend rollup with provenance labels
// (#2903). Hermes tracks credits across accounts by relaying the provider's own
// number; fak reports each spend-relevant figure labeled by WHO AUTHORED it:
//
//   - live/active session counts come from fak's own watchdog registry
//     (sessions.json) — fak authored them, so they are WITNESSED;
//   - the usage-throttled state is the provider's own quota statement relayed
//     through the registry fold — OBSERVED, never a fak claim, and never
//     attributed to a fak action.
//
// The rollup is deliberately DOLLAR-BLIND: no per-account billed-charge source
// is wired today, so no figure claims a USD amount — that honesty is stated in
// the envelope rather than a guessed cost being booked (the same discipline as
// session_spend.go's "no pricing means no debit"). The pure shape + the
// unlabeled-figure gate live in internal/metrics; this file only folds the
// roster into figures and wires the exit code: any unlabeled figure fails the
// run.

import (
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"github.com/anthony-chaudhary/fak/internal/fleetaccounts"
	"github.com/anthony-chaudhary/fak/internal/metrics"
)

const (
	// spendBasisSessions labels a session count: fak's own registry authored it.
	spendBasisSessions = "session count from fak's watchdog registry (sessions.json) — fak-authored, not a billed charge"
	// spendBasisThrottle labels the usage-limit state: the provider's number, relayed.
	spendBasisThrottle = "provider-relayed usage-limit state (1 = throttled) — the provider's own quota statement, not a fak-authored charge"
	// spendDollarNote states the rollup's dollar honesty in one line.
	spendDollarNote = "dollar-blind: no per-account billed-charge source is wired; no figure here is a USD amount"
)

func cmdSpend(argv []string) { os.Exit(runSpend(os.Stdout, os.Stderr, argv)) }

func runSpend(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("spend", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the fak-spend-rollup/1 JSON envelope instead of a table")
	if err := fs.Parse(argv); err != nil {
		return 2
	}

	cwd, _ := os.Getwd()
	repoRoot := findRepoRoot(cwd)
	toolsDir := filepath.Join(repoRoot, "tools")
	paths := fleetaccounts.ResolvePaths(toolsDir)
	pol := fleetaccounts.LoadPolicy(paths)
	reg := fleetaccounts.LoadRegistry(paths.RegistryPath)
	rows := fleetaccounts.AnnotatedRoster(paths.Home, paths.ConfigHome, pol, reg)

	rollup := buildSpendRollup(rows)

	if defects := metrics.GateSpendLabeled(rollup); len(defects) > 0 {
		fmt.Fprintln(stderr, "fak spend: GATE FAILED — unlabeled spend figure(s):")
		for _, d := range defects {
			fmt.Fprintln(stderr, "  "+d)
		}
		return 1
	}

	if *asJSON {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		if err := enc.Encode(rollup); err != nil {
			fmt.Fprintln(stderr, "fak spend: "+err.Error())
			return 1
		}
		return 0
	}

	fmt.Fprintf(stdout, "fak spend — cross-account rollup (%s)\n", rollup.Schema)
	fmt.Fprintln(stdout, spendDollarNote)
	for _, f := range rollup.Figures {
		account := f.Account
		if account == "" {
			account = "(fleet total)"
		}
		fmt.Fprintf(stdout, "%-24s %-18s %8.0f %-8s %-9s %s\n",
			account, f.Metric, f.Value, f.Unit, string(f.Provenance), f.ValuationBasis)
	}
	return 0
}

// buildSpendRollup folds the annotated roster into the labeled rollup: one
// WITNESSED session-count figure and one OBSERVED throttle-state figure per
// worker account that has a runtime row, plus fleet totals whose provenance is
// the weakest of their inputs (arithmetic never upgrades authorship).
func buildSpendRollup(rows []fleetaccounts.Account) metrics.SpendRollup {
	rollup := metrics.SpendRollup{Schema: metrics.SpendRollupSchema, DollarNote: spendDollarNote}
	var totalSessions, totalThrottled float64
	var sessionProv, throttleProv []metrics.SpendProvenance
	for _, row := range rows {
		if row.Kind != fleetaccounts.KindWorker || row.Available == nil {
			continue // no runtime row — nothing to report, and nothing is guessed
		}
		live := 0
		if row.LiveSessions != nil {
			live = *row.LiveSessions
		}
		rollup.Figures = append(rollup.Figures, metrics.SpendFigure{
			Account: row.Tag, Metric: "live_sessions", Value: float64(live),
			Unit: "count", ValuationBasis: spendBasisSessions, Provenance: metrics.SpendWitnessed,
		})
		totalSessions += float64(live)
		sessionProv = append(sessionProv, metrics.SpendWitnessed)

		throttled := 0.0
		if row.Throttled != nil && *row.Throttled {
			throttled = 1
		}
		rollup.Figures = append(rollup.Figures, metrics.SpendFigure{
			Account: row.Tag, Metric: "usage_throttled", Value: throttled,
			Unit: "state", ValuationBasis: spendBasisThrottle, Provenance: metrics.SpendObserved,
		})
		totalThrottled += throttled
		throttleProv = append(throttleProv, metrics.SpendObserved)
	}
	if len(sessionProv) > 0 {
		rollup.Figures = append(rollup.Figures,
			metrics.SpendFigure{
				Metric: "fleet_live_sessions", Value: totalSessions, Unit: "count",
				ValuationBasis: spendBasisSessions,
				Provenance:     metrics.WeakestSpendProvenance(sessionProv...),
			},
			metrics.SpendFigure{
				Metric: "fleet_usage_throttled", Value: totalThrottled, Unit: "count",
				ValuationBasis: spendBasisThrottle,
				Provenance:     metrics.WeakestSpendProvenance(throttleProv...),
			},
		)
	}
	return rollup
}
