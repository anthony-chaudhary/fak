package main

import (
	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// dojo_lever_provider_firsttry.go — the provider-firsttry/first_try_green_rate
// lever (#4494), registered through the additive RegisterLever seam (#5108) so
// the cell lands in its own file with no edit to the central allDojoLevers /
// dojoLeverCatalogBase literals. The pure fold + the one anchored claim live in
// internal/dojo (claim_provider_firsttry.go); this file is only the
// registration plus the thin attempt-ledger adapter the tier-1 core must not
// do itself.
//
// The lever scores the claimed first-try green rate — the fraction of
// dispatched workers whose acceptance gate passes on the FIRST attempt, with
// no retry, per provider — against the dispatch attempt ledger: the workspace
// loop ledger (.fak/loops.jsonl), the same source the dispatch-yield cell
// (#4497) folds. That ledger records each dispatch (a SPAWNED start row keyed
// by the backend that ran it) but carries no per-worker acceptance-gate
// attempt count or green outcome today, so the cell scores UNMEASURED
// honestly (never a fabricated 0.0 rate) until a gate-attempt record lands on
// the ledger row — the concrete extension seam this KPI calibrates against.
// `fak dojo run` folds the cell like any other.
var _ = RegisterLever(dojoLeverInfo{
	Name:    "provider-firsttry",
	Summary: "first-try green rate per provider — acceptance-gate passes with NO retry over dispatch attempts, folded from the dispatch attempt ledger (the workspace loop ledger's SPAWNED rows, the same source dispatch-yield reads). The cross-provider single-shot-quality leaderboard cell (#4494); the ledger records each dispatch but no per-worker acceptance-gate attempt record yet, so the cell scores UNMEASURED honestly until one lands, never a fabricated 0.0",
	Metrics: []dojoMetricInfo{
		{Name: "first_try_green_rate", Theory: "about half of dispatched workers pass their acceptance gate on the first attempt with no retry (claim 0.5 — a seeded estimate the RSI loop recalibrates toward the measured per-provider rates; UNMEASURED until the attempt ledger records per-worker gate attempts)"},
	},
}, func(env dojoLeverEnv) dojo.Lever { return providerFirstTryLever{root: env.Root} })

// providerFirstTryLever folds the dispatch attempt ledger into the
// first-try-green cell. root is the workspace root the loop ledger lives under.
type providerFirstTryLever struct{ root string }

func (providerFirstTryLever) Name() string { return "provider-firsttry" }

func (l providerFirstTryLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return dojo.ProviderFirstTryEpisodes(loadDispatchFirstTryAttempts(l.root)), nil
}

// loadDispatchFirstTryAttempts adapts the dispatch attempt ledger — the
// workspace loop ledger's SPAWNED start rows, one per dispatched worker (the
// same rows the dispatch-yield fold counts as its dispatched population) —
// into the FirstTryAttempt facts the pure fold scores. Each row keys to the
// provider that ran the worker: the row's Principal is the dispatch backend,
// mapped through the shared leaderboard keying (#4505) so this cell's provider
// rows stay comparable with the sibling provider-* cells.
//
// The ledger records the DISPATCH but no per-worker acceptance-gate attempt
// count or green outcome yet, so every row adapts with Attempts 0 (no gate
// attempt recorded) and the pure fold reports the honest UNMEASURED episode
// rather than a fabricated 0.0 rate; the extension seam is a per-worker
// gate-attempt record on the loop-ledger row (#4494). Fail-open: a missing or
// unreadable ledger yields nil attempts and the fold reports the same honest
// UNMEASURED.
func loadDispatchFirstTryAttempts(root string) []dojo.FirstTryAttempt {
	var out []dojo.FirstTryAttempt
	for _, ev := range loadLoopLedgerEvents(root) {
		if ev.Kind != loopmgr.EventStart || ev.Reason != "SPAWNED" {
			continue // only a SPAWNED start row is a dispatched worker
		}
		out = append(out, dojo.FirstTryAttempt{
			Provider: providerTurnsKey(ev.Principal), // the shared leaderboard keying (#4505)
			// No per-worker acceptance-gate attempt record exists on the ledger row
			// yet (#4494): the ledger records the dispatch but not its gate attempts,
			// so Attempts stays 0 and the rate stays honestly UNMEASURED.
			Attempts: 0,
		})
	}
	return out
}
