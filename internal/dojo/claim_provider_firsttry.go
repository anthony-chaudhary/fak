package dojo

import "sort"

// claim_provider_firsttry.go is the provider-firsttry/first_try_green_rate KPI
// cell (#4494): the fraction of dispatched workers whose acceptance gate passes
// on the FIRST attempt — with no retry — grouped by provider, so the dojo can
// state where fak's routed providers sit against each other on single-shot
// quality instead of calibrating only fak-internal levers.
//
// The claim registers through the additive RegisterClaim seam, so this cell
// never edits — and never conflicts on — the central Registry literal.
//
// Layering (why the fold lives here but the lever does not): internal/dojo is
// the gym's pure core (architest pins it tier 1 — the corpus/ledger-scanning
// levers live in the cmd/fak shell). So the SCORING is a pure, total fold over
// FirstTryAttempt values the shell adapts from the dispatch attempt ledger;
// this file reads no ledger and does no I/O, which keeps the fold
// unit-testable without a ledger on disk and keeps the tier legal.

// The one anchored literal for this cell — the single target the RSI loop's
// RECALIBRATE arm rewrites, carried in the cell's own file rather than the
// shared map.
var _ = RegisterClaim("provider-firsttry", "first_try_green_rate", claim(0.5,
	"seed theory (#4494): about half of dispatched workers pass their acceptance gate on the first attempt with no retry — the single-shot-quality KPI that separates providers needing many retries from those that land clean; a genuine estimate the RSI loop recalibrates toward the measured per-provider rates, and the per-provider spread of the same cell is the cross-provider single-shot-quality leaderboard. A provider with no dispatch attempts scores UNMEASURED, never a fabricated 0.0"))

// FirstTryAttempt is one dispatched worker's acceptance-gate outcome, reduced to
// the three facts the fold needs. The shell adapts the dispatch attempt ledger
// into these; the dojo core never learns the ledger's shape. It deliberately does
// NOT redefine the acceptance gate (#4494 out-of-scope): Green is whatever gate
// the worker's own issue declared, already evaluated upstream.
type FirstTryAttempt struct {
	// Provider keys the leaderboard row — the routed provider that ran the worker.
	Provider string
	// Attempts is how many acceptance-gate attempts the worker took (>= 1). A
	// worker that passed on its first attempt took exactly one.
	Attempts int
	// Green reports whether the acceptance gate ultimately passed at all.
	Green bool
}

// FirstTryGreen reports whether this worker passed its acceptance gate without a
// retry — the numerator's membership test. A worker that needed a retry does not
// count even when it eventually went green (the KPI is single-shot quality, not
// eventual success), and a worker that never went green never counts.
func (a FirstTryAttempt) FirstTryGreen() bool { return a.Green && a.Attempts == 1 }

// ProviderFirstTryEpisodes folds dispatch attempts into the dojo's
// (prediction, outcome) pairs for provider-firsttry/first_try_green_rate — one
// episode PER PROVIDER, every episode scored against the SAME registered claim so
// the recalibrate arm still rewrites exactly one literal while the report renders
// the per-provider spread. It mirrors the provider-cost (#4488) and provider-turns
// (#4505) leaderboard folds.
//
// A provider's rate is its first-try greens over ITS OWN dispatch attempts, so a
// noisy neighbour can neither contaminate nor fabricate another provider's number.
// Honesty rules: a ledger with no attempts at all yields ONE honest UNMEASURED
// episode rather than a fabricated 0.0, and a provider with no attempts simply has
// no row to fabricate (it cannot appear in the fold at all). It is pure and total —
// an unkeyed or non-dispatched row is skipped, never an error.
func ProviderFirstTryEpisodes(attempts []FirstTryAttempt) []ScoredInput {
	pred := Registry.MustPredict("provider-firsttry", "first_try_green_rate", "fraction")
	type acc struct {
		total    int
		firstTry int
	}
	sums := map[string]*acc{}
	for _, at := range attempts {
		if at.Provider == "" {
			continue // no billed provider to key the leaderboard row by
		}
		if at.Attempts < 1 {
			continue // not a dispatched attempt — nothing was gated
		}
		a := sums[at.Provider]
		if a == nil {
			a = &acc{}
			sums[at.Provider] = a
		}
		a.total++
		if at.FirstTryGreen() {
			a.firstTry++
		}
	}
	if len(sums) == 0 {
		return []ScoredInput{{
			Prediction: pred,
			Outcome: Outcome{
				Measured: false,
				Source:   "no dispatch attempts in the local attempt ledger — nothing to fold a per-provider first-try green rate from",
			},
		}}
	}
	providers := make([]string, 0, len(sums))
	for p := range sums {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]ScoredInput, 0, len(providers))
	for _, p := range providers {
		a := sums[p]
		out = append(out, ScoredInput{
			Prediction: pred,
			Outcome: Outcome{
				Realized:   float64(a.firstTry) / float64(a.total),
				Provenance: Observed,
				Measured:   true,
				Sample:     a.total,
				Source:     "provider " + p + ": acceptance-gate passes with no retry / dispatch attempts, over the local attempt ledger (OBSERVED — the provider's own dispatch record, not a fak fault)",
			},
		})
	}
	return out
}
