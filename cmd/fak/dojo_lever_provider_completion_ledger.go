package main

import (
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

// dojo_lever_provider_completion_ledger.go — the reconcile-ledger fold of the
// provider-completion/verified_completion_rate KPI cell (#4492), registered
// through the additive RegisterLever seam (#5108) so it lands in its own file
// with no edit to the central allDojoLevers / dojoLeverCatalogBase literals.
//
// Relationship to the sibling provider-completion lever (#4506): ONE cell, ONE
// anchored claim, TWO ground truths. The session-corpus lever scores the cell
// against a corpus PROXY for a verified close (a completed, non-interrupted
// session). This lever scores the SAME registered claim against the reconcile
// ledger — the workspace loop ledger (.fak/loops.jsonl), the ground truth the
// fak-aggregate dispatch-yield cell (#4497) folds: SPAWNED start rows are the
// dispatched issues (each keyed by the dispatch backend that ran the worker,
// through the shared leaderboard keying #4505), and the closure auditor's
// closed_now end-row metrics are the diff-witnessed VERIFIED closes. No second
// registry cell is added: this lever calls MustPredict on the existing
// provider-completion/verified_completion_rate literal (#4506's, the one RSI
// recalibrate anchor), so the one-literal-per-cell rule holds.
//
// Honesty (#4492's assumption): the ledger attributes every DISPATCH to its
// provider, but closed_now is a per-tick AGGREGATE with no per-issue provider
// attribution (the same missing per-worker outcome record the provider-firsttry
// cell #4494 names). So each dispatched provider scores UNMEASURED — carrying
// its witnessed dispatched count as the sample and naming the missing
// provider-attributed close record — never a fabricated rate. The pure fold
// already measures the moment a provider-attributed close record lands on the
// closure-audit row (the extension seam this cell calibrates against).
var _ = RegisterLever(dojoLeverInfo{
	Name:    "provider-completion-ledger",
	Summary: "the reconcile-ledger fold of provider-completion/verified_completion_rate (#4492): dispatched issues that actually closed VERIFIED, per provider, over the workspace loop ledger (.fak/loops.jsonl — the same ground truth dispatch-yield #4497 folds). SPAWNED start rows attribute each dispatch to its provider, but the closure auditor's closed_now is a per-tick aggregate with no per-provider attribution yet, so every dispatched provider scores UNMEASURED honestly (carrying its dispatched count) until a provider-attributed close record lands — never a fabricated rate. Same ONE registered claim as the session-proxy provider-completion lever (#4506); no second cell",
	Metrics: []dojoMetricInfo{
		{Name: "verified_completion_rate", Theory: "about half of the issues dispatched to a provider are actually closed VERIFIED — diff-witnessed closes / dispatched, per provider, from the reconcile ledger (claim 0.5 — the one registered provider-completion estimate; UNMEASURED per provider until the closure-audit row attributes closes per provider)"},
	},
}, func(env dojoLeverEnv) dojo.Lever { return providerCompletionLedgerLever{root: env.Root} })

// providerCompletionLedgerLever folds the workspace loop ledger into the
// provider-completion cell. root is the workspace root the loop ledger lives
// under. The scenario corpus is ignored: the reconcile ledger, not a transcript
// replay, is the ground truth for dispatched-issue closure.
type providerCompletionLedgerLever struct{ root string }

func (providerCompletionLedgerLever) Name() string { return "provider-completion-ledger" }

func (l providerCompletionLedgerLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerCompletionLedgerEpisodes(providerCompletionLedgerFacts(loadLoopLedgerEvents(l.root))), nil
}

// providerDispatchCompletion is one provider's reconcile-ledger completion
// facts, reduced to what the fold needs. ClosesAttributed is the honesty bit
// that separates a genuine "closed nothing" from "the ledger cannot attribute
// closes to this provider": false forces UNMEASURED so today's aggregate-only
// closed_now can never be read as a fabricated per-provider rate (#4492; the
// same missing-outcome-record gap as provider-firsttry #4494).
type providerDispatchCompletion struct {
	// Provider keys the leaderboard row — the dispatch backend that ran the
	// worker, through the shared leaderboard keying (#4505).
	Provider string
	// Dispatched is how many workers the ledger witnessed SPAWNED for this
	// provider — the denominator, always a real ledger fact.
	Dispatched int
	// VerifiedClosed is how many of those dispatched issues the closure auditor
	// reconciled as diff-witnessed VERIFIED closes. Meaningful only when
	// ClosesAttributed is true.
	VerifiedClosed int
	// ClosesAttributed reports whether the ledger attributes VERIFIED closes to
	// this provider at all. False today: closed_now is a per-tick aggregate.
	ClosesAttributed bool
}

// providerCompletionLedgerFacts reduces loop-ledger events to per-provider
// completion facts, sorted by provider for a deterministic fold. Dispatched =
// EventStart rows with reason SPAWNED (one per spawned worker — the same
// population dispatch-yield #4497 counts), keyed by the row's Principal (the
// dispatch backend) through the shared leaderboard keying; an unkeyed row
// (harness-synthetic principal) is skipped, never fabricated into a provider.
// The closure auditor's closed_now end rows are a per-tick AGGREGATE with no
// per-issue provider attribution, so every row carries ClosesAttributed=false
// and zero closes — the fold then scores UNMEASURED naming that seam. It is
// pure over the event slice so it is unit-testable without a ledger on disk.
func providerCompletionLedgerFacts(events []loopmgr.Event) []providerDispatchCompletion {
	dispatched := map[string]int{}
	for _, ev := range events {
		if ev.Kind != loopmgr.EventStart || ev.Reason != "SPAWNED" {
			continue // only a SPAWNED start row is a dispatched worker
		}
		p := providerTurnsKey(ev.Principal) // the shared leaderboard keying (#4505)
		if p == "" {
			continue // no billed provider to key the leaderboard row by
		}
		dispatched[p]++
	}
	providers := make([]string, 0, len(dispatched))
	for p := range dispatched {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]providerDispatchCompletion, 0, len(providers))
	for _, p := range providers {
		out = append(out, providerDispatchCompletion{
			Provider:   p,
			Dispatched: dispatched[p],
			// closed_now on closure-audit end rows is a per-tick aggregate with no
			// per-provider attribution (#4492): closes stay unattributed and the
			// fold scores UNMEASURED rather than fabricating a per-provider rate.
			ClosesAttributed: false,
		})
	}
	return out
}

// providerCompletionLedgerEpisodes folds per-provider reconcile-ledger facts
// into the dojo's (prediction, outcome) pairs for
// provider-completion/verified_completion_rate — one episode PER PROVIDER,
// every episode scored against the SAME registered claim so the recalibrate arm
// still rewrites exactly one literal while the report renders the per-provider
// spread. It is pure and total so the fold is unit-testable without a ledger on
// disk.
//
// Honesty rules (#4492): a ledger with no dispatched (keyed) worker at all
// yields ONE honest UNMEASURED episode, never a fabricated rate. A provider
// whose closes the ledger cannot attribute (ClosesAttributed=false — every
// provider today) scores UNMEASURED carrying its witnessed dispatched count as
// the sample and naming the missing provider-attributed close record. Once the
// ledger attributes closes, the rate is VerifiedClosed/Dispatched, WITNESSED
// (the closure auditor's diff-witnessed reconcile, not a session proxy) — and a
// provider whose every dispatch went unclosed measures a real 0.0, distinct
// from unattributed.
func providerCompletionLedgerEpisodes(rows []providerDispatchCompletion) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-completion", "verified_completion_rate", "fraction")
	out := make([]dojo.ScoredInput, 0, len(rows))
	for _, r := range rows {
		if r.Provider == "" || r.Dispatched < 1 {
			continue // not a dispatched leaderboard row — nothing to score
		}
		if !r.ClosesAttributed {
			out = append(out, dojo.ScoredInput{
				Prediction: pred,
				Outcome: dojo.Outcome{
					Measured: false,
					Sample:   r.Dispatched,
					Source: fmt.Sprintf("provider %s: %d dispatched worker(s) witnessed on the reconcile ledger (SPAWNED rows), but the closure auditor's closed_now is a per-tick aggregate with no per-provider close attribution — verified_completion_rate from the reconcile ledger is UNMEASURED, not a fabricated rate; the extension seam is a provider-attributed VERIFIED-close record on the closure-audit row (#4492)",
						r.Provider, r.Dispatched),
				},
			})
			continue
		}
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   float64(r.VerifiedClosed) / float64(r.Dispatched),
				Provenance: dojo.Witnessed,
				Measured:   true,
				Sample:     r.Dispatched,
				Source: fmt.Sprintf("provider %s: %d diff-witnessed VERIFIED close(s) / %d dispatched worker(s) over the reconcile ledger (WITNESSED — the closure auditor's reconcile, not a session proxy)",
					r.Provider, r.VerifiedClosed, r.Dispatched),
			},
		})
	}
	if len(out) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no SPAWNED start rows in the reconcile ledger — nothing dispatched in this window to fold a per-provider verified completion rate from",
			},
		}}
	}
	return out
}
