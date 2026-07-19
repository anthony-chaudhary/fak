package main

import (
	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// dojo_lever_token_economy.go — the token-economy/tokens_saved_ratio lever (#4487),
// registered through the additive RegisterLever seam (#5108) so the cell lands in
// its own file with no edit to cmd/fak/dojo.go. The pure fold + the one anchored
// claim live in internal/dojo (claim_token_economy.go); this file is only the
// registration plus the thin session-corpus adapter the tier-1 core must not do
// itself.
//
// The lever scores the ONE overall tokens-saved ratio — fak-on vs a no-lever
// baseline over input-side billed tokens. It reads the local multi-provider session
// corpus (the same corpus the provider-* leaderboard cells fold) to sum the ON side,
// but an ordinary session corpus records only fak-ON billing — the no-lever OFF
// counterfactual is never run — so the cell scores UNMEASURED honestly, naming the
// missing paired baseline, until a paired on/off corpus lands. The moment such a
// baseline exists, the same pure fold measures the ratio. `fak dojo run` folds the
// cell like any other registered lever.
var _ = RegisterLever(dojoLeverInfo{
	Name:    "token-economy",
	Summary: "the ONE overall tokens-saved ratio — fak-on vs a no-lever baseline over input-side billed tokens (input + cache_read + cache_creation), the single headline 'what does fak save' number, folded from a PAIRED on/off measurement (never summed per lever, which double-counts overlapping savings). The no-lever OFF baseline is not billed in an ordinary session corpus (the same missing counterfactual as compaction #953), so the cell scores UNMEASURED honestly until a paired baseline lands, never a fabricated ratio (#4487)",
	Metrics: []dojoMetricInfo{
		{Name: "tokens_saved_ratio", Theory: "fak's token-economy levers together save about 30% of the input-side tokens a no-lever baseline would bill (claim 0.30 — a seeded estimate the RSI loop recalibrates toward the measured paired ratio; UNMEASURED until a paired OFF baseline exists)"},
	},
}, func(dojoLeverEnv) dojo.Lever { return tokenEconomyLever{} })

// tokenEconomyLever folds the local session corpus into the overall tokens-saved
// ratio cell. It carries no state: the session corpus, not the scenario replay, is
// the ground truth for billed tokens (the same choice the provider-* leaderboard
// levers make).
type tokenEconomyLever struct{}

func (tokenEconomyLever) Name() string { return "token-economy" }

func (tokenEconomyLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return dojo.TokenEconomyEpisodes(tokenEconomyBilledOnCorpus(loadProviderTurnsSessions())), nil
}

// tokenEconomyBilledOnCorpus reduces the local multi-provider session corpus to
// the paired token corpus the fold needs. It sums the ON-side input-side billed
// tokens (input + cache_read + cache_creation over every billed provider row, using
// the shared provider keying so harness-synthetic rows are skipped), but sets
// BaselinePaired=false: an ordinary session corpus records only fak-ON billing,
// never the no-lever OFF counterfactual, so the fold scores UNMEASURED naming the
// missing baseline rather than fabricating a saving (#4487; the same gap as
// compaction #953). It is pure over the session slice so it is unit-testable
// without a corpus on disk.
func tokenEconomyBilledOnCorpus(sessions []sessionaudit.Session) dojo.PairedTokenCorpus {
	var on int64
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		for model, pm := range s.PerModel {
			if providerTurnsKey(model) == "" {
				continue // harness-synthetic (non-billed) model — no billed tokens to fold
			}
			on += pm.Input + pm.CacheRead + pm.CacheCreate
		}
	}
	return dojo.PairedTokenCorpus{
		OnTokens: uint64(on),
		// No no-lever OFF baseline is billed in an ordinary session corpus (the
		// counterfactual is never run), so the paired baseline is absent and the fold
		// scores UNMEASURED rather than fabricating a saving.
		BaselinePaired: false,
	}
}
