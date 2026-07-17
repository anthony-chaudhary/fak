package main

import (
	"sort"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

// providerTurnsLever is the first cross-provider leaderboard cell (#4505):
// median turns to completion PER PROVIDER, so the dojo can state where fak's
// routed providers sit against each other on this axis instead of calibrating
// only fak-internal levers. Theory is the one registered
// provider-turns/turns_per_task claim; reality folds from the local
// multi-provider session corpus (the transcripts under ~/.claude*/projects),
// one episode per provider keyed by the session's dominant billed model. The
// scenario corpus is ignored: the session corpus, not a replay, is the ground
// truth for turns to completion.
type providerTurnsLever struct{}

func (providerTurnsLever) Name() string { return "provider-turns" }

func (providerTurnsLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerTurnsEpisodesFromSessions(loadProviderTurnsSessions()), nil
}

// providerTurnsSinceDays bounds the corpus fold to a recent window so the
// median reflects current provider behavior (and the scan stays proportionate).
const providerTurnsSinceDays = 30.0

// loadProviderTurnsSessions discovers and analyzes the local session corpus the
// provider-turns fold reads. Subagent transcripts are excluded (a subagent run
// is a delegated slice of its parent task, not a completed task of its own).
// Fail-open: a missing or unreadable corpus yields nil and the fold reports
// itself UNMEASURED, never a fabricated median.
func loadProviderTurnsSessions() []sessionaudit.Session {
	since := providerTurnsSinceDays
	recs, err := sessionaudit.Discover(sessionaudit.DiscoverOptions{SinceDays: &since})
	if err != nil {
		return nil
	}
	sessions := make([]sessionaudit.Session, 0, len(recs))
	for _, rec := range recs {
		sessions = append(sessions, sessionaudit.Analyze(rec.Path))
	}
	return sessions
}

// providerTurnsEpisodesFromSessions adapts the session corpus into the dojo's
// (prediction, outcome) pairs for provider-turns/turns_per_task — one episode
// per provider, every episode scored against the SAME registered claim so the
// recalibrate arm still rewrites exactly one literal while the report renders
// the per-provider spread. It is pure so the fold is unit-testable without a
// corpus on disk. A completed task = a readable session with at least one
// assistant turn; its turn count is the session's assistant turns and its
// provider is the dominant billed model's provider key. A corpus with no
// completed session yields one honest UNMEASURED episode.
func providerTurnsEpisodesFromSessions(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-turns", "turns_per_task", "turns")
	byProvider := map[string][]float64{}
	for _, s := range sessions {
		if s.Error != "" || s.AssistantTurns == 0 {
			continue
		}
		provider := providerTurnsDominantProvider(s)
		if provider == "" {
			continue // only harness-synthetic turns — no billed provider to key by
		}
		byProvider[provider] = append(byProvider[provider], float64(s.AssistantTurns))
	}
	if len(byProvider) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no completed sessions in the local session corpus — nothing to fold a per-provider turn median from",
			},
		}}
	}
	providers := make([]string, 0, len(byProvider))
	for p := range byProvider {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]dojo.ScoredInput, 0, len(providers))
	for _, p := range providers {
		turns := byProvider[p]
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   providerTurnsMedian(turns),
				Provenance: dojo.Observed,
				Measured:   true,
				Sample:     len(turns),
				Source:     "provider " + p + ": median assistant turns per completed session over the local multi-provider corpus (OBSERVED)",
			},
		})
	}
	return out
}

// providerTurnsDominantProvider keys a session by the provider that carried it:
// the provider key with the most billed assistant turns across the session's
// per-model counts (ties break lexicographically for determinism). Empty when
// no turn maps to a billable provider.
func providerTurnsDominantProvider(s sessionaudit.Session) string {
	turnsByProvider := map[string]int64{}
	for model, pm := range s.PerModel {
		key := providerTurnsKey(model)
		if key == "" {
			continue
		}
		turnsByProvider[key] += pm.Turns
	}
	best, bestTurns := "", int64(0)
	for p, n := range turnsByProvider {
		if n > bestTurns || (n == bestTurns && best != "" && p < best) {
			best, bestTurns = p, n
		}
	}
	return best
}

// providerTurnsKey maps a billed model name to its stable leaderboard provider
// key — the six providers the cell compares (#4505) plus "other" so an
// unlisted-but-billed model still folds honestly instead of vanishing. Keys
// stay stable so the recalibrate arm keeps rewriting the one literal while the
// per-provider episodes stay comparable across ticks. Empty for a
// harness-synthetic (non-billed) model.
func providerTurnsKey(model string) string {
	m := strings.ToLower(model)
	if m == "" || m == "?" || m == "<synthetic>" {
		return ""
	}
	for _, b := range []struct {
		key  string
		subs []string
	}{
		{"claude", []string{"claude", "opus", "sonnet", "haiku", "fable"}},
		{"gpt", []string{"gpt", "o1-", "o3-", "o4-", "codex", "davinci"}},
		{"gemini", []string{"gemini", "gemma"}},
		{"deepseek", []string{"deepseek"}},
		{"glm", []string{"glm"}},
		{"kimi", []string{"kimi", "moonshot"}},
	} {
		for _, sub := range b.subs {
			if strings.Contains(m, sub) {
				return b.key
			}
		}
	}
	return "other"
}

// providerTurnsMedian is the median of a non-empty sample — the central
// tendency the cell claims, robust to the long-session tail an agentic corpus
// always carries (a mean would let one 300-turn run swamp a provider's column).
func providerTurnsMedian(values []float64) float64 {
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	mid := len(sorted) / 2
	if len(sorted)%2 == 1 {
		return sorted[mid]
	}
	return (sorted[mid-1] + sorted[mid]) / 2
}

// --- the provider-cache lever -------------------------------------------------

// providerCacheLever is the cross-provider cache-economy cell (#4504): the share
// of a provider's total billed input tokens served as cache reads, PER PROVIDER,
// so the dojo can state where fak sits versus other providers on cache economy
// instead of calibrating only fak-internal cache levers. Theory is the one
// registered provider-cache/cache_read_share claim; reality folds from the same
// local multi-provider session corpus (and the same provider keying) as the
// provider-turns leaderboard cell, one episode per provider. The scenario corpus
// is ignored: the session corpus, not a replay, is the ground truth for billed
// cache share.
type providerCacheLever struct{}

func (providerCacheLever) Name() string { return "provider-cache" }

func (providerCacheLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerCacheEpisodesFromSessions(loadProviderTurnsSessions()), nil
}

// providerCacheEpisodesFromSessions adapts the session corpus into the dojo's
// (prediction, outcome) pairs for provider-cache/cache_read_share — one episode
// per provider, every episode scored against the SAME registered claim so the
// recalibrate arm still rewrites exactly one literal while the report renders
// the per-provider spread. It is pure so the fold is unit-testable without a
// corpus on disk. A provider's share is billed cache_read / (input + cache_read
// + cache_creation) summed over its billed turns — cache reads as a fraction of
// every input token the provider priced. A provider whose rows carry NO cache
// fields at all (neither cache_read nor cache_creation ever billed) scores
// UNMEASURED, never a fabricated 0.0 share: this transcript shape cannot tell
// "billed cold everywhere" apart from "the provider relays no cache-read field"
// (#4490's honesty rule; cache_creation>0 proves the field family is relayed, so
// a real all-cold window still measures as 0.0). A corpus with no billed
// provider at all yields one honest UNMEASURED episode.
func providerCacheEpisodesFromSessions(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-cache", "cache_read_share", "fraction")
	sums := map[string]sessionaudit.ModelCounts{}
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		for model, pm := range s.PerModel {
			key := providerTurnsKey(model) // the shared leaderboard keying (#4505)
			if key == "" {
				continue
			}
			agg := sums[key]
			agg.Turns += pm.Turns
			agg.Input += pm.Input
			agg.CacheRead += pm.CacheRead
			agg.CacheCreate += pm.CacheCreate
			sums[key] = agg
		}
	}
	if len(sums) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no billed provider rows in the local session corpus — nothing to fold a per-provider cache-read share from",
			},
		}}
	}
	providers := make([]string, 0, len(sums))
	for p := range sums {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]dojo.ScoredInput, 0, len(providers))
	for _, p := range providers {
		agg := sums[p]
		if agg.CacheRead == 0 && agg.CacheCreate == 0 {
			out = append(out, dojo.ScoredInput{
				Prediction: pred,
				Outcome: dojo.Outcome{
					Measured: false,
					Sample:   int(agg.Turns),
					Source:   "provider " + p + ": billed turns but no cache_read/cache_creation fields observed — the provider relays no cache billing here, so the share is UNMEASURED rather than a fabricated 0.0",
				},
			})
			continue
		}
		total := agg.Input + agg.CacheRead + agg.CacheCreate
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   float64(agg.CacheRead) / float64(total),
				Provenance: dojo.Observed,
				Measured:   true,
				Sample:     int(agg.Turns),
				Source:     "provider " + p + ": billed cache_read / (input + cache_read + cache_creation) over the local multi-provider corpus (OBSERVED)",
			},
		})
	}
	return out
}

// cacheReadShareLever is the WITNESSED top-line cache-read fraction cell
// (#4498/#4484): ONE overall number — billed cache_read as a share of every
// input-side token across the whole local multi-provider session corpus — so the
// dojo carries a single headline cache-economy figure alongside the per-provider
// provider-cache leaderboard (#4504). Theory is the one registered
// cache-read-share/billed_cache_read_share claim; reality folds from the same
// session corpus (and the same provider keying) as the provider-cache cell,
// aggregated corpus-wide instead of split per provider. The scenario corpus is
// ignored: the session corpus, not a replay, is the ground truth for billed cache
// share.
type cacheReadShareLever struct{}

func (cacheReadShareLever) Name() string { return "cache-read-share" }

func (cacheReadShareLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return cacheReadShareEpisodeFromSessions(loadProviderTurnsSessions()), nil
}

// cacheReadShareEpisodeFromSessions folds the WITNESSED top-line cache-read
// fraction (#4498/#4484): ONE overall episode summing billed cache_read over
// (input + cache_read + cache_creation) across EVERY billed provider row in the
// local session corpus. Where provider-cache/cache_read_share (#4504) splits the
// same tokens per provider into a leaderboard, this is their single aggregate —
// the one number an operator cites for cache economy — scored against the one
// registered cache-read-share claim the RSI loop recalibrates. Pure, so the fold
// is unit-testable without a corpus on disk. Same honesty rule as provider-cache:
// a corpus whose rows carry NO cache fields at all (neither cache_read nor
// cache_creation ever billed) scores UNMEASURED, never a fabricated 0.0 — that
// transcript shape cannot tell "billed cold everywhere" apart from "the provider
// relays no cache-read field" (cache_creation>0 proves the field family is
// relayed, so a real all-cold corpus still measures 0.0). An empty corpus is one
// honest UNMEASURED episode.
func cacheReadShareEpisodeFromSessions(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("cache-read-share", "billed_cache_read_share", "fraction")
	var agg sessionaudit.ModelCounts
	for _, s := range sessions {
		if s.Error != "" {
			continue
		}
		for model, pm := range s.PerModel {
			if providerTurnsKey(model) == "" {
				continue // harness-synthetic (non-billed) model — no billed tokens to fold
			}
			agg.Turns += pm.Turns
			agg.Input += pm.Input
			agg.CacheRead += pm.CacheRead
			agg.CacheCreate += pm.CacheCreate
		}
	}
	if agg.CacheRead == 0 && agg.CacheCreate == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Sample:   int(agg.Turns),
				Source:   "no cache_read/cache_creation fields across the local session corpus — the top-line cache-read share is UNMEASURED rather than a fabricated 0.0",
			},
		}}
	}
	total := agg.Input + agg.CacheRead + agg.CacheCreate
	return []dojo.ScoredInput{{
		Prediction: pred,
		Outcome: dojo.Outcome{
			Realized:   float64(agg.CacheRead) / float64(total),
			Provenance: dojo.Observed,
			Measured:   true,
			Sample:     int(agg.Turns),
			Source:     "billed cache_read / (input + cache_read + cache_creation) summed across EVERY provider row in the local session corpus — the WITNESSED top-line cache-read fraction (OBSERVED)",
		},
	}}
}

// --- the provider-cost lever --------------------------------------------------

// providerCostLever is the cross-provider economics cell (#4488): the billed USD a
// provider spends per completed issue (cost_per_completed_issue), PER PROVIDER, so
// the dojo can state where fak sits versus other providers on the generic
// cost-to-close KPI instead of calibrating only fak-internal levers. Theory is the
// one registered provider-cost/cost_per_completed_issue claim; reality folds from
// the same local multi-provider session corpus (and the same provider keying) as
// the provider-turns and provider-cache leaderboard cells, one episode per
// provider. The scenario corpus is ignored: the session corpus, not a replay, is
// the ground truth for billed cost.
type providerCostLever struct{}

func (providerCostLever) Name() string { return "provider-cost" }

func (providerCostLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerCostEpisodesFromSessions(loadProviderTurnsSessions()), nil
}

// providerCostEpisodesFromSessions adapts the session corpus into the dojo's
// (prediction, outcome) pairs for provider-cost/cost_per_completed_issue — one
// episode per provider, every episode scored against the SAME registered claim so
// the recalibrate arm still rewrites exactly one literal while the report renders
// the per-provider spread. It is pure so the fold is unit-testable without a corpus
// on disk. A completed session = a readable session (Error=="") with at least one
// assistant turn, keyed to its dominant billed provider — the same completed-task
// unit provider-turns (#4505) folds, used here as the corpus proxy for a verified
// close. A provider's cost_per_completed_issue is its billed USD (the EXISTING
// sessionaudit per-model price table, NOT a new one) summed over ITS OWN billed
// models across its completed sessions, divided by that provider's completed-session
// count; billing only the provider's own models keeps a stray cross-provider turn
// from contaminating (or fabricating) a provider's cost. A provider whose completed
// sessions carry NO priced billing — every one of its models is unpriced in the
// sessionaudit table, so its billed USD is 0 — scores UNMEASURED, never a fabricated
// $0.00 per issue: fak having no price for a provider is not the same as a genuinely
// free close (#4490's honesty rule). A corpus with no completed session at all
// yields one honest UNMEASURED episode.
func providerCostEpisodesFromSessions(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-cost", "cost_per_completed_issue", "usd")
	type acc struct {
		count int
		usd   float64
	}
	sums := map[string]*acc{}
	for _, s := range sessions {
		if s.Error != "" || s.AssistantTurns == 0 {
			continue
		}
		provider := providerTurnsDominantProvider(s)
		if provider == "" {
			continue // only harness-synthetic turns — no billed provider to key by
		}
		// Bill only the dominant provider's OWN models, so a stray cross-provider
		// turn never contaminates this provider's cost (and never fabricates a
		// nonzero cost for an otherwise-unpriced provider).
		var usd float64
		for model, pm := range s.PerModel {
			if providerTurnsKey(model) != provider {
				continue
			}
			usd += sessionaudit.ModelCost(model, pm)
		}
		a := sums[provider]
		if a == nil {
			a = &acc{}
			sums[provider] = a
		}
		a.count++
		a.usd += usd
	}
	if len(sums) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no completed sessions in the local session corpus — nothing to fold a per-provider cost per completed issue from",
			},
		}}
	}
	providers := make([]string, 0, len(sums))
	for p := range sums {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]dojo.ScoredInput, 0, len(providers))
	for _, p := range providers {
		a := sums[p]
		if a.usd == 0 {
			out = append(out, dojo.ScoredInput{
				Prediction: pred,
				Outcome: dojo.Outcome{
					Measured: false,
					Sample:   a.count,
					Source:   "provider " + p + ": completed sessions but no priced billing — the sessionaudit price table has no entry for this provider's models, so cost per completed issue is UNMEASURED rather than a fabricated $0.00",
				},
			})
			continue
		}
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   a.usd / float64(a.count),
				Provenance: dojo.Observed,
				Measured:   true,
				Sample:     a.count,
				Source:     "provider " + p + ": billed USD (sessionaudit per-model price table) per completed session over the local multi-provider corpus (OBSERVED)",
			},
		})
	}
	return out
}

// --- the provider-tokens lever ------------------------------------------------

// providerTokensLever is the cross-provider tokens-to-close cell (#4503): the
// TOTAL billed tokens a provider spends per completed issue
// (tokens_per_completed_issue), PER PROVIDER, so the dojo can state where fak
// sits versus other providers on total token spend to close instead of
// calibrating only fak-internal levers. Theory is the one registered
// provider-tokens/tokens_per_completed_issue claim; reality folds from the same
// local multi-provider session corpus (and the same provider keying) as the
// provider-turns, provider-cache, and provider-cost leaderboard cells, one
// episode per provider. The scenario corpus is ignored: the session corpus, not
// a replay, is the ground truth for billed tokens.
type providerTokensLever struct{}

func (providerTokensLever) Name() string { return "provider-tokens" }

func (providerTokensLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerTokensEpisodesFromSessions(loadProviderTurnsSessions()), nil
}

// providerTokensEpisodesFromSessions adapts the session corpus into the dojo's
// (prediction, outcome) pairs for provider-tokens/tokens_per_completed_issue —
// one episode per provider, every episode scored against the SAME registered
// claim so the recalibrate arm still rewrites exactly one literal while the
// report renders the per-provider spread. It is pure so the fold is
// unit-testable without a corpus on disk. A completed session = a readable
// session (Error=="") with at least one assistant turn, keyed to its dominant
// billed provider — the same completed-task unit provider-turns (#4505) and
// provider-cost (#4488) fold, used here as the corpus proxy for a verified
// close. A provider's tokens_per_completed_issue is its TOTAL billed tokens
// (input + output + cache_read + cache_creation) summed over ITS OWN billed
// models across its completed sessions, divided by that provider's
// completed-session count (the mean); counting only the provider's own models
// keeps a stray cross-provider turn from contaminating a provider's total. A
// provider whose completed sessions carry NO billed tokens at all scores
// UNMEASURED, never a fabricated 0 tokens per issue: an empty corpus row is not
// the same as a genuinely token-free close (#4490's honesty rule). A corpus with
// no completed session at all yields one honest UNMEASURED episode.
func providerTokensEpisodesFromSessions(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-tokens", "tokens_per_completed_issue", "tokens")
	type acc struct {
		count  int
		tokens int64
	}
	sums := map[string]*acc{}
	for _, s := range sessions {
		if s.Error != "" || s.AssistantTurns == 0 {
			continue
		}
		provider := providerTurnsDominantProvider(s)
		if provider == "" {
			continue // only harness-synthetic turns — no billed provider to key by
		}
		// Count only the dominant provider's OWN models, so a stray cross-provider
		// turn never contaminates this provider's token total.
		var tokens int64
		for model, pm := range s.PerModel {
			if providerTurnsKey(model) != provider {
				continue
			}
			tokens += pm.Input + pm.Output + pm.CacheRead + pm.CacheCreate
		}
		a := sums[provider]
		if a == nil {
			a = &acc{}
			sums[provider] = a
		}
		a.count++
		a.tokens += tokens
	}
	if len(sums) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no completed sessions in the local session corpus — nothing to fold a per-provider total tokens per completed issue from",
			},
		}}
	}
	providers := make([]string, 0, len(sums))
	for p := range sums {
		providers = append(providers, p)
	}
	sort.Strings(providers)
	out := make([]dojo.ScoredInput, 0, len(providers))
	for _, p := range providers {
		a := sums[p]
		if a.tokens == 0 {
			out = append(out, dojo.ScoredInput{
				Prediction: pred,
				Outcome: dojo.Outcome{
					Measured: false,
					Sample:   a.count,
					Source:   "provider " + p + ": completed sessions but no billed tokens — the corpus rows carry no token counts for this provider's models, so tokens per completed issue is UNMEASURED rather than a fabricated 0",
				},
			})
			continue
		}
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   float64(a.tokens) / float64(a.count),
				Provenance: dojo.Observed,
				Measured:   true,
				Sample:     a.count,
				Source:     "provider " + p + ": total billed tokens (input + output + cache_read + cache_creation) per completed session over the local multi-provider corpus (OBSERVED)",
			},
		})
	}
	return out
}

// --- output + durable ledger I/O -------------------------------------------
