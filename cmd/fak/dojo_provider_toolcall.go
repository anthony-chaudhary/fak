package main

// The provider-toolcall lever (#4493/#4507): first-try tool-call success rate
// PER PROVIDER, folded from the local multi-provider session corpus — the
// cross-provider tool-use-reliability cell (BFCL/tau-bench-style, reframed as a
// calibrated dojo cell scored from the fleet's own tool-call ledger). Theory is
// the one registered provider-toolcall/tool_call_success_rate claim; reality
// folds from the same session corpus (and the same provider keying) as the
// provider-turns/provider-cache/provider-cost/provider-tokens leaderboard
// cells, one episode per provider. The lever + catalog row register through the
// additive RegisterLever seam (#5108) so this cell lands in its own file with
// no edit to the central allDojoLevers / dojoLeverCatalogBase literals. The
// scenario corpus is ignored: the session corpus, not a replay, is the ground
// truth for adjudicated tool calls.

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/sessionaudit"
)

var _ = RegisterLever(dojoLeverInfo{
	Name:    "provider-toolcall",
	Summary: "first-try tool-call success rate per provider (claude/gpt/gemini/deepseek/glm/kimi), folded from the local multi-provider session corpus — the cross-provider tool-use-reliability leaderboard cell, one episode per provider against the one registered claim; a provider whose sessions never called a tool emits no episode, and an empty corpus is UNMEASURED, never a fabricated rate (#4493)",
	Metrics: []dojoMetricInfo{
		{Name: "tool_call_success_rate", Theory: "~90% of a provider's tool calls succeed on the first try — non-errored tool_result over total tool_result across its sessions (claim 0.9 — a seeded estimate the RSI loop recalibrates toward the measured per-provider rates)"},
	},
}, func(dojoLeverEnv) dojo.Lever { return providerToolcallLever{} })

// providerToolcallLever adapts the session corpus into the dojo's
// provider-toolcall/tool_call_success_rate episodes for one run.
type providerToolcallLever struct{}

func (providerToolcallLever) Name() string { return "provider-toolcall" }

func (providerToolcallLever) Episodes(dojo.Scenario) ([]dojo.ScoredInput, error) {
	return providerToolcallEpisodesFromCorpus(loadProviderTurnsSessions()), nil
}

// providerToolcallEpisodesFromCorpus adapts the session corpus into the
// dojo's (prediction, outcome) pairs for provider-toolcall/tool_call_success_rate
// — one episode per provider, every episode scored against the SAME registered
// claim so the recalibrate arm still rewrites exactly one literal while the
// report renders the per-provider spread. It is pure so the fold is
// unit-testable without a corpus on disk.
//
// A provider's rate is (adjudicated tool results - errored tool results) /
// adjudicated tool results, summed over its readable sessions: every
// tool_result the transcript carries is one adjudicated first-try call, and
// every Behavior.ToolErrors entry (including results the transcript never
// attributed to a tool_use, keyed "?") is one first-try failure. Sessions key
// to their dominant billed provider — the same leaderboard keying as
// provider-turns (#4505) — so a harness-synthetic row never decides the
// provider. A provider whose sessions never called a tool emits NO episode:
// with no adjudicated call there is no rate to fold, and a fabricated 1.0
// (or 0.0) would poison the leaderboard. A corpus with no tool-calling billed
// provider at all yields one honest UNMEASURED episode, never a fabricated
// rate (#4490's honesty rule).
func providerToolcallEpisodesFromCorpus(sessions []sessionaudit.Session) []dojo.ScoredInput {
	pred := dojo.Registry.MustPredict("provider-toolcall", "tool_call_success_rate", "fraction")
	type acc struct {
		results int64
		errors  int64
	}
	sums := map[string]*acc{}
	for _, s := range sessions {
		if s.Error != "" {
			continue // an unreadable session never folds
		}
		if s.NToolResult == 0 {
			continue // no adjudicated call — nothing to fold a rate from
		}
		provider := providerTurnsDominantProvider(s)
		if provider == "" {
			continue // only harness-synthetic turns — no billed provider to key by
		}
		a := sums[provider]
		if a == nil {
			a = &acc{}
			sums[provider] = a
		}
		a.results += s.NToolResult
		for _, n := range s.Behavior.ToolErrors {
			a.errors += n
		}
	}
	if len(sums) == 0 {
		return []dojo.ScoredInput{{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Measured: false,
				Source:   "no tool-calling billed sessions in the local session corpus — nothing to fold a per-provider first-try tool-call success rate from",
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
		succeeded := a.results - a.errors
		if succeeded < 0 {
			// More errored results than attributed results (a truncated
			// transcript tail); clamp at zero rather than a negative rate.
			succeeded = 0
		}
		out = append(out, dojo.ScoredInput{
			Prediction: pred,
			Outcome: dojo.Outcome{
				Realized:   float64(succeeded) / float64(a.results),
				Provenance: dojo.Observed,
				Measured:   true,
				Sample:     int(a.results),
				Source:     "provider " + p + ": non-errored tool_result / total tool_result over the local multi-provider corpus (OBSERVED)",
			},
		})
	}
	return out
}
