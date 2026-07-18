package main

import "github.com/anthony-chaudhary/fak/internal/dojo"

// The provider-tokens lever registered through the additive RegisterLever seam
// (#5108) — the worked example the seam ships with: an existing provider lever
// ported off the central allDojoLevers / dojoLeverCatalogBase literals, so its
// lever + catalog row live in a file of their own and a future KPI-cell leaf
// can copy this shape without ever editing cmd/fak/dojo.go. The lever's fold
// logic (providerTokensLever, #4503) stays in dojo_provider_levers.go and its
// claim stays the one anchored provider-tokens/tokens_per_completed_issue
// literal in internal/dojo — this file is only the registration.
var _ = RegisterLever(dojoLeverInfo{
	Name:    "provider-tokens",
	Summary: "total billed tokens per completed issue per provider (claude/gpt/gemini/deepseek/glm/kimi), folded from the local multi-provider session corpus — the cross-provider tokens-to-close leaderboard cell, one episode per provider against the one registered claim; a provider with no billed tokens is UNMEASURED, never a fabricated 0 (#4503)",
	Metrics: []dojoMetricInfo{
		{Name: "tokens_per_completed_issue", Theory: "a provider bills about one million total tokens (input + output + cache_read + cache_creation) per completed issue, the mean total billed tokens per completed session (claim 1000000 — a seeded estimate the RSI loop recalibrates toward the measured per-provider means)"},
	},
}, func(dojoLeverEnv) dojo.Lever { return providerTokensLever{} })
