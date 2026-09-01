package main

import (
	"sort"
	"testing"
)

// TestScoreRoutesCoverTheMetaVerbs pins the `fak score <name>` parent (#1505) to exactly the
// meta-scorecard / RSI subcommands it consolidated. If a route is dropped or a new one is added
// without updating this list, the test reds -- the parent's surface is a contract, not incidental.
//
// The list is reconciled against scoreRoutes as of the flow wiring (#6198): "catchup" left the
// runtime in 0f5ec3f674 (#6022) and "lightgap" arrived in 70d61594c0 (#6348), and neither move
// updated this pin, so the contract had been red on two stale entries before "flow" was added.
func TestScoreRoutesCoverTheMetaVerbs(t *testing.T) {
	want := []string{
		"agent-readiness",
		"brittleness",
		"cache-health",
		"cachevalue-gate",
		"code-quality",
		"conflation",
		"concept-usage",
		"default-value",
		"dogfood",
		"dojo-rsi",
		"flow",
		"focus",
		"generalization",
		"guard-accuracy",
		"guard-rsi",
		"guard-verdict-rsi",
		"issue-hygiene",
		"lightgap",
		"performance-rsi",
		"loop",
		"loop-index",
		"milestone",
		"negation-tax",
		"negation_operator",
		"negframe",
		"product",
		"propagation",
		"qa-process",
		"repo-hygiene",
		"seo",
		"skill-effectiveness",
		"sota-coverage",
		"support-maturity",
		"token-defaults",
		"ui-quality",
		"verifier-exposure",
	}
	got := make([]string, 0, len(scoreRoutes))
	for name, route := range scoreRoutes {
		if route == nil {
			t.Errorf("score route %q is nil -- every subcommand must forward to a handler", name)
		}
		got = append(got, name)
	}
	sort.Strings(got)
	sort.Strings(want)
	if len(got) != len(want) {
		t.Fatalf("scoreRoutes has %d routes, want %d: got %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("scoreRoutes = %v, want %v", got, want)
		}
	}
}

// TestScoreRoutesCoverEveryLegacyScorecardVerb pins the #2247 consolidation: every legacy
// top-level *-scorecard/*-score verb named there routes under `fak score <fold>`, where the
// fold is the legacy verb minus its -scorecard/-score suffix. A scorecard verb reachable only
// at the root is the defect this contract exists to catch -- the namespace, not the root, is
// where scorecards land.
func TestScoreRoutesCoverEveryLegacyScorecardVerb(t *testing.T) {
	legacyToFold := map[string]string{
		"concept-usage-score":           "concept-usage",
		"conflation-scorecard":          "conflation",
		"dogfood-score":                 "dogfood",
		"guard-rsi-scorecard":           "guard-rsi",
		"loop-index-scorecard":          "loop-index",
		"loop-score":                    "loop",
		"milestone-scorecard":           "milestone",
		"performance-rsi-scorecard":     "performance-rsi",
		"product-scorecard":             "product",
		"propagation-scorecard":         "propagation",
		"repo-hygiene-scorecard":        "repo-hygiene",
		"skill-effectiveness-scorecard": "skill-effectiveness",
		"sota-coverage-scorecard":       "sota-coverage",
		"support-maturity-scorecard":    "support-maturity",
		"token-defaults-scorecard":      "token-defaults",
		"ui-quality-scorecard":          "ui-quality",
	}
	for legacy, fold := range legacyToFold {
		route, ok := scoreRoutes[fold]
		if !ok {
			t.Errorf("legacy verb %q has no `fak score %s` route -- the #2247 consolidation contract", legacy, fold)
			continue
		}
		if route == nil {
			t.Errorf("`fak score %s` (legacy %q) routes to a nil handler", fold, legacy)
		}
	}
}
