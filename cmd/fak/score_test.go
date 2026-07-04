package main

import (
	"sort"
	"testing"
)

// TestScoreRoutesCoverTheMetaVerbs pins the `fak score <name>` parent (#1505) to exactly the ten
// meta-scorecard / RSI subcommands it consolidated. If a route is dropped or an eleventh is added
// without updating this list, the test reds -- the parent's surface is a contract, not incidental.
func TestScoreRoutesCoverTheMetaVerbs(t *testing.T) {
	want := []string{
		"conflation",
		"dogfood",
		"dojo-rsi",
		"guard-rsi",
		"guard-verdict-rsi",
		"product",
		"skill-effectiveness",
		"support-maturity",
		"token-defaults",
		"ui-quality",
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
