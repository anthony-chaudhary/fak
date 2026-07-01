package scorecardpane

import (
	"strings"
	"testing"
)

// contextstatus_test.go — issue #1577 witness: `go test ./internal/scorecardpane
// ./cmd/fak -run ContextStatus` (and `-run Info` on the cmd/fak side, per the issue's
// witness). Proves RenderContextStatusLine reflects EVERY named signal — not just
// severity — and that each severity tier renders a visibly distinct line.

// TestRenderContextStatusLineAllSeverityTiers proves the status line leads with the
// distinct severity word for each closed tier, so a caller scanning the
// line can tell the severities apart at a glance (the issue's "concise status
// line" done condition).
func TestRenderContextStatusLineAllSeverityTiers(t *testing.T) {
	tiers := []ContextHealthSeverity{
		ContextHealthFresh, ContextHealthConstrained, ContextHealthQueryNeeded,
		ContextHealthStaleRisk, ContextHealthBudgetDraining,
		ContextHealthObjectiveDrift, ContextHealthResetImminent,
	}
	seen := map[string]bool{}
	for _, sev := range tiers {
		t.Run(string(sev), func(t *testing.T) {
			line := RenderContextStatusLine(ContextStatusSignals{Severity: sev})
			if !strings.Contains(line, "context: "+string(sev)) {
				t.Fatalf("line %q does not lead with severity %q", line, sev)
			}
			if seen[line] {
				t.Fatalf("severity %q rendered a line already seen for another tier: %q", sev, line)
			}
			seen[line] = true
		})
	}
}

// TestRenderContextStatusLineReflectsEveryCounter proves each of the issue's six named
// fields (resident tokens, budget left, cache state, reset count, query-needed count,
// stale assumption count) actually shows up in the rendered line — not just severity —
// so the line cannot silently drop a signal the issue asked for.
func TestRenderContextStatusLineReflectsEveryCounter(t *testing.T) {
	s := ContextStatusSignals{
		Severity:             ContextHealthConstrained,
		ResidentTokens:       12345,
		BudgetTokens:         20000,
		CacheState:           "stable",
		ResetCount:           3,
		QueryNeededCount:     7,
		StaleAssumptionCount: 2,
	}
	line := RenderContextStatusLine(s)

	wantSubstrings := []string{
		"context: constrained",
		"resident 12,345 tok",
		"budget left 7,655 tok", // 20000 - 12345
		"cache stable",
		"resets 3",
		"query-needed 7",
		"stale 2",
	}
	for _, want := range wantSubstrings {
		if !strings.Contains(line, want) {
			t.Errorf("line %q missing expected substring %q", line, want)
		}
	}
}

// TestRenderContextStatusLineZeroValueIsWellFormed proves the zero-value input still
// renders a complete, well-formed line rather than panicking or leaving fields blank —
// severity renders as ContextHealthSeverity's own fail-safe "(unset)", and cache state
// falls back to "unknown" when unreported.
func TestRenderContextStatusLineZeroValueIsWellFormed(t *testing.T) {
	line := RenderContextStatusLine(ContextStatusSignals{})
	want := "context: (unset) · resident 0 tok · budget left 0 tok · cache unknown · resets 0 · query-needed 0 · stale 0"
	if line != want {
		t.Fatalf("RenderContextStatusLine(zero value) = %q, want %q", line, want)
	}
}

// TestRenderContextStatusLineUnknownSeverityFailsSafe proves a foreign/garbage
// severity value renders via ContextHealthSeverity's own "unknown(...)" fail-safe
// instead of being silently accepted as a real tier — the same closed-vocabulary
// discipline contexthealth_test.go proves for ClassifyContextHealth's callers.
func TestRenderContextStatusLineUnknownSeverityFailsSafe(t *testing.T) {
	line := RenderContextStatusLine(ContextStatusSignals{Severity: ContextHealthSeverity("on_fire")})
	if !strings.Contains(line, "context: unknown(on_fire)") {
		t.Fatalf("line %q does not fail safe on a bogus severity", line)
	}
}

// TestContextStatusSignalsBudgetLeftTokensFloorsAtZero proves BudgetLeftTokens never
// goes negative when a session has resident tokens past its declared budget (an
// over-budget session), matching the "cache: not saving yet" convention elsewhere in
// this package of never printing a confusing signed/negative figure where a floor
// reads more plainly.
func TestContextStatusSignalsBudgetLeftTokensFloorsAtZero(t *testing.T) {
	s := ContextStatusSignals{ResidentTokens: 5000, BudgetTokens: 4000}
	if got := s.BudgetLeftTokens(); got != 0 {
		t.Fatalf("BudgetLeftTokens() over-budget = %d, want 0 (floored)", got)
	}

	s2 := ContextStatusSignals{ResidentTokens: 1000, BudgetTokens: 4000}
	if got := s2.BudgetLeftTokens(); got != 3000 {
		t.Fatalf("BudgetLeftTokens() = %d, want 3000", got)
	}
}

// TestRenderContextStatusLineCacheStateTrimsAndDefaults proves CacheState is
// trimmed of surrounding whitespace and falls back to "unknown" when blank, so a
// caller that has not wired a live cache tracker yet renders an honest "unknown"
// rather than an empty gap in the line.
func TestRenderContextStatusLineCacheStateTrimsAndDefaults(t *testing.T) {
	cases := []struct {
		name  string
		state string
		want  string
	}{
		{"blank", "", "cache unknown"},
		{"whitespace-only", "   ", "cache unknown"},
		{"padded", "  stable  ", "cache stable"},
		{"verbatim", "mutated", "cache mutated"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			line := RenderContextStatusLine(ContextStatusSignals{CacheState: tc.state})
			if !strings.Contains(line, tc.want) {
				t.Errorf("CacheState %q: line %q missing %q", tc.state, line, tc.want)
			}
		})
	}
}

// TestGroupThousandsCSMatchesGroupingConvention proves the local thousands-grouping
// helper renders identically to the family convention (cmd/fak/info.go's
// groupThousands) for representative magnitudes, including the negative-floors-to-zero
// guard (this package never renders a negative token count).
func TestGroupThousandsCSMatchesGroupingConvention(t *testing.T) {
	cases := []struct {
		in   int
		want string
	}{
		{0, "0"},
		{7, "7"},
		{999, "999"},
		{1000, "1,000"},
		{12345, "12,345"},
		{1234567, "1,234,567"},
		{-5, "0"},
	}
	for _, tc := range cases {
		if got := groupThousandsCS(tc.in); got != tc.want {
			t.Errorf("groupThousandsCS(%d) = %q, want %q", tc.in, got, tc.want)
		}
	}
}
