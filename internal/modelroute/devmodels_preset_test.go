package modelroute

import (
	"os"
	"testing"
)

// TestDevModelsPresetDecisions is the SEMANTIC witness for the dev-models routing
// preset (examples/routing-presets/dev-models.json). TestRoutingPresetsRoundTrip
// already guards the file against rot (loads + byte-exact round trip); this test
// binds the JSON to the three-way auto-decision it exists to encode — the choice,
// per classified dev aspect, among the three dev routes:
//
//   - fable          — the fast/cheap tier (T2): read/search tool calls, low-
//     complexity aspects, short interactive turns.
//   - opus-4.8       — the balanced strong coder (T1): ordinary implementation and
//     the fail-closed default for unmatched normal work.
//   - opus-ultracode — Opus 4.8 in exhaustive multi-agent orchestration (T0): the
//     hardest work AND the security-release floor (destructive / high-risk aspects),
//     which never drops to a cheap model however small the call looks.
//
// Without this test the preset is data-only: a future edit could silently re-route a
// dev aspect to the wrong tier — most dangerously drop a destructive tool call to
// fable, breaking the security-release floor tierpolicy.go fixes for T0 work — and
// the round-trip test would still pass. Each case asserts BOTH the matched rule and
// the routed primary, so an accidental reorder that changes which rule fires is
// caught even when it happens to route to the same model.
func TestDevModelsPresetDecisions(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/dev-models.json")
	if err != nil {
		t.Fatalf("read dev-models preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse dev-models preset: %v", err)
	}

	cases := []struct {
		name    string
		subject Subject
		primary string // expected routed model id
		rule    string // expected matched rule name ("" == fail-closed default)
	}{
		{
			name:    "ultra-hard aspect escalates to ultracode",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			primary: "opus-ultracode",
			rule:    "dev-ultrahard-ultracode",
		},
		{
			name:    "destructive tool call -> ultracode (security-release floor)",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_branch"},
			primary: "opus-ultracode",
			rule:    "dev-destructive-ultracode",
		},
		{
			// The floor must beat cheap signals: a tiny, low-complexity, interactive
			// delete still routes to ultracode, never fable.
			name:    "tiny low-complexity delete still routes to ultracode",
			subject: Subject{Aspect: AspectToolCall, Tool: "delete_file", Complexity: ComplexityLow, Latency: LatencyInteractive},
			primary: "opus-ultracode",
			rule:    "dev-destructive-ultracode",
		},
		{
			name:    "risk=high label -> ultracode",
			subject: Subject{Aspect: AspectRequest, Labels: map[string]string{"risk": "high"}},
			primary: "opus-ultracode",
			rule:    "dev-high-risk-ultracode",
		},
		{
			// A high-flagged read escalates: the ultra-hard rule precedes the read rule.
			name:    "high-complexity read escalates over the read->fable rule",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file", Complexity: ComplexityHigh},
			primary: "opus-ultracode",
			rule:    "dev-ultrahard-ultracode",
		},
		{
			name:    "read-shaped tool call -> fable",
			subject: Subject{Aspect: AspectToolCall, Tool: "read_file"},
			primary: "fable",
			rule:    "dev-read-fable",
		},
		{
			name:    "search tool call -> fable",
			subject: Subject{Aspect: AspectToolCall, Tool: "search_kb"},
			primary: "fable",
			rule:    "dev-search-fable",
		},
		{
			name:    "low-complexity aspect -> fable",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityLow},
			primary: "fable",
			rule:    "dev-cheap-fable",
		},
		{
			name:    "short interactive turn -> fable",
			subject: Subject{Aspect: AspectRequest, PromptTokens: 512, Latency: LatencyInteractive},
			primary: "fable",
			rule:    "dev-short-interactive-fable",
		},
		{
			// Medium work must land on the T1 model, NOT fall through to low->fable.
			name:    "medium-complexity work -> opus-4.8 (named T1 band)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityMedium},
			primary: "opus-4.8",
			rule:    "dev-normal-opus",
		},
		{
			// Unmatched normal work hits the fail-closed default.
			name:    "unmatched work -> opus-4.8 default",
			subject: Subject{Aspect: AspectQuery},
			primary: "opus-4.8",
			rule:    "",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			d := m.Route(tc.subject)
			if got := d.Plan.Primary(); got != tc.primary {
				t.Errorf("primary = %q, want %q (matched rule %q)", got, tc.primary, d.RuleName)
			}
			if d.RuleName != tc.rule {
				t.Errorf("rule = %q, want %q", d.RuleName, tc.rule)
			}
		})
	}
}
