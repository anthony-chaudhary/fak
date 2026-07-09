package modelroute

import (
	"os"
	"testing"
)

// TestPMFablePresetDecisions is the SEMANTIC witness for the pm-fable routing preset
// (examples/routing-presets/pm-fable.json). TestRoutingPresetsRoundTrip already guards
// the file against rot (loads + byte-exact round trip); this test binds the JSON to the
// two-way auto-decision it exists to encode — the choice, per classified
// project-management aspect, between:
//
//   - fable    — the fast/cheap tier: the steady-state PM pass (triage, next-up
//     ranking, milestone scoring, status rollups, backlog dedup) and the fail-closed
//     default, since a wrong triage costs a re-run, not a bad production write.
//   - opus-4.8 — the strong model: genuinely-hard planning judgment (epic
//     decomposition, cross-cutting replan, milestone re-scope) AND the irreversible
//     close floor (contraction: close/cancel an issue, milestone, or plan), which
//     never drops to the cheap tier however routine the close looks.
//
// This is the concrete-tier sibling of gardening.json (which routes the same class of
// coordination work but with abstract model ids): pm-fable pins project management to
// the real fable/opus-4.8 tiers so "project-management work runs on fable" is a
// witnessed policy, not a comment.
//
// Without this test the preset is data-only: a future edit could silently drop the
// irreversible-close floor to fable, or route the whole PM lane to the expensive tier,
// and the round-trip test would still pass. Each case asserts BOTH the matched rule and
// the routed primary, so an accidental reorder that changes which rule fires is caught
// even when it happens to route to the same model.
func TestPMFablePresetDecisions(t *testing.T) {
	raw, err := os.ReadFile("../../examples/routing-presets/pm-fable.json")
	if err != nil {
		t.Fatalf("read pm-fable preset: %v", err)
	}
	m, err := ParseManifest(raw)
	if err != nil {
		t.Fatalf("parse pm-fable preset: %v", err)
	}

	pm := func(extra map[string]string) map[string]string {
		l := map[string]string{"work_kind": "project_management"}
		for k, v := range extra {
			l[k] = v
		}
		return l
	}

	cases := []struct {
		name    string
		subject Subject
		primary string // expected routed model id
		rule    string // expected matched rule name ("" == fail-closed default)
	}{
		{
			name:    "routine PM aspect -> fable",
			subject: Subject{Aspect: AspectStep, Labels: pm(nil)},
			primary: "fable",
			rule:    "pm-routine-fable",
		},
		{
			// A low-complexity PM aspect is still routine work -> fable.
			name:    "low-complexity PM aspect -> fable (routine, not default)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityLow, Labels: pm(nil)},
			primary: "fable",
			rule:    "pm-routine-fable",
		},
		{
			// Hard planning judgment escalates to the strong model.
			name:    "high-complexity PM judgment -> opus-4.8",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh, Labels: pm(nil)},
			primary: "opus-4.8",
			rule:    "pm-hard-judgment-opus",
		},
		{
			// An irreversible close (contraction) verifies on the strong model.
			name:    "irreversible close -> opus-4.8 (verify floor)",
			subject: Subject{Aspect: AspectState, Labels: pm(map[string]string{"move": "contraction"})},
			primary: "opus-4.8",
			rule:    "pm-irreversible-close-verify",
		},
		{
			// The close floor beats the complexity signal: a routine-looking (even
			// low-complexity) contraction still routes to opus-4.8, never fable.
			name:    "low-complexity contraction still verifies on opus-4.8",
			subject: Subject{Aspect: AspectState, Complexity: ComplexityLow, Labels: pm(map[string]string{"move": "contraction"})},
			primary: "opus-4.8",
			rule:    "pm-irreversible-close-verify",
		},
		{
			// The close rule precedes the hard-judgment rule: a high-complexity
			// contraction still matches the close rule, not the hard-judgment rule.
			name:    "high-complexity contraction fires the close rule, not hard-judgment",
			subject: Subject{Aspect: AspectState, Complexity: ComplexityHigh, Labels: pm(map[string]string{"move": "contraction"})},
			primary: "opus-4.8",
			rule:    "pm-irreversible-close-verify",
		},
		{
			// The label gate: hard work WITHOUT the project_management label does not
			// match any PM rule and hits the fail-closed default (fable).
			name:    "high-complexity non-PM work -> fable default (label gate)",
			subject: Subject{Aspect: AspectStep, Complexity: ComplexityHigh},
			primary: "fable",
			rule:    "",
		},
		{
			// Unlabeled ordinary work hits the fail-closed default.
			name:    "unmatched work -> fable default",
			subject: Subject{Aspect: AspectQuery},
			primary: "fable",
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
