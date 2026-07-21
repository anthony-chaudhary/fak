package policy

// org_precedence_test.go is the failing-first, executable spec for the org-policy
// precedence lattice researched in
// docs/notes/RESEARCH-org-policy-precedence-2026-07-20.md (R3 / #5318, epic #5315,
// grandparent #5170). It enumerates the per-amendment-class truth tables — inputs
// (compiled-in, central, operator) -> resolved verdict — for the RATCHET,
// GATED_WIDEN and FROZEN classes, plus the two load-bearing questions ("can central
// raise a cap the operator lowered?" / "can the operator widen past a central grant?",
// both expected: NO).
//
// It is green-by-skip today: every case is guarded by t.Skip("pending #5322 org
// precedence fold") BEFORE it asserts, so the package stays green while the real fold
// does not yet exist. It is failing-first because the placeholder resolveOrgPrecedence
// returns a sentinel that no expected row matches — the moment #5322 removes the Skip
// guards and points the cases at the real fold, the enumerated verdicts become the
// acceptance test.
//
// Deliberately self-contained: standard `testing` only, and its own local vocabulary
// (orgPrec* identifiers) so it compiles at trunk HEAD independent of the in-flight
// amendment.go registry. The string values mirror amendment.go's AmendmentClass
// vocabulary (FROZEN / RATCHET / GATED_WIDEN) so the two stay legible together.

import "testing"

// orgPrecClass mirrors internal/policy AmendmentClass values (kept local so this
// stub compiles without the in-flight amendment.go registry).
type orgPrecClass string

const (
	orgPrecRatchet    orgPrecClass = "RATCHET"
	orgPrecGatedWiden orgPrecClass = "GATED_WIDEN"
	orgPrecFrozen     orgPrecClass = "FROZEN"
)

// orgPrecContribution is one channel's contribution to a knob in a case row.
//
//   - For RATCHET / FROZEN boolean knobs: Deny reports whether the channel adds a
//     refusal; WidenAttempt reports a (forbidden) attempt to loosen.
//   - For GATED_WIDEN cap knobs: Cap is the ceiling the channel sets, and Set marks
//     whether the channel contributed a value at all (Set=false means "inherit").
type orgPrecContribution struct {
	// boolean-knob fields (RATCHET / FROZEN)
	Deny         bool
	WidenAttempt bool
	// cap-knob fields (GATED_WIDEN)
	Set bool
	Cap int
}

// orgPrecVerdict is the resolved output vocabulary.
type orgPrecVerdict string

const (
	orgPrecAllow orgPrecVerdict = "ALLOW"
	orgPrecDeny  orgPrecVerdict = "DENY"
	// orgPrecUnimplemented is what the placeholder fold returns until #5322 wires
	// the real resolution — it matches no expected row, so removing the skip fails.
	orgPrecUnimplemented orgPrecVerdict = "UNIMPLEMENTED"
)

// orgPrecResolved is the resolved posture: a boolean Verdict for RATCHET/FROZEN and
// a numeric Cap for GATED_WIDEN. Cap == -1 is the unimplemented sentinel.
type orgPrecResolved struct {
	Verdict orgPrecVerdict
	Cap     int
}

// orgPrecInput is a full case: the three reachable channel contributions plus the
// knob's amendment class. agent-self is closed (SELF_AMENDABLE empty today) and so
// contributes nothing — it is intentionally absent from the inputs.
type orgPrecInput struct {
	Class      orgPrecClass
	CompiledIn orgPrecContribution
	Central    orgPrecContribution
	Operator   orgPrecContribution
}

// resolveOrgPrecedence is the PLACEHOLDER fold. #5322 replaces its body with a call
// into the real precedence resolver (union/max-restriction for RATCHET floors, min
// for GATED_WIDEN ceilings, identity for FROZEN), folding
// {compiled-in > central > operator > agent-self}. Until then it returns the
// unimplemented sentinel so every enumerated case below is failing-first.
func resolveOrgPrecedence(_ orgPrecInput) orgPrecResolved {
	return orgPrecResolved{Verdict: orgPrecUnimplemented, Cap: -1}
}

// deny/allow/attempt constructors keep the case tables readable.
func orgPrecAllowC() orgPrecContribution    { return orgPrecContribution{} }
func orgPrecDenyC() orgPrecContribution     { return orgPrecContribution{Deny: true} }
func orgPrecWidenC() orgPrecContribution    { return orgPrecContribution{WidenAttempt: true} }
func orgPrecCapC(n int) orgPrecContribution { return orgPrecContribution{Set: true, Cap: n} }
func orgPrecInheritC() orgPrecContribution  { return orgPrecContribution{} }

// TestOrgPrecedenceRatchet enumerates §2.1 of the research note: a RATCHET boolean
// knob resolves DENY iff ANY channel denies; no channel may un-deny a peer's refusal.
func TestOrgPrecedenceRatchet(t *testing.T) {
	cases := []struct {
		name string
		in   orgPrecInput
		want orgPrecVerdict
	}{
		{"none-restricts", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecAllowC(), Operator: orgPrecAllowC()}, orgPrecAllow},
		{"central-tightens", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecDenyC(), Operator: orgPrecAllowC()}, orgPrecDeny},
		{"operator-tightens", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecAllowC(), Operator: orgPrecDenyC()}, orgPrecDeny},
		{"both-tighten", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecDenyC(), Operator: orgPrecDenyC()}, orgPrecDeny},
		{"operator-cannot-undeny-central", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecDenyC(), Operator: orgPrecWidenC()}, orgPrecDeny},
		{"central-widen-noop-nothing-to-loosen", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecAllowC(), Central: orgPrecWidenC(), Operator: orgPrecAllowC()}, orgPrecAllow},
		{"compiled-floor-denies", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecDenyC(), Central: orgPrecAllowC(), Operator: orgPrecAllowC()}, orgPrecDeny},
		{"no-channel-weakens-floor", orgPrecInput{Class: orgPrecRatchet, CompiledIn: orgPrecDenyC(), Central: orgPrecWidenC(), Operator: orgPrecWidenC()}, orgPrecDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Skip("pending #5322 org precedence fold")
			got := resolveOrgPrecedence(tc.in)
			if got.Verdict != tc.want {
				t.Fatalf("RATCHET %s: got verdict %q, want %q", tc.name, got.Verdict, tc.want)
			}
		})
	}
}

// TestOrgPrecedenceGatedWiden enumerates §2.2 of the research note: a GATED_WIDEN cap
// resolves to min(compiled-in, central, operator) with each set value clamped to the
// channel above it — a widen is a ceiling that only ever descends.
func TestOrgPrecedenceGatedWiden(t *testing.T) {
	cases := []struct {
		name string
		in   orgPrecInput
		want int
	}{
		{"no-widen-sits-at-frozen-cap", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecInheritC(), Operator: orgPrecInheritC()}, 200},
		{"central-grants-ceiling-below-cap", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecCapC(100), Operator: orgPrecInheritC()}, 100},
		{"operator-tightens-below-central", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecCapC(100), Operator: orgPrecCapC(50)}, 50},
		{"operator-cannot-widen-past-central", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecCapC(100), Operator: orgPrecCapC(150)}, 100},
		{"central-cannot-exceed-frozen-cap", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecCapC(300), Operator: orgPrecInheritC()}, 200},
		{"operator-tightens-with-no-central-grant", orgPrecInput{Class: orgPrecGatedWiden, CompiledIn: orgPrecCapC(200), Central: orgPrecInheritC(), Operator: orgPrecCapC(50)}, 50},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Skip("pending #5322 org precedence fold")
			got := resolveOrgPrecedence(tc.in)
			if got.Cap != tc.want {
				t.Fatalf("GATED_WIDEN %s: got cap %d, want %d", tc.name, got.Cap, tc.want)
			}
		})
	}
}

// TestOrgPrecedenceFrozen enumerates §2.3: a FROZEN knob resolves to its compiled-in
// value regardless of any central or operator attempt — no channel moves it.
func TestOrgPrecedenceFrozen(t *testing.T) {
	cases := []struct {
		name string
		in   orgPrecInput
		want orgPrecVerdict
	}{
		{"floor-untouched", orgPrecInput{Class: orgPrecFrozen, CompiledIn: orgPrecDenyC(), Central: orgPrecAllowC(), Operator: orgPrecAllowC()}, orgPrecDeny},
		{"central-tighten-is-noop-on-floor", orgPrecInput{Class: orgPrecFrozen, CompiledIn: orgPrecDenyC(), Central: orgPrecDenyC(), Operator: orgPrecAllowC()}, orgPrecDeny},
		{"central-cannot-weaken-frozen", orgPrecInput{Class: orgPrecFrozen, CompiledIn: orgPrecDenyC(), Central: orgPrecWidenC(), Operator: orgPrecAllowC()}, orgPrecDeny},
		{"operator-cannot-weaken-frozen", orgPrecInput{Class: orgPrecFrozen, CompiledIn: orgPrecDenyC(), Central: orgPrecAllowC(), Operator: orgPrecWidenC()}, orgPrecDeny},
		{"no-combination-moves-frozen", orgPrecInput{Class: orgPrecFrozen, CompiledIn: orgPrecDenyC(), Central: orgPrecWidenC(), Operator: orgPrecWidenC()}, orgPrecDeny},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Skip("pending #5322 org precedence fold")
			got := resolveOrgPrecedence(tc.in)
			if got.Verdict != tc.want {
				t.Fatalf("FROZEN %s: got verdict %q, want %q", tc.name, got.Verdict, tc.want)
			}
		})
	}
}

// TestOrgPrecedenceCentralCannotRaiseOperatorLoweredCap is Q1 of the research note
// (§3): can central raise a cap the operator lowered? Expected: NO. The operator
// lowered the cap to 50; central then grants 180 (> 50). min-wins keeps 50.
func TestOrgPrecedenceCentralCannotRaiseOperatorLoweredCap(t *testing.T) {
	t.Skip("pending #5322 org precedence fold")
	in := orgPrecInput{
		Class:      orgPrecGatedWiden,
		CompiledIn: orgPrecCapC(200),
		Central:    orgPrecCapC(180),
		Operator:   orgPrecCapC(50),
	}
	got := resolveOrgPrecedence(in)
	if got.Cap != 50 {
		t.Fatalf("central must NOT raise the operator-lowered cap: got %d, want 50", got.Cap)
	}
}

// TestOrgPrecedenceOperatorCannotWidenPastCentral is Q2 of the research note (§3):
// can the operator widen past a central grant? Expected: NO. Operator asks for 150
// over a central grant of 100; it clamps to 100.
func TestOrgPrecedenceOperatorCannotWidenPastCentral(t *testing.T) {
	t.Skip("pending #5322 org precedence fold")
	in := orgPrecInput{
		Class:      orgPrecGatedWiden,
		CompiledIn: orgPrecCapC(200),
		Central:    orgPrecCapC(100),
		Operator:   orgPrecCapC(150),
	}
	got := resolveOrgPrecedence(in)
	if got.Cap != 100 {
		t.Fatalf("operator must NOT widen past the central grant: got %d, want 100", got.Cap)
	}
}
