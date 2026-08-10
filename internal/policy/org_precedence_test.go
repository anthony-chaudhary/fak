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
// It was green-by-skip until #5322: every case was guarded by a t.Skip BEFORE it
// asserted, so the package stayed green while the real fold did not exist. #5322
// landed orgprecedence.go, deleted the skips, and pointed resolveOrgPrecedence at
// ResolveOrgPrecedence — so these enumerated verdicts are now the acceptance test
// rather than a promise. If you are changing the lattice, this file is the contract
// you are changing.
//
// The local vocabulary (orgPrec* identifiers) is deliberately KEPT rather than
// rewritten against the production types. The spec should be able to disagree with
// the implementation: if these rows were written in terms of OrgPrecedenceInput and
// OrgAmendVerdict directly, then a rename or a semantic drift in those types would
// silently carry the spec along with it. The adapter below is the single, obvious
// place the two vocabularies meet. The string values mirror amendment.go's
// AmendmentClass vocabulary (FROZEN / RATCHET / GATED_WIDEN) so the two stay legible
// together.

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
)

// orgPrecResolved is the resolved posture: a boolean Verdict for RATCHET/FROZEN and
// a numeric Cap for GATED_WIDEN.
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

// resolveOrgPrecedence adapts a spec row onto the real fold in orgprecedence.go —
// union/max-restriction for RATCHET floors, min for GATED_WIDEN ceilings, identity
// for FROZEN, folding {compiled-in > central > operator > agent-self}.
//
// It translates and asserts nothing. Every judgement below is the production
// resolver's; this function only carries the row across the vocabulary boundary, so
// a failing case is always a real disagreement with the lattice rather than an
// artifact of the harness.
func resolveOrgPrecedence(in orgPrecInput) orgPrecResolved {
	got := ResolveOrgPrecedence(OrgPrecedenceInput{
		Class:      AmendmentClass(in.Class),
		CompiledIn: orgPrecReal(in.CompiledIn),
		Central:    orgPrecReal(in.Central),
		Operator:   orgPrecReal(in.Operator),
	})
	return orgPrecResolved{Verdict: orgPrecVerdict(got.Verdict), Cap: got.Cap}
}

// orgPrecReal is the per-channel half of that translation. The two structs are
// field-identical today and are still written out by hand: a field added to one and
// not the other must break the build here, not quietly resolve to a zero value in a
// security lattice.
func orgPrecReal(c orgPrecContribution) OrgKnobContribution {
	return OrgKnobContribution{Deny: c.Deny, WidenAttempt: c.WidenAttempt, Set: c.Set, Cap: c.Cap}
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
