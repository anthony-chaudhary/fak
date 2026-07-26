// amendment_matrix_test.go is the red-team amendment matrix (#5174, epic #5170,
// Track A). Where amendment_test.go proves EVERY exported adjudicator.Policy
// field is CLASSIFIED (coverage), this file proves the classification is SAFE:
// it walks the full (class × direction × channel) matrix of PolicyKnobRegistry
// and asserts the load-bearing invariant of the whole epic — no amendment
// channel can weaken a FROZEN knob, and no knob may move against its declared
// direction. The registry is the single source of truth the guard consults to
// decide who may amend a surface; if a future edit mis-declares a knob (a
// FROZEN surface that quietly lists an operator channel, a RATCHET knob marked
// widen-capable), that is exactly the self-amendment escape this epic exists to
// forbid, and this test fails loudly rather than letting it ship.
package policy

import "testing"

// mutableChannels is the closed set of channels through which a running system
// (operator or agent) can change policy AFTER the binary ships. ChannelCompiledIn
// is deliberately excluded: changing a compiled-in default requires a code change
// + release, which is the FROZEN contract, not a runtime amendment.
var mutableChannels = map[string]bool{
	ChannelOperatorOverlay:    true,
	ChannelLiveReload:         true,
	ChannelOperatorEscalation: true,
	// ChannelCentral (the org policy plane, epic #5315) is a runtime amendment
	// channel like the operator gates: it changes the effective floor AFTER the
	// binary ships (a signed manifest pulled from a company endpoint), so a
	// FROZEN knob listing it would be exactly the weakening path this matrix
	// forbids.
	ChannelCentral: true,
}

// TestAmendmentMatrixNoChannelWeakensFrozen is the core red-team assertion: for
// every knob in the registry, the (class, direction, channels) triple must be
// mutually consistent AND must not encode a weakening path the class forbids.
func TestAmendmentMatrixNoChannelWeakensFrozen(t *testing.T) {
	if len(PolicyKnobRegistry) == 0 {
		t.Fatal("PolicyKnobRegistry is empty — the amendment matrix has nothing to protect")
	}
	for i, k := range PolicyKnobRegistry {
		label := k.Field
		if label == "" {
			label = "<non-field floor element>"
		}
		switch k.Class {
		case AmendFrozen:
			// The whole point of FROZEN: no runtime channel may move it. Its
			// direction must be frozen and its ONLY channel compiled-in.
			if k.Direction != DirectionFrozen {
				t.Errorf("knob %d (%s): FROZEN knob must be DirectionFrozen, got %q", i, label, k.Direction)
			}
			if len(k.Channels) != 1 || k.Channels[0] != ChannelCompiledIn {
				t.Errorf("knob %d (%s): FROZEN knob must be reachable ONLY via %q, got %v — this is a self-amendment escape",
					i, label, ChannelCompiledIn, k.Channels)
			}
			for _, ch := range k.Channels {
				if mutableChannels[ch] {
					t.Errorf("knob %d (%s): FROZEN knob lists runtime-mutable channel %q — a channel that could WEAKEN a frozen floor",
						i, label, ch)
				}
			}
		case AmendRatchet:
			// RATCHET may be touched at runtime, but only ever to TIGHTEN.
			if k.Direction != DirectionTightenOnly {
				t.Errorf("knob %d (%s): RATCHET knob must be DirectionTightenOnly (never widen), got %q", i, label, k.Direction)
			}
			if len(k.Channels) == 0 {
				t.Errorf("knob %d (%s): RATCHET knob declares no channel — an unreachable knob is a classification bug", i, label)
			}
		case AmendGatedWiden:
			// GATED_WIDEN is the only class that may loosen — but only through a
			// GATED operator channel, never on the agent's own authority. It must
			// widen-only (its zero value is already the tightest posture) and
			// every channel must be a real operator/reload gate.
			if k.Direction != DirectionWidenOnly {
				t.Errorf("knob %d (%s): GATED_WIDEN knob must be DirectionWidenOnly, got %q", i, label, k.Direction)
			}
			if len(k.Channels) == 0 {
				t.Errorf("knob %d (%s): GATED_WIDEN knob declares no channel", i, label)
			}
			for _, ch := range k.Channels {
				if !mutableChannels[ch] {
					t.Errorf("knob %d (%s): GATED_WIDEN channel %q is not a recognized gated operator channel", i, label, ch)
				}
			}
		case AmendSelfAmendable:
			// The declared-but-empty agent-writable frontier. If a knob EVER
			// lands here it means the agent can move policy on its own authority
			// — the exact thing the epic forbids today — so this must stay empty
			// until a deliberate, reviewed design lands it.
			t.Errorf("knob %d (%s): SELF_AMENDABLE is the agent-writable frontier and must be empty today; a knob here is an un-reviewed self-amendment grant", i, label)
		default:
			t.Errorf("knob %d (%s): unknown amendment class %q — not in the closed vocabulary", i, label, k.Class)
		}
	}
}

// TestAmendmentMatrixFrozenSetIsNonEmpty guards against a refactor that quietly
// drops the compiled-in FROZEN floor from the registry (which would make
// TestAmendmentMatrixNoChannelWeakensFrozen vacuously pass for the FROZEN case).
func TestAmendmentMatrixFrozenSetIsNonEmpty(t *testing.T) {
	if got := len(KnobsByClass(AmendFrozen)); got == 0 {
		t.Fatal("no FROZEN knobs in the registry — the compiled-in floor is unenumerated")
	}
}

// TestAmendmentMatrixSelfAmendableEmpty pins the epic's headline posture: the
// agent-writable frontier is declared and EMPTY. A separate named test so the
// intent is legible in a failure report.
func TestAmendmentMatrixSelfAmendableEmpty(t *testing.T) {
	if got := KnobsByClass(AmendSelfAmendable); len(got) != 0 {
		t.Fatalf("SELF_AMENDABLE must be empty today; found %d self-amendable knob(s)", len(got))
	}
}

// knobHasChannel reports whether a knob authorizes the named amendment channel.
func knobHasChannel(k PolicyKnob, channel string) bool {
	for _, ch := range k.Channels {
		if ch == channel {
			return true
		}
	}
	return false
}

// TestAmendmentMatrixCentralChannel is the org-plane (epic #5315) red-team
// slice of the matrix. It proves, against the same single source of truth, the
// three central-channel invariants #5319 must hold:
//
//  1. central may RATCHET a RATCHET knob — every RATCHET knob authorizes it, so
//     the org plane can tighten the floor fleet-wide.
//  2. central may raise a GATED_WIDEN knob but only up to its FROZEN cap — every
//     GATED_WIDEN knob authorizes it, yet (invariant 3) no FROZEN knob does, so
//     the compiled-in floor still caps every central grant.
//  3. central may NOT move any FROZEN knob — no FROZEN knob lists it (FROZEN is
//     compiled-in only), so the org plane can never weaken the core floor.
func TestAmendmentMatrixCentralChannel(t *testing.T) {
	var ratchetSaw, gatedSaw, frozenSaw int
	for i, k := range PolicyKnobRegistry {
		label := k.Field
		if label == "" {
			label = "<non-field floor element>"
		}
		has := knobHasChannel(k, ChannelCentral)
		switch k.Class {
		case AmendRatchet:
			ratchetSaw++
			if !has {
				t.Errorf("knob %d (%s): RATCHET knob must authorize %q so the org plane can tighten it fleet-wide",
					i, label, ChannelCentral)
			}
		case AmendGatedWiden:
			gatedSaw++
			if !has {
				t.Errorf("knob %d (%s): GATED_WIDEN knob must authorize %q so the org plane can raise it (up to the FROZEN cap)",
					i, label, ChannelCentral)
			}
		case AmendFrozen:
			frozenSaw++
			if has {
				t.Errorf("knob %d (%s): FROZEN knob authorizes %q — the org plane must NEVER move a frozen floor; this is the exact weakening path #5319 forbids",
					i, label, ChannelCentral)
			}
		}
	}
	// Guard against a vacuous pass: each arm must have actually seen a knob, so
	// a refactor that empties a class can't make the invariant trivially true.
	if ratchetSaw == 0 {
		t.Error("no RATCHET knobs seen — the central-RATCHET invariant proved nothing")
	}
	if gatedSaw == 0 {
		t.Error("no GATED_WIDEN knobs seen — the central-widen invariant proved nothing")
	}
	if frozenSaw == 0 {
		t.Error("no FROZEN knobs seen — the central-cannot-move-frozen invariant proved nothing")
	}
}

// TestAmendmentMatrixEveryClassHandled ensures the switch in the red-team test
// above covers the entire closed AmendmentClass vocabulary — if a new class is
// added to amendment.go, this fails until the matrix test is taught about it.
func TestAmendmentMatrixEveryClassHandled(t *testing.T) {
	known := map[AmendmentClass]bool{
		AmendFrozen:        true,
		AmendRatchet:       true,
		AmendGatedWiden:    true,
		AmendSelfAmendable: true,
	}
	for _, k := range PolicyKnobRegistry {
		if !known[k.Class] {
			t.Errorf("registry knob has class %q not handled by the amendment matrix test", k.Class)
		}
	}
}
