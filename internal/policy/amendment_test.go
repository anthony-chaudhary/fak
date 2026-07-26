package policy

// The reflection conformance gate for the PolicyKnob amendment-class registry
// (#5171): every exported adjudicator.Policy field must carry exactly one
// registry entry, so adding a knob without declaring how it may be amended
// fails the build gate instead of silently re-scattering the discipline.

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
)

// TestEveryPolicyKnobHasAmendmentClass reflects over adjudicator.Policy and
// fails if any exported field is absent from PolicyKnobRegistry, claimed more
// than once, or if a field-backed entry names a field that no longer exists.
func TestEveryPolicyKnobHasAmendmentClass(t *testing.T) {
	claims := map[string]int{} // field name -> number of registry entries claiming it
	for _, k := range PolicyKnobRegistry {
		if k.Field != "" {
			claims[k.Field]++
		}
	}

	pt := reflect.TypeOf(adjudicator.Policy{})
	var unclassified []string
	for i := 0; i < pt.NumField(); i++ {
		f := pt.Field(i)
		if !f.IsExported() {
			continue
		}
		switch claims[f.Name] {
		case 0:
			unclassified = append(unclassified, f.Name)
		case 1:
			// classified exactly once — the conforming case
		default:
			t.Errorf("adjudicator.Policy field %s is claimed by %d registry entries; want exactly one", f.Name, claims[f.Name])
		}
	}
	if len(unclassified) > 0 {
		t.Errorf("unclassified adjudicator.Policy fields (add a PolicyKnobRegistry row in internal/policy/amendment.go): %v", unclassified)
	}

	// The reverse direction: a field-backed row must name a real, currently
	// exported field, so a rename cannot strand a stale classification.
	for _, k := range PolicyKnobRegistry {
		if k.Field == "" {
			continue
		}
		if f, ok := pt.FieldByName(k.Field); !ok || !f.IsExported() {
			t.Errorf("registry row %q names an adjudicator.Policy field that does not exist (or is unexported); remove or rename it", k.Field)
		}
	}
}

// TestAmendmentClassInvariants pins the closed class/direction/channel
// pairing: FROZEN knobs declare no widening channel (compiled-in only) and the
// frozen direction; RATCHET knobs are tighten-only; GATED_WIDEN knobs are
// widen-only and reachable through at least one gated operator channel; the
// SELF_AMENDABLE frontier is empty today.
func TestAmendmentClassInvariants(t *testing.T) {
	name := func(k PolicyKnob) string {
		if k.Field != "" {
			return k.Field
		}
		return k.Doc
	}
	for _, k := range PolicyKnobRegistry {
		switch k.Class {
		case AmendFrozen:
			if k.Direction != DirectionFrozen {
				t.Errorf("FROZEN knob %s: direction = %q, want %q", name(k), k.Direction, DirectionFrozen)
			}
			for _, ch := range k.Channels {
				if ch != ChannelCompiledIn {
					t.Errorf("FROZEN knob %s declares widening channel %q; FROZEN admits only %q", name(k), ch, ChannelCompiledIn)
				}
			}
		case AmendRatchet:
			if k.Direction != DirectionTightenOnly {
				t.Errorf("RATCHET knob %s: direction = %q, want %q", name(k), k.Direction, DirectionTightenOnly)
			}
		case AmendGatedWiden:
			if k.Direction != DirectionWidenOnly {
				t.Errorf("GATED_WIDEN knob %s: direction = %q, want %q", name(k), k.Direction, DirectionWidenOnly)
			}
			if len(k.Channels) == 0 {
				t.Errorf("GATED_WIDEN knob %s declares no channel; a widening must name its gate", name(k))
			}
		case AmendSelfAmendable:
			t.Errorf("knob %s is SELF_AMENDABLE; the frontier is declared empty today — widening it is an operator decision, not a registry edit", name(k))
		default:
			t.Errorf("knob %s has unknown amendment class %q", name(k), k.Class)
		}
	}
}
