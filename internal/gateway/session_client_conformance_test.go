package gateway

import "testing"

func TestSessionCapabilityCorpusCoversAdvertisedCapabilities(t *testing.T) {
	actions := SessionCapabilityCorpus(sessionClientCapabilities)
	covered := map[string]bool{}
	for _, action := range actions {
		if action.Available {
			covered[action.Capability] = true
		}
		if action.Label == "" {
			t.Fatalf("action %q has no label", action.ID)
		}
		if !action.Available && (action.UnavailableCode == "" || action.UnavailableReason == "" || action.Handoff == "") {
			t.Fatalf("action %q lacks typed refusal/handoff: %+v", action.ID, action)
		}
	}
	for _, capability := range sessionClientCapabilities {
		if !covered[capability] {
			t.Fatalf("advertised capability %q has no reference-client action", capability)
		}
	}
}

func TestSessionCapabilityCorpusIncludesFullPowerCeiling(t *testing.T) {
	want := []string{"observe", "replay", "text_input", "approve", "deny", "pause", "resume", "drain", "close", "checkpoint", "move", "detach", "effect_recovery"}
	all := SessionCapabilityCorpus(sessionCorpusCapabilities())
	seen := map[string]bool{}
	for _, action := range all {
		if action.Available {
			seen[action.Capability] = true
		}
	}
	for _, capability := range want {
		if !seen[capability] {
			t.Errorf("full-power corpus lacks %s", capability)
		}
	}
}
