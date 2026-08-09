package accounts

import (
	"path/filepath"
	"testing"
)

// TestProjectSettingsQuestionTimeoutSeed pins the launch default for the harness's
// "Question auto-continue timeout" (`askUserQuestionTimeout`). The harness default is "never",
// which parks an unattended seat on a question forever, so the projection seeds every seat that
// has not chosen one with ProjectedQuestionTimeout — while leaving a seat that HAS chosen one
// alone, exactly like the seat-owned `model` carve-out.
func TestProjectSettingsQuestionTimeoutSeed(t *testing.T) {
	home := t.TempDir()
	fresh := filepath.Join(home, ".claude-fresh")   // no settings.json yet
	chosen := filepath.Join(home, ".claude-chosen") // seat already picked its own rung
	if err := writeSettingsTestFile(filepath.Join(chosen, "settings.json"), []byte(`{"askUserQuestionTimeout":"10m"}`)); err != nil {
		t.Fatal(err)
	}

	reg := modelDefaultsReg()
	homes := []Home{
		{Name: "fresh", Dir: fresh},
		{Name: "chosen", Dir: chosen},
	}
	if _, ok, err := reg.ProjectSettings(homes, writeSettingsTestFile); err != nil || !ok {
		t.Fatalf("project: ok=%v err=%v", ok, err)
	}

	// A seat the registry says nothing about still gets the auto-continue default.
	freshSettings := readSettings(filepath.Join(fresh, "settings.json"))
	if got := freshSettings[questionTimeoutKey]; got != ProjectedQuestionTimeout {
		t.Errorf("fresh seat %s = %v, want %q", questionTimeoutKey, got, ProjectedQuestionTimeout)
	}
	// The seed is one of the harness's accepted enum rungs — anything else parses back to "never".
	switch ProjectedQuestionTimeout {
	case "60s", "5m", "10m", "never":
	default:
		t.Errorf("ProjectedQuestionTimeout = %q is not an accepted harness value", ProjectedQuestionTimeout)
	}
	// The other defaults still land (the carve-out only touches its own key).
	if perms, _ := freshSettings["permissions"].(map[string]any); perms["defaultMode"] != "bypassPermissions" {
		t.Errorf("fresh seat missing bypass default: %#v", freshSettings)
	}

	// A seat that already chose a rung keeps it.
	chosenSettings := readSettings(filepath.Join(chosen, "settings.json"))
	if got := chosenSettings[questionTimeoutKey]; got != "10m" {
		t.Errorf("chosen seat %s = %v, want 10m (projection clobbered a per-seat choice)", questionTimeoutKey, got)
	}

	// Idempotent: a second projection over the now-seeded roster changes nothing.
	results2, _, err := reg.ProjectSettings(homes, writeSettingsTestFile)
	if err != nil {
		t.Fatalf("re-project: %v", err)
	}
	for _, r := range results2 {
		if r.Changed {
			t.Errorf("second projection changed %s; want idempotent", r.Name)
		}
	}
}

// TestSeatSettingsQuestionTimeoutOverridesRegistryPin proves the seed wins over a registry-pinned
// value for a seat that has not chosen one (same precedence as the `model` carve-out): the
// launch default lives in code, so a stale pin in registry.json cannot quietly reintroduce
// "never" across the roster.
func TestSeatSettingsQuestionTimeoutOverridesRegistryPin(t *testing.T) {
	defaults := map[string]any{questionTimeoutKey: "never"}

	overlay := seatSettingsDefaults(defaults, map[string]any{})
	if got := overlay[questionTimeoutKey]; got != ProjectedQuestionTimeout {
		t.Errorf("unseeded seat overlay %s = %v, want %q", questionTimeoutKey, got, ProjectedQuestionTimeout)
	}
	if got := defaults[questionTimeoutKey]; got != "never" {
		t.Errorf("shared registry defaults were mutated: %s = %v", questionTimeoutKey, got)
	}

	// A seat with its own value drops the key from the overlay entirely, so the merge preserves it.
	overlay = seatSettingsDefaults(defaults, map[string]any{questionTimeoutKey: "5m"})
	if _, present := overlay[questionTimeoutKey]; present {
		t.Errorf("overlay still carries %s for a seat that chose its own: %#v", questionTimeoutKey, overlay)
	}
}
