package accounts

import (
	"path/filepath"
	"testing"
)

// modelDefaultsReg builds a registry whose dos defaults.settings block pins model=claude-fable-5,
// mirroring the real runtime registry (~/.claude-accounts/registry.json) the projection reads.
// This is the exact shape that regressed in #3091: the fable-5 pin (the launch FALLBACK, not the
// primary) was force-merged over every seat's model on each sync/add.
func modelDefaultsReg() Registry {
	return Registry{
		Views: map[string]ViewConfig{
			"dos": {
				Blocks: map[string]any{
					"defaults": map[string]any{
						"settings": map[string]any{
							"model": "claude-fable-5",
							"permissions": map[string]any{
								"defaultMode": "bypassPermissions",
							},
						},
					},
				},
			},
		},
	}
}

// TestProjectSettingsModelCarveOut is the #3091 regression guard for the seat-owned `model` key.
// The settings projection must (a) seed a FRESH seat with the launch default (Opus 4.8), not the
// registry-pinned fable-5, and (b) PRESERVE a per-seat model already in settings.json instead of
// clobbering it back toward the fallback — while still projecting every other default (the bypass
// permission) and staying idempotent on a second run.
func TestProjectSettingsModelCarveOut(t *testing.T) {
	home := t.TempDir()
	fresh := filepath.Join(home, ".claude-fresh")   // no settings.json yet
	pinned := filepath.Join(home, ".claude-pinned") // pre-set per-seat model
	if err := writeSettingsTestFile(filepath.Join(pinned, "settings.json"), []byte(`{"model":"sonnet"}`)); err != nil {
		t.Fatal(err)
	}

	reg := modelDefaultsReg()
	homes := []Home{
		{Name: "fresh", Dir: fresh},
		{Name: "pinned", Dir: pinned},
	}
	if _, ok, err := reg.ProjectSettings(homes, writeSettingsTestFile); err != nil || !ok {
		t.Fatalf("project: ok=%v err=%v", ok, err)
	}

	// (a) The fresh seat inherits the launch default (opus), NOT the registry-pinned fable-5.
	freshSettings := readSettings(filepath.Join(fresh, "settings.json"))
	if got := freshSettings["model"]; got != ProjectedDefaultModel {
		t.Errorf("fresh seat model = %v, want %q (the launch default, not the fable-5 fallback)", got, ProjectedDefaultModel)
	}
	// The carve-out only touches model — the bypass default still lands via the deep-merge.
	if perms, _ := freshSettings["permissions"].(map[string]any); perms["defaultMode"] != "bypassPermissions" {
		t.Errorf("fresh seat missing bypass default: %#v", freshSettings)
	}

	// (b) The pre-set per-seat model survives the projection (not clobbered to fable-5).
	pinnedSettings := readSettings(filepath.Join(pinned, "settings.json"))
	if pinnedSettings["model"] != "sonnet" {
		t.Errorf("pinned seat model = %v, want sonnet (projection clobbered a per-seat model)", pinnedSettings["model"])
	}
	// ...and it still gained the bypass default (the seat-owned carve-out is only for model).
	if perms, _ := pinnedSettings["permissions"].(map[string]any); perms["defaultMode"] != "bypassPermissions" {
		t.Errorf("pinned seat missing bypass default: %#v", pinnedSettings)
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
