package accounts

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

// defaultsReg builds a registry whose dos view carries a defaults.settings block, the exact
// shape ProjectSettings reads. It mirrors viewFixture()'s defaults block but is kept local so a
// change to that fixture cannot silently alter these assertions.
func defaultsReg() Registry {
	return Registry{
		Views: map[string]ViewConfig{
			"dos": {
				Blocks: map[string]any{
					"defaults": map[string]any{
						"settings": map[string]any{
							"skipDangerousModePermissionPrompt": true,
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

func TestDeepMergeJSON(t *testing.T) {
	cases := []struct {
		name        string
		base        map[string]any
		overlay     map[string]any
		want        map[string]any
		wantChanged bool
	}{
		{
			name:        "leaf added to empty base",
			base:        map[string]any{},
			overlay:     map[string]any{"a": "x"},
			want:        map[string]any{"a": "x"},
			wantChanged: true,
		},
		{
			name:        "leaf value overwritten",
			base:        map[string]any{"a": "old"},
			overlay:     map[string]any{"a": "new"},
			want:        map[string]any{"a": "new"},
			wantChanged: true,
		},
		{
			name: "nested recurse keeps sibling",
			base: map[string]any{
				"permissions": map[string]any{"allow": []any{"Bash"}, "defaultMode": "default"},
			},
			overlay: map[string]any{
				"permissions": map[string]any{"defaultMode": "bypassPermissions"},
			},
			want: map[string]any{
				"permissions": map[string]any{"allow": []any{"Bash"}, "defaultMode": "bypassPermissions"},
			},
			wantChanged: true,
		},
		{
			name:        "preserve unlisted keys",
			base:        map[string]any{"theme": "dark", "model": "opus"},
			overlay:     map[string]any{"effortLevel": "xhigh"},
			want:        map[string]any{"theme": "dark", "model": "opus", "effortLevel": "xhigh"},
			wantChanged: true,
		},
		{
			name:        "no change when already equal",
			base:        map[string]any{"a": "x", "n": float64(2)},
			overlay:     map[string]any{"a": "x", "n": float64(2)},
			want:        map[string]any{"a": "x", "n": float64(2)},
			wantChanged: false,
		},
		{
			name:        "map replaces scalar wholesale",
			base:        map[string]any{"k": "scalar"},
			overlay:     map[string]any{"k": map[string]any{"deep": true}},
			want:        map[string]any{"k": map[string]any{"deep": true}},
			wantChanged: true,
		},
		{
			name:        "scalar replaces map wholesale",
			base:        map[string]any{"k": map[string]any{"deep": true}},
			overlay:     map[string]any{"k": "scalar"},
			want:        map[string]any{"k": "scalar"},
			wantChanged: true,
		},
		{
			name:        "empty overlay is no change",
			base:        map[string]any{"a": "x"},
			overlay:     map[string]any{},
			want:        map[string]any{"a": "x"},
			wantChanged: false,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Snapshot base to prove it is not mutated.
			baseCopy := deepCopyMap(tc.base)
			got, changed := deepMergeJSON(tc.base, tc.overlay)
			if changed != tc.wantChanged {
				t.Errorf("changed = %v, want %v", changed, tc.wantChanged)
			}
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("merged = %#v, want %#v", got, tc.want)
			}
			if !reflect.DeepEqual(tc.base, baseCopy) {
				t.Errorf("base was mutated: %#v, want %#v", tc.base, baseCopy)
			}
		})
	}
}

// deepCopyMap makes an independent copy of a JSON-shaped map for the not-mutated assertion.
func deepCopyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		if mv, ok := v.(map[string]any); ok {
			out[k] = deepCopyMap(mv)
		} else {
			out[k] = v
		}
	}
	return out
}

func TestDeepMergeJSONIdempotent(t *testing.T) {
	base := map[string]any{"theme": "dark"}
	overlay := map[string]any{"permissions": map[string]any{"defaultMode": "bypassPermissions"}}
	merged, changed := deepMergeJSON(base, overlay)
	if !changed {
		t.Fatal("first merge should report changed")
	}
	// Feeding the result back through the same overlay must be a no-op.
	again, changed2 := deepMergeJSON(merged, overlay)
	if changed2 {
		t.Errorf("second merge reported changed; want stable")
	}
	if !reflect.DeepEqual(merged, again) {
		t.Errorf("second merge altered the map: %#v vs %#v", again, merged)
	}
}

func TestMarshalSettingsByteStable(t *testing.T) {
	m := map[string]any{
		"theme":       "dark",
		"count":       float64(2), // whole number must serialize as "2", not "2.0"
		"permissions": map[string]any{"defaultMode": "bypassPermissions"},
	}
	want := "{\n" +
		"  \"count\": 2,\n" +
		"  \"permissions\": {\n" +
		"    \"defaultMode\": \"bypassPermissions\"\n" +
		"  },\n" +
		"  \"theme\": \"dark\"\n" +
		"}\n"
	b1, err := marshalSettings(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if string(b1) != want {
		t.Errorf("marshal =\n%q\nwant\n%q", string(b1), want)
	}
	// Byte-stable across runs (sorted keys), the idempotency contract.
	b2, _ := marshalSettings(m)
	if string(b1) != string(b2) {
		t.Errorf("marshal not byte-stable across runs")
	}
}

func TestDefaultsSettings(t *testing.T) {
	if s, ok := defaultsReg().DefaultsSettings(); !ok {
		t.Errorf("expected ok=true for a registry with a defaults.settings block")
	} else if _, ok := s["permissions"]; !ok {
		t.Errorf("defaults.settings missing permissions key: %#v", s)
	}

	// Each missing rung is a clean ok=false, never a panic.
	for _, tc := range []struct {
		name string
		reg  Registry
	}{
		{"no views", Registry{}},
		{"no dos view", Registry{Views: map[string]ViewConfig{"job": {}}}},
		{"no defaults block", Registry{Views: map[string]ViewConfig{"dos": {Blocks: map[string]any{"rotation": map[string]any{}}}}}},
		{"defaults without settings", Registry{Views: map[string]ViewConfig{"dos": {Blocks: map[string]any{"defaults": map[string]any{"other": 1}}}}}},
		{"settings not a map", Registry{Views: map[string]ViewConfig{"dos": {Blocks: map[string]any{"defaults": map[string]any{"settings": "nope"}}}}}},
		{"empty settings map", Registry{Views: map[string]ViewConfig{"dos": {Blocks: map[string]any{"defaults": map[string]any{"settings": map[string]any{}}}}}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, ok := tc.reg.DefaultsSettings(); ok {
				t.Errorf("expected ok=false for %s", tc.name)
			}
		})
	}
}

func TestProjectSettings(t *testing.T) {
	home := t.TempDir()
	fresh := filepath.Join(home, ".claude-fresh")       // no settings.json yet
	existing := filepath.Join(home, ".claude-existing") // pre-existing theme
	if err := os.MkdirAll(existing, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(existing, "settings.json"), []byte(`{"theme":"dark"}`), 0o644); err != nil {
		t.Fatal(err)
	}

	reg := defaultsReg()
	homes := []Home{
		{Name: "fresh", Dir: fresh},
		{Name: "existing", Dir: existing},
		{Name: "tomb", Dir: filepath.Join(home, ".claude-tomb"), Status: StatusTombstoned},
		{Name: "nodir"},
	}

	results, ok, err := reg.ProjectSettings(homes, writeSettingsTestFile)
	if err != nil {
		t.Fatalf("project: %v", err)
	}
	if !ok {
		t.Fatal("expected ok=true (defaults present)")
	}
	byName := map[string]SettingsResult{}
	for _, r := range results {
		byName[r.Name] = r
	}
	if !byName["fresh"].Changed {
		t.Errorf("fresh seat should have changed")
	}
	if !byName["existing"].Changed {
		t.Errorf("existing seat should have changed")
	}
	if byName["tomb"].Skipped != "tombstoned" {
		t.Errorf("tomb Skipped = %q, want tombstoned", byName["tomb"].Skipped)
	}
	if byName["nodir"].Skipped != "no dir" {
		t.Errorf("nodir Skipped = %q, want 'no dir'", byName["nodir"].Skipped)
	}

	// The fresh seat's file exists and carries the bypass default.
	got := readSettings(filepath.Join(fresh, "settings.json"))
	perms, _ := got["permissions"].(map[string]any)
	if perms["defaultMode"] != "bypassPermissions" {
		t.Errorf("fresh settings missing bypass: %#v", got)
	}
	// The existing seat keeps its theme AND gains the bypass (deep merge, not replace).
	got2 := readSettings(filepath.Join(existing, "settings.json"))
	if got2["theme"] != "dark" {
		t.Errorf("existing seat lost theme: %#v", got2)
	}
	perms2, _ := got2["permissions"].(map[string]any)
	if perms2["defaultMode"] != "bypassPermissions" {
		t.Errorf("existing seat missing bypass: %#v", got2)
	}
	// The tombstoned seat's file was never written.
	if _, err := os.Stat(filepath.Join(home, ".claude-tomb", "settings.json")); !os.IsNotExist(err) {
		t.Errorf("tombstoned seat should have no settings.json (err=%v)", err)
	}

	// Second run is a no-op: nothing changed, bytes stable.
	freshBefore, _ := os.ReadFile(filepath.Join(fresh, "settings.json"))
	results2, _, err := reg.ProjectSettings(homes, writeSettingsTestFile)
	if err != nil {
		t.Fatalf("re-project: %v", err)
	}
	for _, r := range results2 {
		if r.Changed {
			t.Errorf("second run changed %s; want idempotent", r.Name)
		}
	}
	freshAfter, _ := os.ReadFile(filepath.Join(fresh, "settings.json"))
	if string(freshBefore) != string(freshAfter) {
		t.Errorf("second run rewrote fresh settings.json")
	}

	// A registry with no defaults block is a clean whole-roster no-op.
	if _, ok, _ := (Registry{}).ProjectSettings(homes, writeSettingsTestFile); ok {
		t.Errorf("expected ok=false for a registry with no defaults.settings")
	}
}

// writeSettingsTestFile is the test writeFn: a plain (non-atomic) write, since a temp tree needs
// no crash safety. Production uses the atomic writeSettingsFile in cmd/fak.
func writeSettingsTestFile(path string, b []byte) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}
