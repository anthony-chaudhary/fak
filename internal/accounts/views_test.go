package accounts

import (
	"strings"
	"testing"
)

// viewFixture is a registry exercising both views: an active default, an active reserved
// seat, an active worker, and a tombstone with full audit fields, plus per-view config.
func viewFixture() Registry {
	tru := true
	return Registry{
		Homes: []Home{
			{Name: "gem8-netra", Dir: `C:\Users\U\.claude-gem8-netra`, Default: true,
				Identity: Identity{Email: "gem8@netra.test"}, Enabled: &tru},
			{Name: "day24-netra", Dir: `C:\Users\U\.claude-day24-netra`, Reserved: true,
				Identity: Identity{Email: "day24@netra.test"}, ChromeProfile: "Profile 3"},
			{Name: "gem7-netra", Dir: `C:\Users\U\.claude-gem7-netra`,
				Identity: Identity{Email: "gem7@netra.test"}, ChromeProfile: "Profile 10"},
			{Name: "q-netra", Status: StatusTombstoned, RehomeTo: "gem8-netra",
				Dir: `C:\Users\U\.claude-q-netra.DELETED`, Identity: Identity{Email: "gem8@netra.test"},
				TombstonedAt: "2026-06-25T15:00:00Z", TombstoneReason: "phantom duplicate of gem8-netra"},
		},
		Views: map[string]ViewConfig{
			"dos": {
				BlockOrder: []string{"rotation", "defaults"},
				Blocks: map[string]any{
					"rotation": map[string]any{"order": "by_reset", "near_cap_util": 0.95},
					"defaults": map[string]any{
						"settings": map[string]any{
							"model":       "opus",
							"effortLevel": "xhigh",
							"permissions": map[string]any{"defaultMode": "bypassPermissions"},
						},
					},
				},
			},
			"job": {
				BlockOrder: []string{"defaults", "rotation", "launch"},
				Blocks: map[string]any{
					"rotation": map[string]any{"order": "by_reset", "near_cap_util": 0.9, "avoid_reserved": true},
					"launch": map[string]any{
						"bypass_permissions": true,
						"extra_flags":        []any{"--effort", "xhigh"},
					},
				},
			},
		},
	}
}

func TestRenderDosView(t *testing.T) {
	got, err := viewFixture().RenderView(ViewDos)
	if err != nil {
		t.Fatalf("render dos: %v", err)
	}
	// Header present.
	if !strings.HasPrefix(got, "# GENERATED from registry.json") {
		t.Errorf("dos view missing generated header:\n%s", got)
	}
	// Active rows present (name + config_dir), reserved/tombstoned NOT in the dos accounts list.
	for _, want := range []string{
		"  - name: gem8-netra\n",
		`    config_dir: "C:\\Users\\U\\.claude-gem8-netra"` + "\n", // backslash+colon path is quoted+escaped
		"  - name: day24-netra\n",
		"  - name: gem7-netra\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("dos view missing %q in:\n%s", want, got)
		}
	}
	// Tombstoned q-netra must NOT appear in the dos active rows.
	if strings.Contains(got, "name: q-netra") {
		t.Errorf("tombstoned q-netra leaked into dos active rows:\n%s", got)
	}
	// Config blocks emitted in order, nested correctly.
	for _, want := range []string{
		"\nrotation:\n",
		"  order: by_reset\n",
		"  near_cap_util: 0.95\n",
		"\ndefaults:\n",
		"  settings:\n",
		"    model: opus\n",
		"      defaultMode: bypassPermissions\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("dos view missing block line %q in:\n%s", want, got)
		}
	}
}

func TestRenderJobView(t *testing.T) {
	got, err := viewFixture().RenderView(ViewJob)
	if err != nil {
		t.Fatalf("render job: %v", err)
	}
	// Active rows carry the richer field set.
	for _, want := range []string{
		"  - name: day24-netra\n",
		"    chrome_profile: Profile 3\n",
		"    email: day24@netra.test\n",
		"    enabled: true\n",
		"    reserved: true\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("job view missing %q in:\n%s", want, got)
		}
	}
	// Tombstone block carries the audit fields.
	for _, want := range []string{
		"\ntombstoned_accounts:\n",
		"  - name: q-netra\n",
		"    enabled: false\n",
		`    tombstoned_at: "2026-06-25T15:00:00Z"` + "\n", // contains ':' -> quoted
		"    tombstone_reason: phantom duplicate of gem8-netra\n",
		"    rehome_to: gem8-netra\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("job view missing tombstone line %q in:\n%s", want, got)
		}
	}
	// A list block (launch.extra_flags) emits as a YAML sequence.
	for _, want := range []string{
		"\nlaunch:\n",
		"  bypass_permissions: true\n",
		"  extra_flags:\n",
		`    - "--effort"` + "\n", // leading '-' is a YAML indicator -> quoted
		"    - xhigh\n",
	} {
		if !strings.Contains(got, want) {
			t.Errorf("job view missing launch line %q in:\n%s", want, got)
		}
	}
	// day24 is reserved; gem7 is not — gem7 must NOT carry a reserved line.
	gem7Block := got[strings.Index(got, "name: gem7-netra"):]
	gem7Block = gem7Block[:strings.Index(gem7Block, "email: gem7@netra.test")+1]
	if strings.Contains(gem7Block, "reserved:") {
		t.Errorf("non-reserved gem7 should not carry a reserved line:\n%s", gem7Block)
	}
}

// TestRenderDeterministic proves the projection is byte-stable across runs (so `check` is a
// meaningful drift detector, not flapping on map iteration order).
func TestRenderDeterministic(t *testing.T) {
	r := viewFixture()
	for _, v := range []ViewName{ViewDos, ViewJob} {
		a, _ := r.RenderView(v)
		for i := 0; i < 5; i++ {
			b, _ := r.RenderView(v)
			if a != b {
				t.Fatalf("view %s render not deterministic across runs", v)
			}
		}
	}
}

// TestYamlScalarQuoting pins the quoting rules: values that would misparse as a non-string
// are quoted; plain identifiers are left bare.
func TestYamlScalarQuoting(t *testing.T) {
	cases := map[string]string{
		"plain-name":           "plain-name",
		"":                     `""`,
		"true":                 `"true"`,
		"has: colon":           `"has: colon"`,
		" leading":             `" leading"`,
		`C:\Users\U\.claude-x`: `"C:\\Users\\U\\.claude-x"`, // backslash + colon -> quoted+escaped
	}
	for in, want := range cases {
		if got := yamlScalar(in); got != want {
			t.Errorf("yamlScalar(%q) = %q, want %q", in, got, want)
		}
	}
}
