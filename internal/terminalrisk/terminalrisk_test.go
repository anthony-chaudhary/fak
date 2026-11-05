package terminalrisk

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestAssessRequiresThreePartRiskAndDirect2DCleans(t *testing.T) {
	risk, err := Assess(Facts{AMDPresent: true, PriorWTRenderCrash: true, SettingsPath: "x", Settings: []byte(`{"profiles":{"list":[]}}`)})
	if err != nil || !risk.Risk {
		t.Fatalf("risk=%+v err=%v", risk, err)
	}
	clean, err := Assess(Facts{AMDPresent: true, PriorWTRenderCrash: true, SettingsPath: "x", Settings: []byte(`{"graphicsAPI":"direct2d"}`)})
	if err != nil || clean.Risk || clean.GraphicsAPI != "direct2d" {
		t.Fatalf("clean=%+v err=%v", clean, err)
	}
}
func TestAssessLegacyAtlasGPUPath(t *testing.T) {
	r, err := Assess(Facts{AMDPresent: true, PriorWTRenderCrash: true, Settings: []byte(`{"experimental":{"useAtlasEngine":true,"rendering":{"software":false}}}`)})
	if err != nil || !r.Risk {
		t.Fatalf("r=%+v err=%v", r, err)
	}
}
func TestApplyDirect2DBacksUpAndReReadsClean(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	raw := []byte(`{"profiles":{"list":[]}}`)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	backup, err := ApplyDirect2D(path, raw, time.Date(2026, 7, 14, 1, 2, 3, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if got, err := os.ReadFile(backup); err != nil || string(got) != string(raw) {
		t.Fatalf("backup=%q err=%v", got, err)
	}
	changed, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	r, err := Assess(Facts{AMDPresent: true, PriorWTRenderCrash: true, Settings: changed})
	if err != nil || r.Risk || r.GraphicsAPI != "direct2d" {
		t.Fatalf("r=%+v err=%v settings=%s", r, err, changed)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestApplyDirect2DRequiresSettingsObject(t *testing.T) {
	for _, tt := range []struct {
		name, raw string
		wantError bool
	}{
		{"null", "null", true},
		{"null with whitespace", " \r\nnull\t", true},
		{"array", "[]", true},
		{"string", `"settings"`, true},
		{"boolean", "true", true},
		{"number", "42", true},
		{"empty object", "{}", false},
	} {
		t.Run(tt.name, func(t *testing.T) {
			dir := t.TempDir()
			path := filepath.Join(dir, "settings.json")
			if err := os.WriteFile(path, []byte(tt.raw), 0o600); err != nil {
				t.Fatal(err)
			}
			backup, err := ApplyDirect2D(path, []byte(tt.raw), time.Date(2026, 7, 14, 1, 2, 3, 0, time.UTC))
			if (err != nil) != tt.wantError {
				t.Fatalf("ApplyDirect2D(%q) error = %v, wantError %v", tt.raw, err, tt.wantError)
			}
			got, readErr := os.ReadFile(path)
			if readErr != nil {
				t.Fatal(readErr)
			}
			if tt.wantError {
				if backup != "" || string(got) != tt.raw {
					t.Fatalf("refusal changed settings or returned backup: settings=%q backup=%q", got, backup)
				}
				entries, readErr := os.ReadDir(dir)
				if readErr != nil || len(entries) != 1 || entries[0].Name() != "settings.json" {
					t.Fatalf("refusal created an artifact: entries=%v err=%v", entries, readErr)
				}
				return
			}
			original, readErr := os.ReadFile(backup)
			if readErr != nil || string(original) != tt.raw {
				t.Fatalf("valid-object backup=%q err=%v", original, readErr)
			}
			report, assessErr := Assess(Facts{AMDPresent: true, PriorWTRenderCrash: true, Settings: got})
			if assessErr != nil || report.Risk || report.GraphicsAPI != Direct2D {
				t.Fatalf("valid-object result=%+v err=%v settings=%s", report, assessErr, got)
			}
		})
	}
}
