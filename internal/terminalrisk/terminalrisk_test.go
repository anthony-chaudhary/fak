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
