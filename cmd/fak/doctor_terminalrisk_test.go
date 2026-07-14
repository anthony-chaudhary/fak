package main

import (
	"bytes"
	"github.com/anthony-chaudhary/fak/internal/terminalrisk"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestDoctorTerminalRiskFlagsAppliesAndRereadsClean(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "settings.json")
	raw := []byte(`{"profiles":{"list":[]}}`)
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	old := gatherTerminalRiskFactsFn
	defer func() { gatherTerminalRiskFactsFn = old }()
	gatherTerminalRiskFactsFn = func(p string) (terminalrisk.Facts, error) {
		b, e := os.ReadFile(p)
		return terminalrisk.Facts{AMDPresent: true, PriorWTRenderCrash: true, SettingsPath: p, Settings: b}, e
	}
	var out, errout bytes.Buffer
	if rc := runDoctorTerminalRisk(&out, &errout, []string{"--settings", path}); rc != 3 {
		t.Fatalf("risk rc=%d out=%s err=%s", rc, out.String(), errout.String())
	}
	out.Reset()
	errout.Reset()
	if rc := runDoctorTerminalRisk(&out, &errout, []string{"--settings", path, "--apply"}); rc != 0 {
		t.Fatalf("apply rc=%d out=%s err=%s", rc, out.String(), errout.String())
	}
	if !strings.Contains(out.String(), "CLEAN") || !strings.Contains(out.String(), "backup:") || !strings.Contains(out.String(), "fully restarted") {
		t.Fatalf("output=%s", out.String())
	}
	changed, _ := os.ReadFile(path)
	if !strings.Contains(string(changed), `"graphicsAPI": "direct2d"`) {
		t.Fatalf("settings=%s", changed)
	}
	matches, _ := filepath.Glob(path + ".fak-backup-*")
	if len(matches) != 1 {
		t.Fatalf("backups=%v", matches)
	}
}
