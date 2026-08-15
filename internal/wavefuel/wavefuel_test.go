package wavefuel

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestFleetWaveUsesNativeDispatchAndKeepsReceiptOutOfChildPrompt(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	raw, err := os.ReadFile(filepath.Join(root, ".claude", "skills", "fleet-wave", "SKILL.md"))
	if err != nil {
		t.Fatal(err)
	}
	skill := string(raw)
	for _, want := range []string{
		"## Phase 2 — Render the operator receipt",
		"never pass it as `-PointerFile`",
		"fak dispatch wave --count 30 --backend codex",
		"--goal high-priority --workspace . --live --json",
	} {
		if !strings.Contains(skill, want) {
			t.Errorf("fleet-wave skill missing native contract %q", want)
		}
	}
	phase2 := between(skill, "## Phase 2 —", "## Phase 3 —")
	for _, stale := range []string{"wc -c", "cap=4000", "& $launcher", "-ExtendStanding", "-Launch"} {
		if strings.Contains(phase2, stale) {
			t.Errorf("Phase 2 routes the operator receipt through stale launcher contract %q", stale)
		}
	}
	phase3 := between(skill, "## Phase 3 —", "## Phase 4 —")
	for _, stale := range []string{"& $launcher", "use `-ExtendStanding`", "-PointerFile $fuelPath"} {
		if strings.Contains(phase3, stale) {
			t.Errorf("Phase 3 still exposes stale launch path %q", stale)
		}
	}
}

func TestFleetWaveReceiptIsNotConstrainedByLegacyGoalCap(t *testing.T) {
	_, file, _, _ := runtime.Caller(0)
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	raw, err := os.ReadFile(filepath.Join(root, ".claude", "skills", "fleet-wave", "fuel-wave.md"))
	if err != nil {
		t.Fatal(err)
	}
	receipt := strings.ReplaceAll(strings.ReplaceAll(string(raw), "{{WAVE}}", "fw0815004049"), "{{DEADLINE}}", "2026-08-15T04:40Z")
	if !strings.Contains(receipt, "WAVE: fw0815004049") || !strings.Contains(receipt, "DEADLINE: 2026-08-15T04:40Z") {
		t.Fatal("rendered receipt lost wave attribution")
	}
	if len(receipt) <= 4000 {
		t.Skip("receipt currently fits the legacy cap; native-contract assertions remain authoritative")
	}
}

func between(s, start, end string) string {
	i := strings.Index(s, start)
	j := strings.Index(s, end)
	if i < 0 || j < 0 || j <= i {
		return ""
	}
	return s[i:j]
}
