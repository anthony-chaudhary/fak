package fleetpane

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"
)

type fakeRunner struct {
	paths map[string]bool
	runs  map[string]RunResult
	seen  []RunRequest
}

func (f *fakeRunner) LookPath(file string) (string, error) {
	if f.paths[file] {
		return file, nil
	}
	return "", errors.New("not found")
}

func (f *fakeRunner) Run(ctx context.Context, req RunRequest) RunResult {
	f.seen = append(f.seen, req)
	key := strings.Join(req.Args, "\x00")
	if res, ok := f.runs[key]; ok {
		return res
	}
	return RunResult{ExitCode: 127, Stderr: "not configured", Err: errors.New("not configured")}
}

func testRoot(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	mustWrite(t, filepath.Join(root, "VERSION"), "v9.9.9\n")
	mustMkdir(t, filepath.Join(root, "tools", "_registry", "machines"))
	return root
}

func mustMkdir(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustWrite(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func fixedOptions(r Runner) Options {
	return Options{
		Runner: r,
		Now: func() time.Time {
			return time.Date(2026, 6, 30, 12, 0, 0, 0, time.UTC)
		},
	}
}

func TestLoadConfigMergesCatalogAndLocalOverrides(t *testing.T) {
	root := testRoot(t)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.example.json"), `{
  "session_window_h": 3,
  "target": 2,
  "registry_dir": "tools/_registry",
  "machine_dir": "tools/_registry/machines",
  "loops": {
    "example": {"enabled": false, "status_cmd": ["examplecmd"]}
  }
}`)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{
  "loops": {
    "shared": {"enabled": true, "status_cmd": ["sharedcmd"], "timeout_s": 7}
  }
}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "control_pane.local.json"), `{
  "target": 5,
  "machine_id": "Host With Spaces",
  "loops": {
    "shared": {"enabled": false, "status_cmd": ["localcmd"], "action": "local wins"}
  }
}`)
	t.Setenv("FLEET_MACHINE_ID", "Env Host")

	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.Target != 5 || cfg.SessionWindowH != 3 || !cfg.LocalExists {
		t.Fatalf("unexpected scalar config: target=%d window=%v local=%v", cfg.Target, cfg.SessionWindowH, cfg.LocalExists)
	}
	if got := MachineID(cfg); got != "env-host" {
		t.Fatalf("MachineID=%q, want env-host", got)
	}
	shared := cfg.Loops["shared"]
	if shared.Enabled || !reflect.DeepEqual(shared.StatusCmd, []string{"localcmd"}) || shared.Action != "local wins" {
		t.Fatalf("local loop override not applied: %+v", shared)
	}
	if !strings.HasSuffix(cfg.RegistryDir, filepath.Join("tools", "_registry")) {
		t.Fatalf("registry dir was not normalized: %s", cfg.RegistryDir)
	}
}

func TestClassifyLoopStatus(t *testing.T) {
	tests := []struct {
		name   string
		res    RunResult
		spec   LoopSpec
		state  string
		detail string
	}{
		{
			name:   "boolean ok",
			res:    RunResult{ExitCode: 0, Stdout: `{"ok": true, "detail": "green"}`},
			state:  "OK",
			detail: "green",
		},
		{
			name:   "boolean action",
			res:    RunResult{ExitCode: 0, Stdout: `{"ok": false, "reason": "needs operator"}`},
			state:  "ACTION",
			detail: "needs operator",
		},
		{
			name:   "verdict folded",
			res:    RunResult{ExitCode: 0, Stdout: `{"verdict": "READY"}`},
			state:  "OK",
			detail: "verdict=READY",
		},
		{
			name:   "nonzero returncode",
			res:    RunResult{ExitCode: 9, Stderr: "boom"},
			state:  "ACTION",
			detail: "boom",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			state, _, detail := ClassifyLoopStatus(tt.res, tt.spec)
			if state != tt.state || detail != tt.detail {
				t.Fatalf("state/detail=(%q,%q), want (%q,%q)", state, detail, tt.state, tt.detail)
			}
		})
	}
}

func TestLoopListCheckAndAuditUseHermeticRunner(t *testing.T) {
	root := testRoot(t)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{
  "loops": {
    "action": {"enabled": true, "status_cmd": ["actioncmd"], "action": "inspect action"},
    "broken": {"enabled": true, "status_cmd": ["missingcmd"]},
    "ok": {"enabled": true, "status_cmd": ["okcmd"]}
  }
}`)
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	runner := &fakeRunner{
		paths: map[string]bool{"okcmd": true, "actioncmd": true},
		runs: map[string]RunResult{
			"okcmd":     {ExitCode: 0, Stdout: `{"ok": true, "detail": "green"}`},
			"actioncmd": {ExitCode: 0, Stdout: `{"ok": false, "reason": "red"}`},
		},
	}
	opts := fixedOptions(runner)

	list := LoopList(cfg, opts)
	if list.OK || list.Blocked != 1 || list.Enabled != 3 {
		t.Fatalf("unexpected loop-list summary: %+v", list)
	}
	check := LoopCheckPlan(context.Background(), cfg, "action", false, false, opts)
	if !check.OK || check.Verdict != "ACTION" || !check.NeedsAction || check.Check.Detail != "red" {
		t.Fatalf("unexpected loop-check: %+v", check)
	}
	audit := LoopAudit(context.Background(), cfg, nil, opts)
	if audit.OK || audit.Counts["healthy"] != 1 || audit.Counts["action"] != 1 || audit.Counts["broken"] != 1 || audit.Counts["total"] != 3 {
		t.Fatalf("unexpected loop-audit: %+v", audit)
	}
}

func TestStatusAndFleetSnapshotFold(t *testing.T) {
	root := testRoot(t)
	mustWrite(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{
  "loops": {
    "ok": {"enabled": true, "status_cmd": ["okcmd"]}
  },
  "supervisor_status_cmd": ["supcmd"]
}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "control_pane.local.json"), `{}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "sessions.json"), `{
  "schema": "sessions/1",
  "generated_utc": "2026-06-30T11:55:00Z",
  "sessions": [
    {"action": "AUTO_RESUME", "category": "live", "disp": "INFRA_AUTH"}
  ],
  "accounts": [
    {"tag": "a", "available": true},
    {"tag": "b", "available": false, "block_kind": "auth", "reason": "login"}
  ]
}`)
	mustWrite(t, filepath.Join(root, "tools", "_registry", "machines", "peer.json"), `{
  "schema": "fleet-control-pane/1",
  "generated_utc": "2026-06-30T11:50:00Z",
  "app_version": "v9.9.9",
  "machine": {"id": "peer", "host": "peer"},
  "registry": {"sessions": 2, "accounts": {"available": 1, "total": 1}},
  "loops": {"count": 0, "checks": []},
  "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync"}},
  "actions": [],
  "verdict": "OK"
}`)
	cfg, err := LoadConfig(root)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	runner := &fakeRunner{
		paths: map[string]bool{"okcmd": true, "supcmd": true},
		runs: map[string]RunResult{
			"okcmd":  {ExitCode: 0, Stdout: `{"ok": true}`},
			"supcmd": {ExitCode: 0, Stdout: `{"verdict": "AT_TARGET", "process": {"alive": true}}`},
		},
	}
	opts := fixedOptions(runner)

	status := CollectStatus(context.Background(), cfg, opts)
	if status.AppVersion != "v9.9.9" || status.Verdict != "OK" {
		t.Fatalf("unexpected status version/verdict: %s %s actions=%v", status.AppVersion, status.Verdict, status.Actions)
	}
	reg := status.Registry
	if intValueDefault(reg["sessions"], 0) != 1 || intValueDefault(reg["auto_resume"], 0) != 1 {
		t.Fatalf("registry summary mismatch: %+v", reg)
	}

	fleet := FleetView(context.Background(), cfg, false, false, opts)
	if fleet.Verdict != "OK" || len(fleet.Machines) != 1 || fleet.Totals["sessions"] != 2 {
		t.Fatalf("unexpected fleet snapshot fold: %+v", fleet)
	}
	text := FleetText(fleet)
	if !strings.Contains(text, "FLEET CONTROL PANE AGGREGATE") || !strings.Contains(text, "peer") {
		t.Fatalf("fleet text missing expected content:\n%s", text)
	}
}
