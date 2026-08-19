package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
)

type infoFleetRunner struct {
	runs map[string]fleetpane.RunResult
}

func (r infoFleetRunner) Run(_ context.Context, req fleetpane.RunRequest) fleetpane.RunResult {
	if v, ok := r.runs[req.Args[0]]; ok {
		return v
	}
	return fleetpane.RunResult{ExitCode: -1, Err: errors.New("unavailable")}
}
func (r infoFleetRunner) LookPath(file string) (string, error) {
	if _, ok := r.runs[file]; ok {
		return file, nil
	}
	return "", errors.New("missing")
}

func writeInfoFleetConfig(t *testing.T, body string) string {
	t.Helper()
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "VERSION"), []byte("test"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(root, "tools", "control_pane.loops.json"), []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return root
}

func TestCollectInfoFleetWorkspaceAttentionFirstAndTyped(t *testing.T) {
	root := writeInfoFleetConfig(t, `{"loops":{"healthy":{"enabled":true,"status_cmd":["healthy"]},"attention":{"enabled":true,"status_cmd":["attention"]},"stale":{"enabled":true,"status_cmd":["stale"]}}}`)
	runner := infoFleetRunner{runs: map[string]fleetpane.RunResult{"healthy": {ExitCode: 0, Stdout: `{"ok":true,"detail":"fresh progress age=2m"}`}, "attention": {ExitCode: 0, Stdout: `{"ok":false,"reason":"no progress age=48m"}`}, "stale": {ExitCode: -1, Err: errors.New("status source unavailable")}}}
	got := collectInfoFleetWorkspace(root, runner, time.Date(2026, 8, 18, 20, 0, 0, 0, time.UTC))
	if got.State != "READY" || len(got.Loops) != 3 {
		t.Fatalf("%+v", got)
	}
	if got.Loops[0].Name != "attention" || got.Loops[0].State != "ACTION" || got.Loops[1].Name != "stale" || got.Loops[1].State != "UNAVAILABLE" || got.Loops[2].State != "OK" {
		t.Fatalf("order=%+v", got.Loops)
	}
}

func TestInfoFleetWorkspaceCapturedRender(t *testing.T) {
	v := guardInfoVars{Fleet: fleetFixture(), FleetWorkspace: &infoFleetWorkspace{State: "READY", GeneratedAt: "2026-08-18T20:00:00Z", Configured: 2, Shown: 2, Loops: []fleetpane.LoopCheck{{Name: "stuck-loop", State: "ACTION", Detail: "no progress age=48m", Enabled: true}, {Name: "healthy-loop", State: "OK", Detail: "fresh progress age=2m", Enabled: true}}}}
	got := renderGuardInfoInteractiveBlock(infoViewState{active: viewFleet}, v, nil, 120, 20)
	for _, want := range []string{"«3 fleet»", "FLEET WORKSPACE · read-only", "machines", "stuck-loop · ACTION · no progress age=48m · next: fak fleetpane loop-check stuck-loop", "healthy-loop · OK · fresh progress age=2m · next: none"} {
		if !strings.Contains(got, want) {
			t.Fatalf("missing %q in captured render:\n%s", want, got)
		}
	}
}

func TestInfoFleetWorkspaceEmptyAndUnavailable(t *testing.T) {
	for _, w := range []*infoFleetWorkspace{{State: "EMPTY", Next: "fak fleetpane loop-list"}, {State: "UNAVAILABLE", Next: "fak fleetpane loop-list"}} {
		rows := strings.Join(fleetWorkspaceRows(guardInfoVars{FleetWorkspace: w}), "\n")
		if !strings.Contains(rows, w.State) || !strings.Contains(rows, "next: fak fleetpane loop-list") {
			t.Fatalf("%s", rows)
		}
	}
}

func TestInfoFixtureFleetSelfcheck(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fixture.json")
	raw := `{"fleet":{"schema":"session-fleet/1","verdict":"ATTENTION","machines":1,"rows":[]},"fleet_workspace":{"state":"READY","generated_at":"2026-08-18T20:00:00Z","configured":1,"shown":1,"loops":[{"name":"loop-a","enabled":true,"state":"ACTION","detail":"stalled age=30m"}]}}`
	if err := os.WriteFile(path, []byte(raw), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runInfoFixtureFrame(&out, &errb, path, "fleet", true, 100, 16); code != 0 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	if !strings.Contains(out.String(), "loop-a · ACTION") || !strings.Contains(out.String(), "fak fleetpane loop-check loop-a") {
		t.Fatalf("%s", out.String())
	}
}

func TestInfoFleetSelfcheckCapturedRun(t *testing.T) {
	var out bytes.Buffer
	if code := runInfoFleetSelfcheck(&out, 100, 16); code != 0 {
		t.Fatal(code)
	}
	for _, want := range []string{"«3 fleet»", "stuck-loop · ACTION", "fak fleetpane loop-check stuck-loop", "healthy-loop · OK", "SELFCHECK OK"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q:\n%s", want, out.String())
		}
	}
}
