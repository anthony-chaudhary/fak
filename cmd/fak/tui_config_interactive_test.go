package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestTUIConfigInteractionSaveResetCaptured(t *testing.T) {
	at := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "console.json")
	var capture bytes.Buffer

	if err := runTUIConfigInteraction(strings.NewReader("csq"), &capture, path, "guard.color", 72, at); err != nil {
		t.Fatalf("save interaction: %v", err)
	}
	saved := buildTUIConfigReport(path, at)
	if !hasTUIConfigSetting(saved.Settings, "guard", "color", "always", "saved") {
		t.Fatalf("saved settings read-back = %+v", saved.Settings)
	}

	if err := runTUIConfigInteraction(strings.NewReader("rq"), &capture, path, "guard.color", 72, at); err != nil {
		t.Fatalf("reset interaction: %v", err)
	}
	final := buildTUIConfigReport(path, at)
	if !hasTUIConfigSetting(final.Settings, "guard", "color", "auto", "built-in") {
		t.Fatalf("reset settings read-back = %+v", final.Settings)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read final config: %v", err)
	}
	if bytes.Contains(data, []byte(`"color"`)) {
		t.Fatalf("reset left guard.color in config: %s", data)
	}

	out := capture.String()
	for _, want := range []string{
		"fak console settings interactive",
		"> guard.color effective=auto source=built-in",
		"status=pending guard.color=always",
		"> guard.color effective=always source=saved",
		"status=saved guard.color=always",
		"status=reset guard.color to built-in",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("captured interaction missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, "\x1b") {
		t.Fatalf("captured interaction contains terminal escape corruption: %q", out)
	}
	for _, line := range strings.Split(strings.TrimSuffix(out, "\n"), "\n") {
		if len([]rune(line)) > 72 {
			t.Fatalf("captured line exceeds width 72 (%d): %q", len([]rune(line)), line)
		}
	}
}

func TestTUIConfigInteractionSavesModelAndEffort(t *testing.T) {
	at := time.Date(2026, 8, 28, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "console.json")
	var capture bytes.Buffer

	// An unset finite select advances to its first option. Three changes select high.
	if err := runTUIConfigInteraction(strings.NewReader("csq"), &capture, path, "agent.model", 96, at); err != nil {
		t.Fatalf("save model interaction: %v", err)
	}
	if err := runTUIConfigInteraction(strings.NewReader("cccsq"), &capture, path, "agent.effort", 96, at); err != nil {
		t.Fatalf("save effort interaction: %v", err)
	}

	saved := buildTUIConfigReport(path, at)
	if !hasTUIConfigSetting(saved.Settings, "agent", "model", "claude-opus-5", "saved") {
		t.Fatalf("saved model read-back = %+v", saved.Settings)
	}
	if !hasTUIConfigSetting(saved.Settings, "agent", "effort", "high", "saved") {
		t.Fatalf("saved effort read-back = %+v", saved.Settings)
	}
	written, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read keyboard-produced config: %v", err)
	}
	wantConfig := "{\n  \"pane_defaults\": {\n    \"agent\": {\n      \"effort\": \"high\",\n      \"model\": \"claude-opus-5\"\n    }\n  }\n}\n"
	if string(written) != wantConfig {
		t.Fatalf("keyboard-produced config bytes:\n%s\nwant:\n%s", written, wantConfig)
	}

	// Close the proof over the real dispatcher: a fresh managed launch must consume
	// the exact file produced by the keyboard interaction above.
	var launchJSON, launchErr bytes.Buffer
	if code := runTUI(&launchJSON, &launchErr, []string{"agent", "--console-config", path, "--json", "--at", "2026-08-28T12:00:00Z"}); code != 0 {
		t.Fatalf("fresh runTUI from keyboard config code=%d stderr=%s", code, launchErr.String())
	}
	var launch tuiAgentReport
	if err := json.Unmarshal(launchJSON.Bytes(), &launch); err != nil {
		t.Fatalf("unmarshal fresh launch: %v\n%s", err, launchJSON.String())
	}
	if launch.Model != "claude-opus-5" || launch.Effort != "high" || !hasTUIArgPair(launch.Launch, "--model", "claude-opus-5") || !hasTUIArgPair(launch.Command, "--effort", "high") {
		t.Fatalf("fresh launch did not consume keyboard config: model=%q effort=%q command=%v launch=%v", launch.Model, launch.Effort, launch.Command, launch.Launch)
	}
	for _, want := range []string{
		"agent.model effective=claude-opus-5 source=saved",
		"status=saved agent.model=claude-opus-5",
		"agent.effort effective=high source=saved",
		"status=saved agent.effort=high",
	} {
		if !strings.Contains(capture.String(), want) {
			t.Fatalf("captured interaction missing %q:\n%s", want, capture.String())
		}
	}
	if strings.Contains(capture.String(), "\x1b") {
		t.Fatalf("captured interaction contains terminal escape corruption: %q", capture.String())
	}
}

func TestTUIConfigInteractionRefusesNonFiniteOrSensitiveSetting(t *testing.T) {
	at := time.Date(2026, 8, 20, 12, 0, 0, 0, time.UTC)
	path := filepath.Join(t.TempDir(), "console.json")
	for _, ref := range []string{"issues.top", "sessions.key"} {
		var out bytes.Buffer
		err := runTUIConfigInteraction(strings.NewReader("q"), &out, path, ref, 72, at)
		if err == nil || !strings.Contains(err.Error(), "not a finite, persistable setting") {
			t.Fatalf("setting %s error = %v", ref, err)
		}
		if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
			t.Fatalf("refused setting %s wrote config (stat=%v)", ref, statErr)
		}
	}
}
