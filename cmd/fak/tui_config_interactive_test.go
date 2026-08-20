package main

import (
	"bytes"
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
