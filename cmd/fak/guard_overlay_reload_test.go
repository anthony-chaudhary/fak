package main

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGuardPolicyReloadReportsMalformedOverlay(t *testing.T) {
	overlay := filepath.Join(t.TempDir(), "allow.json")
	if err := os.WriteFile(overlay, []byte(`{"version":"broken"`), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv(guardAllowOverlayEnv, overlay)

	resp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatalf("reload must tolerate overlay failure: %v", err)
	}
	if !resp.Reloaded || !strings.Contains(resp.Summary, "overlay_error:") || !strings.Contains(resp.Summary, overlay) {
		t.Fatalf("reload response omitted overlay degradation: %+v", resp)
	}
}

func TestGuardPolicyReloadMissingOverlaySilent(t *testing.T) {
	overlay := filepath.Join(t.TempDir(), "missing.json")
	t.Setenv(guardAllowOverlayEnv, overlay)

	resp, err := guardPolicyReloader("")(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(resp.Summary, "overlay_error:") {
		t.Fatalf("missing overlay must remain silent: %+v", resp)
	}
}

func TestSaveGuardAllowOverlayReplacesAtomically(t *testing.T) {
	path := filepath.Join(t.TempDir(), "allow.json")
	if err := os.WriteFile(path, []byte("old"), 0o644); err != nil {
		t.Fatal(err)
	}
	want := guardAllowOverlay{Allow: []string{"search_kb"}}
	if err := saveGuardAllowOverlay(path, want); err != nil {
		t.Fatal(err)
	}
	got, err := loadGuardAllowOverlay(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Allow) != 1 || got.Allow[0] != "search_kb" {
		t.Fatalf("overlay=%+v", got)
	}
	matches, err := filepath.Glob(filepath.Join(filepath.Dir(path), ".allow-*.json.tmp"))
	if err != nil || len(matches) != 0 {
		t.Fatalf("temporary overlay leaked: %v err=%v", matches, err)
	}
}
