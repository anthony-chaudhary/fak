package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGuardRelaunchFilesRestoreGeneratedConfigsAcrossRestart(t *testing.T) {
	root := t.TempDir()
	settings := filepath.Join(root, "fak-guard-precompact-123-1", "claude-precompact-settings.json")
	mcp := filepath.Join(root, "fak-guard-mcp-123-2", "mcp.json")
	want := map[string]string{settings: `{"hooks":{"PreCompact":[]}}`, mcp: `{"mcpServers":{}}`}
	for path, data := range want {
		if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	snapshot, err := captureGuardRelaunchFiles([]string{"claude", "--settings", settings, "--mcp-config=" + mcp, "-p", "work"})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	if got := len(snapshot.files); got != 2 {
		t.Fatalf("captured files = %d, want 2", got)
	}
	if err := os.RemoveAll(filepath.Dir(settings)); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(filepath.Dir(mcp)); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.ensure(); err != nil {
		t.Fatalf("ensure before restart launch: %v", err)
	}
	for path, expected := range want {
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("restart referenced missing config %q: %v", path, err)
		}
		if string(got) != expected {
			t.Fatalf("restored %q = %q, want %q", path, got, expected)
		}
	}
	if err := os.RemoveAll(root); err != nil {
		t.Fatalf("final cleanup: %v", err)
	}
	if _, err := os.Stat(root); !os.IsNotExist(err) {
		t.Fatalf("temp tree survived final cleanup: %v", err)
	}
}

func TestGuardRelaunchFilesDoNotOverwriteExistingConfig(t *testing.T) {
	path := filepath.Join(t.TempDir(), "fak-guard-precompact-123-1", "settings.json")
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("original"), 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := captureGuardRelaunchFiles([]string{"claude", "--settings", path})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("updated"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := snapshot.ensure(); err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != "updated" {
		t.Fatalf("existing config overwritten: %q", got)
	}
}

func TestGuardRelaunchFilesIgnoreCallerOwnedSettings(t *testing.T) {
	path := filepath.Join(t.TempDir(), "settings.json")
	if err := os.WriteFile(path, []byte("caller-owned"), 0o600); err != nil {
		t.Fatal(err)
	}
	snapshot, err := captureGuardRelaunchFiles([]string{"claude", "--settings", path})
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.files) != 0 {
		t.Fatalf("captured %d caller-owned files, want 0", len(snapshot.files))
	}
}
