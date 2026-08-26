//go:build darwin

package sessionjournal

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultPathUsesDarwinUserConfigDir(t *testing.T) {
	t.Setenv(EnvPath, "")
	home := t.TempDir()
	t.Setenv("HOME", home)

	want := filepath.Join(home, "Library", "Application Support", "fak", "session-journal", "events.jsonl")
	if got := DefaultPath(); got != want {
		t.Fatalf("DefaultPath()=%q want Darwin user path %q", got, want)
	}
}

func TestDarwinDefaultPathAcceptsOrdinaryUserAppend(t *testing.T) {
	t.Setenv(EnvPath, "")
	home := t.TempDir()
	t.Setenv("HOME", home)

	event := Event{Kind: KindOpen, ID: "darwin-user", TS: "2026-08-25T00:00:00Z"}
	if err := Append("", event); err != nil {
		t.Fatalf("Append using Darwin default: %v", err)
	}

	wantPath := filepath.Join(home, "Library", "Application Support", "fak", "session-journal", "events.jsonl")
	data, err := os.ReadFile(wantPath)
	if err != nil {
		t.Fatalf("ReadFile(%q): %v", wantPath, err)
	}
	want := "{\"schema\":\"fak.sessionjournal.v1\",\"kind\":\"open\",\"id\":\"darwin-user\",\"ts\":\"2026-08-25T00:00:00Z\"}\n"
	if string(data) != want {
		t.Fatalf("journal contents=%q want %q", data, want)
	}
}

func TestDarwinDefaultPathWithoutHomeUsesMachineStore(t *testing.T) {
	t.Setenv(EnvPath, "")
	t.Setenv("HOME", "")

	want := filepath.Join("/var", "lib", "fak", "session-journal", "events.jsonl")
	if got := DefaultPath(); got != want {
		t.Fatalf("DefaultPath()=%q want home-less fallback %q", got, want)
	}
}
