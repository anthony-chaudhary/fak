//go:build windows

package sessionjournal

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestDefaultPathUsesProgramData(t *testing.T) {
	t.Setenv(EnvPath, "")
	root := t.TempDir()
	t.Setenv("ProgramData", root)
	got := DefaultPath()
	want := filepath.Join(root, "fak", "session-journal", "events.jsonl")
	if got != want {
		t.Fatalf("DefaultPath()=%q want %q", got, want)
	}
}

func TestBootTimePersistsNativeMarker(t *testing.T) {
	t.Setenv(EnvPath, filepath.Join(t.TempDir(), "events.jsonl"))
	approximate := approximateWindowsBootTime(testNow())
	if approximate.IsZero() {
		t.Skip("GetTickCount64 unavailable")
	}

	first, source := BootTime(testNow())
	if source != "gettickcount64" || first.IsZero() {
		t.Fatalf("first=(%v,%q)", first, source)
	}
	second, source := BootTime(testNow())
	if source != "gettickcount64-marker" || !second.Equal(first) {
		t.Fatalf("second=(%v,%q), want marker %v", second, source, first)
	}
	b, err := os.ReadFile(bootMarkerPath())
	if err != nil || !strings.Contains(string(b), "gettickcount64") {
		t.Fatalf("marker=%q err=%v", b, err)
	}
}

func testNow() time.Time { return time.Now().UTC() }
