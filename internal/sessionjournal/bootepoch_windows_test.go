//go:build windows

package sessionjournal

import (
	"errors"
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

func TestBootTimePersistsExactWMIMarker(t *testing.T) {
	t.Setenv(EnvPath, filepath.Join(t.TempDir(), "events.jsonl"))
	approximate := approximateWindowsBootTime(testNow())
	if approximate.IsZero() {
		t.Skip("GetTickCount64 unavailable")
	}
	calls := 0
	old := queryWindowsBootTime
	queryWindowsBootTime = func() (time.Time, error) { calls++; return approximate.Truncate(time.Second), nil }
	t.Cleanup(func() { queryWindowsBootTime = old })

	first, source := BootTime(testNow())
	if source != "wmi-lastbootuptime" || first.IsZero() {
		t.Fatalf("first=(%v,%q)", first, source)
	}
	queryWindowsBootTime = func() (time.Time, error) { calls++; return time.Time{}, errors.New("WMI unavailable") }
	second, source := BootTime(testNow())
	if source != "wmi-lastbootuptime-marker" || !second.Equal(first) {
		t.Fatalf("second=(%v,%q), want marker %v", second, source, first)
	}
	if calls != 1 {
		t.Fatalf("WMI calls=%d want 1", calls)
	}
	b, err := os.ReadFile(bootMarkerPath())
	if err != nil || !strings.Contains(string(b), "wmi-lastbootuptime") {
		t.Fatalf("marker=%q err=%v", b, err)
	}
}

func testNow() time.Time { return time.Now().UTC() }
