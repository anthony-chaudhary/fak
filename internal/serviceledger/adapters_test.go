package serviceledger

import (
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/servicespec"
)

func parseFixture(t *testing.T, name string, parse func(io.Reader, AdapterConfig) ([]Event, error), cfg AdapterConfig) []Event {
	t.Helper()
	f, err := os.Open(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evs, err := parse(f, cfg)
	if err != nil {
		t.Fatal(err)
	}
	if len(evs) == 0 {
		t.Fatalf("fixture %s parsed to zero events", name)
	}
	return evs
}

func timelineOf(t *testing.T, evs []Event) string {
	t.Helper()
	led := Memory()
	if _, _, err := led.AppendAll(evs); err != nil {
		t.Fatal(err)
	}
	var sb strings.Builder
	WriteTimeline(&sb, led.Events())
	return sb.String()
}

// mustGolden compares got with the checked-in golden timeline; run with
// FAK_UPDATE_GOLDEN=1 to regenerate after an intentional format change.
func mustGolden(t *testing.T, name, got string) {
	t.Helper()
	path := filepath.Join("testdata", name)
	if os.Getenv("FAK_UPDATE_GOLDEN") == "1" {
		if err := os.WriteFile(path, []byte(got), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	want, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("missing golden %s (set FAK_UPDATE_GOLDEN=1 to create): %v", name, err)
	}
	normalize := func(s string) string { return strings.ReplaceAll(s, "\r\n", "\n") }
	if normalize(string(want)) != normalize(got) {
		t.Fatalf("golden %s mismatch:\n--- want ---\n%s--- got ---\n%s", name, want, got)
	}
}

func windowsCfg() AdapterConfig {
	return AdapterConfig{Identity: servicespec.Identity{Node: "node-a", Service: "guard"}, Unit: "fakguard"}
}

func TestWindowsEventXMLGoldenTimeline(t *testing.T) {
	evs := parseFixture(t, "windows_eventlog.xml", AdaptWindowsEventXML, windowsCfg())
	got := timelineOf(t, evs)
	mustGolden(t, "windows_timeline.golden", got)
	// Crash-to-ready causality is readable straight off the timeline: boot,
	// ready, host-crash row, SCM crash row, ready again, then intent change.
	for _, marker := range []string{"boot-change", "process-exit class=crash", "readiness phase=ready", "desired-change desired=desired-stopped"} {
		if !strings.Contains(got, marker) {
			t.Errorf("windows timeline lacks %q:\n%s", marker, got)
		}
	}
	// The Application Error command line is ledgered redacted.
	if strings.Contains(got, "deadbeefcafe") || !strings.Contains(got, "--lease-token=[REDACTED]") {
		t.Errorf("windows timeline leaked a command-line secret:\n%s", got)
	}
}

func TestWindowsTaskSchedulerGoldenTimeline(t *testing.T) {
	cfg := AdapterConfig{Identity: servicespec.Identity{Node: "node-a", Service: "guard-sync"}, Unit: `\fak\guard-sync`}
	evs := parseFixture(t, "windows_taskscheduler.xml", AdaptWindowsEventXML, cfg)
	got := timelineOf(t, evs)
	mustGolden(t, "windows_taskscheduler_timeline.golden", got)
	// Scheduled-task causality: launch -> failed run -> relaunch -> clean run.
	idx := -1
	for _, marker := range []string{
		"manager-restart",
		"process-exit class=crash code=2147943645",
		"manager-restart",
		"process-exit class=clean",
	} {
		at := strings.Index(got[idx+1:], marker)
		if at < 0 {
			t.Fatalf("task scheduler timeline lacks ordered marker %q:\n%s", marker, got)
		}
		idx += 1 + at
	}
	if strings.Contains(got, "unrelated-task") {
		t.Fatalf("task-name filter leaked a foreign task:\n%s", got)
	}
}

func TestJournaldGoldenTimeline(t *testing.T) {
	cfg := AdapterConfig{Identity: servicespec.Identity{Node: "node-b", Service: "guard"}, Unit: "fakguard.service"}
	evs := parseFixture(t, "journald.jsonl", AdaptJournaldExport, cfg)
	got := timelineOf(t, evs)
	mustGolden(t, "journald_timeline.golden", got)
	// Ordered causality: ready -> crash -> manager restart -> ready ->
	// boot/incarnation change -> ready -> watchdog timeout.
	idx := -1
	for _, marker := range []string{
		"readiness phase=ready",
		"process-exit class=crash code=1",
		"manager-restart",
		"boot-change",
		"watchdog-timeout",
	} {
		at := strings.Index(got[idx+1:], marker)
		if at < 0 {
			t.Fatalf("journald timeline lacks ordered marker %q:\n%s", marker, got)
		}
		idx += 1 + at
	}
}

func TestLaunchdGoldenTimeline(t *testing.T) {
	cfg := AdapterConfig{Identity: servicespec.Identity{Node: "node-c", Service: "guard"}, Unit: "com.fak.guard"}
	evs := parseFixture(t, "launchd.ndjson", AdaptLaunchdNDJSON, cfg)
	got := timelineOf(t, evs)
	mustGolden(t, "launchd_timeline.golden", got)
	for _, marker := range []string{"manager-restart", "readiness phase=ready", "process-exit class=crash code=2"} {
		if !strings.Contains(got, marker) {
			t.Errorf("launchd timeline lacks %q:\n%s", marker, got)
		}
	}
}

func TestAdapterReplayIsExactOnce(t *testing.T) {
	for _, tc := range []struct {
		name  string
		parse func(io.Reader, AdapterConfig) ([]Event, error)
		cfg   AdapterConfig
	}{
		{"windows_eventlog.xml", AdaptWindowsEventXML, windowsCfg()},
		{"windows_taskscheduler.xml", AdaptWindowsEventXML, AdapterConfig{Identity: servicespec.Identity{Node: "n", Service: "guard-sync"}, Unit: `\fak\guard-sync`}},
		{"journald.jsonl", AdaptJournaldExport, AdapterConfig{Identity: servicespec.Identity{Node: "n", Service: "guard"}, Unit: "fakguard.service"}},
		{"launchd.ndjson", AdaptLaunchdNDJSON, AdapterConfig{Identity: servicespec.Identity{Node: "n", Service: "guard"}, Unit: "com.fak.guard"}},
	} {
		led := Memory()
		first := parseFixture(t, tc.name, tc.parse, tc.cfg)
		ingested, duplicates, err := led.AppendAll(first)
		if err != nil || ingested != len(first) || duplicates != 0 {
			t.Fatalf("%s first ingest: ingested=%d dup=%d err=%v", tc.name, ingested, duplicates, err)
		}
		second := parseFixture(t, tc.name, tc.parse, tc.cfg)
		ingested, duplicates, err = led.AppendAll(second)
		if err != nil || ingested != 0 || duplicates != len(second) {
			t.Fatalf("%s replay was not idempotent: ingested=%d dup=%d err=%v", tc.name, ingested, duplicates, err)
		}
	}
}

func TestAdapterRequiresIdentity(t *testing.T) {
	if _, err := AdaptWindowsEventXML(strings.NewReader(""), AdapterConfig{}); err == nil {
		t.Fatal("adapter accepted an empty identity")
	}
}

func TestWindowsUnitFilterExcludesOtherServices(t *testing.T) {
	cfg := windowsCfg()
	cfg.Unit = "someother"
	f, err := os.Open(filepath.Join("testdata", "windows_eventlog.xml"))
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evs, err := AdaptWindowsEventXML(f, cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range evs {
		if e.Type != EventBootChange {
			t.Fatalf("unit filter leaked a foreign row: %+v", e)
		}
	}
}
