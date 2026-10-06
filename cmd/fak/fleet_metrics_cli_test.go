package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// TestRenderFleetCLIUsageExpositionFoldsJournal pins that the fleet exporter's
// fak_cli_* block carries the readability gauge plus the per-verb families the
// fak-cli-invocation-telemetry board queries.
func TestRenderFleetCLIUsageExpositionFoldsJournal(t *testing.T) {
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	lg, err := usagelog.Open(path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if _, err := lg.Append(usagelog.Row{Verb: "perf", ExitCode: 0, DurationMS: 12}); err != nil {
		t.Fatalf("Append: %v", err)
	}
	lg.Close()

	out := renderFleetCLIUsageExposition(path, time.Now())
	for _, want := range []string{
		"fak_cli_usage_log_readable 1",
		"fak_cli_usage_rows_in_window 1",
		`fak_cli_invocations_total{verb="perf"`,
		"fak_cli_invocation_duration_ms_max",
		"fak_cli_usage_render_timestamp_seconds",
	} {
		if !strings.Contains(out, want) {
			t.Errorf("exposition missing %q:\n%s", want, out)
		}
	}
}

// TestRenderFleetCLIUsageExpositionMissingAndUnreadable pins the honesty split:
// a missing journal is readable with zero rows; an unreadable one reads 0.
func TestRenderFleetCLIUsageExpositionMissingAndUnreadable(t *testing.T) {
	dir := t.TempDir()
	missing := renderFleetCLIUsageExposition(filepath.Join(dir, "nope.jsonl"), time.Now())
	if !strings.Contains(missing, "fak_cli_usage_log_readable 1") || !strings.Contains(missing, "fak_cli_usage_rows_in_window 0") {
		t.Errorf("missing journal must be an honest empty fold:\n%s", missing)
	}
	// A directory in place of the journal cannot be read as a file.
	asDir := filepath.Join(dir, "isdir")
	if err := os.Mkdir(asDir, 0o755); err != nil {
		t.Fatal(err)
	}
	bad := renderFleetCLIUsageExposition(asDir, time.Now())
	if !strings.Contains(bad, "fak_cli_usage_log_readable 0") || strings.Contains(bad, "fak_cli_usage_rows_in_window") {
		t.Errorf("unreadable journal must read 0 and emit no fold:\n%s", bad)
	}
}
