package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFleetMetricsAggregatesRepoPulseReceiptsWithoutDoubleCount(t *testing.T) {
	dir := t.TempDir()
	rows := map[string]string{
		"a.json":       `{"schema":"fak-dispatch-worker/1","issue":10,"pid":100,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":400,"tool_turns_skipped":2,"journal_rows":3}}`,
		"a-retry.json": `{"schema":"fak-dispatch-worker/1","issue":10,"pid":100,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":400,"tool_turns_skipped":2,"journal_rows":3}}`,
		"b.json":       `{"schema":"fak-dispatch-worker/1","issue":11,"pid":101,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":600,"tool_turns_skipped":2,"journal_rows":3}}`,
		"old.json":     `{"schema":"fak-dispatch-worker/1","issue":9,"pid":99}`,
	}
	for name, body := range rows {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	totals := foldDispatchRepoPulseReceipts(dir)
	if totals.Launches != 2 || totals.SavedTokens != 1000 || totals.ToolTurnsSkipped != 4 || totals.JournalRows != 6 || totals.DuplicateRows != 1 {
		t.Fatalf("totals=%+v", totals)
	}
	w := newPromWriter()
	writeRepoPulseMetrics(w, dir)
	raw := w.String()
	for _, want := range []string{"fak_fleet_repo_pulse_launches_total 2", "fak_fleet_repo_pulse_context_tokens_saved_total 1000", "fak_fleet_repo_pulse_tool_turns_skipped_total 4", "fak_fleet_repo_pulse_duplicate_receipts_dropped_total 1"} {
		if !strings.Contains(raw, want) {
			t.Fatalf("missing %q:\n%s", want, raw)
		}
	}
}
