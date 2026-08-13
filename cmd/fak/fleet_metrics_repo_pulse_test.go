package main

import (
	"bytes"
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
	for _, want := range []string{"fak_fleet_repo_pulse_launches_total 2", "fak_fleet_repo_pulse_context_tokens_saved_total 1000", "fak_fleet_repo_pulse_tool_turns_skipped_total 4", "fak_fleet_repo_pulse_duplicate_receipts_dropped_total 1", "fak_fleet_repo_pulse_cohort_ready 0", "fak_fleet_repo_pulse_cohort_sample_deficit 3"} {
		if !strings.Contains(raw, want) {
			t.Fatalf("missing %q:\n%s", want, raw)
		}
	}
}

func TestRepoPulseCohortReadinessRefusesThinEvidence(t *testing.T) {
	dir := t.TempDir()
	if got := assessRepoPulseCohort(dir, 5); got.Verdict != "not-yet" || got.PostLaunches != 0 || !strings.Contains(got.Reason, "need 5") {
		t.Fatalf("got=%+v", got)
	}
}

func TestMicroDogfoodReadinessJSONRefusesWithoutSamples(t *testing.T) {
	var out, errb bytes.Buffer
	code := runMicroDogfoodReadiness(&out, &errb, []string{"--runs-dir", t.TempDir(), "--minimum", "5", "--json"})
	if code != 3 || !strings.Contains(out.String(), `"verdict":"not-yet"`) || !strings.Contains(out.String(), `"post_launches":0`) {
		t.Fatalf("code=%d out=%s err=%s", code, out.String(), errb.String())
	}
}
