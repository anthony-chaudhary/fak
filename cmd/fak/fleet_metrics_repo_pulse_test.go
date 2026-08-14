package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestFleetMetricsAggregatesRepoPulseReceiptsWithoutDoubleCount(t *testing.T) {
	dir := t.TempDir()
	rows := map[string]string{
		"a.json":                        `{"schema":"fak-dispatch-worker/1","issue":10,"pid":100,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":400,"tool_turns_skipped":2,"journal_rows":3}}`,
		"a-retry.json":                  `{"schema":"fak-dispatch-worker/1","issue":10,"pid":100,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":400,"tool_turns_skipped":2,"journal_rows":3}}`,
		"b.json":                        `{"schema":"fak-dispatch-worker/1","issue":11,"pid":101,"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":600,"tool_turns_skipped":2,"journal_rows":3}}`,
		"old.json":                      `{"schema":"fak-dispatch-worker/1","issue":9,"pid":99}`,
		"repo-pulse-launch-12-102.json": `{"schema":"fleet-issue-resolve-dispatch/1","spawned":{"issue":12,"pid":102},"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":700,"tool_turns_skipped":2,"journal_rows":3}}`,
		"last-resolve-tick.json":        `{"schema":"fleet-issue-resolve-dispatch/1","spawned":{"issue":12,"pid":102},"repo_pulse_receipt":{"schema":"fak-dispatch-repo-pulse-receipt/1","saved_tokens":700,"tool_turns_skipped":2,"journal_rows":3}}`,
	}
	for name, body := range rows {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	totals := foldDispatchRepoPulseReceipts(dir)
	if totals.Launches != 3 || totals.SavedTokens != 1700 || totals.ToolTurnsSkipped != 6 || totals.JournalRows != 9 || totals.DuplicateRows != 2 {
		t.Fatalf("totals=%+v", totals)
	}
	w := newPromWriter()
	writeRepoPulseMetrics(w, dir)
	raw := w.String()
	for _, want := range []string{"fak_fleet_repo_pulse_launches_total 3", "fak_fleet_repo_pulse_context_tokens_saved_total 1700", "fak_fleet_repo_pulse_tool_turns_skipped_total 6", "fak_fleet_repo_pulse_duplicate_receipts_dropped_total 2", "fak_fleet_repo_pulse_cohort_ready 0", "fak_fleet_repo_pulse_cohort_sample_deficit 2"} {
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

func TestRepoPulseReadinessFoldsNewestDispatchBlocker(t *testing.T) {
	dir := t.TempDir()
	older := filepath.Join(dir, "last-resolve-tick-claude.json")
	newer := filepath.Join(dir, "last-resolve-tick-codex.json")
	if err := os.WriteFile(older, []byte(`{"action":"launch","verdict":"PASS"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(newer, []byte(`{"action":"refused","verdict":"REFUSE_NO_SEAT","backend":"codex","reason":"seat pool depleted"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	oldTime := time.Now().Add(-time.Minute)
	if err := os.Chtimes(older, oldTime, oldTime); err != nil {
		t.Fatal(err)
	}
	r := assessRepoPulseCohort(dir, 5)
	if r.DispatchBlocker != "REFUSE_NO_SEAT" || r.DispatchEvidence != filepath.Base(newer) || !strings.Contains(r.NextAction, "worker to exit") {
		t.Fatalf("got=%+v", r)
	}
}

func TestRepoPulseReadinessSurfacesMalformedNewestEvidence(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "last-resolve-tick.json")
	if err := os.WriteFile(path, []byte(`not-json`), 0o600); err != nil {
		t.Fatal(err)
	}
	r := assessRepoPulseCohort(dir, 5)
	if r.DispatchBlocker != "dispatch-evidence-unreadable" || !strings.Contains(r.NextAction, "inspect") {
		t.Fatalf("got=%+v", r)
	}
}

func TestMicroDogfoodReadinessJSONIncludesDispatchBlocker(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "last-resolve-tick.json"), []byte(`{"action":"refused","verdict":"REFUSE_NO_SEAT","backend":"codex"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if code := runMicroDogfoodReadiness(&out, &errb, []string{"--runs-dir", dir, "--json"}); code != 3 {
		t.Fatalf("code=%d err=%s", code, errb.String())
	}
	var got repoPulseCohortReadiness
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.DispatchBlocker != "REFUSE_NO_SEAT" || got.PostLaunches != 0 {
		t.Fatalf("got=%+v", got)
	}
}
