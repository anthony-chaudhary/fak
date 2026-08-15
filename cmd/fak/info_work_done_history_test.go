package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func historyRecord(tokens float64, calls uint64, workload, run string, at time.Time) guardInfoWorkHistoryRecord {
	v := workDoneFixtureWith(tokens, calls, 0)
	v.Gateway.UptimeSeconds = 10
	q := guardInfoSessionWorkDoneQuery(v, at)
	return guardInfoWorkHistoryRecordFromQuery(q, workload, run, at)
}

func TestWorkHistoryThreeWindowComparisonAndDiscontinuities(t *testing.T) {
	t0 := time.Date(2026, 8, 14, 20, 0, 0, 0, time.UTC)
	first := historyRecord(100, 5, "compile-agent", "run-a", t0)
	improved := historyRecord(140, 7, "compile-agent", "run-b", t0.Add(time.Minute))
	cmp := guardInfoCompareWorkHistory(improved, []guardInfoWorkHistoryRecord{first})
	if cmp.Status != "improved" || cmp.TokenDelta != 40 || cmp.CallDelta != 2 || cmp.Attribution != "fak_mechanism_change" {
		t.Fatalf("improvement = %#v", cmp)
	}

	shiftedWorkload := historyRecord(200, 9, "docs-agent", "run-c", t0.Add(2*time.Minute))
	cmp = guardInfoCompareWorkHistory(shiftedWorkload, []guardInfoWorkHistoryRecord{first, improved})
	if cmp.Status != "incompatible" || cmp.Attribution != "workload_changed" {
		t.Fatalf("workload shift = %#v", cmp)
	}

	baselineShift := historyRecord(210, 10, "compile-agent", "run-d", t0.Add(3*time.Minute))
	baselineShift.Query.WorkDone.Baseline.ConfigurationSHA256 = "sha256:shifted"
	cmp = guardInfoCompareWorkHistory(baselineShift, []guardInfoWorkHistoryRecord{first, improved, shiftedWorkload})
	if cmp.Status != "incompatible" || cmp.Attribution != "baseline_changed" {
		t.Fatalf("baseline shift = %#v", cmp)
	}
	if row := strings.Join(guardInfoWorkHistoryRows(cmp), "\n"); !strings.Contains(row, "baseline changed") {
		t.Fatalf("discontinuity render: %s", row)
	}
}

func TestWorkHistoryQualityResetAndUnavailableGates(t *testing.T) {
	t0 := time.Unix(0, 0)
	prior := historyRecord(100, 5, "w", "a", t0)
	short := historyRecord(110, 6, "w", "b", t0.Add(time.Second))
	short.Query.Window.DurationNanos = int64(100 * time.Millisecond)
	if got := guardInfoCompareWorkHistory(short, []guardInfoWorkHistoryRecord{prior}); got.Attribution != "quality_gate" {
		t.Fatalf("short window = %#v", got)
	}
	reset := historyRecord(110, 6, "w", "c", t0.Add(2*time.Second))
	reset.Query.Window.Reset = true
	if got := guardInfoCompareWorkHistory(reset, []guardInfoWorkHistoryRecord{prior}); got.Attribution != "reset" {
		t.Fatalf("reset = %#v", got)
	}
	missing := historyRecord(110, 6, "w", "d", t0.Add(3*time.Second))
	missing.Query.WorkDone.Metrics.InputTokensAvoided.Available = false
	if got := guardInfoCompareWorkHistory(missing, []guardInfoWorkHistoryRecord{prior}); got.Attribution != "evidence_unavailable" {
		t.Fatalf("missing = %#v", got)
	}
}

func TestWorkHistoryPersistenceIsBoundedAndPrivacySafe(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history", "work.jsonl")
	secretWorkload, secretRun := "C:/secret/client/repo prompt text", "tool-args=rm-private"
	for i := 0; i < guardInfoWorkHistoryMaxRecords+3; i++ {
		r := historyRecord(float64(i), uint64(i), secretWorkload, secretRun, time.Unix(int64(i), 0))
		if err := guardInfoAppendWorkHistory(path, r); err != nil {
			t.Fatal(err)
		}
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(raw), secretWorkload) || strings.Contains(string(raw), secretRun) || strings.Contains(string(raw), "prompt") || strings.Contains(string(raw), "tool-args") {
		t.Fatalf("history leaked raw identity: %s", raw)
	}
	records, err := guardInfoReadWorkHistory(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != guardInfoWorkHistoryMaxRecords {
		t.Fatalf("retained %d records", len(records))
	}
	if records[0].WorkloadID == "" || !strings.HasPrefix(records[0].RunID, "sha256:") {
		t.Fatalf("identities not hashed: %#v", records[0])
	}
	if info, err := os.Stat(path); err != nil || info.Mode().Perm()&0o077 != 0 {
		t.Fatalf("history permissions = %v err=%v", info.Mode(), err)
	}
}

func TestWorkHistoryExportMatchesComparedRecord(t *testing.T) {
	t0 := time.Unix(100, 0)
	prior, current := historyRecord(10, 1, "w", "r1", t0), historyRecord(20, 2, "w", "r2", t0.Add(time.Second))
	export := guardInfoWorkHistoryExport{Schema: guardInfoWorkHistorySchema, Records: guardInfoComparedHistoryRecords(current, []guardInfoWorkHistoryRecord{prior}), Comparison: guardInfoCompareWorkHistory(current, []guardInfoWorkHistoryRecord{prior})}
	raw, err := json.Marshal(export)
	if err != nil {
		t.Fatal(err)
	}
	var decoded guardInfoWorkHistoryExport
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatal(err)
	}
	if len(decoded.Records) != 2 || decoded.Records[0].RecordedAt != prior.RecordedAt || decoded.Records[1].RecordedAt != current.RecordedAt || decoded.Comparison.PriorRecordedAt != prior.RecordedAt {
		t.Fatalf("export mismatch = %#v", decoded)
	}
}

func TestWorkHistoryCapturedTUIRender(t *testing.T) {
	t0 := time.Unix(100, 0)
	prior, current := historyRecord(10, 1, "w", "r1", t0), historyRecord(25, 3, "w", "r2", t0.Add(time.Second))
	v := workDoneFixtureWith(25, 3, 0)
	cmp := guardInfoCompareWorkHistory(current, []guardInfoWorkHistoryRecord{prior})
	v.WorkHistory = &cmp
	rows := strings.Join(guardInfoWorkDoneRows(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 140), guardPanelFull), "\n")
	if !strings.Contains(rows, "history improved · tokens +15 · calls +2") {
		t.Fatalf("history render:\n%s", rows)
	}
}

func TestWorkHistoryQueryExportsExactPriorAndCurrentRecords(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	prior := historyRecord(10, 1, "w", "r1", time.Now().Add(-time.Minute))
	if err := guardInfoAppendWorkHistory(path, prior); err != nil {
		t.Fatal(err)
	}
	fixture := workDoneFixtureWith(20, 2, 0)
	fixture.Gateway.UptimeSeconds = 10
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _ = json.NewEncoder(w).Encode(fixture) }))
	defer ts.Close()
	c := &claudeMacDebugClient{base: ts.URL, hc: ts.Client()}
	var stdout, stderr bytes.Buffer
	if code := runInfoWorkDoneHistoryQuery(&stdout, &stderr, c, 0, path, "w", "r2"); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var export guardInfoWorkHistoryExport
	if err := json.Unmarshal(stdout.Bytes(), &export); err != nil {
		t.Fatal(err)
	}
	if len(export.Records) != 2 || export.Records[0].RecordedAt != prior.RecordedAt || export.Comparison.Status != "improved" {
		t.Fatalf("history export = %#v", export)
	}
	records, err := guardInfoReadWorkHistory(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 2 {
		t.Fatalf("persisted records=%d", len(records))
	}
}

func TestDecorateWorkHistoryFeedsLiveTUIWithoutPersistingEveryTick(t *testing.T) {
	path := filepath.Join(t.TempDir(), "history.jsonl")
	prior := historyRecord(10, 1, "w", "r1", time.Now().Add(-time.Minute))
	if err := guardInfoAppendWorkHistory(path, prior); err != nil {
		t.Fatal(err)
	}
	c := &claudeMacDebugClient{workHistoryPath: path, workloadKey: "w", runKey: "r2"}
	v := workDoneFixtureWith(20, 2, 0)
	v.Gateway.UptimeSeconds = 10
	c.decorateWorkHistory(&v)
	if v.WorkHistory == nil || v.WorkHistory.Status != "improved" {
		t.Fatalf("live comparison = %#v", v.WorkHistory)
	}
	records, err := guardInfoReadWorkHistory(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(records) != 1 {
		t.Fatalf("watch tick must not append repeatedly: %d", len(records))
	}
}
