package main

import (
	"bytes"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func workDoneFixtureWith(tokens float64, memo, inline uint64) guardInfoVars {
	v := workDoneFixture()
	v.VCache.SavedTokenEquiv = tokens
	v.CacheAttribution.TotalTokenEquiv = tokens
	v.CacheAttribution.ProviderTokenEquiv = tokens * 0.6
	v.CacheAttribution.FakTokenEquiv = tokens * 0.4
	v.CacheAttribution.FakCompactionShedTokens = uint64(tokens * 0.4)
	v.CacheAttribution.FakResponseMemoCalls = memo
	v.CacheAttribution.FakInlineServedCalls = inline
	v.CacheAttribution.FakVDSOAvoidedCalls = memo + inline
	v.Adjudication.E2ELatencySumSeconds = float64(memo + inline)
	v.Adjudication.E2ELatencyCount = memo + inline
	return v
}

func TestGuardInfoSessionWorkDoneQueryContract(t *testing.T) {
	at := time.Date(2026, 8, 14, 20, 1, 2, 3, time.UTC)
	q := guardInfoSessionWorkDoneQuery(workDoneFixture(), at)
	if q.Schema != guardInfoWorkDoneQuerySchema || q.Window.Kind != "session_total" || q.Window.Reset || q.GeneratedAt != at.Format(time.RFC3339Nano) {
		t.Fatalf("session query = %#v", q)
	}
	if q.WorkDone.Metrics.ModelCallsAvoided.IntegerValue == nil || *q.WorkDone.Metrics.ModelCallsAvoided.IntegerValue != 27 {
		t.Fatalf("integer-safe calls = %#v", q.WorkDone.Metrics.ModelCallsAvoided)
	}
}

func TestGuardInfoBoundedWorkDoneQueryDeltasAndOrders(t *testing.T) {
	before := workDoneFixtureWith(100, 3, 2)
	after := workDoneFixtureWith(160, 7, 4)
	start := time.Date(2026, 8, 14, 20, 0, 0, 0, time.UTC)
	q := guardInfoBoundedWorkDoneQuery(before, after, start, start.Add(5*time.Second))
	if q.Window.Kind != "bounded" || q.Window.DurationNanos != int64(5*time.Second) || q.Window.Reset {
		t.Fatalf("window = %#v", q.Window)
	}
	if got := q.WorkDone.Metrics.InputTokensAvoided.Value; got != 60 {
		t.Fatalf("token delta = %v", got)
	}
	if got := q.WorkDone.Metrics.ModelCallsAvoided.Value; got != 6 {
		t.Fatalf("call delta = %v", got)
	}
	ids := make([]string, len(q.WorkDone.Sources))
	for i, source := range q.WorkDone.Sources {
		ids[i] = source.ID
	}
	if got := strings.Join(ids, ","); got != "provider_cache,context_reduction,fak_response_reuse,inline_tool_local" {
		t.Fatalf("source order = %s", got)
	}
}

func TestGuardInfoBoundedWorkDoneQueryMakesResetUnavailable(t *testing.T) {
	before, after := workDoneFixtureWith(100, 4, 2), workDoneFixtureWith(10, 1, 0)
	q := guardInfoBoundedWorkDoneQuery(before, after, time.Unix(0, 0), time.Unix(1, 0))
	if !q.Window.Reset || q.Window.ResetReason != "session_counters_reset" {
		t.Fatalf("reset window = %#v", q.Window)
	}
	for _, metric := range []guardInfoWorkDoneMetric{q.WorkDone.Metrics.InputTokensAvoided, q.WorkDone.Metrics.ModelCallsAvoided, q.WorkDone.Metrics.WaitSecondsAvoided} {
		if metric.Available || metric.Value != 0 || metric.UnavailableReason != "session_counters_reset_during_window" {
			t.Fatalf("reset metric = %#v", metric)
		}
	}
}

func TestGuardInfoWorkDoneQueryHasNoNaNOrImplicitZero(t *testing.T) {
	q := guardInfoSessionWorkDoneQuery(guardInfoVars{}, time.Unix(0, 0))
	raw, err := json.Marshal(q)
	if err != nil {
		t.Fatal(err)
	}
	if bytes.Contains(raw, []byte("NaN")) || !bytes.Contains(raw, []byte(`"unavailable_reason"`)) {
		t.Fatalf("query JSON = %s", raw)
	}
	if integerMetricValue(math.NaN()) != nil || integerMetricValue(-1) != nil {
		t.Fatal("invalid integer metric was accepted")
	}
}

func TestRunInfoWorkDoneQueryIsIntegrationFacingReadPath(t *testing.T) {
	fixture := workDoneFixture()
	var requests atomic.Int64
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		_ = json.NewEncoder(w).Encode(fixture)
	}))
	defer ts.Close()
	c := &claudeMacDebugClient{base: ts.URL, hc: ts.Client()}
	var stdout, stderr bytes.Buffer
	if code := runInfoWorkDoneQuery(&stdout, &stderr, c, 0); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var q guardInfoWorkDoneQuery
	if err := json.Unmarshal(stdout.Bytes(), &q); err != nil {
		t.Fatal(err)
	}
	if requests.Load() != 1 || q.Schema != guardInfoWorkDoneQuerySchema || q.WorkDone.Schema != guardInfoWorkDoneSchema {
		t.Fatalf("integration result = %#v requests=%d", q, requests.Load())
	}
}

func TestWorkDoneTUIAndQueryShareAccountingObject(t *testing.T) {
	v := workDoneFixture()
	q := guardInfoSessionWorkDoneQuery(v, time.Unix(0, 0))
	rows := strings.Join(guardInfoWorkDoneRows(newGuardInfoPanelCtx(v, newGuardInfoTrend(4), 140), guardPanelFull), "\n")
	for _, want := range []string{guardInfoSignedShortCount(q.WorkDone.Metrics.InputTokensAvoided.Value), guardInfoShortCount(int(q.WorkDone.Metrics.ModelCallsAvoided.Value)), q.WorkDone.Baseline.Label} {
		if !strings.Contains(rows, want) {
			t.Fatalf("TUI did not render query value %q:\n%s", want, rows)
		}
	}
}

func TestWorkDoneQueryIgnoresUnknownJSONFields(t *testing.T) {
	raw := []byte(`{"schema":"fak.info.work-done-query/1","future_field":{"anything":true},"window":{"kind":"session_total","end_utc":"2026-08-14T00:00:00Z","duration_nanos":0,"reset":false},"work_done":{"schema":"fak.info.work-done/1","window":"observed_session","baseline":{"id":"direct-provider/v1","label":"direct provider path","revision":1,"effective_utc":"2026-08-14","configuration_sha256":"sha256:x","comparison_scope":"scope","candidate_arm":"candidate","baseline_arm":"baseline"},"metrics":{"input_tokens_avoided":{"available":false,"unit":"input_tokens","evidence":"unavailable","baseline_id":"direct-provider/v1","unavailable_reason":"missing"},"model_calls_avoided":{"available":false,"unit":"model_calls","evidence":"unavailable","baseline_id":"direct-provider/v1","unavailable_reason":"missing"},"wait_seconds_avoided":{"available":false,"unit":"seconds","evidence":"unavailable","baseline_id":"direct-provider/v1","unavailable_reason":"missing"}}}}`)
	var q guardInfoWorkDoneQuery
	if err := json.Unmarshal(raw, &q); err != nil {
		t.Fatalf("forward-compatible decode: %v", err)
	}
	if q.Schema != guardInfoWorkDoneQuerySchema || q.WorkDone.Baseline.ID != guardInfoWorkDoneBaselineID {
		t.Fatalf("decoded query = %#v", q)
	}
}

func TestRunInfoRejectsInvalidWorkDoneFlagCombinations(t *testing.T) {
	for _, argv := range [][]string{{"--work-done-window", "1s"}, {"--work-done-json", "--json"}, {"--work-done-json", "--work-done-window", "-1s"}} {
		var stdout, stderr bytes.Buffer
		if code := runInfo(&stdout, &stderr, argv); code != 2 {
			t.Fatalf("argv=%v code=%d stderr=%s", argv, code, stderr.String())
		}
	}
}
