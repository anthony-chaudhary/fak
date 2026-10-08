package webbench

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:justify why=contract when=changed:internal/webbench/**
// fak-test:runtime fast est=10ms lane=default
func TestFoldServingSamplesExactTokenGoodputContract(t *testing.T) {
	t.Run("full exact coverage", func(t *testing.T) {
		stats := FoldServingSamples([]ServingSample{
			{Status: "ok", EndToEndMillis: 500, OutputTokensExact: servingMeasurementInt(3)},
			{Status: "ok", EndToEndMillis: 1500, OutputTokensExact: servingMeasurementInt(2)},
			{Status: "fail", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(99)},
		}, 2, time.Second)
		servingMeasurementWantFloat(t, "wall seconds", stats.WallSeconds, 2)
		servingMeasurementWantFloat(t, "goodput SLO seconds", stats.GoodputSLOSeconds, 1)
		servingMeasurementWantInt64(t, "successful exact tokens", stats.SuccessfulOutputTokensExact, 5)
		servingMeasurementWantInt64(t, "SLO successful exact tokens", stats.SLOSuccessfulOutputTokensExact, 3)
		servingMeasurementWantMetric(t, "token goodput", stats.GoodputTokensS, 1.5)
		servingMeasurementWantMetric(t, "request goodput", stats.GoodputRPS, 0.5)
	})

	t.Run("zero qualifying tokens is measured", func(t *testing.T) {
		stats := FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: 1500, OutputTokensExact: servingMeasurementInt(0)}}, 2, time.Second)
		servingMeasurementWantInt64(t, "SLO successful exact tokens", stats.SLOSuccessfulOutputTokensExact, 0)
		servingMeasurementWantMetric(t, "token goodput", stats.GoodputTokensS, 0)
		servingMeasurementWantMetric(t, "request goodput", stats.GoodputRPS, 0)
	})

	for _, tc := range []struct {
		name        string
		samples     []ServingSample
		wallSeconds float64
		slo         time.Duration
	}{
		{name: "missing exact coverage", samples: []ServingSample{{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(2)}, {Status: "ok", EndToEndMillis: 200}}, wallSeconds: 1, slo: time.Second},
		{name: "negative exact count", samples: []ServingSample{{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(-1)}}, wallSeconds: 1, slo: time.Second},
		{name: "all requests failed", samples: []ServingSample{{Status: "fail", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(4)}}, wallSeconds: 1, slo: time.Second},
	} {
		t.Run(tc.name, func(t *testing.T) {
			stats := FoldServingSamples(tc.samples, tc.wallSeconds, tc.slo)
			if stats.SuccessfulOutputTokensExact != nil || stats.SLOSuccessfulOutputTokensExact != nil {
				t.Fatalf("invalid exact coverage produced token totals: successful=%v SLO=%v", stats.SuccessfulOutputTokensExact, stats.SLOSuccessfulOutputTokensExact)
			}
			servingMeasurementWantNotMeasured(t, "token goodput", stats.GoodputTokensS)
		})
	}
	if strconv.IntSize == 64 {
		t.Run("exact sum overflow", func(t *testing.T) {
			stats := FoldServingSamples([]ServingSample{
				{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(int(^uint(0) >> 1))},
				{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(1)},
			}, 1, time.Second)
			if stats.SuccessfulOutputTokensExact != nil || stats.SLOSuccessfulOutputTokensExact != nil {
				t.Fatal("overflowing exact sum produced token totals")
			}
			servingMeasurementWantNotMeasured(t, "token goodput", stats.GoodputTokensS)
		})
	}

	for _, wall := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		stats := FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(2)}}, wall, time.Second)
		if stats.WallSeconds != nil {
			t.Fatalf("invalid wall %v retained as %v", wall, *stats.WallSeconds)
		}
		servingMeasurementWantNotMeasured(t, "token goodput", stats.GoodputTokensS)
	}

	stats := FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: 100, OutputTokensExact: servingMeasurementInt(2)}}, 1, 0)
	if stats.GoodputSLOSeconds != nil || stats.SLOSuccessfulOutputTokensExact != nil {
		t.Fatalf("missing SLO produced SLO evidence: seconds=%v tokens=%v", stats.GoodputSLOSeconds, stats.SLOSuccessfulOutputTokensExact)
	}
	servingMeasurementWantNotMeasured(t, "token goodput", stats.GoodputTokensS)
	servingMeasurementWantNotMeasured(t, "request goodput", stats.GoodputRPS)

	stats = FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: math.MaxFloat64, OutputTokensExact: servingMeasurementInt(2)}}, 1, time.Second)
	servingMeasurementWantInt64(t, "large-latency successful exact tokens", stats.SuccessfulOutputTokensExact, 2)
	servingMeasurementWantInt64(t, "large-latency SLO exact tokens", stats.SLOSuccessfulOutputTokensExact, 0)
	servingMeasurementWantMetric(t, "large-latency token goodput", stats.GoodputTokensS, 0)
	servingMeasurementWantMetric(t, "large-latency request goodput", stats.GoodputRPS, 0)

	stats = FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: 0, OutputTokensExact: servingMeasurementInt(1)}}, math.SmallestNonzeroFloat64, time.Second)
	servingMeasurementWantInt64(t, "overflow-rate successful exact tokens", stats.SuccessfulOutputTokensExact, 1)
	servingMeasurementWantInt64(t, "overflow-rate SLO exact tokens", stats.SLOSuccessfulOutputTokensExact, 1)
	servingMeasurementWantNotMeasured(t, "overflow token goodput", stats.GoodputTokensS)
	servingMeasurementWantNotMeasured(t, "overflow request goodput", stats.GoodputRPS)
}

// fak-test:justify why=contract when=changed:internal/webbench/**
// fak-test:runtime fast est=5ms lane=default
func TestServingStatsNullableJSONPreservesLegacyUnknown(t *testing.T) {
	var legacy ServingStats
	if err := json.Unmarshal([]byte(`{"requests":1,"ok":1}`), &legacy); err != nil {
		t.Fatalf("decode legacy stats: %v", err)
	}
	if legacy.WallSeconds != nil || legacy.GoodputSLOSeconds != nil || legacy.SuccessfulOutputTokensExact != nil || legacy.SLOSuccessfulOutputTokensExact != nil || legacy.ObservedMaxInFlight != nil {
		t.Fatal("legacy stats decoded absent measurements as observed values")
	}
	if legacy.ObservedInFlightBasis != "" {
		t.Fatalf("legacy stats decoded overlap basis=%q", legacy.ObservedInFlightBasis)
	}
	raw, err := json.Marshal(legacy)
	if err != nil {
		t.Fatalf("marshal legacy stats: %v", err)
	}
	var object map[string]any
	if err := json.Unmarshal(raw, &object); err != nil {
		t.Fatalf("decode stats object: %v", err)
	}
	for _, key := range []string{"wall_seconds", "goodput_slo_seconds", "successful_output_tokens_exact", "slo_successful_output_tokens_exact", "observed_max_in_flight"} {
		value, present := object[key]
		if present && value != nil {
			t.Errorf("legacy %s=%v, want omitted or null", key, value)
		}
	}
	if _, present := object["goodput_tok_s"]; !present {
		t.Error("goodput_tok_s missing from serialized stats")
	}
	witnessed := FoldServingSamples([]ServingSample{{Status: "ok", EndToEndMillis: 10, OutputTokensExact: servingMeasurementInt(0)}}, 1, time.Second)
	zeroRaw, err := json.Marshal(witnessed)
	if err != nil {
		t.Fatalf("marshal witnessed zero stats: %v", err)
	}
	var zeroRoundTrip ServingStats
	if err := json.Unmarshal(zeroRaw, &zeroRoundTrip); err != nil {
		t.Fatalf("decode witnessed zero stats: %v", err)
	}
	servingMeasurementWantInt64(t, "round-trip successful exact tokens", zeroRoundTrip.SuccessfulOutputTokensExact, 0)
	servingMeasurementWantInt64(t, "round-trip SLO exact tokens", zeroRoundTrip.SLOSuccessfulOutputTokensExact, 0)
	servingMeasurementWantMetric(t, "round-trip token goodput", zeroRoundTrip.GoodputTokensS, 0)
}

// fak-test:justify why=integration when=changed:internal/webbench/**
// fak-test:runtime fast est=250ms lane=default
func TestRunServingParityMeasuresClientHTTPOverlapThroughBodyCompletion(t *testing.T) {
	release := make(chan struct{})
	allEntered := make(chan struct{})
	var releaseOnce sync.Once
	var entered atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		servingMeasurementStartSSE(w)
		if entered.Add(1) == 2 {
			close(allEntered)
		}
		<-release
		servingMeasurementFinishSSE(w, 2)
	}))
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
		server.Close()
	})

	type result struct {
		report *ServingParityReport
		err    error
	}
	done := make(chan result, 1)
	go func() {
		report, err := RunServingParity(context.Background(), ServingParityConfig{
			Model: "fixture", Tracks: []ServingTrackConfig{{Track: TrackOurs, BaseURL: server.URL + "/v1"}},
			Workload: servingMeasurementWorkload("one", "two"), Concurrency: 2, SLO: time.Second, Timeout: 5 * time.Second, Client: server.Client(),
		})
		done <- result{report: report, err: err}
	}()
	servingMeasurementAwait(t, allEntered, "parity requests did not reach the body barrier")
	releaseOnce.Do(func() { close(release) })
	got := <-done
	if got.err != nil {
		t.Fatalf("RunServingParity: %v", got.err)
	}
	if got.report == nil || len(got.report.Tracks) != 1 {
		t.Fatalf("parity tracks=%d, want 1", servingMeasurementParityTrackCount(got.report))
	}
	servingMeasurementWantOverlap(t, got.report.Tracks[0].Stats, 2)
	servingMeasurementWantSerializedOverlap(t, got.report, 2)
}

// fak-test:justify why=integration when=changed:internal/webbench/**
// fak-test:runtime fast est=250ms lane=default
func TestRunServingParityFailureUnwindsBeforeQueuedRequest(t *testing.T) {
	heldEntered := make(chan struct{})
	failureResponded := make(chan struct{})
	afterEntered := make(chan struct{})
	releaseHeld := make(chan struct{})
	var releaseOnce sync.Once
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		content := servingMeasurementRequestContent(r)
		switch content {
		case "held":
			servingMeasurementStartSSE(w)
			close(heldEntered)
			<-releaseHeld
			servingMeasurementFinishSSE(w, 1)
		case "fail":
			<-heldEntered
			http.Error(w, "fixture failure", http.StatusServiceUnavailable)
			close(failureResponded)
		case "after":
			<-failureResponded
			servingMeasurementStartSSE(w)
			close(afterEntered)
			servingMeasurementFinishSSE(w, 1)
		default:
			http.Error(w, "unknown fixture request", http.StatusBadRequest)
		}
	}))
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(releaseHeld) })
		server.Close()
	})

	done := make(chan *ServingParityReport, 1)
	go func() {
		report, _ := RunServingParity(context.Background(), ServingParityConfig{
			Model: "fixture", Tracks: []ServingTrackConfig{{Track: TrackOurs, BaseURL: server.URL + "/v1"}},
			Workload: servingMeasurementWorkload("held", "fail", "after"), Concurrency: 2, SLO: time.Second, Timeout: 5 * time.Second, Client: server.Client(),
		})
		done <- report
	}()
	servingMeasurementAwait(t, afterEntered, "queued request did not start after the failed request")
	releaseOnce.Do(func() { close(releaseHeld) })
	report := <-done
	if report == nil || len(report.Tracks) != 1 {
		t.Fatalf("parity tracks=%d, want 1", servingMeasurementParityTrackCount(report))
	}
	servingMeasurementWantOverlap(t, report.Tracks[0].Stats, 2)
}

// fak-test:justify why=integration when=changed:internal/webbench/**
// fak-test:runtime fast est=250ms lane=default
func TestRunServingSweepSerializesMeasuredClientHTTPOverlap(t *testing.T) {
	release := make(chan struct{})
	allEntered := make(chan struct{})
	var releaseOnce sync.Once
	var calls atomic.Int32
	var overlapped atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		call := calls.Add(1)
		servingMeasurementStartSSE(w)
		if call > 2 {
			if overlapped.Add(1) == 2 {
				close(allEntered)
			}
			<-release
		}
		servingMeasurementFinishSSE(w, 2)
	}))
	t.Cleanup(func() {
		releaseOnce.Do(func() { close(release) })
		server.Close()
	})

	type result struct {
		report *ServingSweepReport
		err    error
	}
	done := make(chan result, 1)
	go func() {
		report, err := RunServingSweep(context.Background(), ServingSweepConfig{
			Model: "fixture", MachineID: "fixture-machine",
			Tracks: []ServingTrackConfig{{Track: TrackOurs, BaseURL: server.URL + "/v1", Model: "fixture"}},
			Contracts: map[ServingTrack]ServingSweepTrackContract{TrackOurs: {
				Track: TrackOurs, Model: "fixture", Engine: "fixture-engine",
				EngineReceiptDigest: "sha256:" + strings.Repeat("a", 64), BatchCapacity: 2, CapacitySource: "fixture",
			}},
			Workload: servingMeasurementWorkload("one", "two"), Concurrencies: []int{1, 2}, GoodputSLO: time.Second,
			Timeout: 5 * time.Second, Client: server.Client(),
		})
		done <- result{report: report, err: err}
	}()
	servingMeasurementAwait(t, allEntered, "sweep concurrency-2 requests did not reach the body barrier")
	releaseOnce.Do(func() { close(release) })
	got := <-done
	if got.err != nil {
		t.Fatalf("RunServingSweep: %v", got.err)
	}
	if got.report == nil || len(got.report.Points) != 2 || len(got.report.Points[1].Tracks) != 1 {
		t.Fatalf("sweep points=%d, want two complete points", servingMeasurementSweepPointCount(got.report))
	}
	stats := got.report.Points[1].Tracks[0].Stats
	servingMeasurementWantOverlap(t, stats, 2)
	servingMeasurementWantSerializedOverlap(t, got.report, 2)
}

func servingMeasurementInt(value int) *int { return &value }

func servingMeasurementWantFloat(t *testing.T, name string, got *float64, want float64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v, want %v", name, got, want)
	}
}

func servingMeasurementWantInt(t *testing.T, name string, got *int, want int) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v, want %d", name, got, want)
	}
}

func servingMeasurementWantInt64(t *testing.T, name string, got *int64, want int64) {
	t.Helper()
	if got == nil || *got != want {
		t.Fatalf("%s=%v, want %d", name, got, want)
	}
}

func servingMeasurementWantMetric(t *testing.T, name string, got ScalarMetric, want float64) {
	t.Helper()
	if got.Status != "measured" || got.Value == nil || *got.Value != want {
		t.Fatalf("%s status=%q value=%v, want measured %v", name, got.Status, got.Value, want)
	}
}

func servingMeasurementWantNotMeasured(t *testing.T, name string, got ScalarMetric) {
	t.Helper()
	if got.Status != "not_measured" || got.Value != nil {
		t.Fatalf("%s status=%q value=%v, want not_measured", name, got.Status, got.Value)
	}
}

func servingMeasurementWantOverlap(t *testing.T, stats ServingStats, want int) {
	t.Helper()
	servingMeasurementWantInt(t, "observed max in-flight", stats.ObservedMaxInFlight, want)
	if stats.ObservedInFlightBasis != "client_http" {
		t.Fatalf("observed in-flight basis=%q, want client_http", stats.ObservedInFlightBasis)
	}
}

func servingMeasurementWantSerializedOverlap(t *testing.T, report any, want int) {
	t.Helper()
	raw, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("marshal report: %v", err)
	}
	if !strings.Contains(string(raw), fmt.Sprintf(`"observed_max_in_flight":%d`, want)) || !strings.Contains(string(raw), `"observed_in_flight_basis":"client_http"`) {
		t.Fatal("serialized report omitted measured client HTTP overlap")
	}
}

func servingMeasurementWorkload(contents ...string) []ServingRequest {
	requests := make([]ServingRequest, len(contents))
	for i, content := range contents {
		requests[i] = ServingRequest{ID: content, Messages: []ChatMessage{{Role: "user", Content: content}}, MaxOutputTokens: 4}
	}
	return requests
}

func servingMeasurementStartSSE(w http.ResponseWriter) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	w.(http.Flusher).Flush()
}

func servingMeasurementFinishSSE(w http.ResponseWriter, outputTokens int) {
	fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
	fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"completion_tokens\":%d}}\n\n", outputTokens)
	fmt.Fprint(w, "data: [DONE]\n\n")
}

func servingMeasurementRequestContent(r *http.Request) string {
	var body struct {
		Messages []ChatMessage `json:"messages"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || len(body.Messages) == 0 {
		return ""
	}
	return body.Messages[len(body.Messages)-1].Content
}

func servingMeasurementAwait(t *testing.T, signal <-chan struct{}, failure string) {
	t.Helper()
	select {
	case <-signal:
	case <-time.After(5 * time.Second):
		t.Fatal(failure)
	}
}

func servingMeasurementParityTrackCount(report *ServingParityReport) int {
	if report == nil {
		return 0
	}
	return len(report.Tracks)
}

func servingMeasurementSweepPointCount(report *ServingSweepReport) int {
	if report == nil {
		return 0
	}
	return len(report.Points)
}
