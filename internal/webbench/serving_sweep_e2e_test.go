package webbench

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:justify why=integration when=changed:internal/webbench/**
// fak-test:runtime fast est=2s lane=default
func TestRunServingSweepE2EMidMarketSSELadder(t *testing.T) {
	concurrencies := []int{1, 4, 8, 16}
	const reqCount = 16
	const modelID = "qwen3.8-27b"
	const tokensPerReq = 4
	const reasoningPerReq = 2

	type trackFixture struct {
		track       ServingTrack
		engine      string
		digest      string
		prefillWait time.Duration
		itlWait     time.Duration
		metricBody  string
		hitRate     float64
	}
	fixtures := []trackFixture{
		{
			track:       TrackOurs,
			engine:      "fak-native",
			digest:      "sha256:" + strings.Repeat("1", 64),
			prefillWait: 2 * time.Millisecond,
			itlWait:     1 * time.Millisecond,
			metricBody:  "fak_gateway_kv_prefix_hit_rate 0.875\n",
			hitRate:     0.875,
		},
		{
			track:       TrackVLLM,
			engine:      "vllm",
			digest:      "sha256:" + strings.Repeat("2", 64),
			prefillWait: 3 * time.Millisecond,
			itlWait:     1 * time.Millisecond,
			metricBody:  "vllm:gpu_prefix_cache_hit_rate{gpu=\"0\"} 0.625\n",
			hitRate:     0.625,
		},
		{
			track:       TrackSGLang,
			engine:      "sglang",
			digest:      "sha256:" + strings.Repeat("3", 64),
			prefillWait: 2 * time.Millisecond,
			itlWait:     1 * time.Millisecond,
			metricBody:  "sglang_prefix_cache_hit_rate 75.0\n",
			hitRate:     0.75,
		},
	}

	dataset := NewDataset([]Instance{
		{TaskID: "mm-01", Description: "Extract midmarket invoice totals from portal table."},
		{TaskID: "mm-02", Description: "Verify inventory SKU count across checkout pages."},
	})
	workload := BuildServingWorkload(dataset, DefaultGeometryModel(), 2, reqCount, 32, "midmarket-shared-prefix")
	if len(workload) != reqCount {
		t.Fatalf("workload len = %d, want %d", len(workload), reqCount)
	}

	tracks := make([]ServingTrackConfig, 0, len(fixtures))
	contracts := make(map[ServingTrack]ServingSweepTrackContract, len(fixtures))
	peakServerInFlight := make(map[ServingTrack]*atomic.Int32, len(fixtures))

	for _, fx := range fixtures {
		fx := fx
		var currentInFlight, maxInFlight atomic.Int32
		peakServerInFlight[fx.track] = &maxInFlight

		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			switch r.URL.Path {
			case "/metrics":
				w.Header().Set("Content-Type", "text/plain")
				fmt.Fprint(w, fx.metricBody)
				return
			case "/v1/chat/completions":
			default:
				http.NotFound(w, r)
				return
			}

			cur := currentInFlight.Add(1)
			for {
				prev := maxInFlight.Load()
				if cur <= prev || maxInFlight.CompareAndSwap(prev, cur) {
					break
				}
			}
			defer currentInFlight.Add(-1)

			var reqBody struct {
				Model         string `json:"model"`
				Stream        bool   `json:"stream"`
				StreamOptions struct {
					IncludeUsage bool `json:"include_usage"`
				} `json:"stream_options"`
			}
			if err := json.NewDecoder(r.Body).Decode(&reqBody); err != nil {
				http.Error(w, err.Error(), http.StatusBadRequest)
				return
			}
			if reqBody.Model != modelID || !reqBody.Stream || !reqBody.StreamOptions.IncludeUsage {
				http.Error(w, "invalid streaming request contract", http.StatusBadRequest)
				return
			}

			w.Header().Set("Content-Type", "text/event-stream")
			flusher, _ := w.(http.Flusher)
			// Role-only framing chunk must not start TTFT.
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\",\"reasoning_content\":null}}]}\n\n")
			flusher.Flush()

			time.Sleep(fx.prefillWait)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"plan \"}}]}\n\n")
			flusher.Flush()

			time.Sleep(fx.itlWait)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"verify\"}}]}\n\n")
			flusher.Flush()

			time.Sleep(fx.itlWait)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"done\"}}]}\n\n")
			flusher.Flush()

			time.Sleep(fx.itlWait)
			fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\".\"}}]}\n\n")
			flusher.Flush()

			fmt.Fprintf(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":%d,\"total_tokens\":%d,\"prompt_tokens_details\":{\"cached_tokens\":96},\"completion_tokens_details\":{\"reasoning_tokens\":%d}}}\n\n", tokensPerReq, 128+tokensPerReq, reasoningPerReq)
			fmt.Fprint(w, "data: [DONE]\n\n")
			flusher.Flush()
		}))
		t.Cleanup(srv.Close)

		tracks = append(tracks, ServingTrackConfig{
			Track:      fx.track,
			BaseURL:    srv.URL + "/v1",
			MetricsURL: srv.URL + "/metrics",
			Model:      modelID,
			ProjectArm: "midmarket-e2e",
		})
		contracts[fx.track] = ServingSweepTrackContract{
			Track:               fx.track,
			Model:               modelID,
			Engine:              fx.engine,
			EngineReceiptDigest: fx.digest,
			BatchCapacity:       16,
			CapacitySource:      "midmarket-1x-gpu:kv-budget",
		}
	}

	report, err := RunServingSweep(context.Background(), ServingSweepConfig{
		GeneratedAt:   "2026-11-28T00:00:00Z",
		MachineID:     "midmarket-1gpu-node",
		Model:         modelID,
		Tracks:        tracks,
		Contracts:     contracts,
		Workload:      workload,
		Concurrencies: concurrencies,
		GoodputSLO:    2 * time.Second,
		TTFTP99Budget: 500 * time.Millisecond,
		ITLP99Budget:  500 * time.Millisecond,
		Timeout:       5 * time.Second,
	})
	if err != nil {
		t.Fatalf("RunServingSweep: %v", err)
	}
	if report.Schema != ServingSweepSchema || !reflect.DeepEqual(report.Workload.Concurrencies, concurrencies) {
		t.Fatalf("report header/concurrencies = %q %v", report.Schema, report.Workload.Concurrencies)
	}
	if len(report.Points) != len(concurrencies) {
		t.Fatalf("points = %d, want %d", len(report.Points), len(concurrencies))
	}

	expectedTokens := int64(reqCount * tokensPerReq)
	for i, wantC := range concurrencies {
		pt := report.Points[i]
		if pt.Concurrency != wantC {
			t.Fatalf("point[%d].Concurrency = %d, want %d", i, pt.Concurrency, wantC)
		}
		if len(pt.Tracks) != len(fixtures) {
			t.Fatalf("point[%d] tracks = %d, want %d", i, len(pt.Tracks), len(fixtures))
		}
		for j, fx := range fixtures {
			tp := pt.Tracks[j]
			if tp.Track != fx.track || tp.Status != "valid" || tp.MeasurementStatus != "measured" || tp.ReasonCode != "" {
				t.Fatalf("c=%d track=%s status=%s/%s code=%q reason=%q", wantC, fx.track, tp.Status, tp.MeasurementStatus, tp.ReasonCode, tp.Reason)
			}
			st := tp.Stats
			if st.Requests != reqCount || st.OK != reqCount || st.Failed != 0 {
				t.Fatalf("c=%d track=%s counts=%+v, want %d/%d/0", wantC, fx.track, st, reqCount, reqCount)
			}
			for name, q := range map[string]QuantileMetric{
				"ttft_ms":       st.TTFTMillis,
				"itl_ms":        st.ITLMillis,
				"tpot_ms":       st.TPOTMillis,
				"end_to_end_ms": st.EndToEndMillis,
			} {
				if q.Status != "measured" || q.P50 == nil || *q.P50 <= 0 || q.P99 == nil || *q.P99 <= 0 {
					t.Fatalf("c=%d track=%s %s = %+v, want measured positive quantiles", wantC, fx.track, name, q)
				}
			}
			if st.WallSeconds == nil || *st.WallSeconds <= 0 {
				t.Fatalf("c=%d track=%s wall_seconds = %v, want > 0", wantC, fx.track, st.WallSeconds)
			}
			if st.TokenCountBasis != "usage.completion_tokens" {
				t.Fatalf("c=%d track=%s basis = %q", wantC, fx.track, st.TokenCountBasis)
			}
			if st.ThroughputTokensS.Status != "measured" || st.ThroughputTokensS.Unit != "usage.completion_tokens/s" || st.ThroughputTokensS.Value == nil || *st.ThroughputTokensS.Value <= 0 {
				t.Fatalf("c=%d track=%s throughput = %+v", wantC, fx.track, st.ThroughputTokensS)
			}
			if st.GoodputRPS.Status != "measured" || st.GoodputRPS.Unit != "requests/s" || st.GoodputRPS.Value == nil || *st.GoodputRPS.Value <= 0 {
				t.Fatalf("c=%d track=%s goodput_rps = %+v", wantC, fx.track, st.GoodputRPS)
			}
			if st.GoodputTokensS.Status != "measured" || st.GoodputTokensS.Unit != "usage.completion_tokens/s" || st.GoodputTokensS.Value == nil || *st.GoodputTokensS.Value <= 0 {
				t.Fatalf("c=%d track=%s goodput_tok_s = %+v", wantC, fx.track, st.GoodputTokensS)
			}
			if st.SuccessfulOutputTokensExact == nil || *st.SuccessfulOutputTokensExact != expectedTokens ||
				st.SLOSuccessfulOutputTokensExact == nil || *st.SLOSuccessfulOutputTokensExact != expectedTokens {
				t.Fatalf("c=%d track=%s exact tokens = %v / %v, want %d", wantC, fx.track, st.SuccessfulOutputTokensExact, st.SLOSuccessfulOutputTokensExact, expectedTokens)
			}
			if st.ObservedMaxInFlight == nil || *st.ObservedMaxInFlight < 1 || *st.ObservedMaxInFlight > wantC || st.ObservedInFlightBasis != "client_http" {
				t.Fatalf("c=%d track=%s overlap = %v (%q)", wantC, fx.track, st.ObservedMaxInFlight, st.ObservedInFlightBasis)
			}
			if wantC == 1 && *st.ObservedMaxInFlight != 1 {
				t.Fatalf("c=1 track=%s observed max in flight = %d, want 1", fx.track, *st.ObservedMaxInFlight)
			}
			if wantC == 16 && *st.ObservedMaxInFlight < 2 {
				t.Fatalf("c=16 track=%s observed max in flight = %d, want concurrent overlap >= 2", fx.track, *st.ObservedMaxInFlight)
			}
			if st.PrefixCacheHitRate.Status != "measured" || st.PrefixCacheHitRate.Value == nil || *st.PrefixCacheHitRate.Value != fx.hitRate {
				t.Fatalf("c=%d track=%s prefix cache hit rate = %+v, want %v", wantC, fx.track, st.PrefixCacheHitRate, fx.hitRate)
			}
		}
	}
	for _, fx := range fixtures {
		if got := peakServerInFlight[fx.track].Load(); got < 2 {
			t.Fatalf("track %s server max in-flight = %d, want >= 2 across ladder", fx.track, got)
		}
	}

	if len(report.Tracks) != len(fixtures) {
		t.Fatalf("track summaries = %d, want %d", len(report.Tracks), len(fixtures))
	}
	for _, summary := range report.Tracks {
		if summary.Status != "measured" || summary.ValidPoints != len(concurrencies) {
			t.Fatalf("track %s summary = %+v, want measured with %d valid points", summary.Track, summary, len(concurrencies))
		}
		if summary.PeakStatus != "measured" || summary.Peak == nil || summary.Peak.ThroughputTokens <= 0 || summary.Peak.ThroughputUnit != "usage.completion_tokens/s" {
			t.Fatalf("track %s peak = status=%q %+v", summary.Track, summary.PeakStatus, summary.Peak)
		}
		if summary.SLAStatus != "measured" || summary.SLAKnee == nil {
			t.Fatalf("track %s SLA knee = status=%q %+v", summary.Track, summary.SLAStatus, summary.SLAKnee)
		}
		for _, claim := range []string{
			fmt.Sprintf("%s serving peak", summary.Track),
			fmt.Sprintf("%s SLA knee", summary.Track),
		} {
			if err := ValidateServingSweepClaim(claim, report); err != nil {
				t.Fatalf("ValidateServingSweepClaim(%q): %v", claim, err)
			}
		}
	}

	artifactPath := filepath.Join(t.TempDir(), "sweep-e2e.json")
	if err := WriteServingSweepReport(report, artifactPath); err != nil {
		t.Fatalf("WriteServingSweepReport: %v", err)
	}
	reloaded, err := LoadServingSweepReport(artifactPath)
	if err != nil {
		t.Fatalf("LoadServingSweepReport: %v", err)
	}
	if err := EvaluateServingSweep(reloaded); err != nil {
		t.Fatalf("EvaluateServingSweep(reloaded): %v", err)
	}
	if len(reloaded.Tracks) != len(fixtures) || reloaded.Points[0].Tracks[0].Stats.TTFTMillis.P50 == nil {
		t.Fatalf("reloaded report lost SSE measurements: %+v", reloaded)
	}
}

// fak-test:justify why=integration when=changed:internal/webbench/**
// fak-test:runtime fast est=250ms lane=default
func TestRunServingSweepE2ENonSSEHonestDistinction(t *testing.T) {
	concurrencies := []int{1, 4, 8, 16}
	const reqCount = 16
	const modelID = "qwen3.8-27b"

	nonSSEServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		time.Sleep(2 * time.Millisecond)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"role":"assistant","reasoning_content":"step one","content":"ok"}}],"usage":{"prompt_tokens":64,"completion_tokens":4,"total_tokens":68,"completion_tokens_details":{"reasoning_tokens":2}}}`)
	}))
	defer nonSSEServer.Close()

	sseServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, _ := w.(http.Flusher)
		time.Sleep(2 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"step \"}}]}\n\n")
		flusher.Flush()
		time.Sleep(1 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"one\"}}]}\n\n")
		flusher.Flush()
		time.Sleep(1 * time.Millisecond)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		flusher.Flush()
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":4,\"total_tokens\":68,\"completion_tokens_details\":{\"reasoning_tokens\":2}}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer sseServer.Close()

	workload := make([]ServingRequest, reqCount)
	for i := 0; i < reqCount; i++ {
		workload[i] = ServingRequest{
			ID:              fmt.Sprintf("req-%02d", i+1),
			Messages:        []ChatMessage{{Role: "user", Content: "solve step"}},
			MaxOutputTokens: 16,
		}
	}

	report, err := RunServingSweep(context.Background(), ServingSweepConfig{
		GeneratedAt: "2026-11-28T00:00:00Z",
		MachineID:   "midmarket-1gpu-node",
		Model:       modelID,
		Tracks: []ServingTrackConfig{
			{Track: TrackOurs, BaseURL: nonSSEServer.URL + "/v1", Model: modelID},
		},
		Contracts: map[ServingTrack]ServingSweepTrackContract{
			TrackOurs: {
				Track:               TrackOurs,
				Model:               modelID,
				Engine:              "fak-native",
				EngineReceiptDigest: "sha256:" + strings.Repeat("a", 64),
				BatchCapacity:       16,
				CapacitySource:      "midmarket-1x-gpu:kv-budget",
			},
		},
		Workload:      workload,
		Concurrencies: concurrencies,
		GoodputSLO:    2 * time.Second,
		TTFTP99Budget: 500 * time.Millisecond,
		ITLP99Budget:  500 * time.Millisecond,
		Timeout:       5 * time.Second,
	})
	if err != nil {
		t.Fatalf("RunServingSweep(non_sse): %v", err)
	}
	for i, pt := range report.Points {
		tp := pt.Tracks[0]
		if tp.Status != "valid" || tp.Stats.OK != reqCount {
			t.Fatalf("point[%d] non_sse status = %+v", i, tp)
		}
		for name, q := range map[string]QuantileMetric{
			"ttft_ms": tp.Stats.TTFTMillis,
			"itl_ms":  tp.Stats.ITLMillis,
			"tpot_ms": tp.Stats.TPOTMillis,
		} {
			if q.Status != "not_measured" || q.P50 != nil || q.P90 != nil || q.P95 != nil || q.P99 != nil {
				t.Fatalf("point[%d] non_sse fabricated %s: %+v", i, name, q)
			}
		}
		if tp.Stats.EndToEndMillis.Status != "measured" || tp.Stats.EndToEndMillis.P50 == nil || *tp.Stats.EndToEndMillis.P50 <= 0 {
			t.Fatalf("point[%d] non_sse end_to_end_ms = %+v", i, tp.Stats.EndToEndMillis)
		}
		if tp.Stats.ThroughputTokensS.Status != "measured" || tp.Stats.GoodputRPS.Status != "measured" || tp.Stats.GoodputTokensS.Status != "measured" {
			t.Fatalf("point[%d] non_sse throughput/goodput = %+v", i, tp.Stats)
		}
	}
	summary := report.Tracks[0]
	if summary.PeakStatus != "measured" || summary.Peak == nil {
		t.Fatalf("non_sse peak = status=%q %+v", summary.PeakStatus, summary.Peak)
	}
	if summary.Peak.TTFTP99Millis != nil || summary.Peak.ITLP99Millis != nil {
		t.Fatalf("non_sse peak invented p99 latencies: %+v", summary.Peak)
	}
	if summary.SLAStatus != "not_identifiable" || summary.SLAKnee != nil {
		t.Fatalf("non_sse invented SLA knee without SSE measurements: status=%q knee=%+v", summary.SLAStatus, summary.SLAKnee)
	}
	if err := ValidateServingSweepClaim("ours SLA knee", report); err == nil {
		t.Fatal("non_sse sweep erroneously certified an SLA knee claim")
	}

	// Parity comparison between SSE (ours) and non_sse (sglang) keeps TTFT/ITL/TPOT tax not_measured.
	parity, err := RunServingParity(context.Background(), ServingParityConfig{
		Model:       modelID,
		Concurrency: 4,
		SLO:         2 * time.Second,
		Timeout:     5 * time.Second,
		Workload:    workload[:4],
		Tracks: []ServingTrackConfig{
			{Track: TrackOurs, BaseURL: sseServer.URL + "/v1", Model: modelID},
			{Track: TrackSGLang, BaseURL: nonSSEServer.URL + "/v1", Model: modelID},
		},
	})
	if err != nil {
		t.Fatalf("RunServingParity(sse vs non_sse): %v", err)
	}
	oursTrack := parity.Tracks[0]
	rawTrack := parity.Tracks[1]
	if oursTrack.Samples[0].StreamMode != "sse" || oursTrack.Samples[0].ReasoningEvents != 2 ||
		oursTrack.Samples[0].Usage == nil || oursTrack.Samples[0].Usage.ReasoningTokens == nil || *oursTrack.Samples[0].Usage.ReasoningTokens != 2 {
		t.Fatalf("ours SSE reasoning accounting missing: %+v", oursTrack.Samples[0])
	}
	if rawTrack.Samples[0].StreamMode != "non_sse" || rawTrack.Samples[0].TTFTMillis != nil || rawTrack.Samples[0].ITLMillis != nil || rawTrack.Samples[0].TPOTMillis != nil {
		t.Fatalf("raw non_sse fabricated stream fields: %+v", rawTrack.Samples[0])
	}
	for _, metricName := range []string{"ttft_ms", "itl_ms", "tpot_ms"} {
		if m := parity.OursVsSGLangTax.Metrics[metricName]; m.Status != "not_measured" || m.Ours != nil || m.Raw != nil {
			t.Fatalf("mixed sse/non_sse tax fabricated %s: %+v", metricName, m)
		}
	}
	if m := parity.OursVsSGLangTax.Metrics["end_to_end_ms"]; m.Status != "measured" {
		t.Fatalf("end_to_end_ms tax = %+v, want measured", m)
	}
	if m := parity.OursVsSGLangTax.Metrics["throughput_tok_s"]; m.Status != "measured" {
		t.Fatalf("throughput_tok_s tax = %+v, want measured", m)
	}
}
