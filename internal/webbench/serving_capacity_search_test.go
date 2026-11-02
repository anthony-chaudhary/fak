package webbench

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s lane=default
func TestCapacitySearchBisectsMaxSustainableQPS(t *testing.T) {
	best, n, err := CapacitySearch(1, 100, .01, 20, func(q float64) (bool, error) { return q <= 37.5, nil })
	if err != nil || best > 37.5 || 37.5-best > .01 || n > 20 {
		t.Fatalf("best=%g probes=%d error=%v", best, n, err)
	}
	sentinel := errors.New("probe failure")
	_, n, err = CapacitySearch(1, 100, .01, 20, func(q float64) (bool, error) { return false, sentinel })
	if !errors.Is(err, sentinel) || n != 1 {
		t.Fatalf("error propagation: probes=%d error=%v", n, err)
	}
	_, n, err = CapacitySearch(1, 100, .01, 2, func(q float64) (bool, error) { return q < 37.5, nil })
	if !errors.Is(err, ErrCapacityProbeBudget) || n != 2 {
		t.Fatalf("budget: probes=%d error=%v", n, err)
	}
	_, n, err = CapacitySearch(math.NaN(), 100, .01, 2, func(q float64) (bool, error) { t.Fatal("invalid input invoked probe"); return true, nil })
	if !errors.Is(err, ErrCapacityInvalid) || n != 0 {
		t.Fatalf("invalid: %v", err)
	}
	cfg := ServingSweepConfig{Workload: []ServingRequest{{}, {}}, TTFTP99Budget: 10 * time.Millisecond, ITLP99Budget: 20 * time.Millisecond}
	a, b := 5., 21.
	stats := ServingStats{Requests: 2, OK: 2, TTFTMillis: QuantileMetric{Status: "measured", P99: &a}, ITLMillis: QuantileMetric{Status: "measured", P99: &b}}
	if pass, e := servingCapacitySLO(stats, cfg); pass || e != nil {
		t.Fatal("independent ITL boundary ignored")
	}
	stats.ITLMillis.P99 = &a
	if pass, e := servingCapacityProbePass(ServingTrackResult{Status: "measured", dispatchQPS: 9, Stats: stats}, 10, cfg); pass || !errors.Is(e, ErrCapacityUnknown) {
		t.Fatal("under-delivery passed")
	}
	stats.ITLMillis.P99 = nil
	if _, e := servingCapacitySLO(stats, cfg); !errors.Is(e, ErrCapacityUnknown) {
		t.Fatal("missing quantile not unknown")
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	cfg = ServingSweepConfig{Model: "synthetic", Tracks: []ServingTrackConfig{{Track: TrackOurs, Model: "synthetic", BaseURL: server.URL}}, Contracts: map[ServingTrack]ServingSweepTrackContract{TrackOurs: {Track: TrackOurs, Model: "synthetic", Engine: "fixture", EngineReceiptDigest: "sha256:" + strings.Repeat("a", 64), BatchCapacity: 2, CapacitySource: "synthetic"}}, Workload: []ServingRequest{{ID: "1", Messages: []ChatMessage{{Role: "user", Content: "one"}}, MaxOutputTokens: 2}, {ID: "2", Messages: []ChatMessage{{Role: "user", Content: "two"}}, MaxOutputTokens: 2}}, TTFTP99Budget: time.Second, Client: server.Client(), Timeout: time.Second, CapacitySearch: &ServingCapacitySearchConfig{MinQPS: 10, MaxQPS: 20, ToleranceQPS: 1, MaxProbes: 2, MaxRequests: 4, MaxDuration: 2 * time.Second, Concurrency: 2}}
	rep, e := RunServingSweep(context.Background(), cfg)
	if e != nil {
		t.Fatal(e)
	}
	got := rep.CapacitySearch
	if got == nil || got.Status != "bounded" || got.Probes != 2 || got.BestPassingQPS == nil || *got.BestPassingQPS != 20 || got.Contract.Engine != "fixture" || got.WorkloadDigest != rep.Workload.Digest || len(rep.Points) != 0 {
		t.Fatalf("consumer result: %+v", got)
	}
	cancelled, cancel := context.WithCancel(context.Background())
	cancel()
	stopped, e := RunServingSweep(cancelled, cfg)
	if e != nil || stopped.CapacitySearch.Status != "unknown" || stopped.CapacitySearch.ReasonCode != "cancelled_or_deadline" || stopped.CapacitySearch.BestPassingQPS != nil {
		t.Fatalf("cancelled search: %v %+v", e, stopped)
	}
	cfg.CapacitySearch.MinQPS = math.SmallestNonzeroFloat64
	if _, e := RunServingSweep(context.Background(), cfg); !errors.Is(e, ErrCapacityInvalid) {
		t.Fatalf("unrepresentable dispatch interval not rejected: %v", e)
	}
	cfg.CapacitySearch.MinQPS = 10
	cfg.CapacitySearch.MaxRequests = 3
	if _, e = RunServingSweep(context.Background(), cfg); !errors.Is(e, ErrCapacityInvalid) {
		t.Fatalf("request budget: %v", e)
	}
}
