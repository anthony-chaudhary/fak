package gateway

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// fak-test:runtime fast est=100ms lane=default
func TestServingPrefixTokenAccounting(t *testing.T) {
	for _, tc := range []struct {
		name, body                       string
		queries, hits, ratio             float64
		haveQueries, haveHits, haveRatio bool
	}{
		{name: "unknown", body: "vllm:num_requests_running 1\n"},
		{name: "query_only", body: "vllm:prefix_cache_queries 10\n", queries: 10, haveQueries: true},
		{name: "hit_only", body: "vllm:prefix_cache_hits 7\n", hits: 7, haveHits: true},
		{name: "zero_queries", body: "vllm:prefix_cache_queries 0\nvllm:prefix_cache_hits 0\n", haveQueries: true, haveHits: true},
		{name: "zero_hits", body: "vllm:prefix_cache_queries 10\nvllm:prefix_cache_hits 0\n", queries: 10, haveQueries: true, haveHits: true, haveRatio: true},
		{name: "token_ratio", body: "vllm:prefix_cache_queries 10\nvllm:prefix_cache_hits 6\n", queries: 10, hits: 6, ratio: .6, haveQueries: true, haveHits: true, haveRatio: true},
		{name: "direct_gauge", body: "vllm:prefix_cache_hit_rate 0.9\nvllm:prefix_cache_queries 10\nvllm:prefix_cache_hits 6\n", queries: 10, hits: 6, ratio: .9, haveQueries: true, haveHits: true, haveRatio: true},
		{name: "separate_tiers", body: "vllm:prefix_cache_queries 10\nvllm:prefix_cache_hits 6\nvllm:external_prefix_cache_queries 20\nvllm:external_prefix_cache_hits 5\nvllm:connector_prefix_cache_queries 30\nvllm:connector_prefix_cache_hits 7\n", queries: 10, hits: 6, ratio: .6, haveQueries: true, haveHits: true, haveRatio: true},
		{name: "overflow", body: "vllm:prefix_cache_queries{rank=\"0\"} 1e308\nvllm:prefix_cache_queries{rank=\"1\"} 1e308\nvllm:prefix_cache_hits 1\n", hits: 1, haveHits: true},
		{name: "invalid_counters", body: "vllm:prefix_cache_queries NaN\nvllm:prefix_cache_hits -1\nvllm:prefix_cache_hits +Inf\nvllm:num_requests_running 1\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			emitter := NewServingScrapeEmitter(ServingMetricLabels{Worker: "tokens", Engine: "vllm", Model: "fixture"})
			// Replacing a scrape must retain resets and missingness, not stale totals.
			if err := emitter.IngestPrometheus("vllm:prefix_cache_queries 100\nvllm:prefix_cache_hits 90\n"); err != nil {
				t.Fatal(err)
			}
			if err := emitter.IngestPrometheus(tc.body); err != nil {
				t.Fatal(err)
			}
			rows := emitter.SnapshotServingMetrics()
			if len(rows) != 1 {
				t.Fatalf("rows=%+v, want one present row", rows)
			}
			for _, check := range []struct {
				name    string
				got     ServingGauge
				value   float64
				present bool
			}{
				{"queries", rows[0].PrefixCacheQueries, tc.queries, tc.haveQueries},
				{"hits", rows[0].PrefixCacheHits, tc.hits, tc.haveHits},
				{"ratio", rows[0].PrefixCacheHitRate, tc.ratio, tc.haveRatio},
			} {
				if check.got.Set != check.present || check.present && check.got.Value != check.value {
					t.Fatalf("%s=%+v, want value=%v present=%v", check.name, check.got, check.value, check.present)
				}
			}
			srv := newTestServer(t)
			srv.SetServingMetricsEmitters(emitter)
			recorder := httptest.NewRecorder()
			srv.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, "/metrics", nil))
			if recorder.Code != http.StatusOK {
				t.Fatalf("/metrics status=%d", recorder.Code)
			}
			out := recorder.Body.String()
			labels := `{worker="tokens",engine="vllm",model="fixture"}`
			for _, check := range []struct {
				name    string
				present bool
			}{
				{"fak_serving_prefix_cache_queries_total", tc.haveQueries},
				{"fak_serving_prefix_cache_hits_total", tc.haveHits},
				{"fak_serving_prefix_cache_hit_rate", tc.haveRatio},
			} {
				if got := strings.Contains(out, check.name+labels+" "); got != check.present {
					t.Fatalf("%s sample present=%v, want %v", check.name, got, check.present)
				}
				if check.present && check.name != "fak_serving_prefix_cache_hit_rate" && !strings.Contains(out, "# TYPE "+check.name+" counter\n") {
					t.Fatalf("%s lost counter type", check.name)
				}
			}
		})
	}

	t.Run("no_cross_source_pair", func(t *testing.T) {
		dst := ServingMetricRow{PrefixCacheQueries: ServingGaugeValue(10)}
		src := ServingMetricRow{PrefixCacheHits: ServingGaugeValue(6)}
		fillServingMetricRowGaps(&dst, src)
		if !dst.PrefixCacheQueries.Set || dst.PrefixCacheHits.Set {
			t.Fatalf("gap fill fabricated a pair: %+v", dst)
		}
		mergeServingMetricRow(&dst, src)
		if dst.PrefixCacheQueries.Set || !dst.PrefixCacheHits.Set || dst.PrefixCacheHitRate.Set {
			t.Fatalf("merge mixed prefix sources: %+v", dst)
		}
	})
}
