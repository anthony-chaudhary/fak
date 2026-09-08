package metrics

import (
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"
)

var metricsServiceExpects = []struct {
	specSource   string
	actualServer string
	match        bool
}{
	{"l3_server_ready", "l3_server_ready", true},
	{"l3_server_active_connections", "l3_server_active_connections", true},
	{"l3_server_uptime_seconds", "l3_server_uptime_seconds", true},
	{"l3_server_active_gb", "l3_server_active_gb", true},
	{"l3_server_entries", "l3_server_entries", true},

	{"l3_server_gets_total", "l3_server_gets_total", true},
	{"l3_server_sets_total", "l3_server_sets_total", true},
	{"l3_server_hits_total", "l3_server_hits_total", true},
	{"l3_server_misses_total", "l3_server_misses_total", true},
	{"l3_server_evictions_total", "l3_server_evictions_total", true},
	{"l3_server_exists_total", "l3_server_exists_total", true},

	{"l3_server_hit_rate_percent", "l3_server_hit_rate_percent", true},
	{"l3_server_eviction_rate_percent", "l3_server_eviction_rate_percent", true},
	{"l3_server_eviction_fail_rate_percent", "l3_server_eviction_fail_rate_percent", true},

	{"l3_server_wire_throughput_gbps_in", "l3_server_wire_throughput_gbps_in", true},
	{"l3_server_wire_throughput_gbps_out", "l3_server_wire_throughput_gbps_out", true},
	{"l3_server_wire_throughput_gbps_total", "l3_server_wire_throughput_gbps_total", true},

	{"l3_server_slab_slot_utilization", "l3_server_slab_slot_utilization", true},
	{"l3_server_slab_effective_gb", "l3_server_slab_effective_gb", true},

	{"l3_client_cache_hit_rate", "l3_client_cache_hit_rate", true},
	{"l3_client_token_usage", "l3_client_token_usage", true},
	{"l3_client_num_running_reqs", "l3_client_num_running_reqs", true},
	{"l3_client_gen_throughput", "l3_client_gen_throughput", true},
	{"l3_client_prefetch_bandwidth_gbps", "l3_client_prefetch_bandwidth_gbps", true},
	{"l3_client_avg_io_latency_ms", "l3_client_avg_io_latency_ms", true},
}

func TestContract_ActualMetricNamesExist(t *testing.T) {
	sources := []string{"collector.go", "client_stats.go"}
	var combined strings.Builder
	for _, name := range sources {
		data, err := os.ReadFile(name)
		if err != nil {
			t.Fatalf("read %s: %v", name, err)
		}
		combined.Write(data)
	}
	src := combined.String()

	for _, m := range metricsServiceExpects {
		if !strings.Contains(src, m.actualServer) {
			t.Errorf("actual metric %q (for spec %q) not found in collector source",
				m.actualServer, m.specSource)
		}
	}
}

func TestContract_DocumentMismatches(t *testing.T) {
	mismatches := 0
	for _, m := range metricsServiceExpects {
		if !m.match || m.specSource != m.actualServer {
			t.Errorf("MISMATCH: metrics-service expects %q but l3-server emits %q",
				m.specSource, m.actualServer)
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Fatalf("%d metric name mismatches between metrics-service Spec and l3-server", mismatches)
	}
}

func TestContract_EmittedMetricsMatchSpec(t *testing.T) {
	startup := &StartupState{}
	startup.Ready.Store(true)
	connReg := NewConnRegistry()
	clientStatsReg := NewClientStatsRegistry()
	startedAt := time.Now().Add(-10 * time.Second)
	c := NewCollector(startup, connReg, clientStatsReg, startedAt)

	c.CacheStateProvider = func() (int64, int64) {
		return 1024 * 1024 * 1024, 42
	}
	c.SlabMetrics = func() ([]SlabClassSnapshot, SlabDetectionSnapshot) {
		classes := []SlabClassSnapshot{
			{Size: 4096, TotalSlots: 100, UsedSlots: 50, SlotUtilization: 0.5},
		}
		det := SlabDetectionSnapshot{
			ModelTotalSlots: 1000,
			ModelClassSize:  4096,
			SlotUtilization: 0.5,
		}
		return classes, det
	}

	sm := NewShardMetrics(0)
	sm.IncrGets()
	sm.IncrSets()
	sm.IncrHits()
	sm.IncrMisses()
	sm.IncrExists()
	sm.IncrEvictions()
	c.SetShards([]*ShardMetrics{sm})

	clientStatsReg.Update(1, "127.0.0.1:1234", map[string]interface{}{
		"cache_hit_rate":          0.95,
		"token_usage":             0.80,
		"num_running_reqs":        4.0,
		"gen_throughput":          100.0,
		"prefetch_bandwidth_gbps": 1.5,
		"avg_io_latency_ms":       2.5,
	})

	time.Sleep(10 * time.Millisecond)

	req := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	rec := httptest.NewRecorder()
	c.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("ServeHTTP returned status %d, expected %d", rec.Code, http.StatusOK)
	}

	emitted := make(map[string]bool)
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "# TYPE ") {
			parts := strings.Fields(line)
			if len(parts) >= 3 {
				emitted[parts[2]] = true
			}
		}
	}

	for _, m := range metricsServiceExpects {
		if !emitted[m.actualServer] {
			t.Errorf("expected emitted metric %q not found in ServeHTTP output", m.actualServer)
		}
	}
}
