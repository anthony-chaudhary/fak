package metrics

import (
	"os"
	"strings"
	"testing"
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

	{"l3_server_gets_total", "l3_ops_gets_total", false},
	{"l3_server_sets_total", "l3_ops_sets_total", false},
	{"l3_server_hits_total", "l3_ops_hits_total", false},
	{"l3_server_misses_total", "l3_ops_misses_total", false},
	{"l3_server_evictions_total", "l3_eviction_total", false},
	{"l3_server_exists_total", "l3_ops_exists_total", false},

	{"l3_server_hit_rate_percent", "l3_ops_hit_rate_percent", false},
	{"l3_server_eviction_rate_percent", "l3_eviction_rate_percent", false},
	{"l3_server_eviction_fail_rate_percent", "l3_eviction_fail_rate_percent", false},

	{"l3_server_wire_throughput_gbps_in", "l3_wire_throughput_gbps_in", false},
	{"l3_server_wire_throughput_gbps_out", "l3_wire_throughput_gbps_out", false},
	{"l3_server_wire_throughput_gbps_total", "l3_wire_throughput_gbps_total", false},

	{"l3_server_slab_slot_utilization", "l3_slab_model_effective_gb", false},
	{"l3_server_slab_effective_gb", "l3_slab_model_effective_gb", false},

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
		if !m.match {
			t.Logf("MISMATCH: metrics-service expects %q but l3-server emits %q",
				m.specSource, m.actualServer)
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Logf("%d metric name mismatches between metrics-service Spec and l3-server", mismatches)
	}
}
