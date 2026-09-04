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
	{"cama_server_ready", "cama_server_ready", true},
	{"cama_server_active_connections", "cama_server_active_connections", true},
	{"cama_server_uptime_seconds", "cama_server_uptime_seconds", true},
	{"cama_server_active_gb", "cama_server_active_gb", true},
	{"cama_server_entries", "cama_server_entries", true},

	{"cama_server_gets_total", "cama_ops_gets_total", false},
	{"cama_server_sets_total", "cama_ops_sets_total", false},
	{"cama_server_hits_total", "cama_ops_hits_total", false},
	{"cama_server_misses_total", "cama_ops_misses_total", false},
	{"cama_server_evictions_total", "cama_eviction_total", false},
	{"cama_server_exists_total", "cama_ops_exists_total", false},

	{"cama_server_hit_rate_percent", "cama_ops_hit_rate_percent", false},
	{"cama_server_eviction_rate_percent", "cama_eviction_rate_percent", false},
	{"cama_server_eviction_fail_rate_percent", "cama_eviction_fail_rate_percent", false},

	{"cama_server_wire_throughput_gbps_in", "cama_wire_throughput_gbps_in", false},
	{"cama_server_wire_throughput_gbps_out", "cama_wire_throughput_gbps_out", false},
	{"cama_server_wire_throughput_gbps_total", "cama_wire_throughput_gbps_total", false},

	{"cama_server_slab_slot_utilization", "cama_slab_model_effective_gb", false},
	{"cama_server_slab_effective_gb", "cama_slab_model_effective_gb", false},

	{"cama_client_cache_hit_rate", "cama_client_cache_hit_rate", true},
	{"cama_client_token_usage", "cama_client_token_usage", true},
	{"cama_client_num_running_reqs", "cama_client_num_running_reqs", true},
	{"cama_client_gen_throughput", "cama_client_gen_throughput", true},
	{"cama_client_prefetch_bandwidth_gbps", "cama_client_prefetch_bandwidth_gbps", true},
	{"cama_client_avg_io_latency_ms", "cama_client_avg_io_latency_ms", true},
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
			t.Logf("MISMATCH: metrics-service expects %q but cama-server emits %q",
				m.specSource, m.actualServer)
			mismatches++
		}
	}
	if mismatches > 0 {
		t.Logf("%d metric name mismatches between metrics-service Spec and cama-server", mismatches)
	}
}
