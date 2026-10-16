package gateway

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

type transientPoolPlanner struct {
	kvMemoryStatsPlanner
	pool agent.DeviceTransientPoolStats
	ok   bool
}

func (p transientPoolPlanner) DeviceTransientPool() (agent.DeviceTransientPoolStats, bool) {
	return p.pool, p.ok
}

func TestDeviceTransientPoolMetrics(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = transientPoolPlanner{pool: agent.DeviceTransientPoolStats{Backend: "vulkan", PooledBuffers: 3, PooledBytes: 4096, PeakPooledBytes: 8192}, ok: true}
	text := srv.renderMetrics()
	for _, want := range []string{
		`fak_gateway_device_transient_pool_bytes{backend="vulkan",kind="pooled"} 4096`,
		`fak_gateway_device_transient_pool_bytes{backend="vulkan",kind="peak"} 8192`,
		`fak_gateway_device_transient_pool_buffers{backend="vulkan"} 3`,
	} {
		if !strings.Contains(text, want+"\n") {
			t.Fatalf("missing %q", want)
		}
	}
}

func TestDeviceTransientPoolMetricsAbsentWithoutPool(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = transientPoolPlanner{}
	if strings.Contains(srv.renderMetrics(), "fak_gateway_device_transient_pool_") {
		t.Fatal("a backend without a transient pool must emit no pool series")
	}
}

func TestGoRuntimeMemoryMetricsOmitUnavailable(t *testing.T) {
	var r runtimeobs.Receipt
	r.Memory.HeapObjectsBytes = runtimeobs.U64{V: 100, OK: true}
	r.Memory.ProcessRSSBytes = runtimeobs.U64{V: 900, OK: true}
	r.GC.Cycles = runtimeobs.U64{V: 7, OK: true}
	r.GC.MemoryLimitBytes = runtimeobs.U64{V: 1 << 62, OK: true}
	r.GC.MemoryLimitUnlimited = true
	var b strings.Builder
	writeGoRuntimeMemoryMetrics(&b, r)
	text := b.String()
	for _, want := range []string{
		`fak_gateway_go_memory_bytes{kind="heap_objects"} 100`,
		`fak_gateway_go_memory_bytes{kind="process_rss"} 900`,
		`fak_gateway_go_gc_cycles_total 7`,
	} {
		if !strings.Contains(text, want+"\n") {
			t.Fatalf("missing %q", want)
		}
	}
	for _, absent := range []string{`kind="runtime_mapped"`, "fak_gateway_go_memory_limit_bytes", "fak_gateway_go_goroutines"} {
		if strings.Contains(text, absent) {
			t.Fatalf("unavailable axis %q rendered", absent)
		}
	}
}

func init() {
	var pinned runtimeobs.Receipt
	pinned.Memory.HeapObjectsBytes = runtimeobs.U64{V: 1 << 20, OK: true}
	pinned.GC.Cycles = runtimeobs.U64{V: 1, OK: true}
	goRuntimeMetricsSample = func() runtimeobs.Receipt { return pinned }
}

func TestGoRuntimeMemoryMetricsOnScrape(t *testing.T) {
	if text := newTestServer(t).renderMetrics(); !strings.Contains(text, `fak_gateway_go_memory_bytes{kind="heap_objects"} 1048576`+"\n") {
		t.Fatal("/metrics does not render the runtime sampler")
	}
}

func TestGoRuntimeMemoryMetricsFromLiveCollector(t *testing.T) {
	var b strings.Builder
	writeGoRuntimeMemoryMetrics(&b, goRuntimeCollector().Sample())
	text := b.String()
	for _, want := range []string{`fak_gateway_go_memory_bytes{kind="heap_objects"} `, "fak_gateway_go_gc_cycles_total "} {
		if !strings.Contains(text, want) {
			t.Fatalf("live scrape missing %q", want)
		}
	}
}
