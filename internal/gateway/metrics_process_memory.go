package gateway

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/runtimeobs"
)

// goRuntimeMetricsSample is the /metrics runtime sampler; the gateway test
// binary pins it so byte-determinism tests do not see a live heap.
var goRuntimeMetricsSample = func() runtimeobs.Receipt { return goRuntimeCollector().Sample() }

// writeGoRuntimeMemoryMetrics renders the same runtimeobs receipt /debug/vars
// serves as go_runtime, so a Prometheus scraper (and the private perf reader)
// sees the Go heap / runtime-mapped / process-RSS split without a second JSON
// endpoint. Unreadable axes are omitted, never rendered as zero.
func writeGoRuntimeMemoryMetrics(b *strings.Builder, r runtimeobs.Receipt) {
	mem := []struct {
		kind string
		v    runtimeobs.U64
	}{
		{"heap_objects", r.Memory.HeapObjectsBytes},
		{"heap_unused", r.Memory.HeapUnusedBytes},
		{"heap_free", r.Memory.HeapFreeBytes},
		{"heap_released", r.Memory.HeapReleasedBytes},
		{"heap_stacks", r.Memory.HeapStacksBytes},
		{"os_stacks", r.Memory.OSStacksBytes},
		{"runtime_mapped", r.Memory.TotalMappedBytes},
		{"process_rss", r.Memory.ProcessRSSBytes},
	}
	header := false
	for _, row := range mem {
		if !row.v.OK {
			continue
		}
		if !header {
			writeHelpType(b, "fak_gateway_go_memory_bytes", "Gateway process memory by kind: Go heap classes and runtime-mapped bytes from runtime/metrics, plus whole-process RSS. process_rss minus runtime_mapped approximates memory mapped outside the Go runtime (weights, cgo/device allocations).", "gauge")
			header = true
		}
		fmt.Fprintf(b, "fak_gateway_go_memory_bytes{kind=\"%s\"} %d\n", row.kind, row.v.V)
	}
	if r.GC.HeapLiveBytes.OK || r.GC.HeapGoalBytes.OK {
		writeHelpType(b, "fak_gateway_go_gc_heap_bytes", "Go GC heap targets: kind=live is the heap marked live by the last cycle, kind=goal the heap size that triggers the next one.", "gauge")
		if r.GC.HeapLiveBytes.OK {
			fmt.Fprintf(b, "fak_gateway_go_gc_heap_bytes{kind=\"live\"} %d\n", r.GC.HeapLiveBytes.V)
		}
		if r.GC.HeapGoalBytes.OK {
			fmt.Fprintf(b, "fak_gateway_go_gc_heap_bytes{kind=\"goal\"} %d\n", r.GC.HeapGoalBytes.V)
		}
	}
	if r.GC.MemoryLimitBytes.OK && !r.GC.MemoryLimitUnlimited {
		writeHelpType(b, "fak_gateway_go_memory_limit_bytes", "GOMEMLIMIT soft memory limit; omitted when unlimited.", "gauge")
		fmt.Fprintf(b, "fak_gateway_go_memory_limit_bytes %d\n", r.GC.MemoryLimitBytes.V)
	}
	if r.GC.Cycles.OK {
		writeHelpType(b, "fak_gateway_go_gc_cycles_total", "Completed Go GC cycles since process start.", "counter")
		fmt.Fprintf(b, "fak_gateway_go_gc_cycles_total %d\n", r.GC.Cycles.V)
	}
	if v, ok := r.GC.Pauses.Quantile(0.99); ok {
		writeHelpType(b, "fak_gateway_go_gc_pause_p99_seconds", "p99 stop-the-world GC pause over the process lifetime (histogram bucket upper bound).", "gauge")
		fmt.Fprintf(b, "fak_gateway_go_gc_pause_p99_seconds %s\n", promFloat(v))
	}
	if r.Sched.Goroutines.OK {
		writeHelpType(b, "fak_gateway_go_goroutines", "Live goroutines in the gateway process.", "gauge")
		fmt.Fprintf(b, "fak_gateway_go_goroutines %d\n", r.Sched.Goroutines.V)
	}
}

// writeDeviceTransientPoolMetrics emits the device backend's request-transient
// buffer pool: bytes retired requests left parked for reuse. The pool holds
// device memory no live tensor owns, so it is the first thing a UMA host gives
// back under pressure; a backend that keeps no pool emits nothing.
func (s *Server) writeDeviceTransientPoolMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.DeviceTransientPoolReporter)
	if !ok {
		return
	}
	st, ok := reporter.DeviceTransientPool()
	if !ok {
		return
	}
	backend := promQuote(st.Backend)
	writeHelpType(b, "fak_gateway_device_transient_pool_bytes", "Device bytes parked in the backend's request-transient reuse pool: kind=pooled now, kind=peak the high-water mark since start.", "gauge")
	fmt.Fprintf(b, "fak_gateway_device_transient_pool_bytes{backend=\"%s\",kind=\"pooled\"} %d\n", backend, st.PooledBytes)
	fmt.Fprintf(b, "fak_gateway_device_transient_pool_bytes{backend=\"%s\",kind=\"peak\"} %d\n", backend, st.PeakPooledBytes)
	writeHelpType(b, "fak_gateway_device_transient_pool_buffers", "Buffers parked in the backend's request-transient reuse pool.", "gauge")
	fmt.Fprintf(b, "fak_gateway_device_transient_pool_buffers{backend=\"%s\"} %d\n", backend, st.PooledBuffers)
}

// writePagedPrefixMetrics emits the device planner's paged prefix page ledger: request
// leases, pages shared with a cached prefix versus freshly allocated, pool-full skips
// (a request that ran without a lease), and the private pages leased for generated
// tokens. A planner with no ledger emits nothing.
func (s *Server) writePagedPrefixMetrics(b *strings.Builder) {
	if s == nil || s.planner == nil {
		return
	}
	reporter, ok := s.planner.(agent.PagedPrefixReporter)
	if !ok {
		return
	}
	st, ok := reporter.PagedPrefixTelemetry()
	if !ok {
		return
	}
	writeHelpType(b, "fak_gateway_paged_prefix_leases_total", "Page leases taken over request prompts.", "counter")
	fmt.Fprintf(b, "fak_gateway_paged_prefix_leases_total %d\n", st.Leases)
	writeHelpType(b, "fak_gateway_paged_prefix_active_leases", "Page leases currently held by in-flight requests.", "gauge")
	fmt.Fprintf(b, "fak_gateway_paged_prefix_active_leases %d\n", st.ActiveLeases)
	writeHelpType(b, "fak_gateway_paged_prefix_pages_total", "Prompt pages bound by leases: kind=shared reused a cached prefix page, kind=allocated took a fresh page.", "counter")
	fmt.Fprintf(b, "fak_gateway_paged_prefix_pages_total{kind=\"shared\"} %d\n", st.SharedPages)
	fmt.Fprintf(b, "fak_gateway_paged_prefix_pages_total{kind=\"allocated\"} %d\n", st.AllocatedPages)
	writeHelpType(b, "fak_gateway_paged_prefix_pool_full_skips_total", "Requests that ran without a page lease because the pool was full: scope=prompt at admission, scope=decode while growing over generated tokens.", "counter")
	fmt.Fprintf(b, "fak_gateway_paged_prefix_pool_full_skips_total{scope=\"prompt\"} %d\n", st.FailOpen)
	fmt.Fprintf(b, "fak_gateway_paged_prefix_pool_full_skips_total{scope=\"decode\"} %d\n", st.DecodeFailOpen)
	writeHelpType(b, "fak_gateway_paged_prefix_decode_pages", "Private pages leased for generated tokens: kind=active held now, kind=total cumulative.", "gauge")
	fmt.Fprintf(b, "fak_gateway_paged_prefix_decode_pages{kind=\"active\"} %d\n", st.ActiveDecodePages)
	fmt.Fprintf(b, "fak_gateway_paged_prefix_decode_pages{kind=\"total\"} %d\n", st.DecodePages)
}
