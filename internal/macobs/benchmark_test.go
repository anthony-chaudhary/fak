package macobs

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var (
	benchSnapshotSink Snapshot
	benchMLXSink      MLXServingTelemetry
	benchPrefixSink   PrefixCacheTelemetry
	benchHeadroomSink HeadroomTelemetry
	benchAnalysisSink AnalysisReport
	benchHardwareSink HardwareTelemetry
	benchLineSink     string
	benchLabelsSink   map[string]string
	benchValSink      float64
	benchOkSink       bool
)

const benchSampleVLLMPrometheus = `
# HELP vllm:num_requests_running Number of requests currently running on GPU.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running 4
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting 1
# HELP vllm:kv_cache_usage_perc Percentage of KV cache memory used.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc 42.5
# HELP vllm:time_to_first_token_seconds_sum Latency to first token.
# TYPE vllm:time_to_first_token_seconds_sum counter
vllm:time_to_first_token_seconds_sum 1.250
vllm:time_to_first_token_seconds_count 10
# HELP vllm:inter_token_latency_seconds_sum Inter-token latency.
# TYPE vllm:inter_token_latency_seconds_sum counter
vllm:inter_token_latency_seconds_sum 0.240
vllm:inter_token_latency_seconds_count 20
# HELP vllm:avg_prompt_throughput_tok_per_s Prompt throughput.
# TYPE vllm:avg_prompt_throughput_tok_per_s gauge
vllm:avg_prompt_throughput_tok_per_s 350.2
# HELP vllm:avg_generation_throughput_tok_per_s Generation throughput.
# TYPE vllm:avg_generation_throughput_tok_per_s gauge
vllm:avg_generation_throughput_tok_per_s 45.8
# HELP vllm:prefix_cache_hits Prefix cache hits.
# TYPE vllm:prefix_cache_hits counter
vllm:prefix_cache_hits 850
# HELP vllm:prefix_cache_queries Prefix cache queries.
# TYPE vllm:prefix_cache_queries counter
vllm:prefix_cache_queries 1000
# HELP vllm:prefix_cache_misses Prefix cache misses.
# TYPE vllm:prefix_cache_misses counter
vllm:prefix_cache_misses 150
# HELP vllm:num_cached_blocks Number of cached blocks.
# TYPE vllm:num_cached_blocks gauge
vllm:num_cached_blocks 4096
# HELP vllm:num_total_blocks Total blocks available.
# TYPE vllm:num_total_blocks gauge
vllm:num_total_blocks 8192
`

const benchSampleMLXLMPrometheus = `
mlx:active_requests 3
mlx:queued_requests 0
mlx:kv_cache_usage_pct 35.0
mlx:ttft_seconds_sum 0.8
mlx:ttft_seconds_count 4
mlx:itl_seconds_sum 0.15
mlx:itl_seconds_count 15
mlx:tokens_per_second{type="prompt"} 280.5
mlx:tokens_per_second{type="generation"} 38.2
mlx:prefix_cache_hit_ratio 0.82
mlx:prefix_cache_hits 450
mlx:prefix_cache_queries 550
`

const benchSampleIORegXML = `<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<array>
	<dict>
		<key>PerformanceStatistics</key>
		<dict>
			<key>Alloc system memory</key>
			<integer>25769803776</integer>
			<key>Allocated PB Size</key>
			<integer>113508352</integer>
			<key>Device Utilization %</key>
			<integer>88</integer>
			<key>In use system memory</key>
			<integer>17179869184</integer>
			<key>Renderer Utilization %</key>
			<integer>85</integer>
			<key>recoveryCount</key>
			<integer>0</integer>
		</dict>
	</dict>
</array>
</plist>`

const benchSampleVMStat = `Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    25000.
Pages active:                                 500000.
Pages inactive:                               400000.
Pages wired down:                             300000.
Pages occupied by compressor:                 120000.
Pageins:                                      8500000.
Pageouts:                                         450.
`

func benchMockRunner() CommandRunner {
	return func(ctx context.Context, name string, args ...string) ([]byte, error) {
		cmdStr := name + " " + strings.Join(args, " ")
		switch {
		case strings.Contains(cmdStr, "hw.memsize"):
			return []byte("38654705664\n"), nil // 36GB
		case strings.Contains(cmdStr, "iogpu.wired_limit_mb"):
			return []byte("27648\n"), nil // 27GB
		case strings.Contains(cmdStr, "vm.swapusage"):
			return []byte("total = 1024.00M used = 0.00M free = 1024.00M\n"), nil
		case strings.Contains(cmdStr, "kern.memorystatus_level"):
			return []byte("75\n"), nil
		case strings.Contains(cmdStr, "vm_stat"):
			return []byte(benchSampleVMStat), nil
		case strings.Contains(cmdStr, "therm"):
			return []byte("Note: No thermal warning level has been recorded\n"), nil
		case strings.Contains(cmdStr, "ps"):
			return []byte("Now drawing from 'AC Power'\n -InternalBattery-0 100%;\n"), nil
		default:
			return nil, nil
		}
	}
}

// ----------------------------------------------------------------------------
// 1. MLX Metrics Collection Benchmarks
// ----------------------------------------------------------------------------

// BenchmarkParseMLXMetrics_VLLM measures parsing representative Prometheus text
// from a vllm-mlx serving instance.
func BenchmarkParseMLXMetrics_VLLM(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv, prefix := ParseMLXMetrics(benchSampleVLLMPrometheus)
		benchMLXSink = srv
		benchPrefixSink = prefix
	}
}

// BenchmarkParseMLXMetrics_MLXLM measures parsing Prometheus text from mlx-lm.
func BenchmarkParseMLXMetrics_MLXLM(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv, prefix := ParseMLXMetrics(benchSampleMLXLMPrometheus)
		benchMLXSink = srv
		benchPrefixSink = prefix
	}
}

// BenchmarkParseMLXMetrics_Empty measures fast-path handling of empty metrics payloads.
func BenchmarkParseMLXMetrics_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv, prefix := ParseMLXMetrics("")
		benchMLXSink = srv
		benchPrefixSink = prefix
	}
}

// BenchmarkParsePrometheusLine measures single metric line parsing and label dictionary extraction.
func BenchmarkParsePrometheusLine(b *testing.B) {
	line := `mlx:tokens_per_second{type="generation",model="qwen2.5-7b",quant="4bit"} 38.2`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		name, labels, val, ok := parsePrometheusLine(line)
		benchLineSink = name
		benchLabelsSink = labels
		benchValSink = val
		benchOkSink = ok
	}
}

// BenchmarkScrapeMLXMetrics measures HTTP scraping of an in-memory MLX metrics server.
func BenchmarkScrapeMLXMetrics(b *testing.B) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(benchSampleVLLMPrometheus))
	}))
	defer ts.Close()

	ctx := context.Background()
	client := ts.Client()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		srv, prefix, err := ScrapeMLXMetrics(ctx, ts.URL, client)
		if err != nil {
			b.Fatal(err)
		}
		benchMLXSink = srv
		benchPrefixSink = prefix
	}
}

// BenchmarkParseMLXMetrics_Parallel measures concurrent Prometheus parsing under multi-goroutine load.
func BenchmarkParseMLXMetrics_Parallel(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var localSrv MLXServingTelemetry
		var localPrefix PrefixCacheTelemetry
		for pb.Next() {
			localSrv, localPrefix = ParseMLXMetrics(benchSampleVLLMPrometheus)
		}
		benchMLXSink = localSrv
		benchPrefixSink = localPrefix
	})
}

// ----------------------------------------------------------------------------
// 2. Headroom Calculation Benchmarks
// ----------------------------------------------------------------------------

// BenchmarkComputeHeadroom_Standard measures unified memory headroom calculation
// for a 7B/8B GQA model with a 4096-token shared prefix.
func BenchmarkComputeHeadroom_Standard(b *testing.B) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 38654705664,       // 36GB
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024, // 27GB
		Available:              true,
	}
	cfg := DefaultHeadroomConfig()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		head := ComputeHeadroom(hw, cfg)
		benchHeadroomSink = head
	}
}

// BenchmarkComputeHeadroom_LargeModel measures headroom modeling for a 70B GQA model
// with a 32K context window and 16K shared prefix.
func BenchmarkComputeHeadroom_LargeModel(b *testing.B) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 137438953472,       // 128GB unified memory
		WiredMemoryLimitBytes:  100 * 1024 * 1024 * 1024, // 100GB wired limit
		Available:              true,
	}
	cfg := HeadroomConfig{
		Layers:             80,
		KVHeads:            8,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   40 * 1024 * 1024 * 1024, // 40GB 4-bit weights
		ContextTokens:      32768,
		SharedPrefixTokens: 16384,
		PrivateTailTokens:  4096,
		OSReserveBytes:     4 * 1024 * 1024 * 1024,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		head := ComputeHeadroom(hw, cfg)
		benchHeadroomSink = head
	}
}

// BenchmarkComputeHeadroom_Constrained measures headroom calculation when memory pool
// is constrained near base model weight and OS reserve limits.
func BenchmarkComputeHeadroom_Constrained(b *testing.B) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 17179869184, // 16GB
		WiredMemoryLimitBytes:  9 * 1024 * 1024 * 1024, // 9GB
		Available:              true,
	}
	cfg := HeadroomConfig{
		Layers:             28,
		KVHeads:            4,
		HeadDim:            128,
		KVBytesPerElement:  2,
		ModelWeightBytes:   5 * 1024 * 1024 * 1024,
		ContextTokens:      8192,
		SharedPrefixTokens: 4096,
		PrivateTailTokens:  2048,
		OSReserveBytes:     3 * 1024 * 1024 * 1024,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		head := ComputeHeadroom(hw, cfg)
		benchHeadroomSink = head
	}
}

// BenchmarkComputeHeadroom_ZeroDefaults measures default sanitization overhead when
// config fields are zero-initialized.
func BenchmarkComputeHeadroom_ZeroDefaults(b *testing.B) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 38654705664,
		Available:              true,
	}
	cfg := HeadroomConfig{}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		head := ComputeHeadroom(hw, cfg)
		benchHeadroomSink = head
	}
}

// BenchmarkComputeHeadroom_Parallel measures concurrent headroom calculation under
// multi-threaded agent coordinator load.
func BenchmarkComputeHeadroom_Parallel(b *testing.B) {
	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 38654705664,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	cfg := DefaultHeadroomConfig()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var localHead HeadroomTelemetry
		for pb.Next() {
			localHead = ComputeHeadroom(hw, cfg)
		}
		benchHeadroomSink = localHead
	})
}

// ----------------------------------------------------------------------------
// 3. Telemetry Analysis Benchmarks (Diagnose)
// ----------------------------------------------------------------------------

// BenchmarkDiagnose_HeadroomOK measures nominal headroom assessment with fine-grained
// bottleneck resolution (GPU utilization and decode token rates).
func BenchmarkDiagnose_HeadroomOK(b *testing.B) {
	hw := HardwareTelemetry{
		DeviceUtilizationPct: 82.0,
		ThermalState:         ThermalNominal,
		PowerSource:          PowerAC,
		Available:            true,
	}
	srv := MLXServingTelemetry{
		ActiveRequests:     2,
		KVCacheUsagePct:    45.0,
		DecodeTokensPerSec: 32.5,
		Available:          true,
	}
	head := HeadroomTelemetry{
		MaxSharedAgents:   16,
		MaxIsolatedAgents: 4,
		Available:         true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 4)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_ReduceConcurrency measures diagnostic evaluation when requested
// agents exceed modeled shared headroom capacity.
func BenchmarkDiagnose_ReduceConcurrency(b *testing.B) {
	hw := HardwareTelemetry{
		Available: true,
	}
	srv := MLXServingTelemetry{
		Available: true,
	}
	head := HeadroomTelemetry{
		MaxSharedAgents:   5,
		MaxIsolatedAgents: 1,
		Available:         true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 12)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_PressureDegrade measures evaluation of critical memorystatus pressure.
func BenchmarkDiagnose_PressureDegrade(b *testing.B) {
	hw := HardwareTelemetry{
		MemoryStatusLevel:      12, // critical pressure <= 15
		InUseSystemMemoryBytes: 26 * 1024 * 1024 * 1024,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{MaxIsolatedAgents: 2, Available: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 4)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_EvictPrefixCache measures KV cache saturation (>90%) diagnostics.
func BenchmarkDiagnose_EvictPrefixCache(b *testing.B) {
	hw := HardwareTelemetry{Available: true}
	srv := MLXServingTelemetry{
		ActiveRequests:  4,
		KVCacheUsagePct: 94.5,
		Available:       true,
	}
	head := HeadroomTelemetry{Available: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 4)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_ThermalThrottled measures SoC thermal throttling diagnostic evaluation.
func BenchmarkDiagnose_ThermalThrottled(b *testing.B) {
	hw := HardwareTelemetry{
		ThermalState:    ThermalSerious,
		CPUThermalLevel: 2,
		GPUThermalLevel: 2,
		Available:       true,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 8)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_SwapCritical measures swap thrashing and disk paging diagnosis.
func BenchmarkDiagnose_SwapCritical(b *testing.B) {
	hw := HardwareTelemetry{
		SwapUsedBytes: 2 * 1024 * 1024 * 1024, // 2GB
		PageOuts:      65000,
		Available:     true,
	}
	srv := MLXServingTelemetry{Available: true}
	head := HeadroomTelemetry{Available: true}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report := Diagnose(hw, srv, head, 4)
		benchAnalysisSink = report
	}
}

// BenchmarkDiagnose_Parallel measures concurrent diagnostic steering under multi-agent load.
func BenchmarkDiagnose_Parallel(b *testing.B) {
	hw := HardwareTelemetry{
		DeviceUtilizationPct: 82.0,
		ThermalState:         ThermalNominal,
		PowerSource:          PowerAC,
		Available:            true,
	}
	srv := MLXServingTelemetry{
		ActiveRequests:     2,
		KVCacheUsagePct:    45.0,
		DecodeTokensPerSec: 32.5,
		Available:          true,
	}
	head := HeadroomTelemetry{
		MaxSharedAgents:   16,
		MaxIsolatedAgents: 4,
		Available:         true,
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		var localReport AnalysisReport
		for pb.Next() {
			localReport = Diagnose(hw, srv, head, 4)
		}
		benchAnalysisSink = localReport
	})
}

// ----------------------------------------------------------------------------
// 4. End-to-End Collector Observation Benchmarks
// ----------------------------------------------------------------------------

// BenchmarkCollector_Observe measures full end-to-end observation pipeline:
// command execution via runner + MLX scrape + headroom computation + diagnosis.
func BenchmarkCollector_Observe(b *testing.B) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(benchSampleVLLMPrometheus))
	}))
	defer ts.Close()

	col := NewCollector(
		WithMetricsURL(ts.URL),
		WithHTTPClient(ts.Client()),
		WithCommandRunner(benchMockRunner()),
		WithRequestedAgents(4),
		WithHeadroomConfig(DefaultHeadroomConfig()),
	)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := col.Observe(ctx)
		if err != nil {
			b.Fatal(err)
		}
		benchSnapshotSink = snap
	}
}

// BenchmarkCollector_Observe_HardwareOnly measures observation pipeline when no
// MLX metrics endpoint is configured (hardware probing + headroom + diagnosis).
func BenchmarkCollector_Observe_HardwareOnly(b *testing.B) {
	col := NewCollector(
		WithMetricsURL(""),
		WithCommandRunner(benchMockRunner()),
		WithRequestedAgents(4),
		WithHeadroomConfig(DefaultHeadroomConfig()),
	)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		snap, err := col.Observe(ctx)
		if err != nil {
			b.Fatal(err)
		}
		benchSnapshotSink = snap
	}
}

// ----------------------------------------------------------------------------
// 5. Hardware Parsing Benchmarks
// ----------------------------------------------------------------------------

// BenchmarkParseIORegXML measures parsing of IOAccelerator PerformanceStatistics XML.
func BenchmarkParseIORegXML(b *testing.B) {
	data := []byte(benchSampleIORegXML)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		alloc, inUse, devUtil, rendUtil, recov, ok := ParseIORegXML(data)
		if !ok {
			b.Fatal("ParseIORegXML failed")
		}
		benchHardwareSink.AllocSystemMemoryBytes = alloc
		benchHardwareSink.InUseSystemMemoryBytes = inUse
		benchHardwareSink.DeviceUtilizationPct = devUtil
		benchHardwareSink.RendererUtilizationPct = rendUtil
		benchHardwareSink.RecoveryCount = recov
	}
}

// BenchmarkParseVMStat measures parsing Mach virtual memory statistics text.
func BenchmarkParseVMStat(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		free, wired, comp, ins, outs, ok := ParseVMStat(benchSampleVMStat)
		if !ok {
			b.Fatal("ParseVMStat failed")
		}
		benchHardwareSink.FreeBytes = free
		benchHardwareSink.WiredBytes = wired
		benchHardwareSink.CompressedBytes = comp
		benchHardwareSink.PageIns = ins
		benchHardwareSink.PageOuts = outs
	}
}

// BenchmarkParseSysctlSwapUsage measures parsing sysctl vm.swapusage line.
func BenchmarkParseSysctlSwapUsage(b *testing.B) {
	line := "total = 24576.00M used = 1024.50M free = 23551.50M (encrypted)\n"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		total, used, ok := ParseSysctlSwapUsage(line)
		if !ok {
			b.Fatal("ParseSysctlSwapUsage failed")
		}
		benchHardwareSink.SwapTotalBytes = total
		benchHardwareSink.SwapUsedBytes = used
	}
}

// ----------------------------------------------------------------------------
// 6. Benchmark Sanity Test
// ----------------------------------------------------------------------------

// TestBenchmarkSanity ensures all benchmark scenarios execute without panics and produce valid outputs.
func TestBenchmarkSanity(t *testing.T) {
	srv, prefix := ParseMLXMetrics(benchSampleVLLMPrometheus)
	if !srv.Available || !prefix.Available {
		t.Fatalf("ParseMLXMetrics failed on sample: srv=%+v prefix=%+v", srv, prefix)
	}

	hw := HardwareTelemetry{
		TotalSystemMemoryBytes: 38654705664,
		WiredMemoryLimitBytes:  27 * 1024 * 1024 * 1024,
		Available:              true,
	}
	head := ComputeHeadroom(hw, DefaultHeadroomConfig())
	if !head.Available || head.MaxSharedAgents <= 0 {
		t.Fatalf("ComputeHeadroom failed on sample: %+v", head)
	}

	report := Diagnose(hw, srv, head, 4)
	if report.Verdict != VerdictHeadroomOK {
		t.Fatalf("Diagnose failed on sample: %+v", report)
	}
}
