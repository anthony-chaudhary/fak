package macobs

import (
	"fmt"
)

// Diagnose evaluates hardware, serving, and headroom telemetry to produce an actionable AnalysisReport.
func Diagnose(hw HardwareTelemetry, srv MLXServingTelemetry, head HeadroomTelemetry, requestedAgents int) AnalysisReport {
	if requestedAgents <= 0 {
		requestedAgents = 1
	}

	// 1. SWAP_CRITICAL: Kernel memory paging severely degrades throughput
	if hw.Available && (hw.SwapUsedBytes > 1024*1024*1024 || (hw.SwapUsedBytes > 0 && hw.PageOuts > 50000)) {
		return AnalysisReport{
			Verdict:           VerdictSwapCritical,
			PrimaryBottleneck: BottleneckSwap,
			BottleneckReason:  fmt.Sprintf("macOS kernel swap is critical (%d MB used, %d pageouts); disk paging stalls unified memory bandwidth", hw.SwapUsedBytes/(1024*1024), hw.PageOuts),
			RecommendedAgents: 1,
			Remediation:       "Immediately unmap inactive models and reduce agent concurrency to 1 to halt disk swap thrashing.",
			Confidence:        1.0,
		}
	}

	// 2. THERMAL_THROTTLED: Apple Silicon SoC thermal throttling
	if hw.Available && (hw.ThermalState == ThermalSerious || hw.ThermalState == ThermalCritical || hw.CPUThermalLevel >= 2 || hw.GPUThermalLevel >= 2) {
		recAgents := requestedAgents / 2
		if recAgents < 1 {
			recAgents = 1
		}
		return AnalysisReport{
			Verdict:           VerdictThermalThrottled,
			PrimaryBottleneck: BottleneckThermal,
			BottleneckReason:  fmt.Sprintf("Apple Silicon SoC is thermally throttled (state: %s, CPU level: %d, GPU level: %d)", hw.ThermalState, hw.CPUThermalLevel, hw.GPUThermalLevel),
			RecommendedAgents: recAgents,
			Remediation:       "Apple Silicon SoC is throttling clock frequencies; throttle request dispatch rates and allow the device to cool.",
			Confidence:        0.95,
		}
	}

	// 3. PRESSURE_DEGRADE: Unified memory pressure approaching system limits
	if hw.Available && ((hw.MemoryStatusLevel > 0 && hw.MemoryStatusLevel <= 15) ||
		(hw.InUseSystemMemoryBytes > 0 && hw.WiredMemoryLimitBytes > 0 && hw.InUseSystemMemoryBytes >= (hw.WiredMemoryLimitBytes*95)/100) ||
		(hw.AllocSystemMemoryBytes > 0 && hw.WiredMemoryLimitBytes > 0 && hw.AllocSystemMemoryBytes >= (hw.WiredMemoryLimitBytes*98)/100)) {
		recAgents := head.MaxIsolatedAgents
		if recAgents < 1 {
			recAgents = 1
		}
		return AnalysisReport{
			Verdict:           VerdictPressureDegrade,
			PrimaryBottleneck: BottleneckMemoryCapacity,
			BottleneckReason:  fmt.Sprintf("Unified memory pressure near limit (memorystatus level: %d, in-use: %d MB / wired limit: %d MB)", hw.MemoryStatusLevel, hw.InUseSystemMemoryBytes/(1024*1024), hw.WiredMemoryLimitBytes/(1024*1024)),
			RecommendedAgents: recAgents,
			Remediation:       "System memory pressure is near eviction limit; degrade non-critical subagent tasks and shed inactive turn context.",
			Confidence:        0.90,
		}
	}

	// 4. EVICT_PREFIX_CACHE: KV cache pool saturated
	if srv.Available && srv.KVCacheUsagePct >= 90.0 {
		recAgents := srv.ActiveRequests
		if recAgents < 1 {
			recAgents = 1
		}
		return AnalysisReport{
			Verdict:           VerdictEvictPrefixCache,
			PrimaryBottleneck: BottleneckCacheSaturation,
			BottleneckReason:  fmt.Sprintf("MLX KV cache pool is %.1f%% full; pending allocations risk queue stalls or OOM failure", srv.KVCacheUsagePct),
			RecommendedAgents: recAgents,
			Remediation:       "KV cache usage is near capacity (>90%); evict stale prefix cache entries and discard completed agent turns.",
			Confidence:        0.95,
		}
	}

	// 5. REDUCE_CONCURRENCY: Requested agents exceeds safe headroom or server queues are backed up
	if requestedAgents > 0 && head.Available && head.MaxSharedAgents > 0 && requestedAgents > head.MaxSharedAgents {
		return AnalysisReport{
			Verdict:           VerdictReduceConcurrency,
			PrimaryBottleneck: BottleneckMemoryCapacity,
			BottleneckReason:  fmt.Sprintf("Requested %d concurrent agents exceeds modeled headroom of %d shared agents", requestedAgents, head.MaxSharedAgents),
			RecommendedAgents: head.MaxSharedAgents,
			Remediation:       fmt.Sprintf("Limit active concurrent subagents to %d to remain within wired unified memory limits.", head.MaxSharedAgents),
			Confidence:        0.85,
		}
	}

	if srv.Available && srv.QueuedRequests > 0 && head.Available && head.MaxSharedAgents > 0 && srv.ActiveRequests >= head.MaxSharedAgents {
		recAgents := head.MaxSharedAgents
		if recAgents < 1 {
			recAgents = 1
		}
		return AnalysisReport{
			Verdict:           VerdictReduceConcurrency,
			PrimaryBottleneck: BottleneckComputePrefill,
			BottleneckReason:  fmt.Sprintf("MLX request queue backed up (%d queued, %d active); prefill latency is elevated", srv.QueuedRequests, srv.ActiveRequests),
			RecommendedAgents: recAgents,
			Remediation:       fmt.Sprintf("Limit active concurrent subagents to %d to clear queued requests.", recAgents),
			Confidence:        0.85,
		}
	}

	// 6. HEADROOM_OK: Nominal operation with fine-grained bottleneck detection
	primaryBottleneck := BottleneckNone
	bottleneckReason := "Unified memory, thermal, and compute metrics are within nominal limits."

	if hw.DeviceUtilizationPct >= 95.0 {
		primaryBottleneck = BottleneckComputePrefill
		bottleneckReason = fmt.Sprintf("GPU utilization is elevated at %.1f%% during prefill forward passes", hw.DeviceUtilizationPct)
	} else if hw.DeviceUtilizationPct >= 80.0 && srv.DecodeTokensPerSec > 0 && srv.DecodeTokensPerSec < 20.0 {
		primaryBottleneck = BottleneckMemoryBandwidth
		bottleneckReason = fmt.Sprintf("High GPU utilization (%.1f%%) with lower decode throughput (%.1f tok/s) indicates memory bandwidth saturation", hw.DeviceUtilizationPct, srv.DecodeTokensPerSec)
	} else if srv.DecodeTokensPerSec > 0 && srv.DecodeTokensPerSec < 10.0 {
		primaryBottleneck = BottleneckComputeDecode
		bottleneckReason = fmt.Sprintf("Decode throughput is %.1f tok/s, indicating compute-bound token generation", srv.DecodeTokensPerSec)
	}

	recAgents := requestedAgents
	if head.Available && head.MaxSharedAgents > 0 {
		if requestedAgents > head.MaxSharedAgents {
			recAgents = head.MaxSharedAgents
		} else {
			recAgents = requestedAgents
		}
	}

	confidence := 0.95
	if !hw.Available && !srv.Available {
		confidence = 0.70
	}

	return AnalysisReport{
		Verdict:           VerdictHeadroomOK,
		PrimaryBottleneck: primaryBottleneck,
		BottleneckReason:  bottleneckReason,
		RecommendedAgents: recAgents,
		Remediation:       "Headroom is nominal; system can support concurrent agent runs or burst tool loops.",
		Confidence:        confidence,
	}
}
