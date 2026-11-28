package modelengine

import (
	"fmt"
	"strconv"
	"strings"
)

// KVUtilizationSnapshot is the Triton-shaped per-response KV-cache utilization
// summary (used/max/tokens_per_block, derived utilization, and token capacity).
type KVUtilizationSnapshot struct {
	UsedBlocks       int     `json:"used_blocks"`
	MaxBlocks        int     `json:"max_blocks"`
	TokensPerBlock   int     `json:"tokens_per_block"`
	KVUtilization    float64 `json:"kv_cache_utilization"`
	MaxTokenCapacity int     `json:"max_token_capacity"`
}

// DeriveKVUtilization computes kv_cache_utilization = used/max (clamped to [0,1])
// and max_token_capacity = max * tokensPerBlock.
func DeriveKVUtilization(used, max, tokensPerBlock int) KVUtilizationSnapshot {
	var util float64
	if max > 0 && used > 0 {
		util = float64(used) / float64(max)
		if util > 1 {
			util = 1
		}
	}
	var capTokens int
	if max > 0 && tokensPerBlock > 0 {
		capTokens = max * tokensPerBlock
	}
	return KVUtilizationSnapshot{
		UsedBlocks:       used,
		MaxBlocks:        max,
		TokensPerBlock:   tokensPerBlock,
		KVUtilization:    util,
		MaxTokenCapacity: capTokens,
	}
}

// ParseTritonKVBlockMetrics parses nv_trt_llm_kv_cache_block_metrics lines
// (used, max, tokens_per_block) and derives KVUtilization + MaxTokenCapacity.
func ParseTritonKVBlockMetrics(promText string) (KVUtilizationSnapshot, bool) {
	var used, maxB, tpb int
	var gotUsed, gotMax bool
	for _, line := range strings.Split(promText, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") || !strings.HasPrefix(line, "nv_trt_llm_kv_cache_block_metrics") {
			continue
		}
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		v, err := strconv.Atoi(fields[len(fields)-1])
		if err != nil {
			continue
		}
		switch {
		case strings.Contains(line, `kv_cache_block_type="used"`):
			used, gotUsed = v, true
		case strings.Contains(line, `kv_cache_block_type="max"`):
			maxB, gotMax = v, true
		case strings.Contains(line, `kv_cache_block_type="tokens_per_block"`):
			tpb = v
		}
	}
	if !gotUsed || !gotMax {
		return KVUtilizationSnapshot{}, false
	}
	return DeriveKVUtilization(used, maxB, tpb), true
}

// KVPreemptionMetricWriter exposes the captured native scheduler's serving metrics.
type KVPreemptionMetricWriter interface {
	WriteKVPreemptionMetrics(*strings.Builder)
	KVUtilization() KVUtilizationSnapshot
}

type nativePreemptionMetricWriter struct {
	scheduler *NativeScheduler
}

// KVPreemptionMetricWriter initializes and captures this engine's scheduler at startup.
// Reading the returned writer never initializes a model or a scheduler.
func (e *Engine) KVPreemptionMetricWriter() KVPreemptionMetricWriter {
	if e == nil {
		return nil
	}
	return nativePreemptionMetricWriter{scheduler: e.nativeScheduler()}
}

func (w nativePreemptionMetricWriter) KVUtilization() KVUtilizationSnapshot {
	if w.scheduler == nil {
		return KVUtilizationSnapshot{}
	}
	stats := w.scheduler.KVPreemptionStats()
	return DeriveKVUtilization(stats.UsedBlocks, stats.MaxBlocks, 0)
}

func (w nativePreemptionMetricWriter) WriteKVPreemptionMetrics(b *strings.Builder) {
	if w.scheduler == nil || b == nil {
		return
	}
	stats := w.scheduler.KVPreemptionStats()
	if stats.MaxBlocks <= 0 {
		return
	}
	writeNativePreemptionMetrics(b, stats)
	u := DeriveKVUtilization(stats.UsedBlocks, stats.MaxBlocks, 0)
	fmt.Fprintf(b, "fak_sched_preempt_kv_utilization %.6f\n", u.KVUtilization)
	fmt.Fprintf(b, "fak_sched_preempt_max_token_capacity %d\n", u.MaxTokenCapacity)
}
