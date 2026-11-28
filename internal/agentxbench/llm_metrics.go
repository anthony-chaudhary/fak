// Structural adaptation of triton-inference-server/perf_analyzer GenAI-Perf
// LLM metric definitions (genai-perf/genai_perf/metrics/llm_metrics.py:37 @ 24f6b9af4db5, BSD-3-Clause).
package agentxbench

import (
	"errors"
	"math"
)

var ErrInvalidLLMTrace = errors.New("agentxbench: invalid LLM request trace")

// LLMRequestTrace captures nanosecond token timestamps for one served request.
type LLMRequestTrace struct {
	TokenTimestampsNS []int64 `json:"token_timestamps_ns"`
	RequestStartNS    int64   `json:"request_start_ns"`
	OutputTokens      int     `json:"output_tokens"`
}

// LLMRequestMetrics holds per-request latency and user-perceived throughput metrics.
type LLMRequestMetrics struct {
	TTFTMS                 float64 `json:"ttft_ms"`
	TTSTMS                 float64 `json:"ttst_ms"`
	ITLMS                  float64 `json:"itl_ms"`
	OutputTokPerSecPerUser float64 `json:"output_tok_per_sec_per_user"`
}

// ComputeLLMRequestMetrics computes TTFT (ms), TTST (ms, time to second token),
// mean ITL (ms), and per-user output tok/s from one request's token timestamps.
func ComputeLLMRequestMetrics(tr LLMRequestTrace) (LLMRequestMetrics, error) {
	if len(tr.TokenTimestampsNS) == 0 {
		return LLMRequestMetrics{}, ErrInvalidLLMTrace
	}
	first := tr.TokenTimestampsNS[0]
	if first < tr.RequestStartNS {
		return LLMRequestMetrics{}, ErrInvalidLLMTrace
	}
	ttftMS := float64(first-tr.RequestStartNS) / 1e6

	var ttstMS, itlMS float64
	if len(tr.TokenTimestampsNS) >= 2 {
		second := tr.TokenTimestampsNS[1]
		if second < first {
			return LLMRequestMetrics{}, ErrInvalidLLMTrace
		}
		ttstMS = float64(second-first) / 1e6
		var itlSum int64
		for i := 1; i < len(tr.TokenTimestampsNS); i++ {
			d := tr.TokenTimestampsNS[i] - tr.TokenTimestampsNS[i-1]
			if d < 0 {
				return LLMRequestMetrics{}, ErrInvalidLLMTrace
			}
			itlSum += d
		}
		itlMS = float64(itlSum) / float64(len(tr.TokenTimestampsNS)-1) / 1e6
	}

	outTokens := tr.OutputTokens
	if outTokens <= 0 {
		outTokens = len(tr.TokenTimestampsNS)
	}
	last := tr.TokenTimestampsNS[len(tr.TokenTimestampsNS)-1]
	durSec := float64(last-tr.RequestStartNS) / 1e9
	tokPerSec := 0.0
	if durSec > 0 {
		tokPerSec = float64(outTokens) / durSec
	}

	return LLMRequestMetrics{
		TTFTMS:                 ttftMS,
		TTSTMS:                 ttstMS,
		ITLMS:                  itlMS,
		OutputTokPerSecPerUser: tokPerSec,
	}, nil
}

// LLMSystemMetrics holds aggregate system output token throughput.
type LLMSystemMetrics struct {
	TotalOutputTokens int     `json:"total_output_tokens"`
	SystemTokPerSec   float64 `json:"system_tok_per_sec"`
}

// ComputeLLMSystemMetrics computes system-wide output tokens/sec over wallDurationNS.
func ComputeLLMSystemMetrics(traces []LLMRequestTrace, wallDurationNS int64) (LLMSystemMetrics, error) {
	if wallDurationNS <= 0 || math.IsNaN(float64(wallDurationNS)) {
		return LLMSystemMetrics{}, ErrInvalidLLMTrace
	}
	total := 0
	for _, tr := range traces {
		n := tr.OutputTokens
		if n <= 0 {
			n = len(tr.TokenTimestampsNS)
		}
		total += n
	}
	return LLMSystemMetrics{
		TotalOutputTokens: total,
		SystemTokPerSec:   float64(total) / (float64(wallDurationNS) / 1e9),
	}, nil
}
