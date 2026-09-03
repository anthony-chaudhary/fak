package modelperfobs

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
)

type Summary struct {
	Schema                string  `json:"schema"`
	Requests              int     `json:"requests"`
	Errors                int     `json:"errors"`
	PromptTokens          int64   `json:"prompt_tokens"`
	CompletionTokens      int64   `json:"completion_tokens"`
	DurationP50MS         float64 `json:"duration_p50_ms"`
	DurationP95MS         float64 `json:"duration_p95_ms"`
	TTFTP50MS             float64 `json:"ttft_p50_ms"`
	TTFTP95MS             float64 `json:"ttft_p95_ms"`
	TPOTP50MS             float64 `json:"tpot_p50_ms"`
	TPOTP95MS             float64 `json:"tpot_p95_ms"`
	OutputTokensPerSecP50 float64 `json:"output_tokens_per_second_p50"`
	OutputTokensPerSecP95 float64 `json:"output_tokens_per_second_p95"`
	LikelyBottleneck      string  `json:"likely_bottleneck"`
	NextCheck             string  `json:"next_check"`
}

func ReadObservations(r io.Reader) ([]Observation, error) {
	var rows []Observation
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 64*1024), 16*1024*1024)
	for s.Scan() {
		var row Observation
		if err := json.Unmarshal(s.Bytes(), &row); err != nil {
			return nil, fmt.Errorf("decode observation %d: %w", len(rows)+1, err)
		}
		if row.Schema != Schema {
			return nil, fmt.Errorf("observation %d: schema %q, want %q", len(rows)+1, row.Schema, Schema)
		}
		rows = append(rows, row)
	}
	return rows, s.Err()
}

func Summarize(rows []Observation) Summary {
	s := Summary{Schema: "fak-model-perf-summary/1", Requests: len(rows)}
	var durations, ttft, tpot, rate []float64
	for _, r := range rows {
		s.PromptTokens += r.PromptTokens
		s.CompletionTokens += r.CompletionTokens
		if r.Error != "" || r.Status >= 400 {
			s.Errors++
		}
		if r.DurationMS > 0 {
			durations = append(durations, r.DurationMS)
		}
		if r.TTFTMS > 0 {
			ttft = append(ttft, r.TTFTMS)
		}
		if r.TPOTMS > 0 {
			tpot = append(tpot, r.TPOTMS)
		}
		if r.OutputTokensPerSec > 0 {
			rate = append(rate, r.OutputTokensPerSec)
		}
	}
	s.DurationP50MS, s.DurationP95MS = percentile(durations, .5), percentile(durations, .95)
	s.TTFTP50MS, s.TTFTP95MS = percentile(ttft, .5), percentile(ttft, .95)
	s.TPOTP50MS, s.TPOTP95MS = percentile(tpot, .5), percentile(tpot, .95)
	s.OutputTokensPerSecP50, s.OutputTokensPerSecP95 = percentile(rate, .5), percentile(rate, .95)
	s.LikelyBottleneck, s.NextCheck = diagnose(s, len(ttft), len(tpot))
	return s
}

func diagnose(s Summary, ttftN, tpotN int) (string, string) {
	if s.Requests == 0 {
		return "no-data", "route an OpenAI-compatible request through the proxy"
	}
	if s.Errors > 0 {
		return "reliability", "group failing observations by status and error before tuning throughput"
	}
	if ttftN == 0 || tpotN == 0 {
		return "missing-stream-timing", "enable streaming and stream_options.include_usage to separate prefill from decode"
	}
	if s.TTFTP50MS > 2*s.TPOTP50MS && s.TTFTP50MS > 500 {
		return "prefill-or-queue", "hold output length constant; sweep prompt length and concurrency to separate prefill from queueing"
	}
	if s.TPOTP50MS > 100 {
		return "decode", "profile device residency, memory bandwidth, batching, and quantized kernels"
	}
	return "workload-orchestration", "join request IDs to agent outcomes and compare useful tokens and wall time per completed task"
}

func WriteMarkdown(w io.Writer, s Summary) error {
	_, err := fmt.Fprintf(w, "# Model performance observation report\n\n- Requests: **%d** (%d errors)\n- Tokens: **%d prompt / %d completion**\n- End-to-end latency p50/p95: **%.1f / %.1f ms**\n- TTFT p50/p95: **%.1f / %.1f ms**\n- TPOT p50/p95: **%.1f / %.1f ms**\n- Output rate p50/p95: **%.2f / %.2f tok/s**\n- Likely bottleneck: **%s**\n- Next check: %s\n", s.Requests, s.Errors, s.PromptTokens, s.CompletionTokens, s.DurationP50MS, s.DurationP95MS, s.TTFTP50MS, s.TTFTP95MS, s.TPOTP50MS, s.TPOTP95MS, s.OutputTokensPerSecP50, s.OutputTokensPerSecP95, s.LikelyBottleneck, s.NextCheck)
	return err
}
