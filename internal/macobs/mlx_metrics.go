package macobs

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"
)

// ScrapeMLXMetrics scrapes a Prometheus metrics endpoint and parses MLX serving telemetry.
func ScrapeMLXMetrics(ctx context.Context, endpointURL string, client *http.Client) (MLXServingTelemetry, PrefixCacheTelemetry, error) {
	if strings.TrimSpace(endpointURL) == "" {
		return MLXServingTelemetry{Available: false}, PrefixCacheTelemetry{Available: false}, nil
	}
	if client == nil {
		client = &http.Client{Timeout: 3 * time.Second}
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpointURL, nil)
	if err != nil {
		return MLXServingTelemetry{Available: false}, PrefixCacheTelemetry{Available: false}, err
	}

	resp, err := client.Do(req)
	if err != nil {
		return MLXServingTelemetry{Available: false}, PrefixCacheTelemetry{Available: false}, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return MLXServingTelemetry{Available: false}, PrefixCacheTelemetry{Available: false}, fmt.Errorf("mlx metrics returned HTTP %d", resp.StatusCode)
	}

	// Limit to 4MB of metrics text to guard against unbounded payloads
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4*1024*1024))
	if err != nil {
		return MLXServingTelemetry{Available: false}, PrefixCacheTelemetry{Available: false}, err
	}

	srv, prefix := ParseMLXMetrics(string(body))
	return srv, prefix, nil
}

// ParseMLXMetrics parses Prometheus text output from vllm-mlx or mlx-lm.
func ParseMLXMetrics(body string) (MLXServingTelemetry, PrefixCacheTelemetry) {
	srv := MLXServingTelemetry{ServerType: "unknown"}
	prefix := PrefixCacheTelemetry{}

	if strings.TrimSpace(body) == "" {
		return srv, prefix
	}

	var ttftSum, ttftCount float64
	var hasTTFTSum, hasTTFTCount bool

	var itlSum, itlCount float64
	var hasITLSum, hasITLCount bool

	var prefixHits, prefixQueries, prefixMisses uint64
	var hasPrefixHits, hasPrefixQueries, hasPrefixMisses bool
	var explicitHitRate float64
	var hasExplicitHitRate bool

	scanner := bufio.NewScanner(strings.NewReader(body))
	parsedAny := false
	isVLLM := false
	isMLX := false

	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		name, labels, val, ok := parsePrometheusLine(line)
		if !ok {
			continue
		}

		if strings.HasPrefix(name, "vllm:") || strings.HasPrefix(name, "vllm_") {
			isVLLM = true
		} else if strings.HasPrefix(name, "mlx:") || strings.HasPrefix(name, "mlx_") {
			isMLX = true
		}

		parsedAny = true

		switch name {
		// Active & Queued Requests
		case "vllm:num_requests_running", "vllm_num_requests_running",
			"mlx:active_requests", "mlx_active_requests", "mlx_lm_active_requests":
			srv.ActiveRequests = int(val)

		case "vllm:num_requests_waiting", "vllm_num_requests_waiting",
			"mlx:queued_requests", "mlx_queued_requests", "mlx_lm_queued_requests":
			srv.QueuedRequests = int(val)

		// KV Cache Usage
		case "vllm:kv_cache_usage_perc", "vllm_kv_cache_usage_perc":
			srv.KVCacheUsagePct = val

		case "vllm:gpu_cache_usage_factor", "vllm_gpu_cache_usage_factor":
			srv.KVCacheUsagePct = val * 100.0

		case "mlx:kv_cache_usage_pct", "mlx_kv_cache_usage_pct":
			srv.KVCacheUsagePct = val

		case "mlx:kv_cache_usage", "mlx_kv_cache_usage":
			if val <= 1.0 && val > 0.0 {
				srv.KVCacheUsagePct = val * 100.0
			} else {
				srv.KVCacheUsagePct = val
			}

		// TTFT
		case "vllm:time_to_first_token_seconds_sum", "vllm_time_to_first_token_seconds_sum",
			"mlx:ttft_seconds_sum", "mlx_ttft_seconds_sum":
			ttftSum = val
			hasTTFTSum = true

		case "vllm:time_to_first_token_seconds_count", "vllm_time_to_first_token_seconds_count",
			"mlx:ttft_seconds_count", "mlx_ttft_seconds_count":
			ttftCount = val
			hasTTFTCount = true

		// ITL / TPOT
		case "vllm:inter_token_latency_seconds_sum", "vllm_inter_token_latency_seconds_sum",
			"vllm:time_per_output_token_seconds_sum", "vllm_time_per_output_token_seconds_sum",
			"vllm:request_time_per_output_token_seconds_sum", "vllm_request_time_per_output_token_seconds_sum",
			"mlx:itl_seconds_sum", "mlx_itl_seconds_sum":
			itlSum = val
			hasITLSum = true

		case "vllm:inter_token_latency_seconds_count", "vllm_inter_token_latency_seconds_count",
			"vllm:time_per_output_token_seconds_count", "vllm_time_per_output_token_seconds_count",
			"vllm:request_time_per_output_token_seconds_count", "vllm_request_time_per_output_token_seconds_count",
			"mlx:itl_seconds_count", "mlx_itl_seconds_count":
			itlCount = val
			hasITLCount = true

		// Throughput
		case "vllm:avg_prompt_throughput_tok_per_s", "vllm_avg_prompt_throughput_tok_per_s",
			"mlx:prompt_tokens_per_sec", "mlx_prompt_tokens_per_sec":
			srv.PromptTokensPerSec = val

		case "vllm:avg_generation_throughput_tok_per_s", "vllm_avg_generation_throughput_tok_per_s",
			"mlx:decode_tokens_per_sec", "mlx_decode_tokens_per_sec", "mlx_lm_tokens_per_second":
			srv.DecodeTokensPerSec = val

		case "mlx:tokens_per_second", "mlx_tokens_per_second":
			if labels["type"] == "prompt" {
				srv.PromptTokensPerSec = val
			} else if labels["type"] == "generation" || labels["type"] == "decode" {
				srv.DecodeTokensPerSec = val
			}

		// Prefix Cache
		case "vllm:prefix_cache_hits", "vllm_prefix_cache_hits",
			"mlx:prefix_cache_hits", "mlx_prefix_cache_hits":
			prefixHits = uint64(val)
			hasPrefixHits = true

		case "vllm:prefix_cache_queries", "vllm_prefix_cache_queries",
			"mlx:prefix_cache_queries", "mlx_prefix_cache_queries":
			prefixQueries = uint64(val)
			hasPrefixQueries = true

		case "vllm:prefix_cache_misses", "vllm_prefix_cache_misses",
			"mlx:prefix_cache_misses", "mlx_prefix_cache_misses":
			prefixMisses = uint64(val)
			hasPrefixMisses = true

		case "vllm:gpu_prefix_cache_hit_rate", "vllm_gpu_prefix_cache_hit_rate",
			"vllm:cpu_prefix_cache_hit_rate", "vllm_cpu_prefix_cache_hit_rate",
			"mlx:prefix_cache_hit_ratio", "mlx_prefix_cache_hit_ratio":
			explicitHitRate = val
			hasExplicitHitRate = true

		case "vllm:num_cached_blocks", "vllm_num_cached_blocks":
			prefix.CachedBlocks = uint64(val)

		case "vllm:num_total_blocks", "vllm_num_total_blocks":
			prefix.QueriedBlocks = uint64(val)
		}
	}

	if isVLLM {
		srv.ServerType = "vllm-mlx"
	} else if isMLX {
		srv.ServerType = "mlx-lm"
	}

	// Compute TTFT average
	if hasTTFTSum && hasTTFTCount && ttftCount > 0 {
		srv.AvgTTFTMs = (ttftSum / ttftCount) * 1000.0
	}

	// Compute ITL average
	if hasITLSum && hasITLCount && itlCount > 0 {
		srv.AvgITLMs = (itlSum / itlCount) * 1000.0
	}

	// Compute Prefix Cache stats
	if hasPrefixHits {
		prefix.Hits = prefixHits
	}
	if hasPrefixQueries {
		prefix.QueriedBlocks = prefixQueries
	}
	if hasPrefixMisses {
		prefix.Misses = prefixMisses
	}

	if hasPrefixHits && hasPrefixQueries && prefixQueries > 0 {
		prefix.HitRatio = float64(prefixHits) / float64(prefixQueries)
		prefix.Available = true
	} else if hasPrefixHits && hasPrefixMisses && (prefixHits+prefixMisses) > 0 {
		total := prefixHits + prefixMisses
		prefix.HitRatio = float64(prefixHits) / float64(total)
		prefix.QueriedBlocks = total
		prefix.Available = true
	} else if hasExplicitHitRate {
		prefix.HitRatio = explicitHitRate
		prefix.Available = true
	}

	srv.PrefixCacheHitRatio = prefix.HitRatio
	srv.Available = parsedAny

	return srv, prefix
}

func parsePrometheusLine(line string) (name string, labels map[string]string, val float64, ok bool) {
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return "", nil, 0, false
	}

	rawName := fields[0]
	valStr := fields[1]

	labels = make(map[string]string)
	if idx := strings.IndexByte(rawName, '{'); idx >= 0 {
		if end := strings.LastIndexByte(rawName, '}'); end > idx {
			labelPart := rawName[idx+1 : end]
			rawName = rawName[:idx]
			for _, item := range strings.Split(labelPart, ",") {
				if k, v, found := strings.Cut(item, "="); found {
					labels[strings.TrimSpace(k)] = strings.Trim(strings.TrimSpace(v), `"`)
				}
			}
		}
	}

	v, err := strconv.ParseFloat(valStr, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return "", nil, 0, false
	}

	return rawName, labels, v, true
}
