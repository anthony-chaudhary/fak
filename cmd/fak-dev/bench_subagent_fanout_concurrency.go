package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"
)

// --- Deterministic concurrency-scaling cell (Issue #1591) ---
//
// This cell issues N same-prefix requests against a loopback httptest server
// with a controllable per-request delay, first sequentially then concurrently,
// and reports the aggregate wall time, the sum-of-request time, and the
// concurrency-scaling ratio. It is a deterministic test-only witness: a
// serializing fake yields a ratio near 1.0 (no scaling), while a genuinely
// concurrent fake yields a ratio near N. It does NOT read server cache
// counters (that is #13029's scope) and does not touch the live HTTP path.

// ConcurrencyScalingCellSchema identifies the deterministic concurrency cell.
const ConcurrencyScalingCellSchema = "fak.benchmark.subagent_fanout_concurrency_scaling/v1"

// ConcurrencyScalingConfig parameterizes the deterministic concurrency cell.
type ConcurrencyScalingConfig struct {
	// FanoutN is the number of same-prefix requests per phase.
	FanoutN int
	// PerRequestDelay is the server-side delay applied to each request so the
	// wall-clock difference between sequential and concurrent execution is
	// measurable and deterministic.
	PerRequestDelay time.Duration
	// PrefixTokens/SuffixTokens/DecodeTokens preserve the frozen trace geometry.
	PrefixTokens int
	SuffixTokens int
	DecodeTokens int
}

// ConcurrencyScalingResult is the machine-readable receipt for the cell.
type ConcurrencyScalingResult struct {
	Schema             string  `json:"schema"`
	FanoutN            int     `json:"fanout_n"`
	PerRequestDelayMs  float64 `json:"per_request_delay_ms"`
	SequentialWallMs   float64 `json:"sequential_wall_ms"`
	ConcurrentWallMs   float64 `json:"concurrent_wall_ms"`
	SumOfRequestTimeMs float64 `json:"sum_of_request_time_ms"`
	ScalingRatio       float64 `json:"scaling_ratio"`
	RequestsIssued     int     `json:"requests_issued"`
}

// RunConcurrencyScalingCell measures the aggregate concurrency-scaling ratio
// against endpoint, issuing FanoutN same-prefix completion requests first
// sequentially and then concurrently. The sequential wall time witnesses the
// serial baseline (N * per-request cost); the concurrent wall time witnesses
// whether the server actually parallelizes. ScalingRatio is
// sequentialWall / concurrentWall, so ~1.0 means serializing and ~N means
// fully concurrent.
func RunConcurrencyScalingCell(ctx context.Context, endpoint string, cfg ConcurrencyScalingConfig) (ConcurrencyScalingResult, error) {
	if cfg.FanoutN <= 0 {
		return ConcurrencyScalingResult{}, fmt.Errorf("concurrency cell requires FanoutN > 0, got %d", cfg.FanoutN)
	}
	if cfg.PerRequestDelay < 0 {
		return ConcurrencyScalingResult{}, fmt.Errorf("concurrency cell requires non-negative PerRequestDelay, got %s", cfg.PerRequestDelay)
	}

	client := &http.Client{Timeout: 120 * time.Second}
	prefixTokens := cfg.PrefixTokens
	if prefixTokens < 1 {
		prefixTokens = 1
	}
	decodeTokens := cfg.DecodeTokens
	if decodeTokens < 1 {
		decodeTokens = 1
	}
	body, err := json.Marshal(map[string]any{
		"model":       DefaultModel,
		"prompt":      strings.Repeat("system instruction root coordinator ", prefixTokens/8) + "subagent task",
		"max_tokens":  decodeTokens,
		"stream":      false,
		"temperature": 0.0,
	})
	if err != nil {
		return ConcurrencyScalingResult{}, fmt.Errorf("marshal request body: %w", err)
	}

	issue := func() (time.Duration, error) {
		req, err := http.NewRequestWithContext(ctx, "POST", endpoint+"/v1/completions", bytes.NewReader(body))
		if err != nil {
			return 0, err
		}
		req.Header.Set("Content-Type", "application/json")
		start := time.Now()
		resp, err := client.Do(req)
		if err != nil {
			return 0, err
		}
		defer resp.Body.Close()
		_, _ = io.Copy(io.Discard, resp.Body)
		if resp.StatusCode != http.StatusOK {
			return 0, fmt.Errorf("HTTP status %d", resp.StatusCode)
		}
		return time.Since(start), nil
	}

	// Phase 1: sequential.
	var sumRequestTime time.Duration
	seqStart := time.Now()
	for i := 0; i < cfg.FanoutN; i++ {
		d, err := issue()
		if err != nil {
			return ConcurrencyScalingResult{}, fmt.Errorf("sequential request %d: %w", i, err)
		}
		sumRequestTime += d
	}
	sequentialWall := time.Since(seqStart)

	// Phase 2: concurrent.
	concStart := time.Now()
	errCh := make(chan error, cfg.FanoutN)
	var wg sync.WaitGroup
	for i := 0; i < cfg.FanoutN; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := issue(); err != nil {
				errCh <- err
			}
		}()
	}
	wg.Wait()
	close(errCh)
	for err := range errCh {
		if err != nil {
			return ConcurrencyScalingResult{}, fmt.Errorf("concurrent request: %w", err)
		}
	}
	concurrentWall := time.Since(concStart)

	ratio := 0.0
	if concurrentWall > 0 {
		ratio = sequentialWall.Seconds() / concurrentWall.Seconds()
	}

	return ConcurrencyScalingResult{
		Schema:             ConcurrencyScalingCellSchema,
		FanoutN:            cfg.FanoutN,
		PerRequestDelayMs:  float64(cfg.PerRequestDelay.Microseconds()) / 1000.0,
		SequentialWallMs:   float64(sequentialWall.Microseconds()) / 1000.0,
		ConcurrentWallMs:   float64(concurrentWall.Microseconds()) / 1000.0,
		SumOfRequestTimeMs: float64(sumRequestTime.Microseconds()) / 1000.0,
		ScalingRatio:       ratio,
		RequestsIssued:     cfg.FanoutN * 2,
	}, nil
}
