// loadgen drives an OpenAI-compatible /v1/chat/completions endpoint at a sweep of
// concurrency levels and reports throughput. The completion response carries token
// counts (usage.completion_tokens) but no timing, so wall-clock is measured here.
//
// This is the pure-Go load matrix for the GPU server benchmark: no Python, no third-party
// deps. It is endpoint-agnostic — it works against fak's own in-kernel engine
// (fak serve --gguf ...), the fak gateway fronting an upstream, or a raw SGLang/vLLM
// server — anything that speaks the OpenAI chat wire.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"sync"
	"time"
)

// chatRequest is the minimal OpenAI chat-completions request.
type chatRequest struct {
	Model       string        `json:"model"`
	Messages    []chatMessage `json:"messages"`
	MaxTokens   int           `json:"max_tokens,omitempty"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Choices []struct {
		Message chatMessage `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// reqResult is one request's outcome.
type reqResult struct {
	latency          time.Duration
	completionTokens int
	err              error
}

// clientInFlight tracks client-side HTTP operations. It does not claim that
// the server admitted the same number of requests into GPU execution.
type clientInFlight struct {
	mu      sync.Mutex
	current int
	peak    int
}

func (f *clientInFlight) begin() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.current++
	if f.current > f.peak {
		f.peak = f.current
	}
}

func (f *clientInFlight) end() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.current--
}

func (f *clientInFlight) peakValue() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.peak
}

// PointResult aggregates one concurrency level.
type PointResult struct {
	Concurrency      int     `json:"concurrency"`
	PeakInFlight     int     `json:"peak_in_flight"` // observed concurrent client HTTP operations, not GPU batch execution
	Requests         int     `json:"requests"`
	Errors           int     `json:"errors"`
	ErrorRate        float64 `json:"error_rate"`
	WallSeconds      float64 `json:"wall_seconds"`
	CompletionTokens int     `json:"completion_tokens"`
	TokensPerSecond  float64 `json:"tokens_per_second"` // aggregate decode throughput
	P50LatencyMS     float64 `json:"p50_latency_ms"`
	P95LatencyMS     float64 `json:"p95_latency_ms"`
	MeanLatencyMS    float64 `json:"mean_latency_ms"`
}

// MatrixResult is the full sweep, shaped to be a drop-in MATRIX.json.
type MatrixResult struct {
	Endpoint     string        `json:"endpoint"`
	Model        string        `json:"model"`
	Stack        string        `json:"stack,omitempty"`
	StartedAtUTC string        `json:"started_at_utc"`
	Canceled     bool          `json:"canceled,omitempty"`
	Points       []PointResult `json:"points"`
}

// Config controls a sweep.
type Config struct {
	URL           string // full /v1/chat/completions URL
	Model         string
	Stack         string
	Prompt        string
	MaxTokens     int
	Concurrencies []int
	RequestsPer   int // requests per concurrency level
	APIKey        string
	HTTPClient    *http.Client
	Now           func() time.Time
}

var (
	errNonPositiveConcurrency   = errors.New("loadgen: concurrency must be positive")
	errRequestsBelowConcurrency = errors.New("loadgen: requests-per must be at least maximum concurrency")
)

// validateConfig rejects sweep shapes that cannot actually put every requested
// client operation in flight. It deliberately preserves point order and duplicates.
func validateConfig(cfg *Config) error {
	maxConcurrency := 0
	for _, concurrency := range cfg.Concurrencies {
		if concurrency <= 0 {
			return fmt.Errorf("%w: got %d", errNonPositiveConcurrency, concurrency)
		}
		if concurrency > maxConcurrency {
			maxConcurrency = concurrency
		}
	}
	if cfg.RequestsPer < maxConcurrency {
		return fmt.Errorf("%w: requests-per=%d maximum-concurrency=%d", errRequestsBelowConcurrency, cfg.RequestsPer, maxConcurrency)
	}
	return nil
}

func (c *Config) client() *http.Client {
	if c.HTTPClient != nil {
		return c.HTTPClient
	}
	return &http.Client{Timeout: 120 * time.Second}
}

func (c *Config) now() time.Time {
	if c.Now != nil {
		return c.Now()
	}
	return time.Now()
}

// oneRequest issues a single chat completion and measures wall-clock + tokens.
func oneRequest(ctx context.Context, cfg *Config, inFlight *clientInFlight) reqResult {
	body, _ := json.Marshal(chatRequest{
		Model:       cfg.Model,
		Messages:    []chatMessage{{Role: "user", Content: cfg.Prompt}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: 0,
		Stream:      false,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, cfg.URL, bytes.NewReader(body))
	if err != nil {
		return reqResult{err: err}
	}
	req.Header.Set("Content-Type", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}
	start := cfg.now()
	inFlight.begin()
	resp, err := cfg.client().Do(req)
	if err != nil {
		inFlight.end()
		return reqResult{err: err, latency: cfg.now().Sub(start)}
	}
	defer func() {
		_ = resp.Body.Close()
		inFlight.end()
	}()
	data, readErr := io.ReadAll(resp.Body)
	lat := cfg.now().Sub(start)
	if readErr != nil {
		return reqResult{err: fmt.Errorf("read response body: %w", readErr), latency: lat}
	}
	if resp.StatusCode >= 300 {
		return reqResult{err: fmt.Errorf("status %d: %.120s", resp.StatusCode, string(data)), latency: lat}
	}
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return reqResult{err: fmt.Errorf("decode: %w", err), latency: lat}
	}
	return reqResult{latency: lat, completionTokens: cr.Usage.CompletionTokens}
}

// runPoint runs `requests` total requests at the given concurrency.
func runPoint(ctx context.Context, cfg *Config, concurrency, requests int) PointResult {
	results := make([]reqResult, requests)
	jobs := make(chan int)
	inFlight := &clientInFlight{}
	var wg sync.WaitGroup
	wallStart := cfg.now()
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = oneRequest(ctx, cfg, inFlight)
			}
		}()
	}
	for i := 0; i < requests; i++ {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	wall := cfg.now().Sub(wallStart)
	result := aggregate(concurrency, requests, wall, results)
	result.PeakInFlight = inFlight.peakValue()
	return result
}

// aggregate folds raw per-request results into a PointResult.
func aggregate(concurrency, requests int, wall time.Duration, results []reqResult) PointResult {
	pr := PointResult{Concurrency: concurrency, Requests: requests, WallSeconds: wall.Seconds()}
	var lats []float64
	var latSum float64
	for _, r := range results {
		if r.err != nil {
			pr.Errors++
			continue
		}
		pr.CompletionTokens += r.completionTokens
		ms := float64(r.latency.Microseconds()) / 1000.0
		lats = append(lats, ms)
		latSum += ms
	}
	if requests > 0 {
		pr.ErrorRate = float64(pr.Errors) / float64(requests)
	}
	if wall.Seconds() > 0 {
		pr.TokensPerSecond = float64(pr.CompletionTokens) / wall.Seconds()
	}
	if n := len(lats); n > 0 {
		sort.Float64s(lats)
		pr.MeanLatencyMS = latSum / float64(n)
		pr.P50LatencyMS = percentile(lats, 50)
		pr.P95LatencyMS = percentile(lats, 95)
	}
	return pr
}

// percentile returns the p-th percentile (0-100) of a sorted slice via
// nearest-rank, matching common benchmark reporting.
func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if len(sorted) == 1 {
		return sorted[0]
	}
	rank := int(math.Ceil(p/100.0*float64(len(sorted)))) - 1
	if rank < 0 {
		rank = 0
	}
	if rank >= len(sorted) {
		rank = len(sorted) - 1
	}
	return sorted[rank]
}

// Run executes the full concurrency sweep.
func Run(ctx context.Context, cfg *Config) MatrixResult {
	mr := MatrixResult{
		Endpoint: cfg.URL, Model: cfg.Model, Stack: cfg.Stack,
		StartedAtUTC: cfg.now().UTC().Format(time.RFC3339),
	}
	for _, c := range cfg.Concurrencies {
		if ctx.Err() != nil {
			mr.Canceled = true
			break
		}
		mr.Points = append(mr.Points, runPoint(ctx, cfg, c, cfg.RequestsPer))
		if ctx.Err() != nil {
			mr.Canceled = true
			break
		}
	}
	return mr
}
