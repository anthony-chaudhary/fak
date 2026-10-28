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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strings"
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
	streamed         bool            // completed via an SSE stream (vs plain-JSON fallback)
	ttft             time.Duration   // start to first non-empty content chunk; 0 if none arrived
	itls             []time.Duration // inter-content-chunk arrival deltas (decode-phase ITL samples)
}

// sseChunk is one `data:` payload of an OpenAI-compatible streaming
// chat-completions response. Only the fields loadgen measures are decoded.
type sseChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
	Usage struct {
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// sseMeasure accumulates per-request stream timings while consuming SSE lines.
type sseMeasure struct {
	firstContent time.Time
	lastContent  time.Time
	itls         []time.Duration
	nContent     int
	usageTokens  int
	sawData      bool
	streamErr    error
	done         bool
}

// feedSSELine folds one SSE line into m at arrival time now. Non-data lines
// (comments, event:, id:, blanks) and malformed data payloads are skipped, never
// fatal; a `data: [DONE]` terminal marks the stream complete.
func (m *sseMeasure) feedSSELine(line string, now time.Time) {
	payload := strings.TrimRight(line, "\r\n")
	if !strings.HasPrefix(payload, "data:") {
		return
	}
	payload = strings.TrimSpace(payload[len("data:"):])
	if payload == "" {
		return
	}
	m.sawData = true
	if payload == "[DONE]" {
		m.done = true
		return
	}
	var chunk sseChunk
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		return // malformed SSE payload: skip, never panic
	}
	if chunk.Error != nil {
		m.streamErr = fmt.Errorf("stream error: %s", chunk.Error.Message)
		return
	}
	if chunk.Usage.CompletionTokens > 0 {
		m.usageTokens = chunk.Usage.CompletionTokens
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content == "" {
			continue
		}
		if m.firstContent.IsZero() {
			m.firstContent = now
		} else {
			m.itls = append(m.itls, now.Sub(m.lastContent))
		}
		m.lastContent = now
		m.nContent++
	}
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
	// TokensPerSecond is end-to-end output throughput (completion tokens / sweep
	// wall; includes per-request prefill/TTFT). The phase-pure decode rate lives in
	// DecodeTokensPerSecond.
	TokensPerSecond float64 `json:"tokens_per_second"`
	P50LatencyMS    float64 `json:"p50_latency_ms"`
	P95LatencyMS    float64 `json:"p95_latency_ms"`
	MeanLatencyMS   float64 `json:"mean_latency_ms"`
	// Phase-separated decode metrics, populated only when requests completed via an
	// SSE stream (Streamed); all zero when the endpoint answered non-streaming.
	Streamed              bool    `json:"streamed,omitempty"`
	TTFTP50MS             float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP95MS             float64 `json:"ttft_p95_ms,omitempty"`
	TPOTMS                float64 `json:"tpot_ms,omitempty"`                  // mean inter-token latency across the point's successful streamed requests
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second,omitempty"` // 1000/TPOTMS when TPOTMS>0, else 0
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

// oneRequest issues a single chat completion as a stream (stream:true) and measures
// wall-clock, time-to-first-token, and inter-content-chunk arrivals. If the endpoint
// answers with a plain JSON body instead of an SSE stream, it falls back to the
// non-streaming shape (streamed=false, no phase timing).
func oneRequest(ctx context.Context, cfg *Config, inFlight *clientInFlight) reqResult {
	body, _ := json.Marshal(chatRequest{
		Model:       cfg.Model,
		Messages:    []chatMessage{{Role: "user", Content: cfg.Prompt}},
		MaxTokens:   cfg.MaxTokens,
		Temperature: 0,
		Stream:      true,
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
	if resp.StatusCode >= 300 {
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return reqResult{err: fmt.Errorf("status %d: %.120s", resp.StatusCode, string(data)), latency: cfg.now().Sub(start)}
	}
	br := bufio.NewReaderSize(resp.Body, 32*1024)
	firstLine, firstErr := br.ReadString('\n')
	ct := strings.ToLower(resp.Header.Get("Content-Type"))
	isSSE := strings.Contains(ct, "text/event-stream")
	isJSON := strings.Contains(ct, "application/json")
	// Classify by content type when declared; otherwise sniff the first line — an
	// SSE stream opens with `data:` lines, a plain-JSON answer does not.
	if isSSE || (!isJSON && strings.HasPrefix(strings.TrimLeft(firstLine, " \t\r"), "data:")) {
		return measureStream(cfg, br, firstLine, start)
	}
	return jsonFallback(cfg, br, firstLine, firstErr, start)
}

// emptyStreamBody is the closed sentinel for a 200 response whose stream carried no
// data lines at all.
var emptyStreamBody = errors.New("empty stream body")

// measureStream consumes an SSE chat-completions body, sampling the arrival time of
// each content chunk. A clean EOF (with or without [DONE]) completes the request;
// malformed lines are skipped and never panic. Completion count prefers
// usage.completion_tokens from the final chunk, falling back to the observed content
// chunk count when the server omits usage.
func measureStream(cfg *Config, br *bufio.Reader, firstLine string, start time.Time) reqResult {
	var m sseMeasure
	if firstLine != "" {
		m.feedSSELine(firstLine, cfg.now())
	}
	for !m.done {
		line, err := br.ReadString('\n')
		if line != "" {
			m.feedSSELine(line, cfg.now())
		}
		if err != nil {
			break // io.EOF ends the stream; transport errors surface below
		}
	}
	lat := cfg.now().Sub(start)
	if m.streamErr != nil {
		return reqResult{err: m.streamErr, latency: lat, streamed: true}
	}
	if !m.sawData {
		return reqResult{err: emptyStreamBody, latency: lat, streamed: true}
	}
	tokens := m.usageTokens
	if tokens == 0 {
		tokens = m.nContent
	}
	rr := reqResult{latency: lat, completionTokens: tokens, streamed: true}
	if !m.firstContent.IsZero() {
		rr.ttft = m.firstContent.Sub(start)
	}
	rr.itls = m.itls
	return rr
}

// jsonFallback parses a plain (non-SSE) chat-completions response — the shape returned
// by endpoints that ignore stream:true. It preserves today's non-streaming behavior:
// wall-clock latency plus usage.completion_tokens, no phase timing.
func jsonFallback(cfg *Config, br *bufio.Reader, firstLine string, firstErr error, start time.Time) reqResult {
	rest, readErr := io.ReadAll(br)
	data := append([]byte(firstLine), rest...)
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return reqResult{err: fmt.Errorf("read response body: %w", readErr), latency: cfg.now().Sub(start)}
	}
	if firstErr != nil && !errors.Is(firstErr, io.EOF) {
		return reqResult{err: fmt.Errorf("read response body: %w", firstErr), latency: cfg.now().Sub(start)}
	}
	if len(bytes.TrimSpace(data)) == 0 {
		return reqResult{err: errors.New("empty response body"), latency: cfg.now().Sub(start)}
	}
	var cr chatResponse
	if err := json.Unmarshal(data, &cr); err != nil {
		return reqResult{err: fmt.Errorf("decode: %w", err), latency: cfg.now().Sub(start)}
	}
	return reqResult{latency: cfg.now().Sub(start), completionTokens: cr.Usage.CompletionTokens}
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
	var ttfts []float64
	var itlSum float64
	var itlN int
	for _, r := range results {
		if r.err != nil {
			pr.Errors++
			continue
		}
		pr.CompletionTokens += r.completionTokens
		// Nanoseconds()/1e6 keeps sub-microsecond loopback timings nonzero; Microseconds() truncates them to 0.
		ms := float64(r.latency.Nanoseconds()) / 1e6
		lats = append(lats, ms)
		latSum += ms
		if !r.streamed {
			continue
		}
		pr.Streamed = true
		if r.ttft > 0 {
			ttfts = append(ttfts, float64(r.ttft.Nanoseconds())/1e6)
		}
		for _, d := range r.itls {
			itlSum += float64(d.Nanoseconds()) / 1e6
			itlN++
		}
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
	if n := len(ttfts); n > 0 {
		sort.Float64s(ttfts)
		pr.TTFTP50MS = percentile(ttfts, 50)
		pr.TTFTP95MS = percentile(ttfts, 95)
	}
	if itlN > 0 {
		pr.TPOTMS = itlSum / float64(itlN)
		if pr.TPOTMS > 0 {
			pr.DecodeTokensPerSecond = 1000 / pr.TPOTMS
		}
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
