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
	latency           time.Duration
	completionTokens  int
	tokenUsageMissing bool
	contentChunks     int
	err               error
	streamed          bool            // completed via an SSE stream (vs plain-JSON fallback)
	ttft              time.Duration   // start to first non-empty content chunk; 0 if none arrived
	chunkIntervals    []time.Duration // client inter-content-chunk arrival deltas; not token timings
	firstChunkTokens  *int            // usage.completion_tokens reported on the first content chunk (per-chunk usage); nil when not reported
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
		CompletionTokens *int `json:"completion_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// sseMeasure accumulates per-request stream timings while consuming SSE lines.
type sseMeasure struct {
	firstContent   time.Time
	lastContent    time.Time
	chunkIntervals []time.Duration
	nContent       int
	usageTokens    int
	usageKnown     bool
	// firstChunkTokens is usage.completion_tokens reported on the chunk that set
	// firstContent (per-chunk usage); nil when the first content chunk carried no
	// usage or no content chunk arrived.
	firstChunkTokens *int
	sawData          bool
	streamErr        error
	done             bool
	// onFirstContent fires exactly once, when the first content chunk lands —
	// the TTFT point where a two-tier caller releases its prefill slot.
	onFirstContent func()
}

// feedSSEMessage folds one complete SSE event body (possibly several physical
// lines) into m at arrival time now, keeping the per-line contract of
// feedSSELine for each line: non-data lines and malformed payloads are skipped,
// never fatal; a `data: [DONE]` terminal marks the stream complete. All lines
// of one event share the event's chunk-arrival stamp.
func (m *sseMeasure) feedSSEMessage(body string, now time.Time) {
	for _, line := range strings.Split(body, "\n") {
		m.feedSSELine(line, now)
	}
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
	if n := chunk.Usage.CompletionTokens; n != nil && *n >= 0 {
		m.usageTokens = *n
		m.usageKnown = true
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content == "" {
			continue
		}
		if m.firstContent.IsZero() {
			m.firstContent = now
			// Per-chunk usage on the first content chunk feeds the aiperf-style ITL
			// divisor; explicit 0 is captured too (DecodeTokenDivisor's non-positive
			// rule handles it).
			if n := chunk.Usage.CompletionTokens; n != nil && *n >= 0 {
				m.firstChunkTokens = n
			}
			if m.onFirstContent != nil {
				m.onFirstContent()
			}
		} else {
			m.chunkIntervals = append(m.chunkIntervals, now.Sub(m.lastContent))
		}
		m.lastContent = now
		m.nContent++
		break // count one content-bearing payload, not one arrival per choice
	}
}

// clientInFlight tracks client-side HTTP operations in two tiers, adapting
// aiperf's two-tier credit model (ai-dynamo/aiperf, src/aiperf/credit/issuer.py:311
// @ ab610add4e8876fca223197596c728e17365f381, Apache-2.0):
//
//   - session tier: one slot per in-flight request, held from request start until
//     the response body closes. Its cap is the sweep's concurrency level.
//   - prefill tier: one slot per request still awaiting its first content chunk
//     (TTFT). It is released at first content while the session slot stays held,
//     so later chunks keep flowing under the session cap. Its cap is
//     Config.PrefillConcurrency; when unset it equals the session cap, so the
//     tier binds only when the operator dials it below the concurrency.
//
// The zero value disables the prefill tier (legacy single-tier behavior), which
// keeps direct oneRequest callers on the old contract. It does not claim that
// the server admitted the same number of requests into GPU execution.
type clientInFlight struct {
	mu          sync.Mutex
	current     int
	peak        int
	prefillHeld int
	prefillPeak int
	sem         chan struct{} // prefill permits; nil disables the prefill tier
}

// newClientInFlight builds a two-tier tracker for one sweep point. prefillCap is
// clamped into (0, sessionCap]; values outside that range fall back to the
// session cap so the prefill tier can never admit more than the session tier.
func newClientInFlight(sessionCap, prefillCap int) *clientInFlight {
	if prefillCap <= 0 || prefillCap > sessionCap {
		prefillCap = sessionCap
	}
	return &clientInFlight{sem: make(chan struct{}, prefillCap)}
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

// beginPrefill acquires a prefill-tier slot, blocking until one frees or ctx ends.
// It reports whether a slot was acquired; with the tier disabled it acquires
// nothing and reports true so callers keep one uniform release path.
func (f *clientInFlight) beginPrefill(ctx context.Context) bool {
	if f.sem == nil {
		return true
	}
	select {
	case f.sem <- struct{}{}:
	case <-ctx.Done():
		return false
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.prefillHeld++
	if f.prefillHeld > f.prefillPeak {
		f.prefillPeak = f.prefillHeld
	}
	return true
}

// endPrefill releases a prefill-tier slot held by beginPrefill. Callers must call
// it exactly once per acquired slot, including every failure path, so a failed
// request cannot strand the permit and starve later requests.
func (f *clientInFlight) endPrefill() {
	if f.sem == nil {
		return
	}
	f.mu.Lock()
	f.prefillHeld--
	f.mu.Unlock()
	<-f.sem
}

func (f *clientInFlight) prefillPeakValue() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.prefillPeak
}

// PointResult aggregates one concurrency level.
type PointResult struct {
	Concurrency               int     `json:"concurrency"`
	PeakInFlight              int     `json:"peak_in_flight"` // observed concurrent client HTTP operations, not GPU batch execution
	Requests                  int     `json:"requests"`
	Errors                    int     `json:"errors"`
	ErrorRate                 float64 `json:"error_rate"`
	WallSeconds               float64 `json:"wall_seconds"`
	CompletionTokens          int     `json:"completion_tokens"` // sum of reported usage only, never inferred from chunks
	TokenUsageMissingRequests int     `json:"token_usage_missing_requests,omitempty"`
	// TokensPerSecond is reported completion tokens / sweep wall (including
	// prefill/TTFT). It stays zero/omitted when any successful stream lacks usage.
	// A missing rate is not a measured zero; inspect TokenUsageMissingRequests.
	TokensPerSecond float64 `json:"tokens_per_second,omitempty"`
	P50LatencyMS    float64 `json:"p50_latency_ms"`
	P95LatencyMS    float64 `json:"p95_latency_ms"`
	MeanLatencyMS   float64 `json:"mean_latency_ms"`
	// Client-observed first-content timing and content-chunk cadence, populated
	// only from successful SSE streams. Buffering and multi-token chunks mean
	// these are not server token/decode timings.
	Streamed                     bool    `json:"streamed,omitempty"`
	TTFTP50MS                    float64 `json:"ttft_p50_ms,omitempty"`
	TTFTP95MS                    float64 `json:"ttft_p95_ms,omitempty"`
	ContentChunks                int     `json:"content_chunks,omitempty"`
	MeanInterContentChunkMS      float64 `json:"mean_inter_content_chunk_ms,omitempty"`
	ContentChunkCadencePerSecond float64 `json:"content_chunk_cadence_per_second,omitempty"` // reciprocal mean interval, not aggregate throughput
	// aiperf-style inter-token latency, derived from the decode window (request
	// latency - TTFT) and server-reported usage; NOT the client chunk-cadence
	// metric above. Populated only from streamed successes with usable usage.
	MeanITLMS float64 `json:"mean_itl_ms,omitempty"`
	ITLP50MS  float64 `json:"itl_p50_ms,omitempty"`
	ITLP95MS  float64 `json:"itl_p95_ms,omitempty"`
	// PerChunkUsageDivisorRequests counts requests whose ITL divisor came from
	// server per-chunk usage rather than the osl-1 fallback.
	PerChunkUsageDivisorRequests int `json:"per_chunk_usage_divisor_requests,omitempty"`
	// Deprecated: SSE content chunks do not establish token boundaries. These
	// legacy fields remain zero/omitted; use the explicitly named chunk metrics.
	TPOTMS                float64 `json:"tpot_ms,omitempty"`
	DecodeTokensPerSecond float64 `json:"decode_tokens_per_second,omitempty"`
	// PrefillConcurrency is this point's effective prefill-tier cap: the number of
	// requests allowed to await first content at once. It equals Concurrency
	// unless Config.PrefillConcurrency dials it lower.
	PrefillConcurrency int `json:"prefill_concurrency,omitempty"`
	// PeakPrefillInFlight is the observed number of client requests simultaneously
	// awaiting their first content chunk. Client-side observation, not server
	// GPU batching.
	PeakPrefillInFlight int `json:"peak_prefill_in_flight,omitempty"`
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
	// PrefillConcurrency caps requests awaiting their first content chunk (TTFT),
	// independently of total in-flight. 0 (default) ties it to each point's
	// concurrency; a smaller value throttles concurrent prompt processing while
	// already-prefilled streams keep decoding under the session cap.
	PrefillConcurrency int
	APIKey             string
	HTTPClient         *http.Client
	Now                func() time.Time
	// Dataset optionally replays stored payload bytes verbatim instead of the
	// synthetic prompt: request i sends payloadSeq(Dataset)[i%n]. The client is
	// owned by the caller; loadgen never closes it.
	Dataset *DatasetClient
	// RatePerSec paces job issuance at this absolute-deadline request rate per
	// concurrency point; 0 disables pacing (issue as fast as workers consume).
	RatePerSec float64
	// MaxCatchUp bounds how far behind schedule the pacer may fall before it
	// re-anchors to now; 0 selects defaultMaxCatchUp, negative is rejected by
	// validateConfig.
	MaxCatchUp time.Duration
}

var (
	errNonPositiveConcurrency   = errors.New("loadgen: concurrency must be positive")
	errRequestsBelowConcurrency = errors.New("loadgen: requests-per must be at least maximum concurrency")
	errNonFiniteRate            = errors.New("loadgen: rate-per-second must be finite")
	errNegativeRate             = errors.New("loadgen: rate-per-second must be non-negative")
	errNegativeMaxCatchUp       = errors.New("loadgen: max-catch-up must be non-negative")
	errPrefillAboveConcurrency  = errors.New("loadgen: prefill-concurrency must not exceed maximum concurrency")
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
	if math.IsNaN(cfg.RatePerSec) || math.IsInf(cfg.RatePerSec, 0) {
		return fmt.Errorf("%w: got %g", errNonFiniteRate, cfg.RatePerSec)
	}
	if cfg.RatePerSec < 0 {
		return fmt.Errorf("%w: got %g", errNegativeRate, cfg.RatePerSec)
	}
	if cfg.MaxCatchUp < 0 {
		return fmt.Errorf("%w: got %s", errNegativeMaxCatchUp, cfg.MaxCatchUp)
	}
	// A prefill slot is always held together with a session slot, so a prefill
	// cap above the largest session cap can never bind; reject it as a typo.
	if cfg.PrefillConcurrency > maxConcurrency {
		return fmt.Errorf("%w: prefill=%d maximum-concurrency=%d", errPrefillAboveConcurrency, cfg.PrefillConcurrency, maxConcurrency)
	}
	return nil
}

// defaultMaxCatchUp bounds how far behind its absolute deadline schedule a paced
// feed may fall before it re-anchors to "now". Upstream aiperf defaults its
// MAX_CATCHUP_SECONDS to 0.01 (request_rate.py / environment.py at the pinned
// commit); zero MaxCatchUp selects this value instead of disabling pacing.
const defaultMaxCatchUp = 10 * time.Millisecond

// pacer paces n issues against absolute deadlines with a bounded catch-up
// window, porting aiperf's constant-rate loop mechanics (ai-dynamo/aiperf,
// src/aiperf/timing/strategies/request_rate.py:234
// @ ab610add4e8876fca223197596c728e17365f381, Apache-2.0):
//
//   - the schedule tracks cumulative absolute targets, so oversleeps never
//     compound into drift;
//   - it re-anchors to "now" only once the backlog exceeds maxCatchUp, so
//     sub-window timer jitter keeps the original schedule while a genuine stall
//     drops its missed ticks instead of firing a burst;
//   - the next target advances BEFORE issuing, so variable issue latency cannot
//     shift subsequent deadlines.
//
// The upstream asyncio high-resolution wait became a context-raced timer
// (pacerSleep); issuance is reported through issue() so the caller owns the
// worker channel.
type pacer struct {
	interval   time.Duration
	maxCatchUp time.Duration
	now        func() time.Time
	sleep      func(context.Context, time.Duration) bool
}

func newPacer(interval, maxCatchUp time.Duration, now func() time.Time, sleep func(context.Context, time.Duration) bool) *pacer {
	if now == nil {
		now = time.Now
	}
	if sleep == nil {
		sleep = pacerSleep
	}
	return &pacer{interval: interval, maxCatchUp: maxCatchUp, now: now, sleep: sleep}
}

// pacerSleep blocks until d elapses or ctx finishes; false means the context
// ended first.
func pacerSleep(ctx context.Context, d time.Duration) bool {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return false
	case <-t.C:
		return true
	}
}

// run issues up to n calls through issue (each returning whether it was
// accepted) on an absolute-deadline schedule: the first deadline is one
// interval after start, each subsequent deadline advances by the interval from
// the PREVIOUS deadline (never from wall-clock wake time, so oversleeps do not
// compound). If the next deadline has fallen more than maxCatchUp in the past,
// the schedule re-anchors to now — missed ticks are dropped, never burst-caught.
// It stops early when ctx finishes or issue rejects, and returns how many
// issues were actually accepted: a rejected feed is never counted, so callers
// can slice their result arrays to the returned count safely. The deadline
// advances BEFORE issuing so issue latency cannot shift subsequent deadlines.
func (p *pacer) run(ctx context.Context, n int, issue func() bool) int {
	next := p.now().Add(p.interval)
	issued := 0
	for i := 0; i < n; i++ {
		if ctx.Err() != nil {
			break
		}
		now := p.now()
		if next.Add(p.maxCatchUp).Before(now) {
			next = now // past the catch-up window: drop the backlog, keep the rate
		}
		if next.After(now) {
			if !p.sleep(ctx, next.Sub(now)) {
				break
			}
		}
		next = next.Add(p.interval) // advance before issue: issue latency must not compound
		if !issue() {
			break // rejected feed: not counted, nothing downstream consumed it
		}
		issued++
	}
	return issued
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
// wall-clock, time-to-first-content, and inter-content-chunk arrivals. When
// bodyOverride is non-nil it is sent verbatim (stored-payload replay); otherwise
// the synthetic chat request is built from cfg. If the endpoint answers with a
// plain JSON body instead of an SSE stream, it falls back to the non-streaming
// shape (streamed=false, no phase timing).
func oneRequest(ctx context.Context, cfg *Config, inFlight *clientInFlight, bodyOverride []byte) reqResult {
	var body []byte
	if bodyOverride != nil {
		body = bodyOverride
	} else {
		body, _ = json.Marshal(chatRequest{
			Model:       cfg.Model,
			Messages:    []chatMessage{{Role: "user", Content: cfg.Prompt}},
			MaxTokens:   cfg.MaxTokens,
			Temperature: 0,
			Stream:      true,
		})
	}
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
	if !inFlight.beginPrefill(ctx) {
		// Canceled while waiting for a prefill slot: release the session slot to
		// keep the tiers symmetric (aiperf releases slots on failure).
		inFlight.end()
		return reqResult{err: ctx.Err(), latency: cfg.now().Sub(start)}
	}
	resp, err := cfg.client().Do(req)
	if err != nil {
		inFlight.endPrefill()
		inFlight.end()
		return reqResult{err: err, latency: cfg.now().Sub(start)}
	}
	defer func() {
		_ = resp.Body.Close()
		inFlight.end()
	}()
	if resp.StatusCode >= 300 {
		inFlight.endPrefill() // no content will arrive: release the prefill slot now
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
		// Arm the TTFT hook on this request's measurer: it releases the prefill
		// slot exactly once, at the first content chunk, while the session slot
		// stays held for the rest of the stream.
		var m sseMeasure
		m.onFirstContent = inFlight.endPrefill
		rr := consumeSSE(&m, cfg, br, firstLine, firstErr, start)
		if m.firstContent.IsZero() {
			inFlight.endPrefill() // stream ended before any content chunk: still release
		}
		return rr
	}
	// Plain-JSON delivery has no incremental TTFT: its first content arrives
	// atomically with the parsed body, so the prefill slot is released once the
	// whole response has been consumed.
	rr := jsonFallback(cfg, br, firstLine, firstErr, start)
	inFlight.endPrefill()
	return rr
}

// emptyStreamBody is the closed sentinel for a 200 response whose stream carried no
// data lines at all.
var emptyStreamBody = errors.New("empty stream body")

// consumeSSE folds an SSE chat-completions body into m, sampling the arrival
// time of each content chunk. The body is parsed incrementally as complete SSE
// events: event delimiters that straddle two network chunks are recovered by
// re-scanning the overlap region, and every event is stamped with the arrival
// time of the chunk that completed it. A clean EOF (with or without [DONE])
// completes the request; malformed lines are skipped and never panic. Observed
// non-EOF read errors fail the request, including errors accompanying partial
// data. Completion tokens come only from usage.completion_tokens; chunk counts
// remain a separate observation. The caller owns m so it can arm
// m.onFirstContent before consuming.
func consumeSSE(m *sseMeasure, cfg *Config, br *bufio.Reader, firstLine string, firstErr error, start time.Time) reqResult {
	p := newSSEStreamParser()
	if firstLine != "" {
		for _, msg := range p.feed([]byte(firstLine), cfg.now()) {
			m.feedSSEMessage(msg.body, msg.at)
		}
	}
	chunkBuf := make([]byte, 32*1024)
	readErr := firstErr
	for readErr == nil && !m.done {
		n, err := br.Read(chunkBuf)
		if n > 0 {
			for _, msg := range p.feed(chunkBuf[:n], cfg.now()) {
				m.feedSSEMessage(msg.body, msg.at)
			}
		}
		readErr = err
	}
	if errors.Is(readErr, io.EOF) {
		for _, msg := range p.finish(cfg.now()) {
			m.feedSSEMessage(msg.body, msg.at)
		}
	}
	lat := cfg.now().Sub(start)
	if readErr != nil && !errors.Is(readErr, io.EOF) {
		return reqResult{err: fmt.Errorf("read stream body: %w", readErr), latency: lat, streamed: true}
	}
	if m.streamErr != nil {
		return reqResult{err: m.streamErr, latency: lat, streamed: true}
	}
	if !m.sawData {
		return reqResult{err: emptyStreamBody, latency: lat, streamed: true}
	}
	rr := reqResult{latency: lat, completionTokens: m.usageTokens, streamed: true,
		tokenUsageMissing: !m.usageKnown, contentChunks: m.nContent}
	if !m.firstContent.IsZero() {
		rr.ttft = m.firstContent.Sub(start)
	}
	rr.chunkIntervals = m.chunkIntervals
	rr.firstChunkTokens = m.firstChunkTokens
	return rr
}

// measureStream consumes an SSE chat-completions body, sampling the arrival time of
// each content chunk. A clean EOF (with or without [DONE]) completes the request;
// malformed lines are skipped and never panic. Observed non-EOF read errors fail
// the request, including errors accompanying partial data. Completion tokens come
// only from usage.completion_tokens; chunk counts remain a separate observation.
func measureStream(cfg *Config, br *bufio.Reader, firstLine string, firstErr error, start time.Time) reqResult {
	var m sseMeasure
	return consumeSSE(&m, cfg, br, firstLine, firstErr, start)
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

// errNoPayloads is reported per request when a dataset-backed sweep has no
// payload bytes to replay.
var errNoPayloads = errors.New("loadgen: dataset contains no payload bytes")

// payloadSeq flattens a dataset's (session, turn) pairs, in index order, into
// the ordered replay sequence a sweep round-robins over.
func payloadSeq(ds *DatasetClient) ([][]byte, error) {
	idx := ds.Index()
	var seq [][]byte
	for _, sid := range idx.ConversationIDs {
		for turn := range idx.PayloadOffsets[sid] {
			b, err := ds.PayloadBytes(sid, turn)
			if err != nil {
				return nil, err
			}
			seq = append(seq, b)
		}
	}
	return seq, nil
}

// prefillCapFor resolves one sweep point's effective prefill-tier cap:
// Config.PrefillConcurrency when dialed below the point's concurrency, else the
// concurrency itself (the tiers move together unless separated).
func prefillCapFor(cfg *Config, concurrency int) int {
	if cfg.PrefillConcurrency > 0 && cfg.PrefillConcurrency < concurrency {
		return cfg.PrefillConcurrency
	}
	return concurrency
}

// runPoint issues up to `requests` requests at the given concurrency. When
// cfg.Dataset is set, request i replays payloadSeq(cfg.Dataset)[i%n] verbatim;
// otherwise the synthetic prompt from cfg is used. With RatePerSec > 0,
// issuance is paced on absolute deadlines with a bounded catch-up window
// (pacer); otherwise jobs issue as fast as workers consume. Feeding stops when
// ctx finishes in both modes, and PointResult.Requests reports the count
// actually issued. With pacing enabled, PointResult.WallSeconds spans the
// paced schedule, so it includes pacing idle time and throughput denominators
// include pacing gaps.
func runPoint(ctx context.Context, cfg *Config, concurrency, requests int) PointResult {
	results := make([]reqResult, requests)
	jobs := make(chan int)
	prefillCap := prefillCapFor(cfg, concurrency)
	inFlight := newClientInFlight(concurrency, prefillCap)
	var wg sync.WaitGroup
	wallStart := cfg.now()
	var (
		bodies [][]byte
		dsErr  error
	)
	if cfg.Dataset != nil {
		bodies, dsErr = payloadSeq(cfg.Dataset)
	}
	for w := 0; w < concurrency; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				if dsErr != nil {
					results[i] = reqResult{err: dsErr}
					continue
				}
				var override []byte
				if len(bodies) > 0 {
					override = bodies[i%len(bodies)]
				} else if cfg.Dataset != nil {
					results[i] = reqResult{err: errNoPayloads}
					continue
				}
				results[i] = oneRequest(ctx, cfg, inFlight, override)
			}
		}()
	}
	feedJob := func(i int) bool {
		select {
		case jobs <- i:
			return true
		case <-ctx.Done():
			return false
		}
	}
	var issued int
	if cfg.RatePerSec > 0 {
		interval := time.Duration(float64(time.Second) / cfg.RatePerSec)
		catchUp := cfg.MaxCatchUp
		if catchUp <= 0 {
			catchUp = defaultMaxCatchUp
		}
		counter := 0
		issued = newPacer(interval, catchUp, cfg.now, nil).run(ctx, requests, func() bool {
			i := counter
			counter++
			return feedJob(i)
		})
	} else {
		for i := 0; i < requests; i++ {
			if !feedJob(i) {
				break
			}
			issued++
		}
	}
	close(jobs)
	wg.Wait()
	wall := cfg.now().Sub(wallStart)
	result := aggregate(concurrency, issued, wall, results[:issued])
	result.PeakInFlight = inFlight.peakValue()
	result.PrefillConcurrency = prefillCap
	result.PeakPrefillInFlight = inFlight.prefillPeakValue()
	return result
}

// aggregate folds raw per-request results into a PointResult.
func aggregate(concurrency, requests int, wall time.Duration, results []reqResult) PointResult {
	pr := PointResult{Concurrency: concurrency, Requests: requests, WallSeconds: wall.Seconds()}
	var lats []float64
	var latSum float64
	var ttfts []float64
	var chunkIntervalSum float64
	var chunkIntervalN int
	var itls []float64
	var itlSum float64
	var perChunkDivisorRequests int
	for _, r := range results {
		if r.err != nil {
			pr.Errors++
			continue
		}
		pr.CompletionTokens += r.completionTokens
		if r.tokenUsageMissing {
			pr.TokenUsageMissingRequests++
		}
		// Nanoseconds()/1e6 keeps sub-microsecond loopback timings nonzero; Microseconds() truncates them to 0.
		ms := float64(r.latency.Nanoseconds()) / 1e6
		lats = append(lats, ms)
		latSum += ms
		if !r.streamed {
			continue
		}
		pr.Streamed = true
		pr.ContentChunks += r.contentChunks
		if r.ttft > 0 {
			ttfts = append(ttfts, float64(r.ttft.Nanoseconds())/1e6)
		}
		for _, d := range r.chunkIntervals {
			chunkIntervalSum += float64(d.Nanoseconds()) / 1e6
			chunkIntervalN++
		}
		// aiperf-style ITL: decode window over the per-chunk-aware decode-token
		// divisor. Skips requests without usable usage or a measured TTFT.
		if !r.tokenUsageMissing && r.completionTokens >= 2 && r.ttft > 0 {
			if d, ok := DecodeTokenDivisor(r.completionTokens, r.firstChunkTokens); ok {
				itl := ITLMS(r.latency, r.ttft, d)
				itls = append(itls, itl)
				itlSum += itl
				if d.PerChunk {
					perChunkDivisorRequests++
				}
			}
		}
	}
	if requests > 0 {
		pr.ErrorRate = float64(pr.Errors) / float64(requests)
	}
	if wall.Seconds() > 0 && pr.TokenUsageMissingRequests == 0 {
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
	if chunkIntervalN > 0 {
		pr.MeanInterContentChunkMS = chunkIntervalSum / float64(chunkIntervalN)
		if pr.MeanInterContentChunkMS > 0 {
			pr.ContentChunkCadencePerSecond = 1000 / pr.MeanInterContentChunkMS
		}
	}
	if n := len(itls); n > 0 {
		sort.Float64s(itls)
		pr.MeanITLMS = itlSum / float64(n)
		pr.ITLP50MS = percentile(itls, 50)
		pr.ITLP95MS = percentile(itls, 95)
	}
	pr.PerChunkUsageDivisorRequests = perChunkDivisorRequests
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
