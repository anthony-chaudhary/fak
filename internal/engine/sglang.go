package engine

// sglang.go — the SGLang EngineDriver adapter (issue #39).
//
// This makes SGLang a first-class, router-dispatchable abi.EngineDriver behind the
// SAME lifecycle seam vLLM uses, instead of a single hardcoded gateway.BaseURL
// proxy. It speaks SGLang's PUBLIC surfaces only — the native /generate streaming
// endpoint for generation, an HTTP radix-residency poll for RadixAttention's
// cache-aware signal, and the Prometheus /metrics scrape — and never forks SGLang
// or links its internals (the issue's non-goal). The three signal lanes mirror the
// vLLM adapter so the two stay symmetric:
//
//   - Generation: POST /generate with stream=true. SGLang's stream sends the
//     CUMULATIVE output text in each chunk (not a delta), with token accounting,
//     the RadixAttention prefix-cache hit count (meta_info.cached_tokens), and the
//     finish reason in meta_info. advance() diffs the cumulative text to emit a
//     per-step delta and is robust to a delta-style server too.
//   - Residency: a poll of a configured radix endpoint feeds the SAME
//     PrefixResidencyIndex the vLLM KV-events lane feeds, so cache-aware
//     power-of-two routing consumes one index across both engines. SGLang has no
//     standardized public per-worker radix dump, so the endpoint shape is a seam a
//     router/bridge (or a test stub) supplies — fak consumes the decoded snapshot.
//   - Metrics: ParseSGLangPrometheus maps sglang:* scheduler metrics onto the SAME
//     fak_serving_* ServingMetricsSnapshot schema the vLLM emitter targets — two
//     emitters, one schema, tagged per-engine/per-worker.
//
// Honesty boundary (unchanged from enginecache): SGLang's public control plane
// resets the whole radix cache (flush_cache), not an exact KV span, so
// enginecache.SupportsExactSpan stays false for EngineSGLang and governance keeps
// flowing through enginecache — this adapter adds NO duplicate flush logic and only
// shares the engine identity "sglang" with the cache referee.

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// SGLangEngineID is the registered engine id for the SGLang adapter. It is the SAME
// string enginecache.EngineSGLang uses so the serving adapter and the cache referee
// agree on engine identity.
const SGLangEngineID = "sglang"

// SGLangConfig wires one SGLang worker through public surfaces only: the native
// /generate endpoint for generation, a radix-residency poll for RadixAttention's
// cache-aware signal, and Prometheus for serving metrics. It deliberately does not
// rely on an SGLang source patch or an in-process Python API.
type SGLangConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	WorkerID   string
	MetricsURL string
	// RadixURL is the per-worker RadixAttention residency endpoint. SGLang exposes
	// no single standardized public radix-dump path, so this is supplied by the
	// router/bridge (or a test stub) and consumed as a decoded snapshot.
	RadixURL string
	Client   *http.Client

	CacheRecorder *CacheEventRecorder
	Residency     *PrefixResidencyIndex
}

// EnvSGLangConfig returns the default SGLang driver configuration. FAK_SGLANG_BASE_URL
// should point at the worker's HTTP root (the host:port sglang.launch_server binds).
func EnvSGLangConfig() SGLangConfig {
	return SGLangConfig{
		BaseURL:    os.Getenv("FAK_SGLANG_BASE_URL"),
		Model:      os.Getenv("FAK_SGLANG_MODEL"),
		APIKey:     os.Getenv("FAK_SGLANG_API_KEY"),
		WorkerID:   envDefault("FAK_SGLANG_WORKER_ID", "sglang"),
		MetricsURL: os.Getenv("FAK_SGLANG_METRICS_URL"),
		RadixURL:   os.Getenv("FAK_SGLANG_RADIX_URL"),
	}
}

// SGLangEngine is the SGLang adapter behind abi.EngineDriver/LifecycleEngine.
type SGLangEngine struct {
	cfg       SGLangConfig
	client    *http.Client
	cache     *CacheEventRecorder
	residency *PrefixResidencyIndex
}

// NewSGLangEngine builds an SGLang driver over public SGLang surfaces.
func NewSGLangEngine(cfg SGLangConfig) *SGLangEngine {
	if cfg.WorkerID == "" {
		cfg.WorkerID = "sglang"
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	cache := cfg.CacheRecorder
	if cache == nil {
		cache = NewCacheEventRecorder()
	}
	residency := cfg.Residency
	if residency == nil {
		residency = NewPrefixResidencyIndex()
	}
	return &SGLangEngine{cfg: cfg, client: client, cache: cache, residency: residency}
}

// Caps advertises the SGLang adapter, the native /generate surface, lifecycle
// streaming, RadixAttention residency ingestion, metrics scrape normalization, and
// the honest whole-prefix cache-control boundary.
func (e *SGLangEngine) Caps() []abi.Capability {
	return []abi.Capability{
		"engine.sglang",
		"engine.sglang.generate",
		"engine.sglang.radix",
		"engine.sglang.metrics",
		"engine.cache.whole-prefix",
		abi.EngineLifecycleCap,
	}
}

// WeightBearing declares that SGLang runs a model-forward, not a deterministic tool.
func (e *SGLangEngine) WeightBearing() bool { return true }

// Residency exposes the per-worker prefix-residency index the radix poll feeds, so a
// router can read it for cache-aware dispatch.
func (e *SGLangEngine) Residency() *PrefixResidencyIndex { return e.residency }

// CacheRecorder exposes the cache-event recorder fed by RadixAttention prefix-cache
// hits surfaced on each /generate turn.
func (e *SGLangEngine) CacheRecorder() *CacheEventRecorder { return e.cache }

// Admit submits one request to SGLang's native /generate with stream=true and
// returns a live request handle whose Tokens channel receives SSE deltas as SGLang
// emits them.
func (e *SGLangEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if strings.TrimSpace(e.cfg.BaseURL) == "" {
		return nil, errors.New("sglang: FAK_SGLANG_BASE_URL or SGLangConfig.BaseURL is required")
	}
	endpoint, err := joinEndpoint(e.cfg.BaseURL, "/generate")
	if err != nil {
		return nil, err
	}
	body, err := e.buildGenerateBody(c, refBytes(ctx, c.Args))
	if err != nil {
		return nil, err
	}
	cctx, cancel := context.WithCancel(ctx)
	req, err := http.NewRequestWithContext(cctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if e.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		cancel()
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("sglang: /generate returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	r := &sglangRequest{
		tokens:   make(chan abi.EngineToken),
		done:     make(chan struct{}),
		cancel:   cancel,
		body:     resp.Body,
		call:     c,
		putCtx:   ctx,
		engine:   SGLangEngineID,
		workerID: e.cfg.WorkerID,
		model:    e.cfg.Model,
		cache:    e.cache,
	}
	go r.pump(cctx)
	return r, nil
}

// Complete drains the live stream and returns the assembled result.
func (e *SGLangEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return completeViaAdmit(ctx, e, c)
}

// buildGenerateBody shapes the native SGLang /generate request. SGLang takes a raw
// `text` prompt (or input_ids/input_embeds) plus sampling_params; it is single-model
// per server, so no model field is injected. A caller-supplied JSON object is passed
// through verbatim with stream forced on; a non-object/empty args synthesizes a
// prompt from the tool name so the offline dispatch chain still drives it.
func (e *SGLangEngine) buildGenerateBody(c *abi.ToolCall, args []byte) ([]byte, error) {
	obj := map[string]json.RawMessage{}
	if json.Unmarshal(args, &obj) != nil || len(obj) == 0 {
		obj = map[string]json.RawMessage{}
	}
	_, hasText := obj["text"]
	_, hasInputIDs := obj["input_ids"]
	_, hasInputEmbeds := obj["input_embeds"]
	if !hasText && !hasInputIDs && !hasInputEmbeds {
		prompt := strings.TrimSpace(toolName(c) + " " + string(args))
		obj["text"] = mustJSON(prompt)
	}
	obj["stream"] = json.RawMessage("true")
	return json.Marshal(obj)
}

// metricsURL derives the SGLang Prometheus endpoint when not configured explicitly.
// SGLang's HTTP root is not an OpenAI /v1 frontend, so no /v1 suffix is stripped.
func (e *SGLangEngine) metricsURL() (string, error) {
	return deriveMetricsURL(e.cfg.MetricsURL, e.cfg.BaseURL, "sglang", "FAK_SGLANG_METRICS_URL", false)
}

func (e *SGLangEngine) radixURL() (string, error) {
	if strings.TrimSpace(e.cfg.RadixURL) != "" {
		return e.cfg.RadixURL, nil
	}
	return "", errors.New("sglang: FAK_SGLANG_RADIX_URL or SGLangConfig.RadixURL is required for radix residency polling")
}

// ScrapeServingMetrics reads SGLang's Prometheus endpoint and normalizes its
// scheduler metrics into fak's shared engine-serving schema (fak_serving_*).
func (e *SGLangEngine) ScrapeServingMetrics(ctx context.Context) (ServingMetricsSnapshot, error) {
	metricsURL, err := e.metricsURL()
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	if e.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return ServingMetricsSnapshot{}, fmt.Errorf("sglang: metrics returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	return ParseSGLangPrometheus(e.cfg.WorkerID, string(raw)), nil
}

// ParseSGLangPrometheus extracts the SGLang scheduler metric names and maps them
// onto the SAME fak_serving_* schema the vLLM emitter targets (two emitters, one
// schema). Unmapped sglang:* names are ignored — only fields the shared schema
// already carries are populated, so the output never leaks SGLang-only metric names.
func ParseSGLangPrometheus(workerID, text string) ServingMetricsSnapshot {
	s := ServingMetricsSnapshot{Engine: SGLangEngineID, WorkerID: firstNonEmpty(workerID, "sglang")}
	var hitRatio *float64
	for _, line := range strings.Split(text, "\n") {
		name, value, ok := parsePromSample(line)
		if !ok {
			continue
		}
		switch name {
		case "sglang:time_to_first_token_seconds_sum":
			s.TTFT.Sum += value
		case "sglang:time_to_first_token_seconds_count":
			s.TTFT.Count += value
		case "sglang:time_per_output_token_seconds_sum":
			s.TPOT.Sum += value
		case "sglang:time_per_output_token_seconds_count":
			s.TPOT.Count += value
		case "sglang:inter_token_latency_seconds_sum":
			s.ITL.Sum += value
		case "sglang:inter_token_latency_seconds_count":
			s.ITL.Count += value
		case "sglang:queue_time_seconds_sum", "sglang:waiting_time_seconds_sum":
			s.Queue.Sum += value
		case "sglang:queue_time_seconds_count", "sglang:waiting_time_seconds_count":
			s.Queue.Count += value
		case "sglang:token_usage", "sglang:kv_cache_usage":
			s.KVCacheUsage += value
		case "sglang:num_running_reqs":
			s.RequestsRunning += value
		case "sglang:num_waiting_reqs":
			s.RequestsWaiting += value
		case "sglang:cache_hit_rate", "sglang:prefix_cache_hit_rate":
			v := value
			if v > 1 && v <= 100 {
				v = v / 100
			}
			hitRatio = &v
		}
	}
	s.PrefixCacheHitRatio = hitRatio
	return s
}

// SGLangRadixSnapshot is the decoded per-worker RadixAttention residency picture fak
// consumes from a poll of the radix endpoint. It is a full snapshot (the resident
// prefix set for the worker), so folding it REPLACES that worker's residency rows.
type SGLangRadixSnapshot struct {
	WorkerID    string                 `json:"worker_id,omitempty"`
	ModelID     string                 `json:"model_id,omitempty"`
	TokenizerID string                 `json:"tokenizer_id,omitempty"`
	TS          float64                `json:"ts,omitempty"`
	Resident    []SGLangResidentPrefix `json:"resident"`
}

// SGLangResidentPrefix is one resident radix prefix/block in a snapshot.
type SGLangResidentPrefix struct {
	Digest    string `json:"digest,omitempty"`
	Hash      string `json:"hash,omitempty"`
	Tokens    int64  `json:"tokens,omitempty"`
	Medium    string `json:"medium,omitempty"`
	BlockSize int    `json:"block_size,omitempty"`
}

// PollRadixResidency fetches one radix-residency snapshot from the configured
// endpoint and folds it into the per-worker prefix-residency index. It is the
// RadixAttention cache-aware signal consumed for cache-aware routing.
func (e *SGLangEngine) PollRadixResidency(ctx context.Context) (SGLangRadixSnapshot, error) {
	radixURL, err := e.radixURL()
	if err != nil {
		return SGLangRadixSnapshot{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, radixURL, nil)
	if err != nil {
		return SGLangRadixSnapshot{}, err
	}
	if e.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return SGLangRadixSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return SGLangRadixSnapshot{}, fmt.Errorf("sglang: radix endpoint returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return SGLangRadixSnapshot{}, err
	}
	var snap SGLangRadixSnapshot
	if err := json.Unmarshal(raw, &snap); err != nil {
		return SGLangRadixSnapshot{}, fmt.Errorf("sglang: decode radix snapshot JSON: %w", err)
	}
	RecordSGLangRadixSnapshot(e.cfg.WorkerID, e.cfg.Model, e.residency, snap)
	return snap, nil
}

// RunRadixPoll polls the radix endpoint on an interval until ctx is cancelled,
// keeping the residency index current. A poll error (other than cancellation) ends
// the loop so the caller can re-establish the subscription.
func (e *SGLangEngine) RunRadixPoll(ctx context.Context, interval time.Duration) error {
	if interval <= 0 {
		interval = time.Second
	}
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		if _, err := e.PollRadixResidency(ctx); err != nil {
			if errors.Is(err, context.Canceled) {
				return nil
			}
			return err
		}
		select {
		case <-ctx.Done():
			return nil
		case <-t.C:
		}
	}
}

// RecordSGLangRadixSnapshot is the pure fold of a radix snapshot into the per-worker
// residency index. The snapshot is a full residency picture, so the worker's prior
// rows are CLEARED before the snapshot rows are stored.
func RecordSGLangRadixSnapshot(worker, model string, idx *PrefixResidencyIndex, snap SGLangRadixSnapshot) []PrefixResidency {
	worker = firstNonEmpty(snap.WorkerID, worker, "sglang")
	model = firstNonEmpty(snap.ModelID, model)
	tokenizer := snap.TokenizerID
	at := time.Unix(0, 0)
	if snap.TS > 0 {
		sec, frac := mathModf(snap.TS)
		at = time.Unix(int64(sec), int64(frac*1e9))
	}
	rows := make([]PrefixResidency, 0, len(snap.Resident))
	for _, p := range snap.Resident {
		digest := firstNonEmpty(p.Digest, p.Hash)
		if digest == "" {
			continue
		}
		rows = append(rows, PrefixResidency{
			WorkerID:    worker,
			Digest:      digest,
			ModelID:     model,
			TokenizerID: tokenizer,
			Medium:      p.Medium,
			Tokens:      p.Tokens,
			BlockSize:   p.BlockSize,
			GroupIdx:    -1,
			UpdatedAt:   at,
		})
	}
	if idx != nil {
		idx.Clear(worker)
		idx.Store(worker, rows...)
	}
	return rows
}

// sglangRequest is one in-flight /generate stream.
type sglangRequest struct {
	tokens chan abi.EngineToken
	done   chan struct{}
	cancel context.CancelFunc
	body   io.ReadCloser

	call     *abi.ToolCall
	putCtx   context.Context
	engine   string
	workerID string
	model    string
	cache    *CacheEventRecorder

	text         strings.Builder
	usage        sglangUsage
	cachedTokens int
	finishReason string
	metaID       string

	res *abi.Result
	err error
}

func (r *sglangRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *sglangRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *sglangRequest) Cancel() { r.cancel() }

func (r *sglangRequest) pump(ctx context.Context) {
	defer r.body.Close()
	defer r.cancel()
	sc := bufio.NewScanner(r.body)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for sc.Scan() {
		if err := ctx.Err(); err != nil {
			r.finish(nil, err)
			return
		}
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			r.finish(r.assemble(), nil)
			return
		}
		var chunk sglangGenChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			r.finish(nil, fmt.Errorf("sglang: decode /generate SSE: %w", err))
			return
		}
		if chunk.MetaInfo.PromptTokens > 0 {
			r.usage.PromptTokens = chunk.MetaInfo.PromptTokens
		}
		if chunk.MetaInfo.CompletionTokens > 0 {
			r.usage.CompletionTokens = chunk.MetaInfo.CompletionTokens
		}
		if chunk.MetaInfo.CachedTokens > 0 {
			r.cachedTokens = chunk.MetaInfo.CachedTokens
		}
		if chunk.MetaInfo.ID != "" {
			r.metaID = chunk.MetaInfo.ID
		}
		if fr := sglangFinishReason(chunk.MetaInfo.FinishReason); fr != "" {
			r.finishReason = fr
		}
		delta := r.advance(chunk.Text)
		if delta == "" {
			continue
		}
		select {
		case r.tokens <- abi.EngineToken{Text: delta}:
		case <-ctx.Done():
			r.finish(nil, ctx.Err())
			return
		}
	}
	if err := sc.Err(); err != nil {
		r.finish(nil, err)
		return
	}
	r.finish(r.assemble(), nil)
}

// advance reconciles SGLang's stream against what has already been emitted. SGLang
// sends the CUMULATIVE output text per chunk, so the delta is the new suffix; a
// delta-style server (text is the increment) is also handled by falling through to
// an append.
func (r *sglangRequest) advance(text string) string {
	if text == "" {
		return ""
	}
	cur := r.text.String()
	if strings.HasPrefix(text, cur) {
		delta := text[len(cur):]
		r.text.Reset()
		r.text.WriteString(text)
		return delta
	}
	r.text.WriteString(text)
	return text
}

func (r *sglangRequest) assemble() *abi.Result {
	tool := ""
	if r.call != nil {
		tool = r.call.Tool
	}
	total := r.usage.PromptTokens + r.usage.CompletionTokens
	body, _ := json.Marshal(struct {
		Tool     string `json:"tool"`
		Engine   string `json:"engine"`
		Worker   string `json:"worker"`
		Endpoint string `json:"endpoint"`
		Model    string `json:"model,omitempty"`
		Text     string `json:"text"`
	}{
		Tool:     tool,
		Engine:   r.engine,
		Worker:   r.workerID,
		Endpoint: "generate",
		Model:    r.model,
		Text:     r.text.String(),
	})
	meta := map[string]string{
		"engine":       r.engine,
		"worker":       r.workerID,
		"endpoint":     "generate",
		"output_chars": strconv.Itoa(r.text.Len()),
	}
	if r.model != "" {
		meta["model"] = r.model
	}
	if r.finishReason != "" {
		meta["finish_reason"] = r.finishReason
	}
	if r.usage.PromptTokens > 0 {
		meta["input_tokens"] = strconv.Itoa(r.usage.PromptTokens)
	}
	if r.usage.CompletionTokens > 0 {
		meta["output_tokens"] = strconv.Itoa(r.usage.CompletionTokens)
	}
	if total > 0 {
		meta["total_tokens"] = strconv.Itoa(total)
	}
	if r.cachedTokens > 0 {
		meta["cached_tokens"] = strconv.Itoa(r.cachedTokens)
		r.recordPrefixHit()
	}
	return &abi.Result{Call: r.call, Payload: putBytes(r.putCtx, body), Status: abi.StatusOK, Meta: meta}
}

// recordPrefixHit folds the turn's RadixAttention prefix-cache hit (meta_info
// cached_tokens) into the shared cache-event stream as a typed KV restore, so a
// genuine prefix reuse is observable as a cache HIT rather than an invisible
// internal SGLang counter.
func (r *sglangRequest) recordPrefixHit() {
	if r.cache == nil || r.cachedTokens <= 0 {
		return
	}
	digest := "sglang-radix-hit:" + r.workerID
	if r.metaID != "" {
		digest += ":" + r.metaID
	}
	r.cache.Record(CacheEvent{
		Direction:    cachemeta.KVRestore,
		SpanDigest:   digest,
		Tokens:       int64(r.cachedTokens),
		ModelID:      r.model,
		PositionMode: cachemeta.PositionPrefixAligned,
		FromTier:     cachemeta.TierHBM,
		ToTier:       cachemeta.TierHBM,
		Owner:        "sglang:" + r.workerID,
		Outcome:      cachemeta.KVTransferOK,
	})
}

func (r *sglangRequest) finish(res *abi.Result, err error) {
	r.res, r.err = res, err
	close(r.tokens)
	close(r.done)
}

type sglangUsage struct {
	PromptTokens     int
	CompletionTokens int
}

// sglangGenChunk is one decoded /generate SSE chunk. text is cumulative; meta_info
// carries token accounting, the RadixAttention prefix-cache hit count, and the
// finish reason (null while streaming, set on the terminal chunk).
type sglangGenChunk struct {
	Text     string `json:"text"`
	MetaInfo struct {
		PromptTokens     int             `json:"prompt_tokens"`
		CompletionTokens int             `json:"completion_tokens"`
		CachedTokens     int             `json:"cached_tokens"`
		FinishReason     json.RawMessage `json:"finish_reason"`
		ID               string          `json:"id"`
	} `json:"meta_info"`
}

// sglangFinishReason normalizes SGLang's finish_reason, which is null while
// streaming and on the terminal chunk is either an object ({"type":"stop"}) or, on
// some builds, a bare string.
func sglangFinishReason(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var obj struct {
		Type string `json:"type"`
	}
	if json.Unmarshal(raw, &obj) == nil {
		return obj.Type
	}
	return ""
}

// DefaultSGLangEngine is registered under "sglang". It is inert until configured with
// FAK_SGLANG_BASE_URL (or replaced in tests via NewSGLangEngine).
var DefaultSGLangEngine = NewSGLangEngine(EnvSGLangConfig())

func init() {
	abi.RegisterEngine(SGLangEngineID, DefaultSGLangEngine)
}

var (
	_ abi.LifecycleEngine = (*SGLangEngine)(nil)
	_ abi.EngineRequest   = (*sglangRequest)(nil)
)
