package engine

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// VLLMEngineID is the registered engine id for the vLLM V1 adapter.
const VLLMEngineID = "vllm"

// VLLMConfig wires one vLLM V1 worker through public surfaces only:
// OpenAI-compatible HTTP for generation, Prometheus for serving metrics, and
// KV-cache event batches for residency. It deliberately does not rely on vLLM
// source patches or internal Python APIs.
//
// Honesty boundary: current vLLM exposes whole-prefix cache reset through its
// public control plane. Exact-span KV governance is not asserted here; it remains
// enginecache.SupportsExactSpan=false and degrades to whole-prefix reset.
type VLLMConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	WorkerID   string
	MetricsURL string
	Client     *http.Client

	// PriorityScheduling advertises that the served vLLM engine runs the V1
	// priority scheduler, so a fak TurnIntent priority may be lowered to the
	// request. When false the adapter emits no priority field (FCFS default).
	PriorityScheduling bool

	CacheRecorder *CacheEventRecorder
	Residency     *PrefixResidencyIndex
	KVEvents      VLLMKVEventSource
}

// EnvVLLMConfig returns the default vLLM driver configuration. FAK_VLLM_BASE_URL
// should point at the worker's OpenAI-compatible root, usually http://host:port/v1.
func EnvVLLMConfig() VLLMConfig {
	return VLLMConfig{
		BaseURL:            os.Getenv("FAK_VLLM_BASE_URL"),
		Model:              os.Getenv("FAK_VLLM_MODEL"),
		APIKey:             os.Getenv("FAK_VLLM_API_KEY"),
		WorkerID:           envDefault("FAK_VLLM_WORKER_ID", "vllm"),
		MetricsURL:         os.Getenv("FAK_VLLM_METRICS_URL"),
		PriorityScheduling: envBool("FAK_VLLM_PRIORITY_SCHEDULING"),
	}
}

// VLLMEngine is a vLLM V1 adapter behind abi.EngineDriver/LifecycleEngine.
type VLLMEngine struct {
	cfg       VLLMConfig
	client    *http.Client
	cache     *CacheEventRecorder
	residency *PrefixResidencyIndex
}

// NewVLLMEngine builds a vLLM driver over public vLLM surfaces.
func NewVLLMEngine(cfg VLLMConfig) *VLLMEngine {
	if cfg.WorkerID == "" {
		cfg.WorkerID = "vllm"
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
	return &VLLMEngine{cfg: cfg, client: client, cache: cache, residency: residency}
}

// Caps advertises the vLLM adapter, the OpenAI HTTP surface, lifecycle streaming,
// KV-event ingestion, metrics scrape normalization, and the honest whole-prefix
// cache-control boundary.
func (e *VLLMEngine) Caps() []abi.Capability {
	return []abi.Capability{
		"engine.vllm",
		"engine.openai",
		"engine.vllm.kv-events",
		"engine.vllm.metrics",
		"engine.cache.whole-prefix",
		abi.EngineLifecycleCap,
	}
}

// WeightBearing declares that vLLM runs a model-forward, not a deterministic tool.
func (e *VLLMEngine) WeightBearing() bool { return true }

// Admit submits one request to vLLM with stream=true and returns a live request
// handle whose Tokens channel receives SSE deltas as vLLM emits them.
func (e *VLLMEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if strings.TrimSpace(e.cfg.BaseURL) == "" {
		return nil, errors.New("vllm: FAK_VLLM_BASE_URL or VLLMConfig.BaseURL is required")
	}
	endpoint, kind, body, err := e.buildOpenAIRequest(ctx, c)
	if err != nil {
		return nil, err
	}
	ctrl := e.deriveVLLMControls(c)
	body = applyVLLMControls(body, ctrl)
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
		return nil, fmt.Errorf("vllm: %s returned %d: %s", kind, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	r := &vllmRequest{
		tokens:      make(chan abi.EngineToken),
		done:        make(chan struct{}),
		cancel:      cancel,
		body:        resp.Body,
		kind:        kind,
		call:        c,
		putCtx:      ctx,
		engine:      VLLMEngineID,
		workerID:    e.cfg.WorkerID,
		model:       e.cfg.Model,
		cacheSalt:   ctrl.cacheSalt,
		priority:    ctrl.priority,
		hasPriority: ctrl.hasPriority,
	}
	go r.pump(cctx)
	return r, nil
}

// RunKVEventSubscription consumes decoded vLLM KV-event batches until ctx is
// cancelled or the source ends. The native vLLM transport is ZMQ/msgpack; fak
// stays dependency-free by taking the decoded batch stream at this seam, so a
// process-local bridge or test fixture can feed the same residency/index logic.
func (e *VLLMEngine) RunKVEventSubscription(ctx context.Context) error {
	if e.cfg.KVEvents == nil {
		return errors.New("vllm: KVEvents source is not configured")
	}
	defer e.cfg.KVEvents.Close()
	for {
		batch, err := e.cfg.KVEvents.Next(ctx)
		if err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, context.Canceled) {
				return nil
			}
			return err
		}
		e.RecordVLLMKVEventBatch(batch)
	}
}

// Complete drains the live stream and returns the assembled result.
func (e *VLLMEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	req, err := e.Admit(ctx, c)
	if err != nil {
		return nil, err
	}
	for range req.Tokens() {
	}
	res, err := req.Result()
	if err != nil {
		return nil, err
	}
	if res != nil && res.Call == nil {
		res.Call = c
	}
	return res, nil
}

func (e *VLLMEngine) buildOpenAIRequest(ctx context.Context, c *abi.ToolCall) (endpoint, kind string, body []byte, err error) {
	args := refBytes(ctx, c.Args)
	kind = vllmEndpointKind(c)
	path := "/chat/completions"
	if kind == "completions" {
		path = "/completions"
	}
	endpoint, err = joinEndpoint(e.cfg.BaseURL, path)
	if err != nil {
		return "", "", nil, err
	}
	if kind == "completions" {
		body, err = e.buildCompletionsBody(c, args)
		return endpoint, kind, body, err
	}
	body, err = e.buildChatBody(c, args)
	return endpoint, kind, body, err
}

func vllmEndpointKind(c *abi.ToolCall) string {
	if c != nil && c.Meta != nil {
		switch strings.ToLower(strings.TrimSpace(c.Meta["openai_endpoint"])) {
		case "completions", "/v1/completions", "/completions":
			return "completions"
		case "chat", "chat/completions", "/v1/chat/completions", "/chat/completions":
			return "chat"
		}
	}
	tool := ""
	if c != nil {
		tool = strings.ToLower(strings.Trim(c.Tool, "/ "))
	}
	if strings.HasSuffix(tool, "completions") && !strings.Contains(tool, "chat") {
		return "completions"
	}
	return "chat"
}

func (e *VLLMEngine) buildChatBody(c *abi.ToolCall, args []byte) ([]byte, error) {
	obj := map[string]json.RawMessage{}
	if json.Unmarshal(args, &obj) != nil || len(obj) == 0 {
		obj = map[string]json.RawMessage{}
	}
	if _, ok := obj["model"]; !ok && e.cfg.Model != "" {
		obj["model"] = mustJSON(e.cfg.Model)
	}
	if _, ok := obj["messages"]; !ok {
		content := strings.TrimSpace(toolName(c) + " " + string(args))
		obj["messages"] = mustJSON([]map[string]string{{"role": "user", "content": content}})
	}
	forceStream(obj)
	return json.Marshal(obj)
}

func (e *VLLMEngine) buildCompletionsBody(c *abi.ToolCall, args []byte) ([]byte, error) {
	obj := map[string]json.RawMessage{}
	if json.Unmarshal(args, &obj) != nil || len(obj) == 0 {
		obj = map[string]json.RawMessage{}
	}
	if _, ok := obj["model"]; !ok && e.cfg.Model != "" {
		obj["model"] = mustJSON(e.cfg.Model)
	}
	if _, ok := obj["prompt"]; !ok {
		prompt := strings.TrimSpace(toolName(c) + " " + string(args))
		obj["prompt"] = mustJSON(prompt)
	}
	forceStream(obj)
	return json.Marshal(obj)
}

func toolName(c *abi.ToolCall) string {
	if c == nil {
		return ""
	}
	return c.Tool
}

func forceStream(obj map[string]json.RawMessage) {
	obj["stream"] = json.RawMessage("true")
	if _, ok := obj["stream_options"]; !ok {
		obj["stream_options"] = json.RawMessage(`{"include_usage":true}`)
	}
}

func mustJSON(v any) json.RawMessage {
	b, _ := json.Marshal(v)
	return b
}

func joinEndpoint(baseURL, suffix string) (string, error) {
	u, err := url.Parse(strings.TrimSpace(baseURL))
	if err != nil {
		return "", err
	}
	if u.Scheme == "" || u.Host == "" {
		return "", fmt.Errorf("vllm: invalid base URL %q", baseURL)
	}
	u.Path = strings.TrimRight(u.Path, "/") + suffix
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

type vllmRequest struct {
	tokens chan abi.EngineToken
	done   chan struct{}
	cancel context.CancelFunc
	body   io.ReadCloser

	kind     string
	call     *abi.ToolCall
	putCtx   context.Context
	engine   string
	workerID string
	model    string

	cacheSalt   string
	priority    string
	hasPriority bool

	text         strings.Builder
	usage        vllmUsage
	finishReason string
	streamModel  string

	res *abi.Result
	err error
}

func (r *vllmRequest) Tokens() <-chan abi.EngineToken { return r.tokens }

func (r *vllmRequest) Result() (*abi.Result, error) {
	<-r.done
	return r.res, r.err
}

func (r *vllmRequest) Cancel() { r.cancel() }

func (r *vllmRequest) pump(ctx context.Context) {
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
		delta, usage, model, finish, err := parseVLLMSSE(data, r.kind)
		if err != nil {
			r.finish(nil, err)
			return
		}
		if usage != nil {
			r.usage = *usage
		}
		if model != "" {
			r.streamModel = model
		}
		if finish != "" {
			r.finishReason = finish
		}
		if delta == "" {
			continue
		}
		r.text.WriteString(delta)
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

func (r *vllmRequest) assemble() *abi.Result {
	tool := ""
	if r.call != nil {
		tool = r.call.Tool
	}
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
		Endpoint: r.kind,
		Model:    firstNonEmpty(r.streamModel, r.model),
		Text:     r.text.String(),
	})
	meta := map[string]string{
		"engine":       r.engine,
		"worker":       r.workerID,
		"endpoint":     r.kind,
		"output_chars": strconv.Itoa(r.text.Len()),
	}
	if model := firstNonEmpty(r.streamModel, r.model); model != "" {
		meta["model"] = model
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
	if r.usage.TotalTokens > 0 {
		meta["total_tokens"] = strconv.Itoa(r.usage.TotalTokens)
	}
	if r.cacheSalt != "" {
		meta["cache_salt"] = r.cacheSalt
	}
	if r.hasPriority {
		meta["priority"] = r.priority
	}
	return &abi.Result{Call: r.call, Payload: putBytes(r.putCtx, body), Status: abi.StatusOK, Meta: meta}
}

func (r *vllmRequest) finish(res *abi.Result, err error) {
	r.res, r.err = res, err
	close(r.tokens)
	close(r.done)
}

type vllmUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type vllmChatSSE struct {
	Model   string `json:"model"`
	Choices []struct {
		Delta struct {
			Content json.RawMessage `json:"content"`
		} `json:"delta"`
		FinishReason any `json:"finish_reason"`
	} `json:"choices"`
	Usage *vllmUsage `json:"usage"`
}

type vllmCompletionSSE struct {
	Model   string `json:"model"`
	Choices []struct {
		Text         string `json:"text"`
		FinishReason any    `json:"finish_reason"`
	} `json:"choices"`
	Usage *vllmUsage `json:"usage"`
}

func parseVLLMSSE(data, kind string) (delta string, usage *vllmUsage, model string, finish string, err error) {
	if kind == "completions" {
		var c vllmCompletionSSE
		if err := json.Unmarshal([]byte(data), &c); err != nil {
			return "", nil, "", "", fmt.Errorf("vllm: decode completions SSE: %w", err)
		}
		if len(c.Choices) > 0 {
			delta = c.Choices[0].Text
			finish = finishString(c.Choices[0].FinishReason)
		}
		return delta, c.Usage, c.Model, finish, nil
	}
	var c vllmChatSSE
	if err := json.Unmarshal([]byte(data), &c); err != nil {
		return "", nil, "", "", fmt.Errorf("vllm: decode chat SSE: %w", err)
	}
	if len(c.Choices) > 0 {
		delta = rawContentText(c.Choices[0].Delta.Content)
		finish = finishString(c.Choices[0].FinishReason)
	}
	return delta, c.Usage, c.Model, finish, nil
}

func rawContentText(raw json.RawMessage) string {
	if len(raw) == 0 || string(raw) == "null" {
		return ""
	}
	var s string
	if json.Unmarshal(raw, &s) == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) == nil {
		var b strings.Builder
		for _, p := range parts {
			if p.Type == "" || p.Type == "text" {
				b.WriteString(p.Text)
			}
		}
		return b.String()
	}
	return ""
}

func finishString(v any) string {
	switch x := v.(type) {
	case string:
		return x
	default:
		return ""
	}
}

// VLLMKVEventBatch is the JSON-shaped mirror of vLLM's KVEventBatch. vLLM
// publishes the live stream over ZMQ/msgpack; this type is the in-process
// normalized form the adapter consumes after transport decoding.
type VLLMKVEventBatch struct {
	TS               float64        `json:"ts"`
	Events           []VLLMKVEvent  `json:"events"`
	DataParallelRank *int           `json:"data_parallel_rank,omitempty"`
	WorkerID         string         `json:"worker_id,omitempty"`
	ModelID          string         `json:"model_id,omitempty"`
	TokenizerID      string         `json:"tokenizer_id,omitempty"`
	Raw              map[string]any `json:"-"`
}

// VLLMKVEvent mirrors vLLM's BlockStored, BlockRemoved, and AllBlocksCleared
// event variants. Block hashes are kept as JSON values because vLLM's
// ExternalBlockHash is versioned; fak only needs a stable digest string.
type VLLMKVEvent struct {
	Type                         string `json:"type,omitempty"`
	Event                        string `json:"event,omitempty"`
	Kind                         string `json:"kind,omitempty"`
	BlockHashes                  []any  `json:"block_hashes,omitempty"`
	ParentBlockHash              any    `json:"parent_block_hash,omitempty"`
	TokenIDs                     []int  `json:"token_ids,omitempty"`
	BlockSize                    int    `json:"block_size,omitempty"`
	LoraID                       *int   `json:"lora_id,omitempty"`
	Medium                       string `json:"medium,omitempty"`
	LoraName                     string `json:"lora_name,omitempty"`
	GroupIdx                     *int   `json:"group_idx,omitempty"`
	KVCacheSpecKind              string `json:"kv_cache_spec_kind,omitempty"`
	KVCacheSpecSlidingWindow     *int   `json:"kv_cache_spec_sliding_window,omitempty"`
	KVCacheSpecSlidingWindowJSON *int   `json:"kv_cache_spec_sliding_window_json,omitempty"`
}

func (ev VLLMKVEvent) eventType() string {
	for _, s := range []string{ev.Type, ev.Event, ev.Kind} {
		s = strings.TrimSpace(s)
		if s != "" {
			return s
		}
	}
	return ""
}

// VLLMKVEventSource is a decoded vLLM KV-event stream. A stdlib-only fak build
// cannot import pyzmq/msgspec, so the transport decoder lives outside this leaf
// and hands the adapter EventBatch-shaped values here.
type VLLMKVEventSource interface {
	Next(ctx context.Context) (VLLMKVEventBatch, error)
	Close() error
}

// VLLMJSONKVEventSource reads one JSON-encoded VLLMKVEventBatch per line. It is
// the dependency-free bridge/test transport for a ZMQ/msgpack subscriber that has
// already decoded vLLM's native EventBatch payloads.
type VLLMJSONKVEventSource struct {
	r io.ReadCloser
	s *bufio.Scanner
}

// NewVLLMJSONKVEventSource wraps an NDJSON batch stream as a KV event source.
func NewVLLMJSONKVEventSource(r io.ReadCloser) *VLLMJSONKVEventSource {
	s := bufio.NewScanner(r)
	s.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	return &VLLMJSONKVEventSource{r: r, s: s}
}

func (s *VLLMJSONKVEventSource) Next(ctx context.Context) (VLLMKVEventBatch, error) {
	for {
		if err := ctx.Err(); err != nil {
			return VLLMKVEventBatch{}, err
		}
		if !s.s.Scan() {
			if err := s.s.Err(); err != nil {
				return VLLMKVEventBatch{}, err
			}
			return VLLMKVEventBatch{}, io.EOF
		}
		line := strings.TrimSpace(s.s.Text())
		if line == "" {
			continue
		}
		var batch VLLMKVEventBatch
		if err := json.Unmarshal([]byte(line), &batch); err != nil {
			return VLLMKVEventBatch{}, fmt.Errorf("vllm: decode KV event batch JSON: %w", err)
		}
		return batch, nil
	}
}

func (s *VLLMJSONKVEventSource) Close() error {
	if s == nil || s.r == nil {
		return nil
	}
	return s.r.Close()
}

// RecordVLLMKVEventBatch folds one decoded vLLM KV-event batch into the
// per-worker residency index and the shared cache-event recorder.
func (e *VLLMEngine) RecordVLLMKVEventBatch(batch VLLMKVEventBatch) []CacheEventResult {
	worker := firstNonEmpty(batch.WorkerID, e.cfg.WorkerID)
	model := firstNonEmpty(batch.ModelID, e.cfg.Model)
	return RecordVLLMKVEventBatch(worker, model, batch.TokenizerID, e.residency, e.cache, batch)
}

// RecordVLLMKVEventBatch is the pure lowering function for vLLM KV events.
func RecordVLLMKVEventBatch(worker, model, tokenizer string, idx *PrefixResidencyIndex, rec *CacheEventRecorder, batch VLLMKVEventBatch) []CacheEventResult {
	if worker == "" {
		worker = "vllm"
	}
	at := time.Unix(0, 0)
	if batch.TS > 0 {
		sec, frac := mathModf(batch.TS)
		at = time.Unix(int64(sec), int64(frac*1e9))
	}
	var out []CacheEventResult
	for _, ev := range batch.Events {
		typ := ev.eventType()
		switch typ {
		case "BlockStored":
			if idx != nil {
				idx.Store(worker, ev.residencyRecords(worker, model, tokenizer, at)...)
			}
			for _, h := range ev.hashDigests() {
				if rec != nil {
					out = append(out, rec.Record(CacheEvent{
						Direction:    cachemeta.KVRestore,
						SpanDigest:   h,
						Tokens:       ev.tokensPerBlock(),
						ModelID:      model,
						TokenizerID:  tokenizer,
						PositionMode: cachemeta.PositionPrefixAligned,
						ToTier:       vllmMediumTier(ev.Medium),
						Owner:        "vllm:" + worker,
						Outcome:      cachemeta.KVTransferOK,
					}))
				}
			}
		case "BlockRemoved":
			if idx != nil {
				idx.Remove(worker, ev.hashDigests()...)
			}
			for _, h := range ev.hashDigests() {
				if rec != nil {
					out = append(out, rec.Record(CacheEvent{
						Direction:    cachemeta.KVOffload,
						SpanDigest:   h,
						Tokens:       ev.tokensPerBlock(),
						ModelID:      model,
						TokenizerID:  tokenizer,
						PositionMode: cachemeta.PositionPrefixAligned,
						FromTier:     vllmMediumTier(ev.Medium),
						ToTier:       cachemeta.TierUnknown,
						Owner:        "vllm:" + worker,
						Outcome:      cachemeta.KVTransferOK,
					}))
				}
			}
		case "AllBlocksCleared":
			if idx != nil {
				idx.Clear(worker)
			}
			if rec != nil {
				out = append(out, rec.Record(CacheEvent{
					Direction:    cachemeta.KVOffload,
					SpanDigest:   "vllm-clear:" + worker,
					ModelID:      model,
					TokenizerID:  tokenizer,
					PositionMode: cachemeta.PositionPrefixAligned,
					FromTier:     cachemeta.TierHBM,
					ToTier:       cachemeta.TierUnknown,
					Owner:        "vllm:" + worker,
					Outcome:      cachemeta.KVTransferOK,
				}))
			}
		}
	}
	return out
}

func (ev VLLMKVEvent) residencyRecords(worker, model, tokenizer string, at time.Time) []PrefixResidency {
	hashes := ev.hashDigests()
	out := make([]PrefixResidency, 0, len(hashes))
	for _, h := range hashes {
		out = append(out, PrefixResidency{
			WorkerID:    worker,
			Digest:      h,
			ModelID:     model,
			TokenizerID: tokenizer,
			Medium:      ev.Medium,
			Tokens:      ev.tokensPerBlock(),
			BlockSize:   ev.BlockSize,
			GroupIdx:    intPtrValue(ev.GroupIdx, -1),
			UpdatedAt:   at,
		})
	}
	return out
}

func (ev VLLMKVEvent) hashDigests() []string {
	out := make([]string, 0, len(ev.BlockHashes))
	for _, h := range ev.BlockHashes {
		d := digestFromAny(h)
		if d != "" {
			out = append(out, d)
		}
	}
	return out
}

func (ev VLLMKVEvent) tokensPerBlock() int64 {
	if ev.BlockSize > 0 {
		return int64(ev.BlockSize)
	}
	if len(ev.TokenIDs) > 0 && len(ev.BlockHashes) > 0 {
		n := len(ev.TokenIDs) / len(ev.BlockHashes)
		if n < 1 {
			n = len(ev.TokenIDs)
		}
		return int64(n)
	}
	return int64(len(ev.TokenIDs))
}

func digestFromAny(v any) string {
	switch x := v.(type) {
	case string:
		return x
	case map[string]any:
		for _, k := range []string{"hash", "hash_value", "block_hash", "digest"} {
			if s, ok := x[k].(string); ok && s != "" {
				return s
			}
		}
	}
	b, err := json.Marshal(v)
	if err != nil || len(b) == 0 || string(b) == "null" {
		return ""
	}
	return cachemeta.DigestBytes(b)
}

func vllmMediumTier(medium string) cachemeta.ResidencyTier {
	switch strings.ToUpper(strings.TrimSpace(medium)) {
	case "GPU", "":
		return cachemeta.TierHBM
	case "CPU":
		return cachemeta.TierDRAM
	default:
		return cachemeta.TierUnknown
	}
}

// PrefixResidency is one worker's current claim that a prefix/KV block is resident.
type PrefixResidency struct {
	WorkerID    string
	Digest      string
	ModelID     string
	TokenizerID string
	Medium      string
	Tokens      int64
	BlockSize   int
	GroupIdx    int
	UpdatedAt   time.Time
}

// PrefixResidencyIndex is the per-worker prefix-residency index fed by vLLM
// KV-cache events.
type PrefixResidencyIndex struct {
	mu      sync.Mutex
	workers map[string]map[string]PrefixResidency
}

func NewPrefixResidencyIndex() *PrefixResidencyIndex {
	return &PrefixResidencyIndex{workers: map[string]map[string]PrefixResidency{}}
}

func (idx *PrefixResidencyIndex) Store(worker string, rows ...PrefixResidency) {
	if idx == nil {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	if idx.workers == nil {
		idx.workers = map[string]map[string]PrefixResidency{}
	}
	if idx.workers[worker] == nil {
		idx.workers[worker] = map[string]PrefixResidency{}
	}
	for _, row := range rows {
		if row.Digest == "" {
			continue
		}
		if row.WorkerID == "" {
			row.WorkerID = worker
		}
		idx.workers[worker][row.Digest] = row
	}
}

func (idx *PrefixResidencyIndex) Remove(worker string, digests ...string) {
	if idx == nil {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	for _, d := range digests {
		delete(idx.workers[worker], d)
	}
}

func (idx *PrefixResidencyIndex) Clear(worker string) {
	if idx == nil {
		return
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	delete(idx.workers, worker)
}

func (idx *PrefixResidencyIndex) Has(worker, digest string) bool {
	if idx == nil {
		return false
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	_, ok := idx.workers[worker][digest]
	return ok
}

func (idx *PrefixResidencyIndex) Snapshot(worker string) []PrefixResidency {
	if idx == nil {
		return nil
	}
	idx.mu.Lock()
	defer idx.mu.Unlock()
	rows := idx.workers[worker]
	out := make([]PrefixResidency, 0, len(rows))
	for _, row := range rows {
		out = append(out, row)
	}
	return out
}

// ScrapeServingMetrics reads vLLM's Prometheus endpoint and normalizes the
// TTFT/TPOT/ITL/queue/KV-util counters into fak's engine-serving schema.
func (e *VLLMEngine) ScrapeServingMetrics(ctx context.Context) (ServingMetricsSnapshot, error) {
	metricsURL, err := e.metricsURL()
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return ServingMetricsSnapshot{}, fmt.Errorf("vllm: metrics returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	return ParseVLLMPrometheus(e.cfg.WorkerID, string(raw)), nil
}

func (e *VLLMEngine) metricsURL() (string, error) {
	if e.cfg.MetricsURL != "" {
		return e.cfg.MetricsURL, nil
	}
	if e.cfg.BaseURL == "" {
		return "", errors.New("vllm: FAK_VLLM_METRICS_URL or BaseURL is required for metrics scrape")
	}
	u, err := url.Parse(e.cfg.BaseURL)
	if err != nil {
		return "", err
	}
	u.Path = strings.TrimRight(strings.TrimSuffix(u.Path, "/v1"), "/") + "/metrics"
	u.RawQuery = ""
	u.Fragment = ""
	return u.String(), nil
}

type metricSumCount struct {
	Count float64
	Sum   float64
}

// ServingMetricsSnapshot is the normalized serving L2 schema for ridden engines.
type ServingMetricsSnapshot struct {
	Engine   string
	WorkerID string
	// WorkerRole is optional. It is set by P/D-aware control planes such as
	// Dynamo when the upstream exposes prefill/decode role labels.
	WorkerRole string

	TTFT  metricSumCount
	TPOT  metricSumCount
	ITL   metricSumCount
	Queue metricSumCount

	KVCacheUsage     float64
	RequestsRunning  float64
	RequestsWaiting  float64
	RequestsSwapped  float64
	RequestSuccesses float64
	PrefixQueries    float64
	PrefixHits       float64

	// PrefixCacheHitRatio is a directly-reported prefix/radix cache-hit ratio
	// (0..1) for engines that expose it as a single gauge (e.g. SGLang's
	// sglang:cache_hit_rate) instead of query/hit counters. It is a pointer so an
	// engine that reports counters instead (vLLM) leaves it nil and emits NO ratio
	// line — a literal 0.0 would read as a measured 0% hit rate, which it is not.
	PrefixCacheHitRatio *float64

	// Optional P/D worker-load gauges. Dynamo exposes these as per-worker signals:
	// active decode blocks and queued prefill tokens. They are not ratios, so they
	// stay separate from KVCacheUsage while still rendering in the fak_serving_* L2
	// namespace with worker labels.
	ActiveDecodeBlocks  *float64
	ActivePrefillTokens *float64
}

// ParseVLLMPrometheus extracts the vLLM metric names used by vLLM V1 and maps
// them onto a stable fak_serving_* schema.
func ParseVLLMPrometheus(workerID, text string) ServingMetricsSnapshot {
	s := ServingMetricsSnapshot{Engine: VLLMEngineID, WorkerID: firstNonEmpty(workerID, "vllm")}
	for _, line := range strings.Split(text, "\n") {
		name, value, ok := parsePromSample(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:time_to_first_token_seconds_sum":
			s.TTFT.Sum += value
		case "vllm:time_to_first_token_seconds_count":
			s.TTFT.Count += value
		case "vllm:request_time_per_output_token_seconds_sum", "vllm:time_per_output_token_seconds_sum":
			s.TPOT.Sum += value
		case "vllm:request_time_per_output_token_seconds_count", "vllm:time_per_output_token_seconds_count":
			s.TPOT.Count += value
		case "vllm:inter_token_latency_seconds_sum":
			s.ITL.Sum += value
		case "vllm:inter_token_latency_seconds_count":
			s.ITL.Count += value
		case "vllm:request_queue_time_seconds_sum":
			s.Queue.Sum += value
		case "vllm:request_queue_time_seconds_count":
			s.Queue.Count += value
		case "vllm:kv_cache_usage_perc":
			s.KVCacheUsage += value
		case "vllm:num_requests_running":
			s.RequestsRunning += value
		case "vllm:num_requests_waiting":
			s.RequestsWaiting += value
		case "vllm:num_requests_swapped":
			s.RequestsSwapped += value
		case "vllm:request_success_total", "vllm:request_success":
			s.RequestSuccesses += value
		case "vllm:prefix_cache_queries":
			s.PrefixQueries += value
		case "vllm:prefix_cache_hits":
			s.PrefixHits += value
		}
	}
	return s
}

type promMetricSample struct {
	name   string
	labels map[string]string
	value  float64
}

func parsePromSample(line string) (name string, value float64, ok bool) {
	s, ok := parsePromMetricSample(line)
	if !ok {
		return "", 0, false
	}
	return s.name, s.value, true
}

func parsePromMetricSample(line string) (promMetricSample, bool) {
	line = strings.TrimSpace(line)
	if line == "" || strings.HasPrefix(line, "#") {
		return promMetricSample{}, false
	}
	fields := strings.Fields(line)
	if len(fields) < 2 {
		return promMetricSample{}, false
	}
	name := fields[0]
	labels := map[string]string{}
	if i := strings.IndexByte(name, '{'); i >= 0 {
		if j := strings.LastIndexByte(name, '}'); j > i {
			if parsed, ok := parsePromSampleLabels(name[i+1 : j]); ok {
				labels = parsed
			}
		}
		name = name[:i]
	}
	v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
	if err != nil {
		return promMetricSample{}, false
	}
	return promMetricSample{name: name, labels: labels, value: v}, true
}

func parsePromSampleLabels(s string) (map[string]string, bool) {
	labels := map[string]string{}
	for strings.TrimSpace(s) != "" {
		s = strings.TrimLeft(s, " \t,")
		eq := strings.IndexByte(s, '=')
		if eq <= 0 {
			return nil, false
		}
		key := strings.TrimSpace(s[:eq])
		s = strings.TrimLeft(s[eq+1:], " \t")
		if !strings.HasPrefix(s, `"`) {
			return nil, false
		}
		value, n, ok := parsePromQuotedLabel(s)
		if !ok {
			return nil, false
		}
		labels[key] = value
		s = s[n:]
	}
	return labels, true
}

func parsePromQuotedLabel(s string) (string, int, bool) {
	var b strings.Builder
	escaped := false
	for i, r := range s {
		if i == 0 {
			continue
		}
		if escaped {
			switch r {
			case 'n':
				b.WriteByte('\n')
			case '\\', '"':
				b.WriteRune(r)
			default:
				b.WriteRune(r)
			}
			escaped = false
			continue
		}
		if r == '\\' {
			escaped = true
			continue
		}
		if r == '"' {
			return b.String(), i + 1, true
		}
		b.WriteRune(r)
	}
	return "", 0, false
}

// Prometheus renders normalized metrics. The values are relabeled as fak_serving_*
// so a vLLM worker and future SGLang/native emitters can share one schema.
func (s ServingMetricsSnapshot) Prometheus() string {
	return ServingMetricsSnapshots{s}.Prometheus()
}

// ServingMetricsSnapshots renders one or more worker rows in the normalized
// fak_serving_* schema without duplicating HELP/TYPE records.
type ServingMetricsSnapshots []ServingMetricsSnapshot

func (rows ServingMetricsSnapshots) Prometheus() string {
	var b strings.Builder
	rows = sortedServingSnapshots(rows)
	writeServingSumCountRows(&b, "fak_serving_ttft_seconds", "Time to first token normalized from the worker serving metrics.", rows, func(s ServingMetricsSnapshot) metricSumCount { return s.TTFT })
	writeServingSumCountRows(&b, "fak_serving_tpot_seconds", "Time per output token normalized from the worker serving metrics.", rows, func(s ServingMetricsSnapshot) metricSumCount { return s.TPOT })
	writeServingSumCountRows(&b, "fak_serving_itl_seconds", "Inter-token latency normalized from the worker serving metrics.", rows, func(s ServingMetricsSnapshot) metricSumCount { return s.ITL })
	writeServingSumCountRows(&b, "fak_serving_queue_seconds", "Queue time normalized from the worker serving metrics.", rows, func(s ServingMetricsSnapshot) metricSumCount { return s.Queue })
	writeGaugeRows(&b, "fak_serving_kv_cache_usage_ratio", "Worker KV-cache usage ratio.", rows, func(s ServingMetricsSnapshot) float64 { return s.KVCacheUsage })
	writeGaugeRows(&b, "fak_serving_requests_running", "Worker running request gauge.", rows, func(s ServingMetricsSnapshot) float64 { return s.RequestsRunning })
	writeGaugeRows(&b, "fak_serving_requests_waiting", "Worker waiting request gauge.", rows, func(s ServingMetricsSnapshot) float64 { return s.RequestsWaiting })
	writeGaugeRows(&b, "fak_serving_requests_swapped", "Worker swapped request gauge.", rows, func(s ServingMetricsSnapshot) float64 { return s.RequestsSwapped })
	writeCounterFloatRows(&b, "fak_serving_request_success_total", "Worker successful request counter.", rows, func(s ServingMetricsSnapshot) float64 { return s.RequestSuccesses })
	writeCounterFloatRows(&b, "fak_serving_prefix_cache_queries_total", "Worker prefix-cache query counter.", rows, func(s ServingMetricsSnapshot) float64 { return s.PrefixQueries })
	writeCounterFloatRows(&b, "fak_serving_prefix_cache_hits_total", "Worker prefix-cache hit counter.", rows, func(s ServingMetricsSnapshot) float64 { return s.PrefixHits })
	writeOptionalGaugeRows(&b, "fak_serving_prefix_cache_hit_ratio", "Directly-reported prefix/radix cache-hit ratio (0..1).", rows, func(s ServingMetricsSnapshot) *float64 { return s.PrefixCacheHitRatio }, false)
	writeOptionalGaugeRows(&b, "fak_serving_worker_active_decode_blocks", "P/D worker active decode KV blocks reported by the ridden control plane.", rows, func(s ServingMetricsSnapshot) *float64 { return s.ActiveDecodeBlocks }, true)
	writeOptionalGaugeRows(&b, "fak_serving_worker_active_prefill_tokens", "P/D worker queued or active prefill tokens reported by the ridden control plane.", rows, func(s ServingMetricsSnapshot) *float64 { return s.ActivePrefillTokens }, true)
	return b.String()
}

func sortedServingSnapshots(rows []ServingMetricsSnapshot) []ServingMetricsSnapshot {
	out := append([]ServingMetricsSnapshot(nil), rows...)
	sort.Slice(out, func(i, j int) bool {
		a, b := out[i], out[j]
		ak := firstNonEmpty(a.Engine, VLLMEngineID) + "\x00" + firstNonEmpty(a.WorkerID, "vllm") + "\x00" + a.WorkerRole
		bk := firstNonEmpty(b.Engine, VLLMEngineID) + "\x00" + firstNonEmpty(b.WorkerID, "vllm") + "\x00" + b.WorkerRole
		return ak < bk
	})
	return out
}

func servingSnapshotLabels(s ServingMetricsSnapshot, includeRole bool) string {
	labels := `engine="` + promLabel(firstNonEmpty(s.Engine, VLLMEngineID)) + `",worker="` + promLabel(firstNonEmpty(s.WorkerID, "vllm")) + `"`
	if includeRole && s.WorkerRole != "" {
		labels += `,role="` + promLabel(s.WorkerRole) + `"`
	}
	return labels
}

func writeServingSumCountRows(b *strings.Builder, name, help string, rows []ServingMetricsSnapshot, pick func(ServingMetricsSnapshot) metricSumCount) {
	writeHelpType(b, name, help, "summary")
	for _, row := range rows {
		labels := servingSnapshotLabels(row, false)
		v := pick(row)
		fmt.Fprintf(b, "%s_sum{%s} %s\n", name, labels, promFloat(v.Sum))
		fmt.Fprintf(b, "%s_count{%s} %s\n", name, labels, promFloat(v.Count))
	}
}

func writeGaugeRows(b *strings.Builder, name, help string, rows []ServingMetricsSnapshot, pick func(ServingMetricsSnapshot) float64) {
	writeHelpType(b, name, help, "gauge")
	for _, row := range rows {
		fmt.Fprintf(b, "%s{%s} %s\n", name, servingSnapshotLabels(row, false), promFloat(pick(row)))
	}
}

func writeCounterFloatRows(b *strings.Builder, name, help string, rows []ServingMetricsSnapshot, pick func(ServingMetricsSnapshot) float64) {
	writeHelpType(b, name, help, "counter")
	for _, row := range rows {
		fmt.Fprintf(b, "%s{%s} %s\n", name, servingSnapshotLabels(row, false), promFloat(pick(row)))
	}
}

func writeOptionalGaugeRows(b *strings.Builder, name, help string, rows []ServingMetricsSnapshot, pick func(ServingMetricsSnapshot) *float64, includeRole bool) {
	var any bool
	for _, row := range rows {
		if pick(row) != nil {
			any = true
			break
		}
	}
	if !any {
		return
	}
	writeHelpType(b, name, help, "gauge")
	for _, row := range rows {
		v := pick(row)
		if v == nil {
			continue
		}
		fmt.Fprintf(b, "%s{%s} %s\n", name, servingSnapshotLabels(row, includeRole), promFloat(*v))
	}
}

func writeHelpType(b *strings.Builder, name, help, typ string) {
	b.WriteString("# HELP " + name + " " + help + "\n")
	b.WriteString("# TYPE " + name + " " + typ + "\n")
}

func promFloat(v float64) string {
	return strconv.FormatFloat(v, 'g', -1, 64)
}

func firstNonEmpty(vals ...string) string {
	for _, v := range vals {
		if v != "" {
			return v
		}
	}
	return ""
}

func envDefault(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func intPtrValue(p *int, def int) int {
	if p == nil {
		return def
	}
	return *p
}

func mathModf(v float64) (intPart, frac float64) {
	i := int64(v)
	return float64(i), v - float64(i)
}

// DefaultVLLMEngine is registered under "vllm". It is inert until configured with
// FAK_VLLM_BASE_URL (or replaced in tests via NewVLLMEngine).
var DefaultVLLMEngine = NewVLLMEngine(EnvVLLMConfig())

func init() {
	abi.RegisterEngine(VLLMEngineID, DefaultVLLMEngine)
}

var (
	_ abi.LifecycleEngine = (*VLLMEngine)(nil)
	_ abi.EngineRequest   = (*vllmRequest)(nil)
)
