package engine

// dynamo.go - NVIDIA Dynamo interop for issue #38.
//
// Dynamo is treated as an external serving control plane. fak does not fork or
// import Dynamo internals; it only speaks Dynamo's public OpenAI-compatible
// frontend for generation and Prometheus metrics for observation. The selected
// posture is fak-governs / Dynamo-routes: Dynamo owns P/D worker placement and KV
// transport, while fak supplies the EngineDriver dispatch point plus governance and
// normalized serving telemetry around that ridden pool.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// DynamoEngineID is the registered engine id for a Dynamo-managed P/D pool.
const DynamoEngineID = "dynamo"

// DynamoConfig wires one Dynamo frontend through public surfaces only: the
// OpenAI-compatible /v1 routes for generation and Prometheus /metrics for P/D
// worker observation.
type DynamoConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	WorkerID   string
	MetricsURL string
	Client     *http.Client
}

// EnvDynamoConfig returns the default Dynamo driver configuration. FAK_DYNAMO_BASE_URL
// should point at Dynamo's OpenAI-compatible frontend root, usually http://host:port/v1.
func EnvDynamoConfig() DynamoConfig {
	return DynamoConfig{
		BaseURL:    os.Getenv("FAK_DYNAMO_BASE_URL"),
		Model:      os.Getenv("FAK_DYNAMO_MODEL"),
		APIKey:     os.Getenv("FAK_DYNAMO_API_KEY"),
		WorkerID:   envDefault("FAK_DYNAMO_WORKER_ID", "dynamo"),
		MetricsURL: os.Getenv("FAK_DYNAMO_METRICS_URL"),
	}
}

// DynamoEngine is an abi.EngineDriver/LifecycleEngine adapter for a Dynamo-managed
// P/D serving pool.
type DynamoEngine struct {
	cfg    DynamoConfig
	client *http.Client
}

// NewDynamoEngine builds a Dynamo driver over public Dynamo surfaces.
func NewDynamoEngine(cfg DynamoConfig) *DynamoEngine {
	if cfg.WorkerID == "" {
		cfg.WorkerID = "dynamo"
	}
	client := cfg.Client
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	return &DynamoEngine{cfg: cfg, client: client}
}

// Caps advertises Dynamo ride-mode support: OpenAI-compatible frontend dispatch,
// lifecycle streaming, Dynamo metrics normalization, and the honest whole-prefix
// governance boundary for a ridden engine whose exact KV span API is not exposed.
func (e *DynamoEngine) Caps() []abi.Capability {
	return []abi.Capability{
		"engine.dynamo",
		"engine.openai",
		"engine.dynamo.pd-router",
		"engine.dynamo.metrics",
		"engine.cache.whole-prefix",
		abi.EngineLifecycleCap,
	}
}

// WeightBearing declares that Dynamo dispatch runs a model-forward, not a deterministic tool.
func (e *DynamoEngine) WeightBearing() bool { return true }

// Admit submits one request to Dynamo's OpenAI-compatible frontend with stream=true
// and returns a live request handle. Dynamo owns routing across its P/D pool behind
// that frontend; fak stays in front as the governance plane.
func (e *DynamoEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if strings.TrimSpace(e.cfg.BaseURL) == "" {
		return nil, errors.New("dynamo: FAK_DYNAMO_BASE_URL or DynamoConfig.BaseURL is required")
	}
	endpoint, kind, body, err := e.buildOpenAIRequest(ctx, c)
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
		return nil, fmt.Errorf("dynamo: %s returned %d: %s", kind, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	r := &vllmRequest{
		tokens:   make(chan abi.EngineToken),
		done:     make(chan struct{}),
		cancel:   cancel,
		body:     resp.Body,
		kind:     kind,
		call:     c,
		putCtx:   ctx,
		engine:   DynamoEngineID,
		workerID: e.cfg.WorkerID,
		model:    e.cfg.Model,
	}
	go r.pump(cctx)
	return r, nil
}

// Complete drains the live stream and returns the assembled result.
func (e *DynamoEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
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

func (e *DynamoEngine) buildOpenAIRequest(ctx context.Context, c *abi.ToolCall) (endpoint, kind string, body []byte, err error) {
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

func (e *DynamoEngine) buildChatBody(c *abi.ToolCall, args []byte) ([]byte, error) {
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

func (e *DynamoEngine) buildCompletionsBody(c *abi.ToolCall, args []byte) ([]byte, error) {
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

func (e *DynamoEngine) metricsURL() (string, error) {
	if e.cfg.MetricsURL != "" {
		return e.cfg.MetricsURL, nil
	}
	if e.cfg.BaseURL == "" {
		return "", errors.New("dynamo: FAK_DYNAMO_METRICS_URL or BaseURL is required for metrics scrape")
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

// ScrapeServingMetrics reads Dynamo's Prometheus endpoint and returns one normalized
// fak_serving_* row per observed worker.
func (e *DynamoEngine) ScrapeServingMetrics(ctx context.Context) (DynamoServingMetrics, error) {
	metricsURL, err := e.metricsURL()
	if err != nil {
		return DynamoServingMetrics{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return DynamoServingMetrics{}, err
	}
	if e.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.cfg.APIKey)
	}
	resp, err := e.client.Do(req)
	if err != nil {
		return DynamoServingMetrics{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return DynamoServingMetrics{}, fmt.Errorf("dynamo: metrics returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return DynamoServingMetrics{}, err
	}
	return ParseDynamoPrometheus(e.cfg.WorkerID, string(raw)), nil
}

// DynamoServingMetrics is Dynamo's multi-worker view normalized into fak's L2
// serving schema.
type DynamoServingMetrics struct {
	Rows []ServingMetricsSnapshot
}

func (m DynamoServingMetrics) Prometheus() string {
	return ServingMetricsSnapshots(m.Rows).Prometheus()
}

// ParseDynamoPrometheus maps Dynamo public Prometheus metrics into fak_serving_*.
// It accepts current Dynamo frontend/worker names and a few stable label aliases so
// the adapter is tolerant of deployment relabeling while still ignoring unknown
// Dynamo-only metrics.
func ParseDynamoPrometheus(defaultWorker, text string) DynamoServingMetrics {
	rows := map[string]*ServingMetricsSnapshot{}
	type avg struct {
		sum   float64
		count float64
	}
	prefixRatio := map[string]avg{}

	rowFor := func(s promMetricSample) *ServingMetricsSnapshot {
		worker := firstNonEmpty(s.labels["worker"], s.labels["worker_id"], s.labels["worker_name"], s.labels["instance"], s.labels["pod"], defaultWorker, "dynamo")
		role := dynamoRole(firstNonEmpty(s.labels["role"], s.labels["worker_type"], s.labels["component"], s.labels["dynamo_component"], s.labels["worker_role"]))
		key := worker + "\x00" + role
		row := rows[key]
		if row == nil {
			row = &ServingMetricsSnapshot{Engine: DynamoEngineID, WorkerID: worker, WorkerRole: role}
			rows[key] = row
		}
		return row
	}

	for _, line := range strings.Split(text, "\n") {
		s, ok := parsePromMetricSample(line)
		if !ok {
			continue
		}
		row := rowFor(s)
		switch s.name {
		case "dynamo_frontend_worker_last_time_to_first_token_seconds":
			row.TTFT.Sum += s.value
			row.TTFT.Count++
		case "dynamo_frontend_worker_last_inter_token_latency_seconds":
			row.ITL.Sum += s.value
			row.ITL.Count++
			row.TPOT.Sum += s.value
			row.TPOT.Count++
		case "dynamo_frontend_worker_active_decode_blocks":
			addServingGauge(&row.ActiveDecodeBlocks, s.value)
			row.RequestsRunning += s.value
		case "dynamo_frontend_worker_active_prefill_tokens":
			addServingGauge(&row.ActivePrefillTokens, s.value)
			row.RequestsWaiting += s.value
		case "dynamo_component_inflight_requests":
			row.RequestsRunning += s.value
		case "dynamo_component_requests_total":
			row.RequestSuccesses += s.value
		case "dynamo_component_request_duration_seconds_sum":
			row.Queue.Sum += s.value
		case "dynamo_component_request_duration_seconds_count":
			row.Queue.Count += s.value
		case "dynamo_component_router_kv_hit_rate_sum":
			a := prefixRatio[servingSnapshotKey(*row)]
			a.sum += s.value
			prefixRatio[servingSnapshotKey(*row)] = a
		case "dynamo_component_router_kv_hit_rate_count":
			a := prefixRatio[servingSnapshotKey(*row)]
			a.count += s.value
			prefixRatio[servingSnapshotKey(*row)] = a
		case "dynamo_frontend_router_kv_hit_rate":
			v := s.value
			if v > 1 && v <= 100 {
				v = v / 100
			}
			row.PrefixCacheHitRatio = &v
		case "dynamo_frontend_router_queue_pending_requests":
			row.RequestsWaiting += s.value
		}
	}

	out := make([]ServingMetricsSnapshot, 0, len(rows))
	for _, row := range rows {
		if a := prefixRatio[servingSnapshotKey(*row)]; a.count > 0 {
			v := a.sum / a.count
			if v > 1 && v <= 100 {
				v = v / 100
			}
			row.PrefixCacheHitRatio = &v
		}
		out = append(out, *row)
	}
	out = sortedServingSnapshots(out)
	return DynamoServingMetrics{Rows: out}
}

func addServingGauge(dst **float64, value float64) {
	if *dst == nil {
		v := 0.0
		*dst = &v
	}
	**dst += value
}

func servingSnapshotKey(row ServingMetricsSnapshot) string {
	return firstNonEmpty(row.Engine, VLLMEngineID) + "\x00" + firstNonEmpty(row.WorkerID, "vllm") + "\x00" + row.WorkerRole
}

func dynamoRole(raw string) string {
	switch strings.ToLower(strings.TrimSpace(raw)) {
	case "prefill", "prefill_worker", "prefill-worker", "prefillworker":
		return "prefill"
	case "decode", "backend", "decode_worker", "decode-worker", "decodeworker":
		return "decode"
	case "frontend", "front-end", "router":
		return "router"
	default:
		return strings.TrimSpace(raw)
	}
}

// DefaultDynamoEngine is registered under "dynamo". It is inert until configured
// with FAK_DYNAMO_BASE_URL or replaced in tests via NewDynamoEngine.
var DefaultDynamoEngine = NewDynamoEngine(EnvDynamoConfig())

func init() {
	abi.RegisterEngine(DynamoEngineID, DefaultDynamoEngine)
}

var (
	_ abi.LifecycleEngine = (*DynamoEngine)(nil)
)
