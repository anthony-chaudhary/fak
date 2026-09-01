package engine

// mlx.go — the MLX ride-adapter (issue #2724, epic #2722).
//
// MLX is Apple Silicon's current SOTA serving path: `mlx-lm`'s server and
// `vllm-mlx` both expose an OpenAI-compatible HTTP frontend (/v1/chat/completions +
// /v1/completions), and vllm-mlx additionally exposes vLLM-format Prometheus
// metrics. This adapter FRONTS such a locally-running server the same "ride, don't
// fork" way fak already fronts vLLM/SGLang/Dynamo/llm-d: it speaks only the public
// OpenAI + Prometheus surfaces and never links MLX internals or reimplements the
// Metal kernel (that native Track B is the epic's other children, out of scope here).
//
// The three lanes mirror the sibling adapters:
//
//   - Generation: buildOpenAIRequest lowers the tool call onto /v1/chat/completions
//     or /v1/completions with stream=true, and the shared vllmRequest pump decodes
//     the OpenAI SSE — the exact path Dynamo/llm-d reuse, because mlx-lm/vllm-mlx
//     speak the same wire.
//   - Metrics: vllm-mlx exposes vLLM-format `vllm:*` metrics, so ParseMLXPrometheus
//     reuses ParseVLLMPrometheus and re-tags the rows engine="mlx" (the same
//     wrap-and-retag llm-d uses). A server that exposes no Prometheus surface
//     (plain mlx-lm's minimal frontend) simply yields an empty snapshot — no
//     fabricated numbers, matching the issue's "document gaps rather than fabricate".
//   - Cache-control honesty fence: MLX's public control plane exposes at most a
//     whole-prefix cache reset (vllm-mlx's prefix-caching stats), never fak's
//     bit-exact middle-span evict, so this advertises engine.cache.whole-prefix and
//     enginecache.SupportsExactSpan stays false for it (the default for any engine
//     not specially cased, exactly as Dynamo and llm-d rely on) — the identical
//     fence the other three ride adapters keep.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
)

// MLXEngineID is the registered engine id for an MLX-fronting server (mlx-lm or
// vllm-mlx) reached over its public OpenAI-compatible frontend.
const MLXEngineID = "mlx"

// MLXConfig wires one MLX server through public surfaces only: the
// OpenAI-compatible /v1 routes for generation and Prometheus /metrics for worker
// observation. It deliberately does not rely on an MLX source patch or an
// in-process Python API.
type MLXConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	WorkerID   string
	MetricsURL string
	Client     *http.Client
}

// EnvMLXConfig returns the default MLX driver configuration. FAK_MLX_BASE_URL should
// point at the server's OpenAI-compatible root, usually http://host:port/v1 (mlx-lm
// defaults to :8080, vllm-mlx to :8000 — swap host/port to match your launch).
func EnvMLXConfig() MLXConfig {
	return MLXConfig{
		BaseURL:    os.Getenv("FAK_MLX_BASE_URL"),
		Model:      os.Getenv("FAK_MLX_MODEL"),
		APIKey:     os.Getenv("FAK_MLX_API_KEY"),
		WorkerID:   envDefault("FAK_MLX_WORKER_ID", "mlx"),
		MetricsURL: os.Getenv("FAK_MLX_METRICS_URL"),
	}
}

type mlxEngineState struct {
	cfg    MLXConfig
	client *http.Client
}

// MLXEngine is an abi.EngineDriver/LifecycleEngine adapter for an MLX-fronting
// server on Apple Silicon.
type MLXEngine struct {
	oneShotLifecycle[mlxEngineState]
}

// NewMLXEngine builds an MLX driver over public mlx-lm/vllm-mlx surfaces.
func NewMLXEngine(cfg MLXConfig) *MLXEngine {
	cfg.WorkerID = defaultWorkerID(cfg.WorkerID, "mlx")
	client := defaultHTTPClient(cfg.Client)
	return &MLXEngine{oneShotLifecycle: newOneShotLifecycle(mlxEngineState{cfg: cfg, client: client})}
}

// Caps advertises MLX ride-mode support: OpenAI-compatible frontend dispatch,
// lifecycle streaming, MLX metrics normalization, and the honest whole-prefix
// governance boundary for a ridden engine whose exact KV span API is not exposed.
func (e *MLXEngine) Caps() []abi.Capability {
	return []abi.Capability{
		"engine.mlx",
		"engine.openai",
		"engine.mlx.metrics",
		"engine.cache.whole-prefix",
		abi.EngineLifecycleCap,
	}
}

// WeightBearing declares that MLX dispatch runs a model-forward, not a deterministic tool.
func (e *MLXEngine) WeightBearing() bool { return true }

// Admit submits one request to the MLX server's OpenAI-compatible frontend with
// stream=true and returns a live request handle. MLX owns Metal execution behind
// that frontend; fak stays in front as the governance plane, recording the served
// result under engine="mlx".
func (e mlxEngineState) admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if strings.TrimSpace(e.cfg.BaseURL) == "" {
		return nil, errors.New("mlx: FAK_MLX_BASE_URL or MLXConfig.BaseURL is required")
	}
	endpoint, kind, body, err := buildOpenAIRequest(ctx, e.cfg.BaseURL, e.cfg.Model, c)
	if err != nil {
		return nil, err
	}
	cctx, cancel, resp, err := postStreamingRequest(ctx, e.client, endpoint, e.cfg.APIKey, body, "mlx", kind)
	if err != nil {
		return nil, err
	}
	r := &vllmRequest{
		tokens:   make(chan abi.EngineToken),
		done:     make(chan struct{}),
		cancel:   cancel,
		body:     resp.Body,
		kind:     kind,
		call:     c,
		putCtx:   ctx,
		engine:   MLXEngineID,
		workerID: e.cfg.WorkerID,
		model:    e.cfg.Model,
	}
	go r.pump(cctx)
	return r, nil
}

func (e *MLXEngine) metricsURL() (string, error) {
	return deriveMetricsURL(e.state.cfg.MetricsURL, e.state.cfg.BaseURL, "mlx", "FAK_MLX_METRICS_URL", true)
}

// ScrapeServingMetrics reads the MLX server's Prometheus endpoint and normalizes it
// under engine="mlx" in fak's serving schema. vllm-mlx exposes the vLLM-format
// surface ParseVLLMPrometheus already reads; a plain mlx-lm server that exposes no
// Prometheus endpoint returns a non-200 here (an honest error), and a server that
// exposes an empty/foreign surface normalizes to an empty snapshot rather than
// fabricated counters.
func (e *MLXEngine) ScrapeServingMetrics(ctx context.Context) (ServingMetricsSnapshot, error) {
	metricsURL, err := e.metricsURL()
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	if e.state.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.state.cfg.APIKey)
	}
	resp, err := e.state.client.Do(req)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return ServingMetricsSnapshot{}, fmt.Errorf("mlx: metrics returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := readHTTPAdapterResponse(resp.Body, maxHTTPAdapterResponseBytes)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	return ParseMLXPrometheus(e.state.cfg.WorkerID, string(raw)), nil
}

// ParseMLXPrometheus maps vLLM-format worker metrics from a vllm-mlx deployment into
// fak's normalized serving schema, preserving mlx as the engine identity. It wraps
// ParseVLLMPrometheus because vllm-mlx exposes the same `vllm:*` metric names; a
// server that exposes none of them normalizes to an empty (engine="mlx") snapshot,
// never a fabricated metric.
func ParseMLXPrometheus(workerID, text string) ServingMetricsSnapshot {
	snap := ParseVLLMPrometheus(strmatch.FirstNonEmpty(workerID, MLXEngineID), text)
	snap.Engine = MLXEngineID
	if snap.WorkerID == "" || snap.WorkerID == VLLMEngineID {
		snap.WorkerID = strmatch.FirstNonEmpty(workerID, MLXEngineID)
	}
	return snap
}

// DefaultMLXEngine is registered under "mlx". It is inert until configured with
// FAK_MLX_BASE_URL (or replaced in tests via NewMLXEngine).
var DefaultMLXEngine = NewMLXEngine(EnvMLXConfig())

func init() {
	abi.RegisterEngine(MLXEngineID, DefaultMLXEngine)
}

var (
	_ abi.LifecycleEngine = (*MLXEngine)(nil)
)
