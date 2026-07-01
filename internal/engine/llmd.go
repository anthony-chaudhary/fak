package engine

// llmd.go - llm-d interop.
//
// llm-d is treated as a ridden Kubernetes serving stack: llm-d owns the Gateway
// API / EPP routing layer, vLLM workers, P/D placement, and KV/cache policy behind
// its public frontend. fak stays in front as the governance plane and exposes a
// named EngineDriver so route manifests and syscall dispatch can select "llm-d"
// directly instead of pretending the deployment is only a raw vLLM worker.

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// LLMDEngineID is the registered engine id for an llm-d managed serving pool.
const LLMDEngineID = "llm-d"

// LLMDConfig wires one llm-d OpenAI-compatible frontend through public surfaces
// only: /v1 routes for generation, optional Prometheus /metrics for vLLM worker
// observation, and optional decoded vLLM KV events for residency.
type LLMDConfig struct {
	BaseURL    string
	Model      string
	APIKey     string
	WorkerID   string
	MetricsURL string
	Client     *http.Client

	CacheRecorder *CacheEventRecorder
	Residency     *PrefixResidencyIndex
	KVEvents      VLLMKVEventSource
}

// EnvLLMDConfig returns the default llm-d driver configuration. FAK_LLMD_BASE_URL
// should point at the llm-d Gateway API frontend's OpenAI-compatible /v1 root.
// FAK_LLM_D_* aliases are accepted because the upstream project name includes a
// hyphen, while environment variables cannot.
func EnvLLMDConfig() LLMDConfig {
	return LLMDConfig{
		BaseURL:    envFirst("FAK_LLMD_BASE_URL", "FAK_LLM_D_BASE_URL"),
		Model:      envFirst("FAK_LLMD_MODEL", "FAK_LLM_D_MODEL"),
		APIKey:     envFirst("FAK_LLMD_API_KEY", "FAK_LLM_D_API_KEY"),
		WorkerID:   firstNonEmpty(envFirst("FAK_LLMD_WORKER_ID", "FAK_LLM_D_WORKER_ID"), LLMDEngineID),
		MetricsURL: envFirst("FAK_LLMD_METRICS_URL", "FAK_LLM_D_METRICS_URL"),
	}
}

// LLMDEngine is an abi.EngineDriver/LifecycleEngine adapter for an llm-d managed
// fleet. It reuses the vLLM-compatible request and metrics lowerings because llm-d
// frontends route OpenAI-compatible requests to vLLM workers.
type LLMDEngine struct {
	cfg  LLMDConfig
	vllm *VLLMEngine
}

// NewLLMDEngine builds an llm-d driver over public llm-d/vLLM-compatible surfaces.
func NewLLMDEngine(cfg LLMDConfig) *LLMDEngine {
	if cfg.WorkerID == "" {
		cfg.WorkerID = LLMDEngineID
	}
	vllm := NewVLLMEngine(VLLMConfig{
		BaseURL:       cfg.BaseURL,
		Model:         cfg.Model,
		APIKey:        cfg.APIKey,
		WorkerID:      cfg.WorkerID,
		MetricsURL:    cfg.MetricsURL,
		Client:        cfg.Client,
		CacheRecorder: cfg.CacheRecorder,
		Residency:     cfg.Residency,
		KVEvents:      cfg.KVEvents,
	})
	return &LLMDEngine{cfg: cfg, vllm: vllm}
}

// Caps advertises llm-d ride-mode support: OpenAI-compatible frontend dispatch,
// lifecycle streaming, Gateway API / EPP managed routing, vLLM-worker metrics, KV
// event ingestion, and the honest whole-prefix governance boundary for an
// external engine whose exact KV span API is not exposed.
func (e *LLMDEngine) Caps() []abi.Capability {
	return []abi.Capability{
		"engine.llm-d",
		"engine.openai",
		"engine.llm-d.gateway-api",
		"engine.llm-d.epp-router",
		"engine.vllm.metrics",
		"engine.vllm.kv-events",
		"engine.cache.whole-prefix",
		abi.EngineLifecycleCap,
	}
}

// WeightBearing declares that llm-d dispatch runs a model-forward, not a deterministic tool.
func (e *LLMDEngine) WeightBearing() bool { return true }

// Admit submits one request to the llm-d OpenAI-compatible frontend with
// stream=true and returns a live request handle. llm-d owns placement behind that
// frontend; fak records the served result under engine="llm-d".
func (e *LLMDEngine) Admit(ctx context.Context, c *abi.ToolCall) (abi.EngineRequest, error) {
	if strings.TrimSpace(e.cfg.BaseURL) == "" {
		return nil, errors.New("llm-d: FAK_LLMD_BASE_URL or LLMDConfig.BaseURL is required")
	}
	endpoint, kind, body, err := buildOpenAIRequest(ctx, e.cfg.BaseURL, e.cfg.Model, c)
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
	resp, err := e.vllm.client.Do(req)
	if err != nil {
		cancel()
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		_ = resp.Body.Close()
		cancel()
		return nil, fmt.Errorf("llm-d: %s returned %d: %s", kind, resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	r := &vllmRequest{
		tokens:   make(chan abi.EngineToken),
		done:     make(chan struct{}),
		cancel:   cancel,
		body:     resp.Body,
		kind:     kind,
		call:     c,
		putCtx:   ctx,
		engine:   LLMDEngineID,
		workerID: e.cfg.WorkerID,
		model:    e.cfg.Model,
	}
	go r.pump(cctx)
	return r, nil
}

// Complete drains the live stream and returns the assembled result.
func (e *LLMDEngine) Complete(ctx context.Context, c *abi.ToolCall) (*abi.Result, error) {
	return completeViaAdmit(ctx, e, c)
}

// RunKVEventSubscription consumes decoded vLLM KV-event batches from the llm-d
// worker plane when a deployment supplies that bridge.
func (e *LLMDEngine) RunKVEventSubscription(ctx context.Context) error {
	return e.vllm.RunKVEventSubscription(ctx)
}

func (e *LLMDEngine) metricsURL() (string, error) {
	return deriveMetricsURL(e.cfg.MetricsURL, e.cfg.BaseURL, "llm-d", "FAK_LLMD_METRICS_URL", true)
}

// ScrapeServingMetrics reads the llm-d/vLLM Prometheus endpoint and normalizes it
// under engine="llm-d" in fak's serving schema.
func (e *LLMDEngine) ScrapeServingMetrics(ctx context.Context) (ServingMetricsSnapshot, error) {
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
	resp, err := e.vllm.client.Do(req)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1024))
		return ServingMetricsSnapshot{}, fmt.Errorf("llm-d: metrics returned %d: %s", resp.StatusCode, strings.TrimSpace(string(raw)))
	}
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return ServingMetricsSnapshot{}, err
	}
	return ParseLLMDPrometheus(e.cfg.WorkerID, string(raw)), nil
}

// ParseLLMDPrometheus maps vLLM-style worker metrics from an llm-d deployment into
// fak's normalized serving schema, preserving llm-d as the engine identity.
func ParseLLMDPrometheus(workerID, text string) ServingMetricsSnapshot {
	snap := ParseVLLMPrometheus(firstNonEmpty(workerID, LLMDEngineID), text)
	snap.Engine = LLMDEngineID
	if snap.WorkerID == "" || snap.WorkerID == VLLMEngineID {
		snap.WorkerID = firstNonEmpty(workerID, LLMDEngineID)
	}
	return snap
}

func envFirst(keys ...string) string {
	for _, key := range keys {
		if v := os.Getenv(key); strings.TrimSpace(v) != "" {
			return v
		}
	}
	return ""
}

// DefaultLLMDEngine is registered under "llm-d". It is inert until configured
// with FAK_LLMD_BASE_URL / FAK_LLM_D_BASE_URL or replaced in tests via NewLLMDEngine.
var DefaultLLMDEngine = NewLLMDEngine(EnvLLMDConfig())

func init() {
	abi.RegisterEngine(LLMDEngineID, DefaultLLMDEngine)
}

var (
	_ abi.LifecycleEngine = (*LLMDEngine)(nil)
)
