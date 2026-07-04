package engine

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestVLLMLiveWorkerSmoke is the env-gated LIVE witness for issue #40 scope
// item 1 ("Completion over OpenAI HTTP — drive a live vLLM V1 worker's
// /v1/chat/completions and /completions ... via the adapter") plus the live
// half of scope item 3 (Prometheus scrape → the fak_serving_* L2 schema).
//
// The GPU-free tests front a MOCK upstream: TestVLLMHTTPAdapterStreamsChatAndCompletions
// (engine) and the gateway drop-in twins (TestChatProxyFrontsVLLMAndSGLangServedToolCalls,
// TestMessagesWireFrontsVLLMServedToolCalls) prove the protocol lowering without a
// GPU. What they cannot cover is acceptance item 2's "through a **live** vLLM-V1
// worker ... real worker, not a mock" — and until now nothing exercised
// EnvVLLMConfig()/VLLMEngine against a real endpoint at all, so that half was
// "would need measurement on real hardware" with no runnable path.
//
// This test IS that path. Point it at a real vLLM V1 worker and it drives the
// adapter end-to-end over the worker's public OpenAI + Prometheus surfaces:
//
//	FAK_VLLM_BASE_URL=http://<host>:8000/v1 \
//	FAK_VLLM_MODEL=<served-model-id> \
//	[FAK_VLLM_METRICS_URL=http://<host>:8000/metrics] \
//	[FAK_VLLM_API_KEY=<key>] \
//	go test ./internal/engine -run TestVLLMLiveWorkerSmoke -count=1 -v
//
// Without FAK_VLLM_BASE_URL / FAK_VLLM_MODEL it SKIPS: GPU-free CI proves it
// compiles and the gate is wired; it does NOT prove live serving (that needs a
// GPU worker this dev box does not have). This is the "prototype behind an
// explicit gate" the gen/second-next frame calls for — never a fabricated pass.
//
// Generation frame (gen/second-next — architectural option, not default exposure):
//   - Promotion evidence (toward gen/next): this test PASSING against a real
//     worker is the live half of acceptance item 2 + scope item 3; together with
//     the sibling parity-overhead bench (acceptance item 6, still GPU-gated) it
//     clears the last hardware-gated items on the ride-vLLM path.
//   - Demotion/retirement evidence: if the Track-B native engine ships the base
//     serving items (continuous batching, paged KV, prefix cache) first, the
//     ride's stated value — "serve many GPU nodes before the native engine is
//     ready" — collapses; this gate should be retired with the adapter, not
//     promoted.
//   - Invalidating assumption: it assumes the served worker speaks the public
//     OpenAI /chat/completions + /completions SSE contract and Prometheus surface
//     the adapter lowers to. If a vLLM release changes that wire, this test is the
//     intended failure site — the live contract must be re-witnessed, not assumed.
func TestVLLMLiveWorkerSmoke(t *testing.T) {
	cfg := EnvVLLMConfig()
	if strings.TrimSpace(cfg.BaseURL) == "" || strings.TrimSpace(cfg.Model) == "" {
		t.Skip("live vLLM smoke: set FAK_VLLM_BASE_URL and FAK_VLLM_MODEL to a real vLLM V1 worker; skipped GPU-free")
	}
	e := NewVLLMEngine(cfg)

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	// Scope item 1: /v1/chat/completions through the adapter against the live worker.
	chat, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(
			`{"messages":[{"role":"user","content":"Reply with the single word: ok"}],"max_tokens":16,"temperature":0}`)},
	})
	if err != nil {
		t.Fatalf("live chat Complete: %v", err)
	}
	assertLiveVLLMResult(t, ctx, chat, "chat")

	// Scope item 1: /v1/completions through the same adapter against the live worker.
	comp, err := e.Complete(ctx, &abi.ToolCall{
		Tool: "completions",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(
			`{"prompt":"Reply with the single word: ok","max_tokens":16,"temperature":0}`)},
		Meta: map[string]string{"openai_endpoint": "completions"},
	})
	if err != nil {
		t.Fatalf("live completions Complete: %v", err)
	}
	assertLiveVLLMResult(t, ctx, comp, "completions")

	// Scope item 3 (live half): the worker's Prometheus normalizes into the shared
	// fak_serving_* L2 schema, per-worker labelled. vLLM V1 always exposes /metrics;
	// ScrapeServingMetrics derives it from BaseURL when FAK_VLLM_METRICS_URL is unset.
	snap, err := e.ScrapeServingMetrics(ctx)
	if err != nil {
		t.Fatalf("live ScrapeServingMetrics: %v", err)
	}
	prom := snap.Prometheus()
	if !strings.Contains(prom, `engine="vllm"`) || !strings.Contains(prom, "fak_serving_") {
		t.Fatalf("live metrics did not normalize into the fak_serving_* schema:\n%s", prom)
	}
}

// assertLiveVLLMResult asserts a live Complete returned an adapter-tagged,
// non-empty assembled result — the same Result contract the GPU-free
// assertVLLMResult pins, minus the fixed mock text/token counts a real worker
// cannot promise.
func assertLiveVLLMResult(t *testing.T, ctx context.Context, res *abi.Result, endpoint string) {
	t.Helper()
	if res == nil || res.Status != abi.StatusOK {
		t.Fatalf("live %s result = %+v, want StatusOK", endpoint, res)
	}
	if res.Meta["engine"] != VLLMEngineID || res.Meta["endpoint"] != endpoint {
		t.Fatalf("live %s result not adapter-tagged: %+v", endpoint, res.Meta)
	}
	body := res.Payload.Inline
	if res.Payload.Kind != abi.RefInline {
		resolver := abi.ActiveResolver()
		if resolver == nil {
			t.Fatalf("live %s payload was %v but ActiveResolver is nil", endpoint, res.Payload.Kind)
		}
		b, err := resolver.Resolve(ctx, res.Payload)
		if err != nil {
			t.Fatalf("resolve live %s payload: %v", endpoint, err)
		}
		body = b
	}
	if !strings.Contains(string(body), `"text":"`) || len(strings.TrimSpace(string(body))) == 0 {
		t.Fatalf("live %s payload missing assembled text: %s", endpoint, body)
	}
}
