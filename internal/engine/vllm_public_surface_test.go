package engine

import (
	"context"
	"net/url"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/enginecache"
)

// TestVLLMAdapterConstructsOnlyPublicVLLMEndpoints is the executable
// category-error guard for issue #40 Scope item 5 / Acceptance item 5:
//
//	"No vLLM source is forked or patched — only public HTTP + KV-event +
//	 metrics surfaces are touched (asserted in review)."
//
// The driver doc (docs/serving/vllm-v1-adapter.md, §"The three public surfaces")
// states this in prose. Before this test that promise was enforced only by human
// review; an edit that pointed buildOpenAIRequest or deriveMetricsURL at a
// vLLM-internal / forked / debug endpoint (the exact non-goal category error the
// issue forbids) would pass every existing gate silently. This test makes the
// promise fail closed: it drives the adapter's real path constructors and asserts
// every HTTP path it can emit is in the documented public vLLM V1 allowlist and
// sits at or above the public boundary (no /internal, /debug, or path traversal).
//
// Surface ownership (why the allowlist is exactly these three):
//   - /v1/chat/completions, /v1/completions — the OpenAI-compatible generation
//     frontend this adapter lowers onto (buildOpenAIRequest).
//   - /metrics — the Prometheus scrape target (deriveMetricsURL), derived by
//     stripping /v1 or taken from an explicit FAK_VLLM_METRICS_URL override.
//   - The whole-prefix control plane POST /reset_prefix_cache is NOT constructed
//     by this leaf: it is owned and witnessed by internal/enginecache
//     (TestInvalidateVLLMResetsPrefixCache, proofs_witness_test.go). The engine
//     identity that binds that lowering to this driver is pinned by
//     TestVLLMGovernanceResolvesToEngineVLLM; re-asserted minimally below so this
//     guard fails if the two leaves drift apart.
//   - The KV-cache-events surface constructs NO HTTP path at all: it is consumed
//     at the decoded-batch VLLMKVEventSource seam (RunKVEventSubscription), so the
//     adapter structurally cannot reach a private vLLM endpoint through it.
func TestVLLMAdapterConstructsOnlyPublicVLLMEndpoints(t *testing.T) {
	const base = "http://vllm-host:8000/v1"

	// publicVLLMGenAndMetrics is the closed allowlist of HTTP paths THIS leaf is
	// permitted to construct. Adding an entry is the conscious "this is a public
	// vLLM surface" review gate the acceptance item names — made executable here.
	publicVLLMGenAndMetrics := map[string]bool{
		"/v1/chat/completions": true,
		"/v1/completions":      true,
		"/metrics":             true,
	}

	got := map[string]bool{}
	addPath := func(what, raw string) {
		u, err := url.Parse(raw)
		if err != nil {
			t.Fatalf("%s: parse %q: %v", what, raw, err)
		}
		// A ridden engine must never be reached below its public boundary: any of
		// these markers means the adapter is speaking to a forked/internal surface.
		for _, forbidden := range []string{"..", "/internal", "/debug", "/private", "/admin"} {
			if strings.Contains(u.Path, forbidden) {
				t.Fatalf("%s constructs non-public vLLM path %q (contains %q) — #40 item 5 category error", what, u.Path, forbidden)
			}
		}
		got[u.Path] = true
	}

	// 1. Generation frontend — chat route.
	chatEndpoint, chatKind, _, err := buildOpenAIRequest(context.Background(), base, "m", &abi.ToolCall{
		Tool: "chat",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"messages":[{"role":"user","content":"hi"}]}`)},
	})
	if err != nil {
		t.Fatalf("buildOpenAIRequest chat: %v", err)
	}
	if chatKind != "chat" {
		t.Fatalf("chat call routed to kind %q, want chat", chatKind)
	}
	addPath("chat frontend", chatEndpoint)

	// 2. Generation frontend — completions route (must not silently collapse to chat).
	compEndpoint, compKind, _, err := buildOpenAIRequest(context.Background(), base, "m", &abi.ToolCall{
		Tool: "completions",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"prompt":"hi"}`)},
		Meta: map[string]string{"openai_endpoint": "completions"},
	})
	if err != nil {
		t.Fatalf("buildOpenAIRequest completions: %v", err)
	}
	if compKind != "completions" {
		t.Fatalf("completions call routed to kind %q, want completions", compKind)
	}
	addPath("completions frontend", compEndpoint)

	// 3. Prometheus scrape — derived from base (strips /v1) and explicit override.
	derived, err := deriveMetricsURL("", base, "vllm", "FAK_VLLM_METRICS_URL", true)
	if err != nil {
		t.Fatalf("deriveMetricsURL derived: %v", err)
	}
	addPath("metrics (derived)", derived)
	explicit, err := deriveMetricsURL("http://vllm-host:8000/metrics", base, "vllm", "FAK_VLLM_METRICS_URL", true)
	if err != nil {
		t.Fatalf("deriveMetricsURL explicit: %v", err)
	}
	addPath("metrics (explicit)", explicit)

	// Every constructed path must be in the public allowlist (no category error) ...
	for p := range got {
		if !publicVLLMGenAndMetrics[p] {
			t.Fatalf("adapter constructs path %q outside the public vLLM allowlist — #40 item 5 category error", p)
		}
	}
	// ... and every declared public surface must actually be exercised, so a
	// surface silently dropped from the adapter (leaving the allowlist a dead
	// letter) is caught too. Guard and allowlist stay in lockstep.
	for p := range publicVLLMGenAndMetrics {
		if !got[p] {
			t.Fatalf("public vLLM surface %q is declared but no adapter path constructs it: %v", p, sortedKeys(got))
		}
	}

	// The whole-prefix control-plane lowering (POST /reset_prefix_cache, owned by
	// internal/enginecache) binds to THIS driver only if the two leaves agree on
	// engine identity. Pin that binding so a rename on either side fails here.
	if VLLMEngineID != string(enginecache.EngineVLLM) {
		t.Fatalf("adapter engine id %q != enginecache identity %q — the /reset_prefix_cache control path would not bind to this driver", VLLMEngineID, enginecache.EngineVLLM)
	}
}
