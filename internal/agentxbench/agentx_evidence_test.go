package agentxbench_test

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agentxbench"
)

// wireOf reads a receipt through its JSON wire shape, the form other tools and
// committed witnesses consume.
func wireOf(t *testing.T, r *agentxbench.AgentXReceipt) map[string]any {
	t.Helper()
	b, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("marshal receipt: %v", err)
	}
	var wire map[string]any
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("unmarshal receipt: %v", err)
	}
	return wire
}

// A self-check run never measured an endpoint; its receipt must say so and must
// not read as a verified measurement.
// fak-test:runtime fast est=50ms lane=default
func TestMockReceiptReadsSimulatedOnTheWire(t *testing.T) {
	r, err := agentxbench.NewRunner(nil).Run(context.Background(), agentxbench.Config{
		AgentCount: 2, TurnsPerAgent: 2, MaxTokens: 8, DeterministicSeed: 7, MockExecution: true,
	})
	if err != nil {
		t.Fatalf("mock run: %v", err)
	}
	wire := wireOf(t, r)
	if wire["execution_mode"] != "mock" {
		t.Fatalf("execution_mode = %v, want mock", wire["execution_mode"])
	}
	if wire["validation_status"] == "VERIFIED_PASS" {
		t.Fatalf("mock receipt reads VERIFIED_PASS")
	}
	if wire["upstream_comparable"] != false {
		t.Fatalf("upstream_comparable = %v, want false", wire["upstream_comparable"])
	}
	if hw, _ := wire["hardware"].(string); hw == "GCP A100-SXM4-40GB" {
		t.Fatalf("mock receipt claims hardware %q", hw)
	}
}

// The shape of the committed docs/_witnesses/agentx-gcp-a100-qwen38 receipt:
// no execution mode and every response the self-check sentinel. It must not
// verify.
// fak-test:runtime fast est=10ms lane=default
func TestLegacyReceiptWithoutModeIsUnverifiable(t *testing.T) {
	legacy := `{"schema":"fak.agentx-benchmark-receipt.v1","benchmark_id":"agentx-legacy","hardware":"GCP A100-SXM4-40GB",
	"model":"Qwen3.8-27B-Q4_K_M","engine":"fak-inkernel-cuda","agent_count":1,"turns_per_agent":1,
	"aggregated":{"total_requests":1},
	"requests":[{"request_id":"agent-1-t0","agent_id":"agent-1","turn_index":0,"prompt_tokens":1024,"completion_tokens":2,
	"client_phases":{"queue_wait_ms":5,"agent_execution_ms":200,"evaluation_ms":3,"total_lifecycle_ms":208},
	"interactivity":{"ttft_ms":1150,"normalized_interactivity_tok_per_sec":60},
	"token_timestamps_unix_nano":[1,2],"success":true,"response_text":"Action verified. Tool execution succeeded."}]}`
	var r agentxbench.AgentXReceipt
	if err := json.Unmarshal([]byte(legacy), &r); err != nil {
		t.Fatalf("unmarshal legacy receipt: %v", err)
	}
	errs := agentxbench.ValidateReceipt(&r)
	for _, e := range errs {
		if strings.HasPrefix(e, "EXECUTION_MODE_MISSING:") {
			return
		}
	}
	t.Fatalf("legacy receipt not refused for its missing execution mode: %v", errs)
}

// A live run in which every request failed measured nothing and must not verify.
// fak-test:runtime fast est=50ms lane=default
func TestLiveRunWithNoSuccessfulRequestFails(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer server.Close()
	r, err := agentxbench.NewRunner(server.Client()).Run(context.Background(), agentxbench.Config{
		EndpointURL: server.URL, Model: "m", Engine: "test-engine", Hardware: "test-host",
		AgentCount: 1, TurnsPerAgent: 2, MaxTokens: 4, DeterministicSeed: 1,
	})
	if err != nil {
		t.Fatalf("live run: %v", err)
	}
	if wire := wireOf(t, r); wire["validation_status"] == "VERIFIED_PASS" {
		t.Fatalf("all-failed live run reads VERIFIED_PASS: %v", wire["validation_errors"])
	}
}

// Usage without prompt_tokens_details reports prompt and completion counts but
// says nothing about cache hits, so the receipt must not read as full usage.
// fak-test:runtime fast est=100ms lane=default
func TestUsageWithoutCacheDetailsIsCacheUnreported(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\n")
		fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":1}}\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	defer server.Close()
	r, err := agentxbench.NewRunner(server.Client()).Run(context.Background(), agentxbench.Config{
		EndpointURL: server.URL, Model: "m", Engine: "test-engine", Hardware: "test-host",
		AgentCount: 1, TurnsPerAgent: 1, MaxTokens: 4, DeterministicSeed: 1,
	})
	if err != nil {
		t.Fatalf("live run: %v", err)
	}
	if got := wireOf(t, r)["token_count_source"]; got != "server_usage_cache_unreported" {
		t.Fatalf("token_count_source = %v, want server_usage_cache_unreported", got)
	}
}

// A live turn's token counts and cache hits come from the server's usage block,
// and TTFT starts at the first generated content, not at a role-only frame.
// fak-test:runtime fast est=200ms lane=default
func TestLiveTurnUsesServerUsageAndFirstContentTTFT(t *testing.T) {
	const contentDelay = 40 * time.Millisecond
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		opts, _ := body["stream_options"].(map[string]any)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
		flusher.Flush()
		time.Sleep(contentDelay)
		for _, tok := range []string{"ok", " done"} {
			fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", tok)
			flusher.Flush()
		}
		if opts["include_usage"] == true {
			fmt.Fprint(w, "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":777,\"completion_tokens\":2,\"prompt_tokens_details\":{\"cached_tokens\":512}}}\n\n")
		}
		fmt.Fprint(w, "data: [DONE]\n\n")
		flusher.Flush()
	}))
	defer server.Close()

	r, err := agentxbench.NewRunner(server.Client()).Run(context.Background(), agentxbench.Config{
		EndpointURL: server.URL, Model: "m", Engine: "test-engine", Hardware: "test-host",
		AgentCount: 1, TurnsPerAgent: 2, MaxTokens: 4, DeterministicSeed: 1,
	})
	if err != nil {
		t.Fatalf("live run: %v", err)
	}
	wire := wireOf(t, r)
	if wire["execution_mode"] != "live" || wire["token_count_source"] != "server_usage" {
		t.Fatalf("execution_mode=%v token_count_source=%v, want live/server_usage", wire["execution_mode"], wire["token_count_source"])
	}
	for _, req := range r.Requests {
		if req.PromptTokens != 777 || req.CachedTokens != 512 || req.CompletionTokens != 2 {
			t.Fatalf("%s: prompt=%d cached=%d completion=%d, want 777/512/2 from usage",
				req.RequestID, req.PromptTokens, req.CachedTokens, req.CompletionTokens)
		}
		if req.Interactivity.TTFTMS < float64(contentDelay.Milliseconds()) {
			t.Fatalf("%s: TTFT %.2fms counted the role-only frame (content arrived after %v)", req.RequestID, req.Interactivity.TTFTMS, contentDelay)
		}
	}
}
