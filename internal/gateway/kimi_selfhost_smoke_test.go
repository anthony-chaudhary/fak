package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"
)

// Kimi K2 self-hosted baseline smoke (sibling of deepseek_selfhost_smoke_test.go).
//
// This is the OPTIONAL live half of the self-host baseline described in
// docs/benchmarks/KIMI-SELFHOST-BASELINE-RUNBOOK.md. It exercises the
// OpenAI-compatible upstream contract that `fak serve --provider
// openai-compatible --base-url $KIMI_SELFHOST_BASE_URL` forwards to — the
// gateway's proxy planner posts to `BaseURL + "/chat/completions"`, so this
// smoke drives the same wire against a real vLLM/SGLang/NIM Kimi K2 server
// brought up by scripts/dgx-kimi-serve.sh.
//
// It is KEYLESS from fak's perspective: the test skips cleanly (never fails)
// when KIMI_SELFHOST_BASE_URL is unset, so `go test ./internal/gateway` stays
// green on a box with no model server. When the env var IS set it collects the
// minimum evidence for a "supported self-hosted route":
//
//  1. readiness    — GET  {base}/models returns 200 with a model roster
//  2. non-stream   — POST {base}/chat/completions (stream:false) returns
//     content + a usage block
//  3. streaming    — POST {base}/chat/completions (stream:true) yields SSE
//     deltas terminated by [DONE]
//
// NO throughput/latency headline is claimed here — that requires a real tuned
// baseline per the runbook. This is a wire-readiness witness only.
//
// Note the route split: Kimi K3 is Moonshot's CLOUD API (not self-hostable) and
// is witnessed by internal/agent/kimi_k3_test.go against the Moonshot wire; this
// file witnesses the SELF-HOST K2 wire fak fronts. They are different targets.
const (
	kimiSelfhostBaseEnv  = "KIMI_SELFHOST_BASE_URL"
	kimiSelfhostModelEnv = "KIMI_SELFHOST_MODEL"
	kimiSelfhostKeyEnv   = "KIMI_SELFHOST_API_KEY" // optional; empty ⇒ no auth header
	kimiSelfhostDefModel = "moonshotai/Kimi-K2-Instruct"
)

// kimiSelfhostConfig reads the env-provided upstream shape, or signals skip when
// the base URL is unset. The base URL is used verbatim as the OpenAI-compatible
// root (typically ending in "/v1"), matching how the gateway proxy planner
// treats Config.BaseURL.
func kimiSelfhostConfig(t *testing.T) (base, model, key string) {
	t.Helper()
	base = strings.TrimRight(strings.TrimSpace(os.Getenv(kimiSelfhostBaseEnv)), "/")
	if base == "" {
		t.Skipf("%s unset — skipping the optional live Kimi K2 self-host smoke "+
			"(set it to a vLLM/SGLang OpenAI-compatible base URL, e.g. http://host:8000/v1, to run)",
			kimiSelfhostBaseEnv)
	}
	model = strings.TrimSpace(os.Getenv(kimiSelfhostModelEnv))
	if model == "" {
		model = kimiSelfhostDefModel
	}
	key = strings.TrimSpace(os.Getenv(kimiSelfhostKeyEnv))
	return base, model, key
}

func kimiSelfhostDo(t *testing.T, req *http.Request, key string) *http.Response {
	t.Helper()
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	client := &http.Client{Timeout: 120 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("request %s %s: %v", req.Method, req.URL, err)
	}
	return resp
}

// kimiSelfhostSkipIfOverloaded turns an upstream capacity signal into a clean
// skip rather than a failure. A shared or busy serving node throttling this
// probe (HTTP 429/503, or a gRPC-style "ResourceExhausted" body — the exact
// shape a pooled vLLM/SGLang/NIM endpoint returns under load) is NOT a fak wire
// defect: the wire can only be judged when the upstream actually served the
// request. This keeps a transient capacity 503 from reddening the suite while
// still failing hard on a genuine wire break (4xx contract errors, malformed
// bodies, missing [DONE]).
func kimiSelfhostSkipIfOverloaded(t *testing.T, status int, body []byte) {
	t.Helper()
	if status == http.StatusTooManyRequests || status == http.StatusServiceUnavailable ||
		bytes.Contains(body, []byte("ResourceExhausted")) {
		t.Skipf("upstream capacity throttle (HTTP %d) — not a fak wire defect; "+
			"retry against a dedicated Kimi serving node to witness this rung: %s",
			status, bytes.TrimSpace(body))
	}
}

// TestKimiSelfHostReadiness is the readiness rung: the served engine answers
// /models with the roster fak needs to route. Skips cleanly when no upstream is
// configured.
func TestKimiSelfHostReadiness(t *testing.T) {
	base, model, key := kimiSelfhostConfig(t)

	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, base+"/models", nil)
	if err != nil {
		t.Fatal(err)
	}
	resp := kimiSelfhostDo(t, req, key)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s/models = %d, want 200 (upstream not ready): %s", base, resp.StatusCode, body)
	}

	var roster struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &roster); err != nil {
		t.Fatalf("readiness roster not JSON: %v\n%s", err, body)
	}
	if len(roster.Data) == 0 {
		t.Fatalf("readiness roster is empty (no served model ids): %s", body)
	}
	ids := make([]string, 0, len(roster.Data))
	served := false
	for _, m := range roster.Data {
		ids = append(ids, m.ID)
		if m.ID == model {
			served = true
		}
	}
	// A roster that does not name the configured model is a routing mismatch worth
	// surfacing honestly, not a hard failure — the operator may front an alias.
	if !served {
		t.Logf("readiness OK, but configured model %q is not in the roster %v — "+
			"set %s to a served id before quoting a result", model, ids, kimiSelfhostModelEnv)
	}
	t.Logf("readiness OK: %d model(s) served: %v", len(ids), ids)
}

// TestKimiSelfHostNonStreaming is the non-streaming completion rung: one tiny
// chat completion returns content and an honest usage block.
func TestKimiSelfHostNonStreaming(t *testing.T) {
	base, model, key := kimiSelfhostConfig(t)

	reqBody, _ := json.Marshal(map[string]any{
		"model":    model,
		"messages": []map[string]string{{"role": "user", "content": "Reply with the single word: ready"}},
		// 64, not 16: a Kimi "thinking" variant can spend the budget on
		// reasoning_content and leave message.content empty. 64 gives visible
		// content room while staying a trivially cheap probe.
		"max_tokens":  64,
		"temperature": 0,
		"stream":      false,
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, base+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp := kimiSelfhostDo(t, req, key)
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	kimiSelfhostSkipIfOverloaded(t, resp.StatusCode, body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("non-streaming completion = %d, want 200: %s", resp.StatusCode, body)
	}

	var out struct {
		Choices []struct {
			Message struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"message"`
			FinishReason string `json:"finish_reason"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
			TotalTokens      int `json:"total_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(body, &out); err != nil {
		t.Fatalf("completion not JSON: %v\n%s", err, body)
	}
	if len(out.Choices) == 0 {
		t.Fatalf("non-streaming completion returned no choices: %s", body)
	}
	// A thinking-mode turn is a live wire even when message.content is empty: the
	// tokens land in reasoning_content instead. Treat either as content evidence;
	// only both-empty is a real wire failure.
	msg := out.Choices[0].Message
	if strings.TrimSpace(msg.Content) == "" && strings.TrimSpace(msg.ReasoningContent) == "" {
		t.Fatalf("non-streaming completion returned no content and no reasoning_content: %s", body)
	}
	if strings.TrimSpace(msg.Content) == "" && strings.TrimSpace(msg.ReasoningContent) != "" {
		t.Logf("non-streaming completion produced reasoning_content but empty content — " +
			"thinking-mode consumed the token budget; wire is live (raise max_tokens for visible content)")
	}
	// Usage/counter behavior, documented honestly rather than assumed: a compliant
	// engine reports token counts; a build that omits them is a recorded gap, not a
	// synthesized number.
	if out.Usage.TotalTokens == 0 {
		t.Logf("non-streaming completion OK but usage block is absent/zero — record "+
			"the counter gap honestly in the runbook (prompt=%d completion=%d)",
			out.Usage.PromptTokens, out.Usage.CompletionTokens)
	} else {
		t.Logf("non-streaming completion OK: finish=%q usage prompt=%d completion=%d total=%d",
			out.Choices[0].FinishReason, out.Usage.PromptTokens, out.Usage.CompletionTokens, out.Usage.TotalTokens)
	}
}

// TestKimiSelfHostStreaming is the streaming rung: a stream:true request yields
// incremental SSE deltas terminated by the [DONE] sentinel, assembling non-empty
// content.
func TestKimiSelfHostStreaming(t *testing.T) {
	base, model, key := kimiSelfhostConfig(t)

	reqBody, _ := json.Marshal(map[string]any{
		"model":          model,
		"messages":       []map[string]string{{"role": "user", "content": "Count: one two three"}},
		"max_tokens":     24,
		"temperature":    0,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	})
	req, err := http.NewRequestWithContext(context.Background(), http.MethodPost, base+"/chat/completions", bytes.NewReader(reqBody))
	if err != nil {
		t.Fatal(err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	resp := kimiSelfhostDo(t, req, key)
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		kimiSelfhostSkipIfOverloaded(t, resp.StatusCode, body)
		t.Fatalf("streaming completion = %d, want 200: %s", resp.StatusCode, body)
	}

	var content strings.Builder
	var deltas int
	var sawDone bool
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		payload := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if payload == "[DONE]" {
			sawDone = true
			break
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
			continue // usage-only or keep-alive frames are not fatal
		}
		for _, c := range chunk.Choices {
			if c.Delta.Content != "" {
				content.WriteString(c.Delta.Content)
				deltas++
			}
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatalf("reading stream: %v", err)
	}
	if !sawDone {
		t.Fatalf("stream never reached [DONE] (%d content deltas so far)", deltas)
	}
	if strings.TrimSpace(content.String()) == "" {
		t.Fatalf("stream reached [DONE] but assembled no content across %d deltas", deltas)
	}
	t.Logf("streaming completion OK: %d content deltas, %d chars assembled", deltas, content.Len())
}
