package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// llamaTemplateRole500 is the body llama.cpp's server returned for a `role: "narrator"` message
// under the Qwen3.8 template (witnessed 2026-10-09 on Strix Halo, #13784).
const llamaTemplateRole500 = `{"error":{"code":500,"message":"Error: Jinja Exception: Unexpected message role.","type":"server_error"}}`

func TestUpstreamErrorStatus_TemplateRenderIsTyped(t *testing.T) {
	status, code, msg := upstreamErrorStatus(&agent.UpstreamStatusError{Status: http.StatusInternalServerError, Body: llamaTemplateRole500})
	if status != http.StatusBadGateway || code != codeUpstreamTemplateRender {
		t.Fatalf("template 500 = (%d, %q), want (502, %q)", status, code, codeUpstreamTemplateRender)
	}
	if !strings.Contains(msg, "Unexpected message role") {
		t.Fatalf("message %q does not carry the Jinja reason", msg)
	}
	// A secret echoed into a template body is scrubbed, never forwarded.
	_, _, msg = upstreamErrorStatus(&agent.UpstreamStatusError{Status: 500,
		Body: `{"error":{"message":"Jinja Exception: bad role sk-ant-abcdefghijklmnopqrstuvwxyz0123"}}`})
	if strings.Contains(msg, "sk-ant-") {
		t.Fatalf("token-shaped run leaked: %q", msg)
	}
	// A plain 500 keeps the historical untyped envelope.
	if _, code, _ := upstreamErrorStatus(&agent.UpstreamStatusError{Status: 500, Body: `{"error":{"message":"Internal Server Error"}}`}); code != "" {
		t.Fatalf("generic 500 code = %q, want untyped", code)
	}
}

// TestChatTemplateRender500TypedOnBothPaths drives a real gateway against a stub llama that
// answers every request with its template-render 500, on the buffered AND the streamed
// chat-completions paths: each must fail fast (one upstream attempt) with the typed code and
// the Jinja reason. The stub is the test-only trigger for the buffered path — live, fak folds an
// unknown role before llama's template sees it, so only the stream reached the template there.
func TestChatTemplateRender500TypedOnBothPaths(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterAdjudicator(0, toolAdj{})

			var hits int32
			llama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				atomic.AddInt32(&hits, 1)
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusInternalServerError)
				_, _ = io.WriteString(w, llamaTemplateRole500)
			}))
			t.Cleanup(llama.Close)

			srv, err := New(Config{EngineID: "test", Model: "m", BaseURL: llama.URL + "/v1", Provider: "openai", VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			ts := httptest.NewServer(srv.Handler())
			t.Cleanup(ts.Close)

			body, _ := json.Marshal(map[string]any{
				"model":    "m",
				"stream":   stream,
				"messages": []map[string]string{{"role": "narrator", "content": "once upon a time"}},
			})
			start := time.Now()
			resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
			if err != nil {
				t.Fatal(err)
			}
			defer resp.Body.Close()
			raw, _ := io.ReadAll(resp.Body)
			if elapsed := time.Since(start); elapsed > 2*time.Second {
				t.Fatalf("template 500 took %s — want fail-fast", elapsed)
			}
			if resp.StatusCode != http.StatusBadGateway {
				t.Fatalf("status = %d, want 502: %s", resp.StatusCode, raw)
			}
			var env struct {
				Error struct {
					Code    string `json:"code"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if err := json.Unmarshal(raw, &env); err != nil {
				t.Fatalf("decode envelope: %v (%s)", err, raw)
			}
			if env.Error.Code != codeUpstreamTemplateRender || !strings.Contains(env.Error.Message, "Unexpected message role") {
				t.Fatalf("envelope = %s, want code %q carrying the Jinja reason", raw, codeUpstreamTemplateRender)
			}
			if got := atomic.LoadInt32(&hits); got != 1 {
				t.Fatalf("upstream attempts = %d, want exactly 1", got)
			}
		})
	}
}

// TestLiveLlamaBufferedTemplateRender500 is the opt-in LIVE witness for the buffered path
// (#13784). It sends a non-streaming `role: "narrator"` turn straight through the agent planner
// — bypassing the gateway's role normalization, which otherwise keeps the unknown role away from
// llama's template on this path — and checks the surfaced error folds to the typed envelope.
//
//	FAK_LIVE_LLAMA_URL=http://strix:8091/v1 go test ./internal/gateway -run TestLiveLlamaBufferedTemplateRender500 -v
func TestLiveLlamaBufferedTemplateRender500(t *testing.T) {
	base := os.Getenv("FAK_LIVE_LLAMA_URL")
	if base == "" {
		t.Skip("set FAK_LIVE_LLAMA_URL to a llama-server /v1 base to run the live buffered template-500 witness")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := agent.NewHTTPPlanner(base, os.Getenv("FAK_LIVE_LLAMA_MODEL"), "").
		Complete(ctx, []agent.Message{{Role: "narrator", Content: "once upon a time"}}, nil)
	var se *agent.UpstreamStatusError
	if !errors.As(err, &se) {
		t.Fatalf("err = %v, want an *UpstreamStatusError from the template render", err)
	}
	status, code, msg := upstreamErrorStatus(err)
	t.Logf("live buffered: upstream HTTP %d -> client (%d, %q, %q)", se.Status, status, code, msg)
	if code != codeUpstreamTemplateRender {
		t.Fatalf("code = %q, want %q (upstream body %q)", code, codeUpstreamTemplateRender, se.Body)
	}
}
