package gateway

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestMessagesProxyModelPassthrough proves Config.ModelPassthrough on the Anthropic
// /v1/messages wire of a single OpenAI proxy: armed, the client's requested model
// reaches the upstream body on both the buffered and the streamed path, and an
// omitted model still sends the configured --model; unarmed, every request keeps
// the historical bytes and sends --model. This is the Claude Code front door
// contract: a client choosing boostrun.medium must not be silently served
// boostrun.large because the front door was launched with --model boostrun.large.
//
// fak-test:runtime fast est=1s
func TestMessagesProxyModelPassthrough(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	var mu sync.Mutex
	var gotModel string
	var gotStream bool
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var req struct {
			Model  string `json:"model"`
			Stream bool   `json:"stream"`
		}
		if err := json.Unmarshal(raw, &req); err != nil {
			t.Errorf("decode upstream request: %v\n%s", err, raw)
			return
		}
		mu.Lock()
		gotModel = req.Model
		gotStream = req.Stream
		mu.Unlock()
		if req.Stream {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"model":"served","choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}]}`+"\n\n")
			_, _ = io.WriteString(w, `data: {"model":"served","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"served","choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`))
	}))
	defer upstream.Close()

	post := func(t *testing.T, base string, body map[string]any) {
		t.Helper()
		mu.Lock()
		gotModel = ""
		gotStream = false
		mu.Unlock()
		raw, _ := json.Marshal(body)
		resp, err := http.Post(base+"/v1/messages", "application/json", bytes.NewReader(raw))
		if err != nil {
			t.Fatal(err)
		}
		out, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			t.Fatalf("status = %d, want 200: %s", resp.StatusCode, out)
		}
	}
	upstreamModel := func() (string, bool) {
		mu.Lock()
		defer mu.Unlock()
		return gotModel, gotStream
	}
	msg := func(model string, stream bool) map[string]any {
		b := map[string]any{
			"max_tokens": 64,
			"messages":   []map[string]any{{"role": "user", "content": "hi"}},
			"stream":     stream,
		}
		if model != "" {
			b["model"] = model
		}
		return b
	}

	for _, tc := range []struct {
		name        string
		passthrough bool
		model       string
		stream      bool
		want        string
	}{
		{"armed buffered forwards client model", true, "caller-model", false, "caller-model"},
		{"armed streamed forwards client model", true, "caller-model", true, "caller-model"},
		{"armed omitted model sends configured default", true, "", false, "configured-default"},
		{"unarmed buffered keeps configured model", false, "caller-model", false, "configured-default"},
		{"unarmed streamed keeps configured model", false, "caller-model", true, "configured-default"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			srv, err := New(Config{EngineID: "test", Model: "configured-default", BaseURL: upstream.URL, Provider: "openai", VDSO: true, ModelPassthrough: tc.passthrough})
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(srv.Close)
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			post(t, ts.URL, msg(tc.model, tc.stream))
			got, streamed := upstreamModel()
			if got != tc.want {
				t.Errorf("upstream model = %q, want %q", got, tc.want)
			}
			// A streamed case must reach the upstream as a stream, or it proved only the
			// buffered fallback and not plannerSampleOpts on the live streaming path.
			if streamed != tc.stream {
				t.Errorf("upstream stream = %v, want %v", streamed, tc.stream)
			}
		})
	}
}
