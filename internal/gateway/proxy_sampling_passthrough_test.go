package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// TestProxiedChatSamplingReachesUpstream pins what a proxied OpenAI chat turn puts on
// the upstream wire. A client that omits temperature must not be served greedy
// decoding the gateway invented: Halo llama.cpp upstreams then repeated identical
// tool-call bundles. Explicit sampling and repetition penalties must arrive intact,
// as must the llama.cpp/vLLM extensions top_k, min_p and chat_template_kwargs that
// Halo Qwen's recommended sampling and enable_thinking ride on.
func TestProxiedChatSamplingReachesUpstream(t *testing.T) {
	cases := []struct {
		name   string
		fields string
		want   map[string]any
		absent []string
	}{
		{name: "omitted", fields: ``, absent: []string{"temperature", "presence_penalty", "frequency_penalty", "top_k", "min_p", "chat_template_kwargs"}},
		{name: "explicit-zero", fields: `"temperature":0,`, want: map[string]any{"temperature": 0.0}},
		{name: "explicit-sampling", fields: `"temperature":0.7,"top_p":0.8,"presence_penalty":1.5,"frequency_penalty":0.25,`,
			want: map[string]any{"temperature": 0.7, "top_p": 0.8, "presence_penalty": 1.5, "frequency_penalty": 0.25}},
		{name: "qwen-extensions", fields: `"top_k":20,"min_p":0.05,"chat_template_kwargs":{"enable_thinking":true},`,
			want: map[string]any{"top_k": 20.0, "min_p": 0.05, "chat_template_kwargs": map[string]any{"enable_thinking": true}}},
	}
	for _, stream := range []bool{false, true} {
		for _, tc := range cases {
			t.Run(fmt.Sprintf("%s/stream=%t", tc.name, stream), func(t *testing.T) {
				var mu sync.Mutex
				var got map[string]any
				up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					raw, _ := io.ReadAll(r.Body)
					var body map[string]any
					if err := json.Unmarshal(raw, &body); err != nil {
						t.Errorf("decode upstream body: %v", err)
					}
					mu.Lock()
					got = body
					mu.Unlock()
					usage := `"usage":{"prompt_tokens":3,"completion_tokens":1,"total_tokens":4}`
					if s, _ := body["stream"].(bool); s {
						w.Header().Set("Content-Type", "text/event-stream")
						fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":null}]}`+"\n\n")
						fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],`+usage+`}`+"\n\n")
						fmt.Fprint(w, "data: [DONE]\n\n")
						return
					}
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprint(w, `{"choices":[{"index":0,"finish_reason":"stop","message":{"role":"assistant","content":"ok"}}],`+usage+`}`)
				}))
				defer up.Close()

				srv := newTestServer(t)
				srv.planner = agent.NewHTTPPlanner(up.URL+"/v1", "qwen", "")
				ts := httptest.NewServer(srv.Handler())
				defer ts.Close()

				req := fmt.Sprintf(`{"model":"qwen",%s"stream":%t,"messages":[{"role":"user","content":"hi"}]}`, tc.fields, stream)
				resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(req))
				if err != nil {
					t.Fatalf("post: %v", err)
				}
				_, _ = io.ReadAll(resp.Body)
				resp.Body.Close()
				if resp.StatusCode != http.StatusOK {
					t.Fatalf("chat status = %d, want 200", resp.StatusCode)
				}
				mu.Lock()
				defer mu.Unlock()
				if got == nil {
					t.Fatal("upstream never received the turn")
				}
				for _, k := range tc.absent {
					if v, ok := got[k]; ok {
						t.Errorf("upstream %s = %v, want absent", k, v)
					}
				}
				for k, want := range tc.want {
					if v, ok := got[k]; !ok || !reflect.DeepEqual(v, want) {
						t.Errorf("upstream %s = %v (present=%t), want %v", k, v, ok, want)
					}
				}
			})
		}
	}
}
