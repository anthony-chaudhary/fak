package gateway

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/policy"
)

// streamedUpstreamBody posts one stream:true chat request through the live
// streaming emitter and returns the body the gateway sent upstream.
func streamedUpstreamBody(t *testing.T, srv *Server, upstream *httptest.Server, captured func() []byte, body string) []byte {
	t.Helper()
	srv.planner = agent.NewHTTPPlanner(upstream.URL+"/v1", "looped-cap-model", "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	out, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d: %s", resp.StatusCode, out)
	}
	got := captured()
	if len(got) == 0 {
		t.Fatal("upstream saw no request")
	}
	return got
}

func loopCaptureUpstream(t *testing.T) (*httptest.Server, func() []byte) {
	t.Helper()
	var mu sync.Mutex
	var last []byte
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		mu.Lock()
		last = b
		mu.Unlock()
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"}}],"object":"chat.completion.chunk"}`+"\n\n")
		fmt.Fprint(w, `data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"object":"chat.completion.chunk"}`+"\n\n")
		fmt.Fprint(w, "data: [DONE]\n\n")
	}))
	t.Cleanup(up.Close)
	return up, func() []byte {
		mu.Lock()
		defer mu.Unlock()
		return append([]byte(nil), last...)
	}
}

func upstreamXArgsLoops(t *testing.T, body []byte) (int, bool) {
	t.Helper()
	var wire struct {
		Stream    bool `json:"stream"`
		VLLMXArgs *struct {
			Loops *int `json:"loops"`
		} `json:"vllm_xargs"`
	}
	if err := json.Unmarshal(body, &wire); err != nil {
		t.Fatalf("upstream body %s: %v", body, err)
	}
	if !wire.Stream {
		t.Fatalf("upstream request was not streamed: %s", body)
	}
	if wire.VLLMXArgs == nil || wire.VLLMXArgs.Loops == nil {
		return 0, false
	}
	return *wire.VLLMXArgs.Loops, true
}

// TestStreamedChatForwardsLoopDepth proves a stream:true request on a looped model
// carries the resolved depth upstream as vllm_xargs.loops, as the buffered path
// does, while a non-looped streamed request keeps its pre-seam upstream body.
// fak-test:runtime fast est=300ms lane=default
func TestStreamedChatForwardsLoopDepth(t *testing.T) {
	t.Setenv("FAK_MODEL_CAPABILITIES", `{"looped-cap-model":{"max_loops":6}}`)
	up, captured := loopCaptureUpstream(t)
	const msgs = `"messages":[{"role":"user","content":"hi"}]`

	cases := []struct {
		name string
		body string
		rule *policy.LoopDepthRule
		want int
	}{
		{"explicit loops", `{"model":"looped-cap-model","stream":true,"loops":3,` + msgs + `}`, nil, 3},
		{"vllm_xargs loops", `{"model":"looped-cap-model","stream":true,"vllm_xargs":{"loops":2},` + msgs + `}`, nil, 2},
		{"default full depth", `{"model":"looped-cap-model","stream":true,` + msgs + `}`, nil, 6},
		{"policy clamp", `{"model":"looped-cap-model","stream":true,"loops":5,` + msgs + `}`, &policy.LoopDepthRule{MaxLoopDepth: 4}, 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := newTestServer(t)
			if err := srv.SetLoopDepthRule(tc.rule); err != nil {
				t.Fatal(err)
			}
			body := streamedUpstreamBody(t, srv, up, captured, tc.body)
			got, ok := upstreamXArgsLoops(t, body)
			if !ok || got != tc.want {
				t.Fatalf("upstream vllm_xargs.loops = %d (present=%v), want %d; body %s", got, ok, tc.want, body)
			}
		})
	}

	t.Run("non-looped is byte-identical", func(t *testing.T) {
		plain := streamedUpstreamBody(t, newTestServer(t), up, captured, `{"model":"plain-model","stream":true,`+msgs+`}`)
		if _, ok := upstreamXArgsLoops(t, plain); ok || strings.Contains(string(plain), `"loops"`) {
			t.Fatalf("non-looped streamed request carried loops upstream: %s", plain)
		}
		withLoopsOne := streamedUpstreamBody(t, newTestServer(t), up, captured, `{"model":"plain-model","stream":true,"loops":1,`+msgs+`}`)
		if string(withLoopsOne) != string(plain) {
			t.Fatalf("loops:1 on a non-looped model changed the upstream body:\n got %s\nwant %s", withLoopsOne, plain)
		}
	})
}
