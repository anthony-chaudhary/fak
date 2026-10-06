package gateway

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fakeLlamaServer answers chat completions the way llama.cpp's server does: usage plus
// an engine `timings` object (250ms of decode for 5 tokens), on a buffered response and
// on the final usage chunk of a stream. It holds each request past the reported decode
// time so the engine phases fit inside the gateway's wall clock.
func fakeLlamaServer(t *testing.T) *httptest.Server {
	t.Helper()
	const usage = `"usage":{"completion_tokens":5,"prompt_tokens":40,"total_tokens":45,"prompt_tokens_details":{"cached_tokens":30}}`
	const timings = `"timings":{"cache_n":30,"prompt_n":10,"prompt_ms":50,"predicted_n":5,"predicted_ms":250}`
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		time.Sleep(350 * time.Millisecond)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}],"object":"chat.completion.chunk"}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"object":"chat.completion.chunk"}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[],"object":"chat.completion.chunk",`+usage+`,`+timings+`}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hello"}}],"object":"chat.completion",`+usage+`,`+timings+`}`)
	}))
	t.Cleanup(up.Close)
	return up
}

// TestProxiedTurnTimingsFeedLatencySeries proves a proxied /v1/chat/completions
// turn whose upstream reports llama.cpp timings lands in the TTFT/TPOT/prefill/decode
// series end to end, on both the buffered and the streamed client wire. Before, every
// such turn counted as unmeasured, so an always-on appliance scraped ttft_turns_total 0.
func TestProxiedTurnTimingsFeedLatencySeries(t *testing.T) {
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream=%t", stream), func(t *testing.T) {
			srv := newTestServer(t)
			srv.planner = agent.NewHTTPPlanner(fakeLlamaServer(t).URL+"/v1", "qwen", "")
			ts := httptest.NewServer(srv.Handler())
			defer ts.Close()

			req := fmt.Sprintf(`{"model":"qwen","stream":%t,"messages":[{"role":"user","content":"hi"}]}`, stream)
			resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(req))
			if err != nil {
				t.Fatalf("post: %v", err)
			}
			_, _ = io.ReadAll(resp.Body)
			resp.Body.Close()
			if resp.StatusCode != http.StatusOK {
				t.Fatalf("chat status = %d, want 200", resp.StatusCode)
			}

			text := getMetrics(t, ts.URL+"/metrics", "")
			for _, want := range []string{
				"fak_gateway_inference_ttft_turns_total 1\n",
				"fak_gateway_inference_ttft_seconds_count 1\n",
				"fak_gateway_inference_tpot_seconds_count 1\n",
				// decode = 5 completion tokens / the engine's 250ms decode phase.
				"fak_gateway_inference_decode_tokens_per_second 20\n",
			} {
				if !strings.Contains(text, want) {
					t.Fatalf("proxied turn missing %q\n--- metrics ---\n%s", want, proxiedInferenceLines(text))
				}
			}
		})
	}
}

func proxiedInferenceLines(text string) string {
	var b strings.Builder
	for _, line := range strings.Split(text, "\n") {
		if strings.HasPrefix(line, "fak_gateway_inference_") {
			b.WriteString(line + "\n")
		}
	}
	return b.String()
}
