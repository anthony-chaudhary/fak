package gateway

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// timingsRelayUpstream answers /v1/chat/completions like llama-server: the stream
// ends with a choices:[] usage chunk carrying the engine `timings` object, and the
// buffered body carries the same object at top level.
func timingsRelayUpstream(t *testing.T) *httptest.Server {
	t.Helper()
	tail := `"usage":{"completion_tokens":16,"prompt_tokens":20,"total_tokens":36},` +
		`"timings":{"cache_n":16,"prompt_n":4,"prompt_ms":1464.766,"prompt_per_token_ms":366.19,"prompt_per_second":2.73,` +
		`"predicted_n":16,"predicted_ms":1348.4,"predicted_per_token_ms":89.89,"predicted_per_second":11.12,"draft_n":10,"draft_n_accepted":10}`
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if strings.Contains(string(body), `"stream":true`) {
			w.Header().Set("Content-Type", "text/event-stream")
			fmt.Fprint(w, `data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"hello"}}],"object":"chat.completion.chunk"}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[{"finish_reason":"stop","index":0,"delta":{}}],"object":"chat.completion.chunk"}`+"\n\n")
			fmt.Fprint(w, `data: {"choices":[],"object":"chat.completion.chunk",`+tail+`}`+"\n\n")
			fmt.Fprint(w, "data: [DONE]\n\n")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"finish_reason":"stop","index":0,"message":{"role":"assistant","content":"hello"}}],"object":"chat.completion",`+tail+`}`)
	}))
	t.Cleanup(up.Close)
	return up
}

// completeOnlyPlanner hides agent.StreamingPlanner so the gateway takes the
// buffered stream emitter.
type completeOnlyPlanner struct{ agent.Planner }

type relayedTimingsFrame struct {
	Choices []json.RawMessage `json:"choices"`
	Timings *agent.Timings    `json:"timings"`
}

func timingsRelayFrames(t *testing.T, planner agent.Planner, includeUsage bool) []relayedTimingsFrame {
	t.Helper()
	srv := newTestServer(t)
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(ts.Close)
	body := `{"model":"qwen","stream":true,"messages":[{"role":"user","content":"hi"}]}`
	if includeUsage {
		body = `{"model":"qwen","stream":true,"stream_options":{"include_usage":true},"messages":[{"role":"user","content":"hi"}]}`
	}
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatalf("post: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status = %d: %s", resp.StatusCode, b)
	}
	var frames []relayedTimingsFrame
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		payload, ok := strings.CutPrefix(strings.TrimSpace(sc.Text()), "data:")
		if !ok || strings.TrimSpace(payload) == "[DONE]" {
			continue
		}
		var f relayedTimingsFrame
		if err := json.Unmarshal([]byte(payload), &f); err != nil {
			t.Fatalf("frame %q: %v", payload, err)
		}
		frames = append(frames, f)
	}
	return frames
}

// TestStreamedChatRelaysUpstreamTimings proves a proxied llama-server stream keeps
// the engine `timings` object on exactly one terminal SSE chunk (the usage-only
// chunk when opted in, else the finish chunk) on both the live and buffered emitters.
// fak-test:runtime fast est=500ms lane=default
func TestStreamedChatRelaysUpstreamTimings(t *testing.T) {
	up := timingsRelayUpstream(t).URL + "/v1"
	planners := map[string]func() agent.Planner{
		"live":     func() agent.Planner { return agent.NewHTTPPlanner(up, "qwen", "") },
		"buffered": func() agent.Planner { return completeOnlyPlanner{agent.NewHTTPPlanner(up, "qwen", "")} },
	}
	for name, mk := range planners {
		for _, includeUsage := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/include_usage=%v", name, includeUsage), func(t *testing.T) {
				frames := timingsRelayFrames(t, mk(), includeUsage)
				if len(frames) == 0 {
					t.Fatal("no SSE frames")
				}
				var carriers []int
				for i, f := range frames {
					if f.Timings != nil {
						carriers = append(carriers, i)
					}
				}
				if len(carriers) != 1 || carriers[0] != len(frames)-1 {
					t.Fatalf("timings carriers = %v of %d frames, want exactly the last", carriers, len(frames))
				}
				last := frames[len(frames)-1]
				if wantEmpty := includeUsage; wantEmpty != (len(last.Choices) == 0) {
					t.Fatalf("terminal carrier choices = %d, include_usage = %v", len(last.Choices), includeUsage)
				}
				got := *last.Timings
				if got.PromptN != 4 || got.PredictedN != 16 || got.CacheN != 16 || got.DraftN != 10 || got.DraftNAccepted != 10 {
					t.Fatalf("timings = %+v, want prompt_n=4 predicted_n=16 cache_n=16 draft_n=10 draft_n_accepted=10", got)
				}
			})
		}
	}
}
