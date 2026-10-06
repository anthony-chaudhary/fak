package gateway

// fak-test:runtime fast est=1s

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type timedStubPlanner struct {
	comp  *agent.Completion
	delay time.Duration
}

func (p timedStubPlanner) Complete(ctx context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	time.Sleep(p.delay)
	c := *p.comp
	return &c, nil
}
func (timedStubPlanner) Model() string { return "timed-stub" }

func postChatForTimings(t *testing.T, url string) ChatResponse {
	t.Helper()
	body := `{"model":"test-model","messages":[{"role":"user","content":"hi"}]}`
	resp, err := http.Post(url+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("status %d: %s", resp.StatusCode, b)
	}
	var out ChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	return out
}

func metricValue(t *testing.T, url, series string) string {
	t.Helper()
	resp, err := http.Get(url + "/metrics")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	b, _ := io.ReadAll(resp.Body)
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, series+" ") {
			return strings.TrimPrefix(line, series+" ")
		}
	}
	t.Fatalf("series %s absent from /metrics", series)
	return ""
}

func TestNativeChatTurnFillsTTFTAndEchoesTimings(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = timedStubPlanner{delay: 60 * time.Millisecond, comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 100, CompletionTokens: 10, TotalTokens: 110, PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 40}},
		Timings:      agent.NewTimings(100, 40, 10, 0.030, 0.020),
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	out := postChatForTimings(t, ts.URL)
	if out.Timings == nil {
		t.Fatal("native turn response carries no timings")
	}
	if out.Timings.CacheN != 40 || out.Timings.PromptN != 60 || out.Timings.PredictedN != 10 {
		t.Fatalf("timings counts = %+v, want cache_n=40 prompt_n=60 predicted_n=10", *out.Timings)
	}
	if got := metricValue(t, ts.URL, "fak_gateway_inference_ttft_turns_total"); got != "1" {
		t.Fatalf("ttft_turns_total = %s, want 1", got)
	}
	if got := metricValue(t, ts.URL, "fak_gateway_inference_ttft_seconds_count"); got != "1" {
		t.Fatalf("ttft_seconds_count = %s, want 1", got)
	}
	if got := metricValue(t, ts.URL, "fak_gateway_inference_tpot_seconds_count"); got != "1" {
		t.Fatalf("tpot_seconds_count = %s, want 1", got)
	}
}

func TestBufferedChatTurnWithoutTimingsLeavesTTFTUnmeasured(t *testing.T) {
	srv := newTestServer(t)
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "hello"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 100, CompletionTokens: 10, TotalTokens: 110},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	if out := postChatForTimings(t, ts.URL); out.Timings != nil {
		t.Fatalf("buffered turn invented timings: %+v", *out.Timings)
	}
	if got := metricValue(t, ts.URL, "fak_gateway_inference_ttft_turns_total"); got != "0" {
		t.Fatalf("ttft_turns_total = %s, want 0", got)
	}
}

func TestCompletionTTFTIsEverythingBeforeDecode(t *testing.T) {
	tm := agent.NewTimings(100, 0, 10, 0.030, 0.020)
	if got := completionTTFT(tm, 100*time.Millisecond); got != 80*time.Millisecond {
		t.Fatalf("ttft = %v, want 80ms (dur - predicted)", got)
	}
	if got := completionTTFT(tm, 10*time.Millisecond); got != 30*time.Millisecond {
		t.Fatalf("ttft = %v, want prompt_ms fallback 30ms when decode exceeds dur", got)
	}
	if got := completionTTFT(nil, time.Second); got != 0 {
		t.Fatalf("nil timings ttft = %v, want 0", got)
	}
}
