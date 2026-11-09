package gateway

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=100ms lane=default
func TestServedTokenHistogramsReachMetrics(t *testing.T) {
	srv := newTestServer(t)
	srv.logf = func(string, ...any) {}
	srv.planner = stubPlanner{comp: &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "done"},
		FinishReason: "stop",
		Usage: agent.Usage{
			PromptTokens: 150, CompletionTokens: 40, TotalTokens: 190,
			PromptTokensDetails: &agent.UsageTokenDetails{CachedTokens: 50},
		},
	}}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	check := func(turns int) {
		t.Helper()
		text := getMetrics(t, ts.URL+"/metrics", "")
		lines := strings.Split(text, "\n")
		for _, h := range []struct {
			name string
			sum  int
		}{
			{"input", turns * 100}, // Same normalized uncached axis as the counter.
			{"output", turns * 40},
		} {
			name := "fak_gateway_inference_" + h.name + "_tokens"
			for _, want := range []string{
				"# TYPE " + name + " histogram",
				fmt.Sprintf("%s_count %d", name, turns),
				fmt.Sprintf("%s_sum %d", name, h.sum),
				fmt.Sprintf("%s_bucket{le=\"+Inf\"} %d", name, turns),
				fmt.Sprintf("%s_bucket{le=\"1310.72\"} %d", name, turns),
			} {
				found := false
				for _, line := range lines {
					if line == want {
						found = true
						break
					}
				}
				if !found {
					t.Fatalf("turns=%d: /metrics missing %q", turns, want)
				}
			}
			series := 0
			for _, line := range lines {
				if strings.HasPrefix(line, name+"_bucket{") || strings.HasPrefix(line, name+"_sum ") || strings.HasPrefix(line, name+"_count ") {
					series++
				}
			}
			if series != tokenHistogramBucketCount+3 {
				t.Fatalf("%s series=%d, want bounded %d", name, series, tokenHistogramBucketCount+3)
			}
		}
		for _, want := range []string{
			fmt.Sprintf("fak_gateway_inference_prompt_tokens_total %d\n", turns*100),
			fmt.Sprintf("fak_gateway_inference_completion_tokens_total %d\n", turns*40),
			fmt.Sprintf("fak_gateway_inference_cached_prompt_tokens_total %d\n", turns*50),
			fmt.Sprintf("fak_gateway_inference_e2e_seconds_count %d\n", turns),
			"fak_gateway_inference_ttft_seconds_count 0\n",
		} {
			if !strings.Contains(text, want) {
				t.Fatalf("existing accounting changed: missing %q", want)
			}
		}
	}
	check(0)
	for i := 1; i <= 2; i++ {
		var resp ChatResponse
		code := postJSON(t, ts.URL+"/v1/chat/completions", ChatRequest{
			Model: "test-model", Messages: []agent.Message{{Role: agent.RoleUser, Content: fmt.Sprintf("turn %d", i)}},
		}, &resp)
		if code != http.StatusOK {
			t.Fatalf("chat status=%d, want 200", code)
		}
		check(i)
		check(i) // Scraping must not account another turn.
	}

	t.Run("scaled snapshots ignore omitted counts and own their slices", func(t *testing.T) {
		m := newGatewayMetrics(time.Now())
		m.scaleTokenHistograms(4096)
		m.observeInference(100, 40, 0, 0, "stop", time.Second)
		before := m.inferenceSnapshotData()
		m.observeInference(0, -1, 0, 0, "stop", time.Second)
		m.observeInference(-1, 0, 0, 0, "stop", time.Second)
		m.observeInference(300, 4096, 0, 0, "stop", time.Second)
		m.scaleTokenHistograms(8192) // Populated ladders must not silently change.
		after := m.inferenceSnapshotData()
		if before.inputTokenHist.count != 1 || before.inputTokenHist.sum != 100 || before.inputTokenHist.buckets[7] != 1 {
			t.Fatalf("prior snapshot changed: %+v", before.inputTokenHist)
		}
		if after.inputTokenHist.count != 2 || after.inputTokenHist.sum != 400 || after.outputTokenHist.count != 2 || after.outputTokenHist.sum != 4136 {
			t.Fatalf("omitted counts entered histogram: input=%+v output=%+v", after.inputTokenHist, after.outputTokenHist)
		}
		if after.promptTok != after.inputTokenHist.sum || after.complTok != after.outputTokenHist.sum || after.e2eHist.count != 4 {
			t.Fatalf("counter or latency accounting drift: %+v", after)
		}
		if after.inputTokenHist.edges[99] != 4096 || after.inputTokenHist.buckets[1] != 0 || after.inputTokenHist.buckets[2] != 1 || after.inputTokenHist.buckets[7] != 2 || after.outputTokenHist.buckets[99] != 2 {
			t.Fatalf("scaled cumulative buckets changed: %+v", after)
		}
		after.inputTokenHist.edges[0] = -1
		after.inputTokenHist.buckets[0] = 999
		after.outputTokenHist.edges[0] = -1
		after.outputTokenHist.buckets[0] = 999
		fresh := m.inferenceSnapshotData()
		if fresh.inputTokenHist.edges[0] != 40.96 || fresh.inputTokenHist.buckets[0] != 0 || fresh.outputTokenHist.edges[0] != 40.96 || fresh.outputTokenHist.buckets[0] != 1 {
			t.Fatal("scrape snapshot aliases live token histogram slices")
		}
	})

	t.Run("concurrent observations and scrapes share counter lock", func(t *testing.T) {
		// Exercise lazy initialization, including the synthetic-turn convention.
		m := &gatewayMetrics{}
		m.observeInferenceServedTimedFrom(perfSource{synthetic: true}, localityUnknown, "", 10, 5, 0, 0, "stop", time.Second, 0)
		var wg sync.WaitGroup
		for range 4 {
			wg.Go(func() {
				for range 16 {
					m.observeInference(10, 5, 0, 0, "stop", time.Second)
					snap := m.inferenceSnapshotData()
					if snap.promptTok != snap.inputTokenHist.sum || snap.complTok != snap.outputTokenHist.sum {
						t.Error("scrape observed mismatched counter and histogram totals")
					}
				}
			})
		}
		wg.Wait()
		final := m.inferenceSnapshotData()
		if final.inputTokenHist.count != 65 || final.outputTokenHist.count != 65 || final.promptTok != 650 || final.complTok != 325 || final.syntheticTurns != 1 || final.e2eHist.count != 64 {
			t.Fatalf("concurrent/synthetic accounting drift: %+v", final)
		}
	})
}
