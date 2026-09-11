package taskrun

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestModelObserverContract(t *testing.T) {
	t.Run("accounts identity and usage for every successful response", func(t *testing.T) {
		responses := []string{
			`{"model":"fixture-model","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`{"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`{"model":"wrong-model","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`{"model":"fixture-model","usage":{}}`,
		}
		var next atomic.Int64
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			i := int(next.Add(1)) - 1
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, responses[i])
		}))
		defer upstream.Close()
		observer, err := startModelObserver(context.Background(), upstream.URL, "fixture-model", filepath.Join(t.TempDir(), "events.jsonl"), 12)
		if err != nil {
			t.Fatal(err)
		}
		for range responses {
			if status := postObserver(t, observer.Endpoint()+"/v1/chat/completions", `{}`); status != http.StatusOK {
				t.Fatalf("successful response status = %d", status)
			}
		}
		got, err := observer.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got.SuccessfulRequests != 4 || got.IdentityKnownRequests != 2 || got.IdentityUnknownRequests != 1 || got.IdentityMismatchedRequests != 1 || got.ModelIdentityStatus != "mismatched" {
			t.Fatalf("per-response identity coverage = %+v", got)
		}
		if got.UsageKnownRequests != 3 || got.UsageUnknownRequests != 1 || got.UsageInvalidRequests != 1 {
			t.Fatalf("usage accounting accepted an empty/non-token tuple: %+v", got)
		}

		missing := []string{
			`{"model":"fixture-model","usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
			`{"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}`,
		}
		next.Store(0)
		upstream.Config.Handler = http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			i := int(next.Add(1)) - 1
			w.Header().Set("Content-Type", "application/json")
			io.WriteString(w, missing[i])
		})
		second, err := startModelObserver(context.Background(), upstream.URL, "fixture-model", filepath.Join(t.TempDir(), "events.jsonl"), 12)
		if err != nil {
			t.Fatal(err)
		}
		for range missing {
			postObserver(t, second.Endpoint()+"/v1/chat/completions", `{}`)
		}
		coverage, err := second.Close()
		if err != nil {
			t.Fatal(err)
		}
		if coverage.ModelIdentityStatus != "unknown" || coverage.IdentityUnknownRequests != 1 {
			t.Fatalf("incomplete identity coverage = %+v, want unknown", coverage)
		}
	})

	t.Run("shared meter records real cross-task upstream overlap", func(t *testing.T) {
		entered := make(chan struct{}, 2)
		release := make(chan struct{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			entered <- struct{}{}
			<-release
			writeObserverCompletion(w, "fixture-model", true)
		}))
		defer upstream.Close()
		meter := &modelConcurrencyMeter{}
		observers := make([]*modelObserver, 2)
		for i := range observers {
			var err error
			observers[i], err = startModelObserver(context.Background(), upstream.URL, "fixture-model", filepath.Join(t.TempDir(), "events.jsonl"), 12)
			if err != nil {
				t.Fatal(err)
			}
			observers[i].meter = meter
			request, _ := http.NewRequest(http.MethodGet, observers[i].Endpoint()+"/v1/chat/completions", nil)
			response, err := http.DefaultClient.Do(request)
			if err != nil {
				t.Fatal(err)
			}
			response.Body.Close()
			if response.StatusCode < 400 {
				t.Fatalf("observer %d admitted forbidden ingress", i)
			}
		}
		var wg sync.WaitGroup
		wg.Add(2)
		for _, observer := range observers {
			go func(observer *modelObserver) {
				defer wg.Done()
				_ = postObserverStatus(observer.Endpoint()+"/v1/chat/completions", `{}`)
			}(observer)
		}
		<-entered
		<-entered
		close(release)
		wg.Wait()
		for i, observer := range observers {
			got, err := observer.Close()
			if err != nil {
				t.Fatal(err)
			}
			if got.PeakHTTPInFlight != 1 || got.UpstreamRequests != 1 || got.SuccessfulRequests != 1 || got.Requests != 1 {
				t.Fatalf("observer %d local/ingress accounting = %+v", i, got)
			}
		}
		if got := meter.observedPeak(); got != 2 {
			t.Fatalf("shared upstream peak = %d, want real overlap 2", got)
		}
	})

	t.Run("counts planner retries and fails closed at limit", func(t *testing.T) {
		var upstreamCalls atomic.Int64
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			id := upstreamCalls.Add(1)
			if id == 1 {
				http.Error(w, "retry", http.StatusInternalServerError)
				return
			}
			writeObserverCompletion(w, "fixture-model", true)
		}))
		defer upstream.Close()
		path := filepath.Join(t.TempDir(), "events.jsonl")
		observer, err := startModelObserver(context.Background(), upstream.URL, "fixture-model", path, 12)
		if err != nil {
			t.Fatal(err)
		}
		planner := agent.NewHTTPPlanner(observer.Endpoint(), "fixture-model", "secret-must-not-appear")
		if _, err := planner.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "private prompt must not appear"}}, nil); err != nil {
			t.Fatalf("planner retry through observer: %v", err)
		}
		for i := 0; i < 10; i++ { // retry pair plus ten requests reaches the per-task ceiling.
			if status := postObserver(t, observer.Endpoint()+"/v1/chat/completions", `{}`); status != http.StatusOK {
				t.Fatalf("request %d before ceiling status = %d", i+1, status)
			}
		}
		if status := postObserver(t, observer.Endpoint()+"/v1/chat/completions", `{}`); status < 400 {
			t.Fatalf("request 13 status = %d, want refusal before upstream", status)
		}
		got, err := observer.Close()
		if err == nil || !got.LimitExceeded || got.Requests != 13 || upstreamCalls.Load() != 12 {
			t.Fatalf("limit observation = %+v err=%v upstream=%d", got, err, upstreamCalls.Load())
		}
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if lines := bytes.Count(bytes.TrimSpace(body), []byte("\n")) + 1; lines != 2*got.Requests {
			t.Fatalf("event rows = %d, want paired start/terminal rows for %d requests", lines, got.Requests)
		}
		sum := sha256.Sum256(body)
		if got.EventsPath != path || got.EventsDigest != hex.EncodeToString(sum[:]) {
			t.Fatalf("durable event artifact mismatch: %+v want digest %s", got, hex.EncodeToString(sum[:]))
		}
		for _, forbidden := range []string{"private prompt must not appear", "secret-must-not-appear", "authorization", "choices"} {
			if strings.Contains(strings.ToLower(string(body)), strings.ToLower(forbidden)) {
				t.Fatalf("observer event artifact leaked %q: %s", forbidden, body)
			}
		}
	})

	t.Run("records HTTP overlap and honest model usage provenance", func(t *testing.T) {
		entered := make(chan struct{}, 2)
		release := make(chan struct{})
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			entered <- struct{}{}
			<-release
			writeObserverCompletion(w, "wrong-model", true)
		}))
		defer upstream.Close()
		observer, err := startModelObserver(context.Background(), upstream.URL, "fixture-model", filepath.Join(t.TempDir(), "events.jsonl"), 12)
		if err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		wg.Add(2)
		for range 2 {
			go func() { defer wg.Done(); _ = postObserverStatus(observer.Endpoint()+"/v1/chat/completions", `{}`) }()
		}
		<-entered
		<-entered
		close(release)
		wg.Wait()
		got, err := observer.Close()
		if err != nil {
			t.Fatal(err)
		}
		if got.PeakHTTPInFlight != 2 || got.InternalGPUConcurrency != nil || got.ModelIdentityStatus == "matched" || len(got.ObservedModels) != 1 || got.ObservedModels[0] != "wrong-model" || got.UsageKnownRequests != 2 {
			t.Fatalf("overlap/provenance observation = %+v", got)
		}
	})

	t.Run("missing identity usage forbidden routes and cancellation", func(t *testing.T) {
		upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/redirect" {
				http.Redirect(w, r, "/v1/chat/completions", http.StatusFound)
				return
			}
			io.WriteString(w, `{"choices":[{"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		}))
		defer upstream.Close()
		ctx, cancel := context.WithCancel(context.Background())
		observer, err := startModelObserver(ctx, upstream.URL, "fixture-model", filepath.Join(t.TempDir(), "events.jsonl"), 12)
		if err != nil {
			t.Fatal(err)
		}
		client := &http.Client{CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse }}
		for _, probe := range []struct{ method, url string }{
			{http.MethodGet, observer.Endpoint() + "/v1/chat/completions"},
			{http.MethodPost, observer.Endpoint() + "/wrong"},
			{http.MethodPost, observer.Endpoint() + "/v1/chat/completions?drift=1"},
		} {
			req, _ := http.NewRequest(probe.method, probe.url, strings.NewReader(`{}`))
			resp, err := client.Do(req)
			if err != nil {
				t.Fatal(err)
			}
			resp.Body.Close()
			if resp.StatusCode < 400 {
				t.Fatalf("forbidden route %s %s status = %d", probe.method, probe.url, resp.StatusCode)
			}
		}
		if status := postObserver(t, observer.Endpoint()+"/v1/chat/completions", `{}`); status != http.StatusOK {
			t.Fatalf("valid request status = %d", status)
		}
		cancel()
		got, err := observer.Close()
		if err == nil || got.ModelIdentityStatus == "matched" || got.UsageUnknownRequests != 1 {
			t.Fatalf("canceled unknown observation = %+v err=%v", got, err)
		}
		if _, err := http.Post(observer.Endpoint()+"/v1/chat/completions", "application/json", strings.NewReader(`{}`)); err == nil {
			t.Fatal("observer still accepted connections after cancellation/close")
		}
	})
}

func writeObserverCompletion(w http.ResponseWriter, model string, usage bool) {
	response := map[string]any{"model": model, "choices": []any{map[string]any{"message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}}
	if usage {
		response["usage"] = map[string]any{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(response)
}

func postObserver(t *testing.T, url, body string) int {
	t.Helper()
	status := postObserverStatus(url, body)
	if status == 0 {
		t.Fatal("observer request failed")
	}
	return status
}

func postObserverStatus(url, body string) int {
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Post(url, "application/json", strings.NewReader(body))
	if err != nil {
		return 0
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode
}
