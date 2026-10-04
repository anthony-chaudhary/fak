package gateway

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
	"github.com/anthony-chaudhary/fak/internal/tokenizer"
)

// fak-test:runtime fast est=50ms lane=default
func TestProxyComputeCanonicalUpstreamDeclarations(t *testing.T) {
	for _, c := range []struct {
		name, base     string
		replicas, want []string
		refuse         bool
	}{
		{name: "none"}, {name: "whitespace-only-base", base: " \t\n"},
		{name: "trimmed-base", base: " http://127.0.0.1:18091/v1 ", want: []string{"http://127.0.0.1:18091/v1"}},
		{name: "replica-only", replicas: []string{" http://127.0.0.1:18091/v1 "}, want: []string{"http://127.0.0.1:18091/v1"}},
		{name: "all-upstreams", base: "http://127.0.0.1:18091/v1", replicas: []string{" http://127.0.0.1:18092/v1 "}, want: []string{"http://127.0.0.1:18091/v1", "http://127.0.0.1:18092/v1"}},
		{name: "empty-replica", replicas: []string{""}, refuse: true},
		{name: "empty-replica-after-valid", base: "http://127.0.0.1:18091/v1", replicas: []string{" \t"}, refuse: true},
		// Presence classification intentionally preserves the canonical parser's
		// contract. URL/provider validation belongs to the actual planner consumer.
		{name: "syntax-validation-deferred", base: "not-a-url", want: []string{"not-a-url"}},
	} {
		t.Run(c.name, func(t *testing.T) {
			got, err := proxyBaseURLs(Config{BaseURL: c.base, ReplicaBaseURLs: c.replicas})
			if c.refuse {
				if err == nil {
					t.Fatal("empty upstream replica accepted")
				}
				return
			}
			if err != nil || !reflect.DeepEqual(got, append([]string{}, c.want...)) {
				t.Fatalf("canonical upstream decision drifted: %v/%v", got, err)
			}
		})
	}
}

// fak-test:runtime fast est=250ms lane=default
func TestProxyComputeActualGatewayForwardsRemoteChat(t *testing.T) {
	for _, mode := range []string{"base-only", "replica-only", "base-and-replica"} {
		t.Run(mode, func(t *testing.T) {
			var calls atomic.Int32
			handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
					http.NotFound(w, r)
					return
				}
				var body struct {
					Model    string          `json:"model"`
					Messages []agent.Message `json:"messages"`
				}
				if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Model != "remote-test-model" || len(body.Messages) != 1 || body.Messages[0].Content != "remote route invocation" {
					t.Error("actual proxy request lost model/message binding")
					w.WriteHeader(http.StatusBadRequest)
					return
				}
				calls.Add(1)
				w.Header().Set("Content-Type", "application/json")
				fmt.Fprint(w, `{"model":"remote-test-model","choices":[{"message":{"role":"assistant","content":"remote-answer-from-upstream"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
			})
			upstream := httptest.NewServer(handler)
			secondary := httptest.NewServer(handler)
			defer secondary.Close()
			defer upstream.Close()
			abi.ResetForTest()
			abi.RegisterRegionBackend(inlineBackend{})
			abi.RegisterEngine("test", echoEngine{})
			abi.RegisterAdjudicator(0, toolAdj{})
			cfg := Config{EngineID: "test", Model: "remote-test-model", Provider: "openai"}
			wantPlanner := "proxy"
			switch mode {
			case "base-only":
				cfg.BaseURL = " " + upstream.URL + "/v1 "
			case "replica-only":
				cfg.ReplicaBaseURLs = []string{" " + upstream.URL + "/v1 "}
			case "base-and-replica":
				cfg.BaseURL = upstream.URL + "/v1"
				cfg.ReplicaBaseURLs = []string{secondary.URL + "/v1"}
				wantPlanner = "replica"
			}
			srv, err := New(cfg)
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			health := httptest.NewRecorder()
			srv.Handler().ServeHTTP(health, httptest.NewRequest(http.MethodGet, "/healthz", nil))
			var status map[string]any
			if err := json.Unmarshal(health.Body.Bytes(), &status); err != nil || status["planner"] != wantPlanner {
				t.Fatalf("actual serving planner unavailable: %v/%v", status, err)
			}
			endpoint := httptest.NewServer(srv.Handler())
			defer endpoint.Close()
			var response ChatResponse
			code := postJSON(t, endpoint.URL+"/v1/chat/completions", ChatRequest{Model: "remote-test-model", Messages: []agent.Message{{Role: agent.RoleUser, Content: "remote route invocation"}}}, &response)
			if code != http.StatusOK || len(response.Choices) != 1 || response.Choices[0].Message.Content != "remote-answer-from-upstream" || calls.Load() != 1 {
				t.Fatalf("remote request not forwarded exactly once: status%d calls%d", code, calls.Load())
			}
		})
	}
}

// fak-test:runtime fast est=100ms lane=default
func TestProxyComputeGatewayMixedPlannerAndMalformedRemainDistinct(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})
	cfg := Config{EngineID: "test", Model: "remote-test-model", BaseURL: "http://127.0.0.1:18091/v1", InKernelModel: &model.Model{}, Tokenizer: &tokenizer.Tokenizer{}}
	srv, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	if plannerKind(srv.planner) != "dual" {
		t.Fatal("mixed resident model treated as remote-only")
	}
	cfg.InKernelModel = nil
	cfg.Tokenizer = nil
	cfg.ReplicaBaseURLs = []string{" "}
	if _, err := New(cfg); err == nil || !strings.Contains(err.Error(), "replica") {
		t.Fatal("actual gateway accepted empty declared replica")
	}
	cfg.ReplicaBaseURLs = nil
	cfg.Provider = "unknown-provider"
	if _, err := New(cfg); err == nil {
		t.Fatal("presence classification bypassed actual provider validation")
	}
}
