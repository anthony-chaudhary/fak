package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/engine"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

type responsesRouteWireCall struct {
	path, authorization string
	model               string
	stream              bool
	temperature         float64
}

func newResponsesRouteWire(t *testing.T, name string) (*httptest.Server, func() []responsesRouteWireCall) {
	t.Helper()
	var mu sync.Mutex
	var calls []responsesRouteWireCall
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model       string  `json:"model"`
			Stream      bool    `json:"stream"`
			Temperature float64 `json:"temperature"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "invalid fixture request", http.StatusBadRequest)
			return
		}
		mu.Lock()
		calls = append(calls, responsesRouteWireCall{r.URL.Path, r.Header.Get("Authorization"), req.Model, req.Stream, req.Temperature})
		mu.Unlock()
		if !req.Stream {
			w.Header().Set("Content-Type", "application/json")
			_, _ = fmt.Fprintf(w, `{"model":"served-%s","choices":[{"message":{"role":"assistant","content":"reply-%s"},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1}}`, name, name)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: {\"model\":\"served-%s\",\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"reply-%s\"},\"finish_reason\":null}]}\n\n", name, name)
		w.(http.Flusher).Flush()
		_, _ = io.WriteString(w, "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,\"completion_tokens\":1}}\n\ndata: [DONE]\n\n")
	}))
	t.Cleanup(server.Close)
	return server, func() []responsesRouteWireCall {
		mu.Lock()
		defer mu.Unlock()
		return append([]responsesRouteWireCall(nil), calls...)
	}
}

// Exercise authentication, real roster binding, the public Responses handler and
// real HTTPPlanner transport. No internal context binding is injected here.
// fak-test:runtime fast est=1s lane=default
func TestResponsesLiveHonorsAuthenticatedRosterRoute(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterEngine("mock", engine.MockEngine)
	abi.RegisterAdjudicator(0, toolAdj{})
	boot, bootCalls := newResponsesRouteWire(t, "boot")
	a, aCalls := newResponsesRouteWire(t, "a")
	b, bCalls := newResponsesRouteWire(t, "b")
	t.Setenv("FAK_RESPONSES_ROUTE_A_TEST_KEY", "route-a-fixture-key")
	t.Setenv("FAK_RESPONSES_ROUTE_B_TEST_KEY", "route-b-fixture-key")
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{
			{ID: "account-a", Kind: modelroute.KindOpenAI, BaseURL: a.URL, CredEnv: "FAK_RESPONSES_ROUTE_A_TEST_KEY", Principals: []string{"tenant-a"}},
			{ID: "account-b", Kind: modelroute.KindOpenAI, BaseURL: b.URL, CredEnv: "FAK_RESPONSES_ROUTE_B_TEST_KEY", Principals: []string{"tenant-b"}},
		},
		Bindings: []modelroute.Binding{
			{Model: "public-a", Account: "account-a", UpstreamModel: "alias-a"},
			{Model: "public-b", Account: "account-b", UpstreamModel: "alias-b"},
		},
		Default: "account-a",
	}
	srv, err := New(Config{
		EngineID: "test", Model: "boot-model", BaseURL: boot.URL,
		Provider: "openai-compatible", APIKey: "boot-fixture-key", RouteAccounts: roster,
		KeyPrincipals: map[string]string{"caller-a": "tenant-a", "caller-b": "tenant-b"},
	})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(srv.Close)
	handler := srv.Handler()
	for _, tc := range []struct {
		name, model, key, target, alias, authorization string
		stream, denied                                 bool
	}{
		{"live-a", "public-a", "caller-a", "a", "alias-a", "Bearer route-a-fixture-key", true, false},
		{"live-b", "public-b", "caller-b", "b", "alias-b", "Bearer route-b-fixture-key", true, false},
		{"buffered-a", "public-a", "caller-a", "a", "alias-a", "Bearer route-a-fixture-key", false, false},
		{"denied-live-b", "public-b", "caller-a", "", "", "", true, true},
		{"unbound-live", "boot-model", "caller-a", "boot", "boot-model", "Bearer boot-fixture-key", true, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			before := map[string]int{"boot": len(bootCalls()), "a": len(aCalls()), "b": len(bCalls())}
			body := fmt.Sprintf(`{"model":%q,"input":"route me","stream":%t,"store":false,"temperature":0.25,"max_output_tokens":8}`, tc.model, tc.stream)
			requestCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			req := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(body)).WithContext(requestCtx)
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("x-api-key", tc.key)
			w := httptest.NewRecorder()
			handler.ServeHTTP(w, req)
			after := map[string][]responsesRouteWireCall{"boot": bootCalls(), "a": aCalls(), "b": bCalls()}
			for name, calls := range after {
				want := before[name]
				if !tc.denied && name == tc.target {
					want++
				}
				if len(calls) != want {
					t.Errorf("%s upstream calls=%d, want %d; status=%d body=%s", name, len(calls), want, w.Code, w.Body.String())
				}
			}
			if tc.denied {
				if w.Code != http.StatusForbidden {
					t.Fatalf("denied account status=%d, want 403", w.Code)
				}
				return
			}
			if w.Code != http.StatusOK || !strings.Contains(w.Body.String(), "reply-"+tc.target) {
				t.Fatalf("status=%d body=%s, want selected-account reply", w.Code, w.Body.String())
			}
			calls := after[tc.target]
			if len(calls) != before[tc.target]+1 {
				t.Fatal("selected upstream did not receive exactly one request")
			}
			call := calls[len(calls)-1]
			if call.path != "/chat/completions" || call.authorization != tc.authorization || call.model != tc.alias || call.stream != tc.stream || call.temperature != 0.25 {
				t.Fatalf("selected wire call = %+v, want account auth, alias, stream=%v and temperature=0.25", call, tc.stream)
			}
			if tc.stream {
				terminals := 0
				for _, line := range strings.Split(w.Body.String(), "\n") {
					if !strings.HasPrefix(line, "data: ") {
						continue
					}
					var event struct {
						Type     string            `json:"type"`
						Response responsesResponse `json:"response"`
					}
					if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event); err != nil {
						t.Fatal(err)
					}
					if event.Type == "response.created" || event.Type == "response.completed" {
						if event.Response.Model != tc.model {
							t.Errorf("%s public model=%q, want requested %q", event.Type, event.Response.Model, tc.model)
						}
					}
					if event.Type == "response.completed" {
						terminals++
						if event.Response.OutputText != "reply-"+tc.target {
							t.Errorf("terminal text=%q, want selected-account reply", event.Response.OutputText)
						}
					}
				}
				if terminals != 1 {
					t.Fatalf("completed terminals=%d, want 1", terminals)
				}
			}
		})
	}
}

type responsesBoundGatePlanner struct {
	*scriptedLiveResponsesPlanner
	provider     agent.Provider
	force        bool
	gateCalls    int
	gateProvider agent.Provider
	gateForce    bool
}

func (p *responsesBoundGatePlanner) Provider() agent.Provider     { return p.provider }
func (p *responsesBoundGatePlanner) IsForceResponsesStream() bool { return p.force }
func (p *responsesBoundGatePlanner) ResponsesWireStreamsLive(provider agent.Provider, force bool) bool {
	p.gateCalls++
	p.gateProvider, p.gateForce = provider, force
	return p.gateAllows
}

// Exposing only these interfaces deliberately hides Provider and force metadata.
type responsesOpaqueGatePlanner struct {
	agent.StreamingPlanner
	responsesLiveGate
}

// These are seam-level controls. Current public Responses ingress does not accept
// the injected raw-body, API-key or beta sampling carriers used below.
// fak-test:runtime fast est=100ms lane=default
func TestResponsesLiveUsesBoundPlannerGateAndOptions(t *testing.T) {
	for _, tc := range []struct {
		name                       string
		bound, supports, allow     bool
		force, opaque, wantHandled bool
	}{
		{"bound-force-true", true, true, true, true, false, true},
		{"bound-force-false", true, true, true, false, false, true},
		{"bound-gate-refuses", true, true, false, false, false, false},
		{"bound-streaming-unsupported", true, false, true, true, false, false},
		{"bound-provider-unknown", true, true, true, true, true, false},
		{"unbound-preserves-boot", false, true, true, false, false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			newPlanner := func(force bool) *responsesBoundGatePlanner {
				return &responsesBoundGatePlanner{
					scriptedLiveResponsesPlanner: &scriptedLiveResponsesPlanner{
						supports: true, gateAllows: true, fragments: []string{"selected"},
						comp: &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "selected"}, FinishReason: "stop"},
					},
					provider: agent.ProviderOpenAIResponses, force: force,
				}
			}
			boot, selected := newPlanner(!tc.force), newPlanner(tc.force)
			selected.supports, selected.gateAllows = tc.supports, tc.allow
			srv := newTestServer(t)
			srv.planner = boot
			ctx := context.Background()
			if tc.bound {
				var planner agent.Planner = selected
				if tc.opaque {
					planner = responsesOpaqueGatePlanner{selected, selected}
				}
				ctx = context.WithValue(ctx, chatRouteContextKey{}, &chatRouteBinding{
					RequestedModel: "public-model", Target: modelroute.Target{UpstreamModel: "bound-alias"}, Planner: planner,
				})
			} else {
				selected = boot
			}
			opts := []agent.SampleOpt{agent.WithMaxTokens(7), func(p *agent.SampleParams) {
				p.Model = "client-override"
				p.UpstreamAPIKey = "client-fixture-key"
				p.RawRequestBody = []byte(`{"model":"raw-override"}`)
				p.UpstreamBeta = "client-fixture-beta"
			}}
			w := httptest.NewRecorder()
			persisted := 0
			handled := srv.streamResponsesLive(ctx, w, "public-model", "route-test", servedSessionTurn{}, nil, responsesLiveTurn{
				messages: []agent.Message{{Role: agent.RoleUser, Content: "hi"}}, sampleOpts: opts,
			}, newResponsesRestoreContinuation(nil), func(string, agent.Message) { persisted++ })
			if handled != tc.wantHandled {
				t.Fatalf("handled=%v, want %v", handled, tc.wantHandled)
			}
			if tc.bound && (boot.gateCalls != 0 || len(boot.streamParams) != 0 || len(boot.bufferedParams) != 0) {
				t.Fatal("bound turn consulted or executed the boot planner")
			}
			wantGateCalls := 1
			if !tc.supports || tc.opaque {
				wantGateCalls = 0
			}
			if selected.gateCalls != wantGateCalls {
				t.Fatalf("selected gate calls=%d, want %d", selected.gateCalls, wantGateCalls)
			}
			if wantGateCalls != 0 && (selected.gateProvider != selected.provider || selected.gateForce != selected.force) {
				t.Fatalf("gate inputs=(%q, %v), want selected (%q, %v)", selected.gateProvider, selected.gateForce, selected.provider, selected.force)
			}
			if !tc.wantHandled {
				if w.Body.Len() != 0 || len(w.Header()) != 0 || persisted != 0 || len(selected.streamParams) != 0 {
					t.Fatal("fallback must remain uncommitted without execution or persistence")
				}
				return
			}
			if len(selected.streamParams) != 1 || len(selected.bufferedParams) != 0 || persisted != 1 {
				t.Fatalf("selected stream calls=%d buffered=%d persisted=%d, want 1/0/1", len(selected.streamParams), len(selected.bufferedParams), persisted)
			}
			got := selected.streamParams[0]
			if got.MaxTokens == nil || *got.MaxTokens != 7 || got.PerTokenStream == nil || !*got.PerTokenStream {
				t.Fatalf("sampling budget/per-token contract changed: %+v", got)
			}
			if tc.bound {
				if got.Model != "bound-alias" || got.UpstreamAPIKey != "" || got.RawRequestBody != nil || got.UpstreamBeta != "" {
					t.Fatalf("route-final options not enforced: %+v", got)
				}
			} else if got.Model != "client-override" || got.UpstreamAPIKey != "client-fixture-key" || len(got.RawRequestBody) == 0 || got.UpstreamBeta != "client-fixture-beta" {
				t.Fatalf("unbound options changed: %+v", got)
			}
			var original agent.SampleParams
			for _, opt := range opts {
				opt(&original)
			}
			if original.Model != "client-override" || original.PerTokenStream != nil || original.UpstreamAPIKey != "client-fixture-key" || len(original.RawRequestBody) == 0 {
				t.Fatal("live route mutated the caller's sampling options")
			}
		})
	}
}
