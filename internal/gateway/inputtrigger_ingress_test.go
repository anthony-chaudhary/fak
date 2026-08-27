package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
	"github.com/anthony-chaudhary/fak/internal/modelroute/inputtrigger"
)

type inputTriggerIngressPlanner struct {
	calls          int
	got            agent.SampleParams
	nativeEngine   string
	nativeModel    string
	nativeFallback bool
}

func (p *inputTriggerIngressPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	var sample agent.SampleParams
	for _, opt := range opts {
		opt(&sample)
	}
	p.got = sample
	nativeEngine := p.nativeEngine
	if nativeEngine == "" {
		nativeEngine = TurnIngressEngine
	}
	nativeModel := p.nativeModel
	if nativeModel == "" {
		nativeModel = TurnIngressModel
	}
	completion := &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Model:        TurnIngressModel,
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}
	if sample.NativeInferenceReceipt {
		completion.NativeInference = &agent.NativeInferenceReceipt{
			TokenIDs:       []int{38},
			TokenLogprobs:  []float64{-0.25},
			Model:          nativeModel,
			Engine:         nativeEngine,
			Backend:        "test-native",
			ForwardPath:    "fak/native/qwen38",
			FallbackActive: p.nativeFallback,
		}
	}
	return completion, nil
}

func (*inputTriggerIngressPlanner) Model() string { return TurnIngressModel }

func (*inputTriggerIngressPlanner) StreamingSupported() bool { return true }

func (p *inputTriggerIngressPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	completion, err := p.Complete(ctx, messages, tools, opts...)
	if err == nil {
		err = sink(completion.Message.Content)
	}
	return completion, err
}

func inputTriggerGatewayManifest(model string) *modelroute.Manifest {
	plan := modelroute.Plan{Members: []modelroute.Member{{Model: model, Role: "primary"}}}
	return &modelroute.Manifest{
		Version: modelroute.Version,
		Default: plan,
		Rules: []modelroute.Rule{
			{
				Name:  "request-user",
				Match: modelroute.Match{Aspect: modelroute.AspectRequest, InputTrigger: inputtrigger.UserMessage},
				Plan:  plan,
			},
			{
				Name:  "request-tool-result",
				Match: modelroute.Match{Aspect: modelroute.AspectRequest, InputTrigger: inputtrigger.ToolResult},
				Plan:  plan,
			},
		},
	}
}

func newInputTriggerGateway(t *testing.T, manifest *modelroute.Manifest) (*Server, *inputTriggerIngressPlanner) {
	t.Helper()
	srv := newTestServer(t)
	srv.route = modelroute.NewLive(manifest)
	planner := &inputTriggerIngressPlanner{}
	srv.planner = planner
	return srv, planner
}

func postInputTriggerChat(t *testing.T, srv *Server, req ChatRequest) *httptest.ResponseRecorder {
	t.Helper()
	body, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Marshal request: %v", err)
	}
	httpReq := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))
	httpReq.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, httpReq)
	return rec
}

func TestInputTriggerGatewayIngressReceiptReplayEndToEnd(t *testing.T) {
	cases := []struct {
		name       string
		messages   []agent.Message
		explicit   *inputtrigger.Explicit
		wantClass  inputtrigger.Trigger
		wantSource inputtrigger.Provenance
		wantRule   string
		wantMeta   int
	}{
		{
			name:       "user",
			messages:   []agent.Message{{Role: agent.RoleUser, Content: "USER_CONTENT_SECRET"}},
			explicit:   &inputtrigger.Explicit{Classification: inputtrigger.UserMessage, Provenance: inputtrigger.ProvenanceExplicit},
			wantClass:  inputtrigger.UserMessage,
			wantSource: inputtrigger.ProvenanceExplicit,
			wantRule:   "request-user",
		},
		{
			name: "tool-result",
			messages: []agent.Message{
				{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "call_1", Type: "function", Function: agent.Func{Name: "lookup", Arguments: `{}`}}}},
				{Role: agent.RoleTool, Name: "lookup", ToolCallID: "call_1", Content: "TOOL_RESULT_SECRET"},
			},
			explicit: &inputtrigger.Explicit{
				Classification: inputtrigger.ToolResult,
				Provenance:     inputtrigger.ProvenanceExplicit,
				Metadata: map[string]string{
					"authorization": "Bearer TOOL_METADATA_SECRET",
					"model":         "llama.cpp/fallback",
				},
			},
			wantClass:  inputtrigger.ToolResult,
			wantSource: inputtrigger.ProvenanceExplicit,
			wantRule:   "request-tool-result",
			wantMeta:   2,
		},
		{
			name:     "retry",
			messages: []agent.Message{{Role: agent.RoleUser, Content: "RETRY_CONTENT_SECRET"}},
			explicit: &inputtrigger.Explicit{
				Classification: inputtrigger.UserMessage,
				Provenance:     inputtrigger.ProvenanceRetry,
				Metadata:       map[string]string{"attempt": "2"},
			},
			wantClass:  inputtrigger.UserMessage,
			wantSource: inputtrigger.ProvenanceRetry,
			wantRule:   "request-user",
			wantMeta:   1,
		},
		{
			name:       "compatibility-missing",
			messages:   []agent.Message{{Role: agent.RoleUser, Content: "COMPAT_CONTENT_SECRET"}},
			wantClass:  inputtrigger.UserMessage,
			wantSource: inputtrigger.ProvenanceCompatibilityMissing,
			wantRule:   "request-user",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, planner := newInputTriggerGateway(t, inputTriggerGatewayManifest(TurnIngressModel))
			rec := postInputTriggerChat(t, srv, ChatRequest{
				Messages: tc.messages,
				Fak: &FakRequestExt{
					NativeInferenceReceipt: true,
					InputTrigger:           tc.explicit,
				},
			})
			if rec.Code != http.StatusOK {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if planner.calls != 1 || planner.got.Model != TurnIngressModel {
				t.Fatalf("planner calls/model = %d/%q, want 1/%q", planner.calls, planner.got.Model, TurnIngressModel)
			}
			var response ChatResponse
			if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
				t.Fatalf("Unmarshal response: %v", err)
			}
			if response.Fak == nil || response.Fak.InputTriggerRoute == nil {
				t.Fatalf("missing input trigger route receipt: %s", rec.Body.String())
			}
			receipt := response.Fak.InputTriggerRoute
			assertInputTrigger(t, receipt.Trigger(), tc.wantClass, tc.wantSource, tc.wantMeta)
			route := receipt.Route()
			if route.Decision.RuleName != tc.wantRule || route.Decision.Subject.InputTrigger != tc.wantClass ||
				route.Decision.Subject.Labels[inputTriggerProvenanceLabel] != string(tc.wantSource) {
				t.Fatalf("route lost trigger: rule=%q class=%q provenance=%q", route.Decision.RuleName,
					route.Decision.Subject.InputTrigger, route.Decision.Subject.Labels[inputTriggerProvenanceLabel])
			}
			rawReceipt, err := json.Marshal(receipt)
			if err != nil {
				t.Fatalf("Marshal route receipt: %v", err)
			}
			replayed, err := ReplayInputTriggerRouteReceipt(rawReceipt)
			if err != nil {
				t.Fatalf("ReplayInputTriggerRouteReceipt: %v", err)
			}
			assertInputTrigger(t, replayed.Trigger(), tc.wantClass, tc.wantSource, tc.wantMeta)
			if replayed.Route().Digest != route.Digest {
				t.Fatalf("replay route digest=%q, want %q", replayed.Route().Digest, route.Digest)
			}
			if receipt.Engine() != TurnIngressEngine || receipt.Model() != TurnIngressModel ||
				receipt.RouteIdentity() != TurnIngressRouteIdentity {
				t.Fatalf("route execution=%q/%q (%q)", receipt.Engine(), receipt.Model(), receipt.RouteIdentity())
			}
			native := response.Fak.NativeInferenceReceipt
			if native == nil || native.Engine != TurnIngressEngine || native.Model != TurnIngressModel || native.FallbackActive {
				t.Fatalf("native execution receipt=%+v, want fak_native Qwen3.8 without fallback", native)
			}
			wire := rec.Body.String()
			for _, forbidden := range []string{
				"CONTENT_SECRET", "RESULT_SECRET", "METADATA_SECRET", "authorization", "llama",
			} {
				if strings.Contains(wire, forbidden) {
					t.Fatalf("response retained forbidden input %q: %s", forbidden, wire)
				}
			}
		})
	}
}

func TestInputTriggerGatewayIngressInvalidExplicitFailsBeforePlanner(t *testing.T) {
	cases := []struct {
		name     string
		explicit *inputtrigger.Explicit
	}{
		{
			name: "classification-mismatch",
			explicit: &inputtrigger.Explicit{
				Classification: inputtrigger.ToolResult,
				Provenance:     inputtrigger.ProvenanceExplicit,
			},
		},
		{
			name: "metadata-over-limit",
			explicit: &inputtrigger.Explicit{
				Classification: inputtrigger.UserMessage,
				Provenance:     inputtrigger.ProvenanceExplicit,
				Metadata: map[string]string{
					"1": "x", "2": "x", "3": "x", "4": "x", "5": "x",
					"6": "x", "7": "x", "8": "x", "9": "SECRET_METADATA_VALUE",
				},
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, planner := newInputTriggerGateway(t, inputTriggerGatewayManifest(TurnIngressModel))
			rec := postInputTriggerChat(t, srv, ChatRequest{
				Messages: []agent.Message{{Role: agent.RoleUser, Content: "INVALID_CONTENT_SECRET"}},
				Fak:      &FakRequestExt{InputTrigger: tc.explicit},
			})
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
			}
			if planner.calls != 0 {
				t.Fatalf("invalid trigger reached planner %d times", planner.calls)
			}
			if strings.Contains(rec.Body.String(), "SECRET") {
				t.Fatalf("invalid-trigger error reflected caller data: %s", rec.Body.String())
			}
		})
	}
}

func TestInputTriggerGatewayIngressRejectsFallbackRoute(t *testing.T) {
	srv, planner := newInputTriggerGateway(t, inputTriggerGatewayManifest("llama.cpp/fallback"))
	rec := postInputTriggerChat(t, srv, ChatRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "route"}},
		Fak: &FakRequestExt{InputTrigger: &inputtrigger.Explicit{
			Classification: inputtrigger.UserMessage,
			Provenance:     inputtrigger.ProvenanceExplicit,
		}},
	})
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if planner.calls != 0 {
		t.Fatalf("fallback route reached planner %d times", planner.calls)
	}
	if strings.Contains(strings.ToLower(rec.Body.String()), "llama") {
		t.Fatalf("fallback route leaked into client receipt: %s", rec.Body.String())
	}
}

func TestInputTriggerGatewayIngressRejectsMismatchedNativeExecution(t *testing.T) {
	srv, planner := newInputTriggerGateway(t, inputTriggerGatewayManifest(TurnIngressModel))
	planner.nativeEngine = "llama.cpp"
	planner.nativeModel = "fallback"
	planner.nativeFallback = true
	rec := postInputTriggerChat(t, srv, ChatRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "route"}},
		Fak: &FakRequestExt{
			NativeInferenceReceipt: true,
			InputTrigger: &inputtrigger.Explicit{
				Classification: inputtrigger.UserMessage,
				Provenance:     inputtrigger.ProvenanceExplicit,
			},
		},
	})
	if rec.Code != http.StatusBadGateway {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	if planner.calls != 1 {
		t.Fatalf("planner calls=%d, want 1", planner.calls)
	}
	if strings.Contains(strings.ToLower(rec.Body.String()), "llama") {
		t.Fatalf("mismatched native identity leaked into client response: %s", rec.Body.String())
	}
}

func TestInputTriggerGatewayIngressStreamsSameRouteReceipt(t *testing.T) {
	srv, planner := newInputTriggerGateway(t, inputTriggerGatewayManifest(TurnIngressModel))
	rec := postInputTriggerChat(t, srv, ChatRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "STREAM_CONTENT_SECRET"}},
		Stream:   true,
		Fak: &FakRequestExt{InputTrigger: &inputtrigger.Explicit{
			Classification: inputtrigger.UserMessage,
			Provenance:     inputtrigger.ProvenanceExplicit,
		}},
	})
	if rec.Code != http.StatusOK || planner.calls != 1 {
		t.Fatalf("status/calls=%d/%d body=%s", rec.Code, planner.calls, rec.Body.String())
	}
	var final ChatStreamResponse
	for _, line := range strings.Split(rec.Body.String(), "\n") {
		data, ok := strings.CutPrefix(line, "data: ")
		if !ok || data == "[DONE]" {
			continue
		}
		var chunk ChatStreamResponse
		if err := json.Unmarshal([]byte(data), &chunk); err == nil && chunk.Fak != nil {
			final = chunk
		}
	}
	if final.Fak == nil || final.Fak.InputTriggerRoute == nil {
		t.Fatalf("stream omitted input-trigger receipt: %s", rec.Body.String())
	}
	raw, err := json.Marshal(final.Fak.InputTriggerRoute)
	if err != nil {
		t.Fatal(err)
	}
	replayed, err := ReplayInputTriggerRouteReceipt(raw)
	if err != nil {
		t.Fatal(err)
	}
	assertInputTrigger(t, replayed.Trigger(), inputtrigger.UserMessage, inputtrigger.ProvenanceExplicit, 0)
	if strings.Contains(rec.Body.String(), "CONTENT_SECRET") {
		t.Fatalf("stream receipt leaked prompt: %s", rec.Body.String())
	}
}

func TestInputTriggerRouteReplayRejectsForgedIdentity(t *testing.T) {
	srv, _ := newInputTriggerGateway(t, inputTriggerGatewayManifest(TurnIngressModel))
	rec := postInputTriggerChat(t, srv, ChatRequest{
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "route"}},
		Fak: &FakRequestExt{
			NativeInferenceReceipt: true,
			InputTrigger: &inputtrigger.Explicit{
				Classification: inputtrigger.UserMessage,
				Provenance:     inputtrigger.ProvenanceExplicit,
			},
		},
	})
	if rec.Code != http.StatusOK {
		t.Fatalf("status=%d body=%s", rec.Code, rec.Body.String())
	}
	var response ChatResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &response); err != nil {
		t.Fatal(err)
	}
	raw, err := json.Marshal(response.Fak.InputTriggerRoute)
	if err != nil {
		t.Fatal(err)
	}
	for _, mutation := range [][]byte{
		bytes.Replace(raw, []byte(`"engine":"fak_native"`), []byte(`"engine":"llama.cpp"`), 1),
		bytes.Replace(raw, []byte(`"model":"Qwen3.8"`), []byte(`"model":"fallback"`), 1),
		bytes.Replace(raw, []byte(`"route_identity":"fak_native/Qwen3.8"`), []byte(`"route_identity":"llama.cpp/fallback"`), 1),
	} {
		if _, err := ReplayInputTriggerRouteReceipt(mutation); err == nil {
			t.Fatalf("forged execution identity replayed: %s", mutation)
		}
	}
}

func assertInputTrigger(t *testing.T, got inputtrigger.InputTrigger, class inputtrigger.Trigger, provenance inputtrigger.Provenance, metadata int) {
	t.Helper()
	if err := got.Validate(); err != nil {
		t.Fatalf("trigger validation: %v", err)
	}
	if got.Classification() != class || got.Provenance() != provenance || got.MetadataCount() != metadata {
		t.Fatalf("trigger=%q/%q metadata=%d, want %q/%q metadata=%d",
			got.Classification(), got.Provenance(), got.MetadataCount(), class, provenance, metadata)
	}
}
