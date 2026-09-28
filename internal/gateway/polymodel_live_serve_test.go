package gateway

import (
	"context"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

// polymodelReachabilityPlanner identifies the exact resident binding that ran.
// It deliberately has no knowledge of the pool or gateway implementation.
type polymodelReachabilityPlanner struct {
	model string
	calls atomic.Int32
}

func (p *polymodelReachabilityPlanner) Model() string { return p.model }

func (p *polymodelReachabilityPlanner) Complete(
	_ context.Context,
	_ []agent.Message,
	_ []agent.ToolDef,
	_ ...agent.SampleOpt,
) (*agent.Completion, error) {
	p.calls.Add(1)
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "served:" + p.model},
		FinishReason: "stop",
		Model:        p.model,
	}, nil
}

// TestPolymodelLiveServeReachability proves the production HTTP surface selects
// the backend bound to the request's exact model ID. Distinct response bodies
// and call counters rule out response relabeling or a shared default backend.
func TestPolymodelLiveServeReachability(t *testing.T) {
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	t.Run("two configured IDs route exactly and unknown IDs are refused", func(t *testing.T) {
		t.Setenv("FAK_POLYMODEL", "1")
		alpha := &polymodelReachabilityPlanner{model: "resident-alpha"}
		beta := &polymodelReachabilityPlanner{model: "resident-beta"}
		srv, err := New(Config{
			EngineID: "test",
			Model:    "legacy-default",
			VDSO:     true,
			PolymodelBindings: []PolymodelBinding{
				{ModelID: alpha.model, Planner: alpha, WeightBytes: 7},
				{ModelID: beta.model, Planner: beta, WeightBytes: 11},
			},
		})
		if err != nil {
			t.Fatalf("New with two resident bindings: %v", err)
		}
		defer srv.Close()

		httpSrv := httptest.NewServer(srv.Handler())
		defer httpSrv.Close()

		for _, tc := range []struct {
			model string
			want  string
		}{
			{model: alpha.model, want: "served:" + alpha.model},
			{model: beta.model, want: "served:" + beta.model},
		} {
			var got ChatResponse
			status := postJSON(t, httpSrv.URL+"/v1/chat/completions", ChatRequest{
				Model:    tc.model,
				Messages: []agent.Message{{Role: agent.RoleUser, Content: "which resident served me?"}},
			}, &got)
			if status != http.StatusOK {
				t.Fatalf("POST model %q status = %d, want 200", tc.model, status)
			}
			if len(got.Choices) != 1 || got.Choices[0].Message.Content != tc.want {
				t.Fatalf("POST model %q response = %#v, want content %q", tc.model, got, tc.want)
			}
			if got.Model != tc.model {
				t.Fatalf("POST model %q returned model %q, want exact served model", tc.model, got.Model)
			}
		}
		if got := alpha.calls.Load(); got != 1 {
			t.Fatalf("alpha planner calls = %d, want 1", got)
		}
		if got := beta.calls.Load(); got != 1 {
			t.Fatalf("beta planner calls = %d, want 1", got)
		}

		status := postJSON(t, httpSrv.URL+"/v1/chat/completions", ChatRequest{
			Model:    "not-resident",
			Messages: []agent.Message{{Role: agent.RoleUser, Content: "do not substitute me"}},
		}, nil)
		if status < http.StatusBadRequest {
			t.Fatalf("unknown model status = %d, want a 4xx or 5xx refusal", status)
		}
		if alpha.calls.Load() != 1 || beta.calls.Load() != 1 {
			t.Fatalf("unknown model reached a resident: alpha=%d beta=%d", alpha.calls.Load(), beta.calls.Load())
		}
	})

	t.Run("ordinary single-model configuration still serves", func(t *testing.T) {
		t.Setenv("FAK_POLYMODEL", "1")
		srv, err := New(Config{
			EngineID: "test",
			Model:    "legacy-default",
			VDSO:     true,
		})
		if err != nil {
			t.Fatalf("New with ordinary single-model config: %v", err)
		}
		defer srv.Close()

		httpSrv := httptest.NewServer(srv.Handler())
		defer httpSrv.Close()

		var got ChatResponse
		status := postJSON(t, httpSrv.URL+"/v1/chat/completions", ChatRequest{
			Model:    "legacy-default",
			Messages: []agent.Message{{Role: agent.RoleUser, Content: "legacy request"}},
		}, &got)
		if status != http.StatusOK {
			t.Fatalf("single-model POST status = %d, want 200", status)
		}
		if got.Model != "legacy-default" {
			t.Fatalf("single-model response model = %q, want legacy-default", got.Model)
		}
	})
}
