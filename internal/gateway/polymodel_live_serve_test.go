package gateway

import (
	"context"
	"errors"
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

// TestPolymodelLiveServeReachability proves the public request surface reaches
// the pool-backed planner installed by New. The two planners are independent
// oracles: observing their call counts demonstrates exact-ID dispatch rather
// than response relabeling. The flag-off subtest protects the legacy constructor.
func TestPolymodelLiveServeReachability(t *testing.T) {
	abi.RegisterRegionBackend(inlineBackend{})
	abi.RegisterEngine("test", echoEngine{})
	abi.RegisterAdjudicator(0, toolAdj{})

	alpha := &polymodelReachabilityPlanner{model: "resident-alpha"}
	beta := &polymodelReachabilityPlanner{model: "resident-beta"}
	bindings := []PolymodelBinding{
		{ModelID: alpha.model, Planner: alpha, WeightBytes: 7},
		{ModelID: beta.model, Planner: beta, WeightBytes: 11},
	}

	t.Run("flag on routes exact resident IDs and refuses an unknown ID", func(t *testing.T) {
		t.Setenv("FAK_POLYMODEL", "1")
		srv, err := New(Config{
			EngineID:          "test",
			Model:             "legacy-default",
			VDSO:              true,
			PolymodelBindings: bindings,
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
		if status == http.StatusOK {
			t.Fatal("unknown model status = 200, want an explicit refusal")
		}
		if alpha.calls.Load() != 1 || beta.calls.Load() != 1 {
			t.Fatalf("unknown model reached a resident: alpha=%d beta=%d", alpha.calls.Load(), beta.calls.Load())
		}

		_, err = srv.planner.Complete(context.Background(), nil, nil, agent.WithModel("not-resident"))
		var notHeld *PolymodelModelNotHeldError
		if !errors.As(err, &notHeld) {
			t.Fatalf("unknown model error = %T %v, want *PolymodelModelNotHeldError", err, err)
		}
	})

	t.Run("flag off preserves the legacy planner", func(t *testing.T) {
		t.Setenv("FAK_POLYMODEL", "off")
		srv, err := New(Config{
			EngineID:          "test",
			Model:             "legacy-default",
			VDSO:              true,
			PolymodelBindings: bindings,
		})
		if err != nil {
			t.Fatalf("New with polymodel disabled: %v", err)
		}
		defer srv.Close()

		if _, ok := srv.planner.(*agent.MockPlanner); !ok {
			t.Fatalf("flag-off planner = %T, want legacy *agent.MockPlanner", srv.planner)
		}
		if alpha.calls.Load() != 1 || beta.calls.Load() != 1 {
			t.Fatalf("flag-off construction reached a resident: alpha=%d beta=%d", alpha.calls.Load(), beta.calls.Load())
		}
	})
}
