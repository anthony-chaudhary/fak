package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/model"
)

type recurrentEvictOncePlanner struct {
	calls int
}

func (p *recurrentEvictOncePlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	if p.calls == 1 {
		panic(&model.RecurrentEvictUnsupportedError{Layers: []int{0, 1, 2}})
	}
	return &agent.Completion{
		Message:      agent.Message{Role: agent.RoleAssistant, Content: "ok"},
		FinishReason: "stop",
		Usage:        agent.Usage{PromptTokens: 1, CompletionTokens: 1, TotalTokens: 2},
	}, nil
}

func (*recurrentEvictOncePlanner) Model() string { return "recurrent-evict-once" }

func TestChatCompletionRecurrentEvictPanicReturns409AndServerSurvives(t *testing.T) {
	srv := newTestServer(t)
	planner := &recurrentEvictOncePlanner{}
	srv.planner = planner
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	status, body := postRecurrentEvictChat(t, ts.URL)
	if status != http.StatusConflict {
		t.Fatalf("recurrent eviction status = %d, want 409; body=%s", status, body)
	}
	for _, want := range []string{"in_kernel_recurrent_evict_unsupported", "fresh session", "--reset-on-budget"} {
		if !strings.Contains(body, want) {
			t.Fatalf("409 body missing %q: %s", want, body)
		}
	}

	status, body = postRecurrentEvictChat(t, ts.URL)
	if status != http.StatusOK {
		t.Fatalf("server did not survive recurrent eviction panic: status=%d body=%s", status, body)
	}
	if planner.calls != 2 {
		t.Fatalf("planner calls = %d, want 2", planner.calls)
	}
}

func postRecurrentEvictChat(t *testing.T, base string) (int, string) {
	t.Helper()
	raw, err := json.Marshal(ChatRequest{
		Model:    "test-model",
		Messages: []agent.Message{{Role: agent.RoleUser, Content: "hello"}},
	})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := http.Post(base+"/v1/chat/completions", "application/json", bytes.NewReader(raw))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	return resp.StatusCode, string(body)
}
