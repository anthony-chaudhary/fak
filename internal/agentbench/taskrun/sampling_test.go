package taskrun

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fak-test:runtime fast est=200ms lane=default
func TestTaskPlannerSendsExplicitSampling(t *testing.T) {
	var mu sync.Mutex
	var bodies [][]byte
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		mu.Lock()
		bodies = append(bodies, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"id":"c1","object":"chat.completion","model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"done"},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`)
	}))
	defer server.Close()

	temp, topP, topK, maxTokens, seed := 0.7, 0.8, 20, 512, int64(1234)
	want := Sampling{Temperature: &temp, TopP: &topP, TopK: &topK, MaxTokens: &maxTokens, Seed: &seed}
	planner, opts, err := newTaskClient(server.URL+"/v1", "m", want)
	if err != nil {
		t.Fatal(err)
	}
	p := &recordingPlanner{inner: planner, sampling: opts}
	callerTemp := 0.0
	if _, err := p.Complete(context.Background(), []agent.Message{{Role: agent.RoleUser, Content: "hi"}}, nil, agent.WithTemperature(&callerTemp), agent.WithMaxTokens(64)); err != nil {
		t.Fatal(err)
	}
	if len(bodies) != 1 {
		t.Fatalf("requests = %d, want 1", len(bodies))
	}
	gotJSON, _ := json.Marshal(sentSampling(bodies[0]))
	wantJSON, _ := json.Marshal(want)
	if string(gotJSON) != string(wantJSON) {
		t.Fatalf("sent sampling = %s, want %s", gotJSON, wantJSON)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestSamplingValidateRejectsOutOfRange(t *testing.T) {
	neg, zero, over, negK := -0.1, 0.0, 1.5, -1
	for name, s := range map[string]Sampling{
		"temperature": {Temperature: &neg},
		"top_p zero":  {TopP: &zero},
		"top_p over":  {TopP: &over},
		"top_k":       {TopK: &negK},
		"max_tokens":  {MaxTokens: &negK},
	} {
		if s.Validate() == nil {
			t.Fatalf("%s accepted", name)
		}
	}
	if (Sampling{}).Validate() != nil {
		t.Fatal("empty sampling rejected")
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestObserverRecordsDistinctSentSampling(t *testing.T) {
	o := &modelObserver{}
	o.observeRequest([]byte(`{"model":"m","temperature":0.2,"top_k":40,"seed":7}`))
	o.observeRequest([]byte(`{"model":"m","seed":7,"top_k":40,"temperature":0.2,"messages":[]}`))
	o.observeRequest([]byte(`{"model":"m","temperature":0}`))
	got, _ := json.Marshal(o.samplingSent)
	if string(got) != `[{"temperature":0.2,"top_k":40,"seed":7},{"temperature":0}]` {
		t.Fatalf("sampling_sent = %s", got)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestChildReceiptFileCarriesTranscriptBeyondStdoutCap(t *testing.T) {
	big := strings.Repeat("x", 4*maxChildOut)
	r := ChildReceipt{Schema: "fak.agentbench.task-child.v2", PlannerCalls: 1, Turns: []PlannerTurn{{Messages: []agent.Message{{Role: agent.RoleTool, ToolCallID: "c1", Content: big}}}}, ToolCalls: ToolCallMetrics{Measured: true, Total: 1}}
	path := filepath.Join(t.TempDir(), "child.json")
	var stdout strings.Builder
	if err := writeChildReceipt(path, &stdout, r); err != nil {
		t.Fatal(err)
	}
	if stdout.Len() != 0 {
		t.Fatalf("receipt leaked onto stdout (%d bytes)", stdout.Len())
	}
	got, err := readChildReceipt(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Turns) != 1 || len(got.Turns[0].Messages) != 1 || got.Turns[0].Messages[0].Content != big || got.ToolCalls.Total != 1 {
		t.Fatal("receipt did not round-trip through the receipt file")
	}
	if err := writeChildReceipt(path, &stdout, r); err == nil {
		t.Fatal("existing receipt file was overwritten")
	}
}
