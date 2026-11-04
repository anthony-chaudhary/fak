package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

type monotonicResponsesPlanner struct {
	calls chan []agent.Message
}

func (p *monotonicResponsesPlanner) Complete(ctx context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	select {
	case p.calls <- append([]agent.Message(nil), messages...):
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "done"}, FinishReason: "stop"}, nil
}

func (*monotonicResponsesPlanner) Model() string { return "test-model" }

// fak-test:runtime fast est=1s
func TestResponsesCompactionMonotonicOverHTTP(t *testing.T) {
	srv := newTestServer(t)
	planner := &monotonicResponsesPlanner{calls: make(chan []agent.Message, 1)}
	srv.planner = planner
	// A low explicit budget crosses the same boundary the default CLI budget
	// crosses for sufficiently long histories, without a giant fixture.
	srv.compactHistoryBudget = 1
	server := httptest.NewServer(srv.Handler())
	t.Cleanup(server.Close)
	client := &http.Client{Timeout: 5 * time.Second}
	const trace = "responses-monotonic"
	const system = "Pinned system instructions."
	large := strings.Repeat("An ordinary historical explanation. ", 200)
	input := []agent.Message{
		{Role: agent.RoleUser, Content: "Initial task."},
		{Role: agent.RoleAssistant, Content: "ok"},
		{Role: agent.RoleAssistant, Content: large},
		{Role: agent.RoleUser, Content: "yes"},
		{Role: agent.RoleAssistant, Content: "Recent answer."},
		{Role: agent.RoleUser, Content: "Continue."},
	}
	post := func(messages []agent.Message) []agent.Message {
		t.Helper()
		items := make([]map[string]any, 0, len(messages))
		for _, m := range messages {
			items = append(items, map[string]any{"type": "message", "role": m.Role, "content": m.Content})
		}
		body, err := json.Marshal(map[string]any{"model": "test-model", "instructions": system, "input": items})
		if err != nil {
			t.Fatal(err)
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, server.URL+"/v1/responses", bytes.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set(traceHeader, trace)
		resp, err := client.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		responseBody, err := io.ReadAll(resp.Body)
		if err != nil || resp.StatusCode != http.StatusOK {
			t.Fatalf("status=%d body=%s error=%v", resp.StatusCode, responseBody, err)
		}
		select {
		case messages := <-planner.calls:
			return messages
		default:
			t.Fatal("HTTP request did not reach the planner")
			return nil
		}
	}

	first := post(input)
	before := append([]agent.Message{{Role: agent.RoleSystem, Content: system}}, input...)
	if len(first) != len(before) {
		t.Fatalf("message count=%d, want %d", len(first), len(before))
	}
	ref := ""
	for i := range first {
		if i == 3 { // the only profitable, unprotected middle message
			var ok bool
			ref, _, _, ok = agent.ParseResponsesCompactStub(first[i].Content)
			if !ok || first[i].Content == large {
				t.Fatal("long middle message was not compacted")
			}
			continue
		}
		if !reflect.DeepEqual(first[i], before[i]) {
			t.Fatalf("short/protected message %d changed", i)
		}
	}
	if agent.EstimateResponsesMessagesTokens(first) >= agent.EstimateResponsesMessagesTokens(before) {
		t.Fatal("first HTTP compaction did not reduce estimated tokens")
	}
	restored, err := srv.restoreFromStash(trace, ref)
	if err != nil || restored.Bytes != large {
		t.Fatalf("original payload restore failed: %v", err)
	}

	// Echo the compacted history through the actual public ingress. Newly
	// eligible short messages and the prior stub must retain their exact bytes.
	nextInput := append(append([]agent.Message(nil), first[1:]...),
		agent.Message{Role: agent.RoleAssistant, Content: "Another short answer."},
		agent.Message{Role: agent.RoleUser, Content: "Continue again."})
	second := post(nextInput)
	wantSecond := append([]agent.Message{{Role: agent.RoleSystem, Content: system}}, nextInput...)
	if !reflect.DeepEqual(second, wantSecond) {
		t.Fatal("later HTTP turn rewrote a prior decision or expanded short context")
	}
	restored, err = srv.restoreFromStash(trace, ref)
	if err != nil || restored.Bytes != large {
		t.Fatalf("later turn lost the original restore target: %v", err)
	}
}
