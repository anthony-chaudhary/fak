package gateway

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

func offeredWireNames(names ...string) agent.OfferedTools {
	return agent.NewOfferedTools(offeredWireDefs(names...))
}

func offeredWireDefs(names ...string) []agent.ToolDef {
	tools := make([]agent.ToolDef, len(names))
	for i, name := range names {
		tools[i] = agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name, Parameters: json.RawMessage(`{"type":"object"}`)}}
	}
	return tools
}

// fak-test:runtime fast est=1s lane=default
func TestOfferedToolHTTPAndStreamBoundary(t *testing.T) {
	const content = `before <tool_call>{"name":"Bash","arguments":{}}</tool_call> after`
	for _, mode := range []string{"buffered", "stream-json", "stream-sse"} {
		t.Run(mode, func(t *testing.T) {
			up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				quoted, _ := json.Marshal(content)
				if mode == "stream-sse" {
					w.Header().Set("Content-Type", "text/event-stream")
					fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%s},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", quoted)
				} else {
					w.Header().Set("Content-Type", "application/json")
					fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}]}`, quoted)
				}
			}))
			defer up.Close()
			planner := agent.NewHTTPPlanner(up.URL, "model", "")
			for _, tools := range [][]agent.ToolDef{nil, offeredWireDefs("Read")} {
				var comp *agent.Completion
				var err error
				var streamed strings.Builder
				if mode == "buffered" {
					comp, err = planner.Complete(context.Background(), nil, tools)
				} else {
					comp, err = planner.CompleteStream(context.Background(), func(piece string) error { streamed.WriteString(piece); return nil }, nil, tools)
				}
				if err != nil {
					t.Fatal(err)
				}
				if comp.Message.Content != content || len(comp.Message.ToolCalls) != 0 || comp.ToolCallsDropped {
					t.Fatalf("unoffered content changed: %+v", comp)
				}
				if mode != "buffered" && streamed.String() != content {
					t.Fatalf("stream = %q, want %q", streamed.String(), content)
				}
			}
		})
	}
}

// fak-test:runtime fast est=1s lane=default
func TestOfferedToolGatewayExtensionSnapshot(t *testing.T) {
	name := "read_file"
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		payload, _ := json.Marshal(map[string]any{"name": name, "arguments": map[string]any{}})
		content, _ := json.Marshal("<tool_call>" + string(payload) + "</tool_call>")
		fmt.Fprintf(w, `{"choices":[{"message":{"content":%s},"finish_reason":"tool_calls"}]}`, content)
	}))
	defer up.Close()
	planner := agent.NewHTTPPlanner(up.URL, "model", "")
	tools := offeredWireDefs("read")
	bound := withTextToolOffer(context.Background(), tools)
	tools[0].Function.Name = "write"
	for _, tc := range []struct {
		name  string
		ctx   context.Context
		tools []agent.ToolDef
		calls int
	}{
		{"alias", bound, offeredWireDefs("read"), 1},
		{"duplicate target", withTextToolOffer(bound, offeredWireDefs("read", "read")), offeredWireDefs("read", "read"), 0},
		{"empty inherited", bound, nil, 0},
		{"replaced request", withTextToolOffer(bound, offeredWireDefs("write")), offeredWireDefs("write"), 0},
		{"declared exact", withTextToolOffer(bound, offeredWireDefs("read_file", "read", "read")), offeredWireDefs("read_file", "read", "read"), 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			comp, err := planner.Complete(tc.ctx, nil, tc.tools)
			if err != nil {
				t.Fatal(err)
			}
			if len(comp.Message.ToolCalls) != tc.calls {
				t.Fatalf("calls=%d want=%d", len(comp.Message.ToolCalls), tc.calls)
			}
			if tc.calls == 0 && (!comp.ToolCallsDropped || comp.ToolCallsDroppedReason != abi.ReasonUnknownTool) {
				t.Fatalf("typed refusal lost: %+v", comp)
			}
		})
	}
	name = "mcp__guard__fak_context_restore"
	comp, err := planner.Complete(withTextToolOffer(context.Background(), offeredWireDefs("bash")), nil, offeredWireDefs("bash"))
	if err != nil || comp == nil || len(comp.Message.ToolCalls) != 1 {
		t.Fatalf("served restore dialect: %+v %v", comp, err)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestOfferedToolGatewayTypedRefusal(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"<tool_call>{\"name\":\"unoffered\",\"arguments\":{}}</tool_call>"},"finish_reason":"tool_calls"}]}`)
	}))
	defer up.Close()
	srv := newTestServer(t)
	srv.planner = agent.NewHTTPPlanner(up.URL, "model", "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	var result map[string]any
	body := json.RawMessage(`{"model":"model","messages":[{"role":"user","content":"inspect"}],"tools":[{"type":"function","function":{"name":"allow_a","parameters":{"type":"object"}}}]}`)
	code := postJSON(t, ts.URL+"/v1/chat/completions", body, &result)
	encoded, _ := json.Marshal(result)
	if code != http.StatusBadGateway || !strings.Contains(string(encoded), "UNKNOWN_TOOL") {
		t.Fatalf("status=%d body=%s", code, encoded)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestOfferedToolAliasThroughHandler(t *testing.T) {
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"choices":[{"message":{"content":"<tool_call>{\"name\":\"read_file\",\"arguments\":{\"path\":\"go.mod\"}}</tool_call>"},"finish_reason":"stop"}]}`)
	}))
	defer up.Close()
	srv := newTestServer(t)
	abi.RegisterAdjudicator(1, readAdj{})
	srv.planner = agent.NewHTTPPlanner(up.URL, "model", "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	var resp ChatResponse
	body := json.RawMessage(`{"model":"model","messages":[{"role":"user","content":"inspect"}],"tools":[{"type":"function","function":{"name":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]}`)
	if code := postJSON(t, ts.URL+"/v1/chat/completions", body, &resp); code != http.StatusOK {
		t.Fatalf("status=%d", code)
	}
	if len(resp.Choices) != 1 || len(resp.Choices[0].Message.ToolCalls) != 1 || resp.Choices[0].Message.ToolCalls[0].Function.Name != "read" {
		t.Fatalf("alias was not admitted under offered name: %+v", resp)
	}
}

// fak-test:runtime fast est=1s lane=default
func TestOfferedRestoreThroughResponsesContinuation(t *testing.T) {
	const trace = "offered-restore-continuation"
	const saved = "bounded saved evidence"
	id := ctxplan.Digest([]byte(saved))
	var requests atomic.Int32
	var sawRestore atomic.Bool
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		text := "Finished."
		if requests.Add(1) == 1 {
			text = `<tool_call>{"name":"mcp__guard__fak_context_restore","arguments":{"id":"` + id + `"}}</tool_call>`
		} else if strings.Contains(string(body), saved) {
			sawRestore.Store(true)
		}
		quoted, _ := json.Marshal(text)
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprintf(w, `{"choices":[{"message":{"role":"assistant","content":%s},"finish_reason":"stop"}]}`, quoted)
	}))
	defer up.Close()
	srv := newTestServer(t)
	srv.stashRestore(trace, id, "evidence", []byte(saved))
	srv.planner = agent.NewHTTPPlanner(up.URL, "model", "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	resp := postResponsesTrace(t, ts.URL, trace, map[string]any{"model": "model", "input": "Use saved evidence.", "tools": []map[string]any{{"type": "function", "name": "allow_a", "parameters": map[string]any{"type": "object"}}}})
	if requests.Load() != 2 || !sawRestore.Load() || resp.OutputText != "Finished." {
		t.Fatalf("continuation requests=%d restored=%v response=%+v", requests.Load(), sawRestore.Load(), resp)
	}
	if strings.Contains(resp.OutputText, saved) {
		t.Fatal("restored bytes escaped as answer")
	}
}

// fak-test:runtime fast est=1s lane=default
func TestUnofferedStreamThroughHandler(t *testing.T) {
	const text = `before <tool_call>{"name":"Bash","arguments":{}}</tool_call> after`
	up := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		quoted, _ := json.Marshal(text)
		fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%s},\"finish_reason\":null}]}\n\ndata: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n", quoted)
	}))
	defer up.Close()
	srv := newTestServer(t)
	srv.planner = agent.NewHTTPPlanner(up.URL, "model", "")
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()
	body := `{"model":"model","stream":true,"messages":[{"role":"user","content":"inspect"}],"tools":[{"type":"function","function":{"name":"Read","parameters":{"type":"object"}}}]}`
	resp, err := http.Post(ts.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d body=%s", resp.StatusCode, raw)
	}
	var out strings.Builder
	for _, line := range strings.Split(string(raw), "\n") {
		if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
			continue
		}
		var chunk struct {
			Choices []struct {
				Delta struct {
					Content   string            `json:"content"`
					ToolCalls []json.RawMessage `json:"tool_calls"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if err := json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &chunk); err != nil {
			t.Fatal(err)
		}
		for _, choice := range chunk.Choices {
			if len(choice.Delta.ToolCalls) != 0 {
				t.Fatal("unoffered call escaped")
			}
			out.WriteString(choice.Delta.Content)
		}
	}
	if out.String() != text {
		t.Fatalf("wire=%q want=%q", out.String(), text)
	}
}
