package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"
)

// fak-test:runtime fast est=1s
func TestHTTPPlannerResponsesNamespaceRoundTrip(t *testing.T) {
	t.Setenv("FAK_PROVIDER_EXTRA_BODY_JSON", "")
	t.Setenv("FAK_PLANNER_MAX_ATTEMPTS", "1")
	for _, tc := range []struct {
		name        string
		namespaces  []string
		restoreJSON bool
	}{
		{name: "direct_namespaced", namespaces: []string{"mcp__fak_guard"}},
		{name: "restored_namespaced", namespaces: []string{"mcp__fak_guard"}, restoreJSON: true},
		{name: "restored_same_name_distinct_namespaces", namespaces: []string{"left", "right"}, restoreJSON: true},
		{name: "restored_omitted_namespace", namespaces: []string{""}, restoreJSON: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const args = `{"city":"SFO"}`
			var requests atomic.Int32
			ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				n := requests.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/responses" {
					t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
				}
				var request struct {
					Input []map[string]json.RawMessage `json:"input"`
				}
				if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
					t.Errorf("decode request: %v", err)
				}
				w.Header().Set("Content-Type", "application/json")
				if n == 1 {
					items := make([]map[string]any, 0, len(tc.namespaces))
					for i, ns := range tc.namespaces {
						item := map[string]any{"type": "function_call", "call_id": fmt.Sprintf("call_%d", i), "name": "lookup", "arguments": args}
						if ns != "" {
							item["namespace"] = ns
						}
						items = append(items, item)
					}
					_ = json.NewEncoder(w).Encode(map[string]any{"status": "completed", "output": items})
					return
				}
				if n != 2 {
					t.Errorf("unexpected extra request %d", n)
				}
				calls, outputs := 0, 0
				for _, item := range request.Input {
					field := func(key string) string { var value string; _ = json.Unmarshal(item[key], &value); return value }
					switch field("type") {
					case "function_call":
						if calls >= len(tc.namespaces) {
							t.Errorf("extra call: %v", item)
							continue
						}
						wantNS := tc.namespaces[calls]
						_, present := item["namespace"]
						if field("namespace") != wantNS || present != (wantNS != "") || field("call_id") != fmt.Sprintf("call_%d", calls) || field("name") != "lookup" || field("arguments") != args {
							t.Errorf("next-turn function identity changed: %v; namespace=%q", item, wantNS)
						}
						calls++
					case "function_call_output":
						if _, present := item["namespace"]; present {
							t.Error("namespace leaked onto tool result")
						}
						if field("call_id") != fmt.Sprintf("call_%d", outputs) || field("output") != "sunny" {
							t.Errorf("tool result link changed: %v", item)
						}
						outputs++
					}
				}
				if calls != len(tc.namespaces) || outputs != len(tc.namespaces) {
					t.Errorf("next-turn calls=%d outputs=%d, expected %d each", calls, outputs, len(tc.namespaces))
				}
				_, _ = w.Write([]byte(`{"status":"completed","output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"done"}]}]}`))
			}))
			defer ts.Close()
			planner, err := NewProviderHTTPPlanner(string(ProviderOpenAIResponses), ts.URL, "gpt-test", "loopback-only")
			if err != nil {
				t.Fatal(err)
			}
			planner.Client = ts.Client()
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			messages := []Message{{Role: RoleUser, Content: "look up SFO"}}
			comp, err := planner.Complete(ctx, messages, adapterTestTools())
			if err != nil || comp == nil {
				t.Fatalf("first completion=%+v error=%v", comp, err)
			}
			if len(comp.Message.ToolCalls) != len(tc.namespaces) {
				t.Fatalf("calls=%+v", comp.Message.ToolCalls)
			}
			for i, ns := range tc.namespaces {
				call := comp.Message.ToolCalls[i]
				if call.Function.Namespace != ns || call.ID != fmt.Sprintf("call_%d", i) {
					t.Fatalf("first identity changed: %+v", call)
				}
			}
			messages = append(messages, comp.Message)
			for _, call := range comp.Message.ToolCalls {
				messages = append(messages, Message{Role: RoleTool, Name: call.Function.Name, ToolCallID: call.ID, Content: "sunny"})
			}
			if tc.restoreJSON {
				wire, err := json.Marshal(messages)
				if err != nil {
					t.Fatal(err)
				}
				messages = nil
				if err := json.Unmarshal(wire, &messages); err != nil {
					t.Fatal(err)
				}
			}
			comp, err = planner.Complete(ctx, messages, adapterTestTools())
			if err != nil || comp == nil || comp.Message.Content != "done" || comp.FinishReason != "stop" {
				t.Fatalf("second completion=%+v error=%v", comp, err)
			}
			if requests.Load() != 2 {
				t.Fatalf("requests=%d, want two", requests.Load())
			}
		})
	}
}

// fak-test:runtime fast est=1ms
func TestFuncNamespaceDecodeCompatibility(t *testing.T) {
	for _, tc := range []struct{ name, raw, namespace, args string }{
		{name: "object", raw: `{"name":"lookup","namespace":"guard","arguments":{"x":1}}`, namespace: "guard", args: `{"x":1}`},
		{name: "string", raw: `{"name":"lookup","namespace":"guard","arguments":"{\"x\":1}"}`, namespace: "guard", args: `{"x":1}`},
		{name: "null_arguments", raw: `{"name":"lookup","namespace":"guard","arguments":null}`, namespace: "guard"},
		{name: "omitted_namespace", raw: `{"name":"lookup","arguments":{}}`, args: `{}`},
		{name: "empty_namespace", raw: `{"name":"lookup","namespace":"","arguments":{}}`, args: `{}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := Func{Namespace: "old", Arguments: "old"}
			if err := json.Unmarshal([]byte(tc.raw), &f); err != nil {
				t.Fatal(err)
			}
			if f.Name != "lookup" || f.Namespace != tc.namespace || f.Arguments != tc.args {
				t.Fatalf("decoded function=%+v", f)
			}
		})
	}
}
