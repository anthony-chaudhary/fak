package gateway

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// fak-test:runtime fast est=1s lane=default
func TestNativeHTTPChildInheritsBoundPlanner(t *testing.T) {
	t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
	t.Setenv("FAK_CHILD_ROUTE_TEST_KEY", "selected-test-key")
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})

	var bootCalls, selectedParents, selectedChildren atomic.Int32
	boot := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bootCalls.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"wrong-boot-child"},"finish_reason":"stop"}]}`))
	}))
	defer boot.Close()
	selected := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model    string          `json:"model"`
			Messages []agent.Message `json:"messages"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		if req.Model != "selected-upstream-model" || r.Header.Get("Authorization") != "Bearer selected-test-key" {
			t.Errorf("wrong selected account wire: model=%q auth=%q", req.Model, r.Header.Get("Authorization"))
			http.Error(w, "wrong selected account", http.StatusBadRequest)
			return
		}
		var task string
		results := map[string]string{}
		for _, m := range req.Messages {
			if m.Role == agent.RoleUser && strings.HasPrefix(m.Content, "route-") {
				task = m.Content
			}
			if m.Role == agent.RoleTool {
				results[m.Name] = m.Content
			}
		}
		message := agent.Message{Role: agent.RoleAssistant}
		finish := "stop"
		switch {
		case strings.HasPrefix(task, "route-child"):
			selectedChildren.Add(1)
			message.Content = "selected-child-result"
		case strings.HasPrefix(task, "route-parent"):
			selectedParents.Add(1)
			if raw, ok := results[agent.ToolTaskWait]; ok {
				var receipt agent.TaskWaitReceipt
				if err := json.Unmarshal([]byte(raw), &receipt); err != nil {
					http.Error(w, err.Error(), http.StatusBadRequest)
					return
				}
				child := receipt.Tasks["route-child-id"]
				if child == nil || child.State != agent.TaskStateCompleted || child.Result != "selected-child-result" {
					http.Error(w, "child did not complete on the selected planner", http.StatusBadRequest)
					return
				}
				message.Content = "joined-selected-child"
			} else {
				tool := agent.ToolTaskSpawn
				args := map[string]any{"task_id": "route-child-id", "prompt": "route-child", "read_only": true}
				if _, ok := results[agent.ToolTaskSpawn]; ok {
					tool = agent.ToolTaskWait
					args = map[string]any{"task_id": "route-child-id", "timeout_ms": 5000}
				}
				raw, _ := json.Marshal(args)
				message.ToolCalls = []agent.ToolCall{{ID: "call-" + tool, Type: "function", Function: agent.Func{Name: tool, Arguments: string(raw)}}}
				finish = "tool_calls"
			}
		default:
			http.Error(w, "unexpected task", http.StatusBadRequest)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{{"message": message, "finish_reason": finish}},
			"usage":   map[string]int{"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
		})
	}))
	defer selected.Close()
	roster := &modelroute.Roster{
		Version: modelroute.RosterVersion,
		Accounts: []modelroute.Account{{ID: "selected", Kind: modelroute.KindOpenAI, BaseURL: selected.URL,
			CredEnv: "FAK_CHILD_ROUTE_TEST_KEY", Principals: []string{"tenant-child"}}},
		Bindings: []modelroute.Binding{{Model: "requested-model", Account: "selected", UpstreamModel: "selected-upstream-model"}},
		Default:  "selected",
	}
	srv, err := New(Config{EngineID: "localtools", Model: "boot-model", Provider: "openai-compatible", BaseURL: boot.URL,
		APIKey: "boot-test-key", Native: true, NativeMaxTurns: 4, NativeCodeWorkspace: t.TempDir(), VDSO: true,
		RouteAccounts: roster, KeyPrincipals: map[string]string{"gateway-test-key": "tenant-child"}})
	if err != nil {
		t.Fatal(err)
	}
	defer srv.Close()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	body := []byte(`{"model":"requested-model","max_tokens":128,"messages":[{"role":"user","content":"route-parent"}]}`)
	req := httptest.NewRequest(http.MethodPost, "/v1/messages", bytes.NewReader(body)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "gateway-test-key")
	req.Header.Set("X-Trace-Id", "native-bound-child-route")
	recorder := httptest.NewRecorder()
	srv.Handler().ServeHTTP(recorder, req)
	if recorder.Code != http.StatusOK || !bytes.Contains(recorder.Body.Bytes(), []byte("joined-selected-child")) {
		t.Fatalf("bound native request failed: status=%d body=%s", recorder.Code, recorder.Body.Bytes())
	}
	if bootCalls.Load() != 0 || selectedParents.Load() != 3 || selectedChildren.Load() != 1 {
		t.Fatalf("planner calls boot/parent/child=%d/%d/%d, want 0/3/1", bootCalls.Load(), selectedParents.Load(), selectedChildren.Load())
	}
}
