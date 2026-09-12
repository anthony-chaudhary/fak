package gateway

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
)

type servedChildIndependentPlanner struct {
	ready       chan string
	statusSeen  chan string
	allStarted  chan struct{}
	release     map[string]chan struct{}
	cancelled   chan string
	childCalls  atomic.Int32
	cancelAlpha bool
}

func (p *servedChildIndependentPlanner) Model() string { return "served-child-witness" }

func (p *servedChildIndependentPlanner) StreamingSupported() bool { return true }

func (p *servedChildIndependentPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	completion, err := p.Complete(ctx, messages, tools, opts...)
	if err == nil && completion != nil && completion.Message.Content != "" && sink != nil {
		err = sink(completion.Message.Content)
	}
	return completion, err
}

func (p *servedChildIndependentPlanner) Complete(ctx context.Context, messages []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	var task string
	for _, m := range messages {
		if m.Role == agent.RoleUser && (strings.HasPrefix(m.Content, "served-parent:") || strings.HasPrefix(m.Content, "served-child:")) {
			task = m.Content
			break
		}
	}
	names := map[string]bool{}
	for _, tool := range tools {
		names[tool.Function.Name] = true
	}
	final := func(text string) (*agent.Completion, error) {
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: text}, FinishReason: "stop"}, nil
	}
	if strings.HasPrefix(task, "served-child:") {
		name := strings.TrimPrefix(task, "served-child:")
		if names["client_only_capability"] {
			return nil, fmt.Errorf("client capability leaked into child")
		}
		for _, tool := range []string{agent.ToolTaskSpawn, agent.ToolTaskStatus, agent.ToolTaskWait, agent.ToolTaskCancel} {
			if names[tool] {
				return nil, fmt.Errorf("nested task capability leaked: %s", tool)
			}
		}
		if !names["Read"] {
			return nil, fmt.Errorf("server-owned Read absent from child")
		}
		gate, ok := p.release[name]
		if !ok {
			return nil, fmt.Errorf("unexpected child task %q", task)
		}
		p.childCalls.Add(1)
		p.ready <- name
		select {
		case <-gate:
			return final("child-result:" + name)
		case <-ctx.Done():
			p.cancelled <- name
			// Model outstanding child effects: parent cleanup must join this exit.
			<-gate
			return nil, ctx.Err()
		}
	}
	if !strings.HasPrefix(task, "served-parent:") {
		return nil, fmt.Errorf("unexpected owned task %q", task)
	}
	name := strings.TrimPrefix(task, "served-parent:")
	for _, tool := range []string{agent.ToolTaskSpawn, agent.ToolTaskStatus, agent.ToolTaskWait, agent.ToolTaskCancel} {
		if !names[tool] {
			return nil, fmt.Errorf("server did not advertise %s", tool)
		}
	}
	results := map[string]string{}
	for _, m := range messages {
		if m.Role == agent.RoleTool {
			results[m.Name] = m.Content
		}
	}
	call := func(tool string, args any) (*agent.Completion, error) {
		body, err := json.Marshal(args)
		if err != nil {
			return nil, err
		}
		return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, ToolCalls: []agent.ToolCall{{ID: "same-call-" + tool, Type: "function", Function: agent.Func{Name: tool, Arguments: string(body)}}}}, FinishReason: "tool_calls"}, nil
	}
	if raw, ok := results[agent.ToolTaskCancel]; ok {
		var receipt agent.TaskCancelReceipt
		if err := json.Unmarshal([]byte(raw), &receipt); err != nil || !receipt.Cancelled {
			return nil, fmt.Errorf("cancel receipt %s: %v", raw, err)
		}
		return final("cancelled:" + name)
	}
	if raw, ok := results[agent.ToolTaskWait]; ok {
		var receipt agent.TaskWaitReceipt
		if err := json.Unmarshal([]byte(raw), &receipt); err != nil {
			return nil, err
		}
		child := receipt.Tasks["same-child"]
		if child == nil || child.State != agent.TaskStateCompleted || child.Result != "child-result:"+name {
			return nil, fmt.Errorf("cross-parent or incomplete wait: %s", raw)
		}
		return final(child.Result.(string))
	}
	if raw, ok := results[agent.ToolTaskStatus]; ok {
		var receipt agent.TaskStatusReceipt
		if err := json.Unmarshal([]byte(raw), &receipt); err != nil {
			return nil, err
		}
		if receipt.Total != 1 || len(receipt.Tasks) != 1 || receipt.Tasks[0].Prompt != "served-child:"+name || receipt.Tasks[0].State != agent.TaskStateRunning {
			return nil, fmt.Errorf("cross-parent or stale status: %s", raw)
		}
		p.statusSeen <- name
		if p.cancelAlpha && name == "alpha" {
			return call(agent.ToolTaskCancel, map[string]any{"task_id": "same-child"})
		}
		return call(agent.ToolTaskWait, map[string]any{"task_id": "same-child", "timeout_ms": 5000})
	}
	if _, ok := results[agent.ToolTaskSpawn]; ok {
		select {
		case <-p.allStarted:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
		return call(agent.ToolTaskStatus, map[string]any{"task_id": "same-child"})
	}
	return call(agent.ToolTaskSpawn, map[string]any{"task_id": "same-child", "idempotency_key": "same-key", "prompt": "served-child:" + name, "read_only": false})
}

type servedChildHTTPResult struct {
	name   string
	status int
	body   []byte
	err    error
}

func TestNativeHTTPChildTasksAreRequestScopedAndJoined(t *testing.T) {
	for _, tc := range []struct{ cancelAlpha, stream, requestCancel bool }{{false, false, false}, {true, false, false}, {false, true, false}, {true, true, false}, {false, false, true}, {false, true, true}} {
		cancelAlpha, stream, requestCancel := tc.cancelAlpha, tc.stream, tc.requestCancel
		t.Run(fmt.Sprintf("cancel_alpha_%v_stream_%v_request_cancel_%v", cancelAlpha, stream, requestCancel), func(t *testing.T) {
			t.Setenv("FAK_SESSION_LEDGER_DIR", t.TempDir())
			agent.Configure()
			abi.RegisterRegionBackend(inlineBackend{})
			p := &servedChildIndependentPlanner{
				ready: make(chan string, 2), statusSeen: make(chan string, 2), allStarted: make(chan struct{}), cancelled: make(chan string, 2),
				release: map[string]chan struct{}{"alpha": make(chan struct{}), "beta": make(chan struct{})}, cancelAlpha: cancelAlpha,
			}
			var startOnce, alphaOnce, betaOnce sync.Once
			start := func() { startOnce.Do(func() { close(p.allStarted) }) }
			releaseAlpha := func() { alphaOnce.Do(func() { close(p.release["alpha"]) }) }
			releaseBeta := func() { betaOnce.Do(func() { close(p.release["beta"]) }) }
			ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
			alphaCtx, cancelRequestAlpha := context.WithCancel(ctx)
			defer cancelRequestAlpha()
			srv, err := New(Config{EngineID: "localtools", Model: p.Model(), Native: true, NativeMaxTurns: 8, NativeCodeWorkspace: t.TempDir(), VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			srv.planner = p
			ts := httptest.NewServer(srv.Handler())
			defer func() {
				cancel()
				start()
				releaseAlpha()
				releaseBeta()
				ts.Close()
				srv.Close()
			}()
			results := make(chan servedChildHTTPResult, 2)
			for _, name := range []string{"alpha", "beta"} {
				go func(name string) {
					body, _ := json.Marshal(map[string]any{
						"model": p.Model(), "max_tokens": 128, "stream": stream, "messages": []map[string]string{{"role": "user", "content": "served-parent:" + name}},
						"tools": []map[string]any{{"name": "client_only_capability", "description": "A client declaration, never an owned child capability.", "input_schema": map[string]any{"type": "object", "properties": map[string]any{}}}},
					})
					reqCtx := ctx
					if requestCancel && name == "alpha" {
						reqCtx = alphaCtx
					}
					req, _ := http.NewRequestWithContext(reqCtx, http.MethodPost, ts.URL+"/v1/messages", bytes.NewReader(body))
					req.Header.Set("Content-Type", "application/json")
					req.Header.Set("X-Trace-Id", fmt.Sprintf("served-child-independent-%t-%t-%s", cancelAlpha, stream, name))
					// Direct HTTP handler dispatch lets this witness observe server cleanup,
					// which a cancelled network client cannot wait for.
					if requestCancel {
						recorder := httptest.NewRecorder()
						srv.Handler().ServeHTTP(recorder, req)
						results <- servedChildHTTPResult{name: name, status: recorder.Code, body: recorder.Body.Bytes()}
						return
					}
					resp, err := ts.Client().Do(req)
					if err != nil {
						results <- servedChildHTTPResult{name: name, err: err}
						return
					}
					data, readErr := io.ReadAll(resp.Body)
					resp.Body.Close()
					results <- servedChildHTTPResult{name: name, status: resp.StatusCode, body: data, err: readErr}
				}(name)
			}
			awaitBoth := func(ch <-chan string, label string) {
				t.Helper()
				seen := map[string]bool{}
				for len(seen) < 2 {
					select {
					case name := <-ch:
						seen[name] = true
					case early := <-results:
						t.Fatalf("request ended before %s: %+v body=%s", label, early, early.body)
					case <-time.After(5 * time.Second):
						t.Fatalf("both parents did not reach %s; seen=%v", label, seen)
					}
				}
			}
			awaitBoth(p.ready, "real child planner entry")
			start()
			awaitBoth(p.statusSeen, "isolated running status")
			if requestCancel {
				cancelRequestAlpha()
			}
			if cancelAlpha || requestCancel {
				select {
				case name := <-p.cancelled:
					if name != "alpha" {
						t.Fatalf("wrong child cancelled: %s", name)
					}
				case <-time.After(5 * time.Second):
					t.Fatal("alpha child cancellation not observed")
				}
				select {
				case early := <-results:
					t.Fatalf("parent returned before child effects ceased: %s", early.name)
				case <-time.After(50 * time.Millisecond):
				}
			}
			releaseAlpha()
			var first servedChildHTTPResult
			select {
			case first = <-results:
			case <-time.After(5 * time.Second):
				t.Fatal("alpha request did not finish")
			}
			if first.name != "alpha" {
				t.Fatalf("beta ended before release: %+v", first)
			}
			check := func(result servedChildHTTPResult, want string) {
				t.Helper()
				if result.err != nil || result.status != 200 {
					t.Fatalf("%s: status=%d err=%v body=%s", result.name, result.status, result.err, result.body)
				}
				var response anthropicMessageResponse
				body := result.body
				if stream {
					body = nil
					for _, frame := range readAnthropicSSE(t, bufio.NewReader(bytes.NewReader(result.body))) {
						if frame.event == "message_stop" {
							body = []byte(frame.data)
						}
					}
				}
				if err := json.Unmarshal(body, &response); err != nil {
					t.Fatal(err)
				}
				if response.Fak == nil || response.Fak.NativeArm == nil {
					t.Fatalf("missing native arm: %s", result.body)
				}
				arm := response.Fak.NativeArm
				if arm.FinalAnswer != want || arm.ToolCalls != 3 || arm.Denies != 0 || arm.HitTurnCap {
					t.Fatalf("unexpected owned-loop result: %+v", arm)
				}
			}
			wantAlpha := "child-result:alpha"
			if cancelAlpha {
				wantAlpha = "cancelled:alpha"
			}
			if !requestCancel {
				check(first, wantAlpha)
			}
			select {
			case early := <-results:
				t.Fatalf("peer ended before independent release: %+v", early)
			case <-time.After(50 * time.Millisecond):
			}
			releaseBeta()
			select {
			case second := <-results:
				check(second, "child-result:beta")
			case <-time.After(5 * time.Second):
				t.Fatal("beta request did not finish")
			}
			if got := p.childCalls.Load(); got != 2 {
				t.Fatalf("real child planner calls=%d, want 2", got)
			}
		})
	}
}

func TestNativeChildTasksRequireOwnedWorkspace(t *testing.T) {
	agent.Configure()
	abi.RegisterRegionBackend(inlineBackend{})
	for _, stream := range []bool{false, true} {
		t.Run(fmt.Sprintf("stream_%v", stream), func(t *testing.T) {
			srv, err := New(Config{EngineID: "localtools", Model: "test-model", Native: true, NativeMaxTurns: 2, VDSO: true})
			if err != nil {
				t.Fatal(err)
			}
			defer srv.Close()
			srv.planner = servedChildActivationPlanner{plannerFunc(func(_ context.Context, _ []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
				for _, tool := range tools {
					switch tool.Function.Name {
					case agent.ToolTaskSpawn, agent.ToolTaskStatus, agent.ToolTaskWait, agent.ToolTaskCancel:
						return nil, fmt.Errorf("task capability armed without workspace: %s", tool.Function.Name)
					}
				}
				return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "no children"}, FinishReason: "stop"}, nil
			})}
			var metrics agent.ArmMetrics
			if stream {
				metrics, err = srv.runNativeArmStreamSeed(context.Background(), nativeWireSeed{Task: "hello"}, "no-workspace-stream", nil, nil)
			} else {
				metrics, err = srv.runNativeArmSeed(context.Background(), nativeWireSeed{Task: "hello"}, "no-workspace-buffered")
			}
			if err != nil || metrics.FinalAnswer != "no children" {
				t.Fatalf("metrics=%+v err=%v", metrics, err)
			}
		})
	}
}

type servedChildActivationPlanner struct{ plannerFunc }

func (p servedChildActivationPlanner) StreamingSupported() bool { return true }
func (p servedChildActivationPlanner) CompleteStream(ctx context.Context, sink agent.StreamSink, messages []agent.Message, tools []agent.ToolDef, opts ...agent.SampleOpt) (*agent.Completion, error) {
	completion, err := p.Complete(ctx, messages, tools, opts...)
	if err == nil && completion != nil && sink != nil {
		err = sink(completion.Message.Content)
	}
	return completion, err
}
