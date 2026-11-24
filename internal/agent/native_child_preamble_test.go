package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type prefixStableRequest struct {
	messages []Message
	tools    []ToolDef
}

type prefixStableParentPlanner struct {
	step  int
	first prefixStableRequest
}

func (p *prefixStableParentPlanner) Model() string { return "prefix-stable-parent" }

func (p *prefixStableParentPlanner) Complete(_ context.Context, messages []Message, tools []ToolDef, _ ...SampleOpt) (*Completion, error) {
	call := func(name string, args any) (*Completion, error) {
		raw, _ := json.Marshal(args)
		return &Completion{Message: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: fmt.Sprintf("parent-%d", p.step), Type: "function", Function: Func{Name: name, Arguments: string(raw)}}}}, FinishReason: "tool_calls"}, nil
	}
	stage := p.step
	p.step++
	switch stage {
	case 0:
		p.first = prefixStableRequest{append([]Message(nil), messages...), append([]ToolDef(nil), tools...)}
		return call(ToolTaskSpawn, TaskSpawnRequest{TaskID: "explore-child", Prompt: "List the package entry points.", Description: "map entry points", SubagentType: "explore", ReadOnly: true})
	case 1:
		all := true
		return call(ToolTaskWait, TaskWaitRequest{TaskID: "explore-child", WaitAll: &all, TimeoutMs: 5000})
	default:
		return &Completion{Message: Message{Role: RoleAssistant, Content: "parent done"}, FinishReason: "stop"}, nil
	}
}

type prefixStableChildPlanner struct {
	mu       sync.Mutex
	requests []prefixStableRequest
	reply    Message
}

func (p *prefixStableChildPlanner) Model() string { return "prefix-stable-child" }

func (p *prefixStableChildPlanner) Complete(_ context.Context, messages []Message, tools []ToolDef, _ ...SampleOpt) (*Completion, error) {
	p.mu.Lock()
	p.requests = append(p.requests, prefixStableRequest{append([]Message(nil), messages...), append([]ToolDef(nil), tools...)})
	p.mu.Unlock()
	finish := "stop"
	if len(p.reply.ToolCalls) > 0 {
		finish = "tool_calls"
	}
	return &Completion{Message: p.reply, FinishReason: finish}, nil
}

func prefixStableTool(name string) ToolDef {
	return ToolDef{Type: "function", Function: ToolDefFunction{
		Name:        name,
		Description: name + " tool",
		Parameters:  json.RawMessage(`{"type":"object","properties":{"path":{"type":"string"}}}`),
	}}
}

func runPrefixStableParent(t *testing.T, child *prefixStableChildPlanner) (*prefixStableParentPlanner, *TaskItem) {
	t.Helper()
	DisarmTaskTools()
	t.Cleanup(DisarmTaskTools)
	catalog := []ToolDef{prefixStableTool(codetools.ToolRead), prefixStableTool(codetools.ToolWrite), prefixStableTool(codetools.ToolGrep)}
	runner := NewNativeChildTaskRunner(child, 3, nil, catalog, adjudicator.Policy{Posture: adjudicator.PostureFailClosed})
	var item *TaskItem
	wrapped := func(ctx context.Context, req ChildTaskRunRequest) (any, error) {
		result, err := runner(ctx, req)
		item = &TaskItem{ID: req.TaskID, Result: result}
		if err != nil {
			item.Error = err.Error()
		}
		return result, err
	}
	parent := &prefixStableParentPlanner{}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	if _, err := RunArm(ctx, parent, "parent task", true, 4, nil,
		WithSystemPrompt("Parent standing instruction shared by every subagent."),
		WithToolCatalog(catalog),
		WithChildTaskRunner(1, 1, wrapped)); err != nil {
		t.Fatal(err)
	}
	if item == nil {
		t.Fatal("child never ran")
	}
	return parent, item
}

// fak-test:runtime fast est=1s
func TestNativeChildPrefixStablePreamble(t *testing.T) {
	child := &prefixStableChildPlanner{reply: Message{Role: RoleAssistant, Content: "child done"}}
	parent, item := runPrefixStableParent(t, child)
	if item.Error != "" || item.Result != "child done" {
		t.Fatalf("child result changed: %+v", item)
	}
	if len(child.requests) != 1 {
		t.Fatalf("child requests = %d, want 1", len(child.requests))
	}
	got := child.requests[0]

	parentRendered := renderChatMLTools(parent.first.messages, parent.first.tools)
	childRendered := renderChatMLTools(got.messages, got.tools)
	shared := inKernelSharedPrefixText(parentRendered)
	if shared == "" || !strings.Contains(shared, ToolTaskSpawn) {
		t.Fatalf("parent has no system/tools block: %q", parentRendered)
	}
	if !strings.HasPrefix(childRendered, shared) {
		t.Fatalf("child diverges inside the parent's system/tools block\nparent: %q\nchild:  %q", shared, childRendered[:min(len(childRendered), len(shared))])
	}
	parentWire, _ := json.Marshal(parent.first.tools)
	childWire, _ := json.Marshal(got.tools)
	if string(parentWire) != string(childWire) {
		t.Fatalf("child tool schemas differ from the parent's on the wire")
	}

	user := got.messages[len(got.messages)-1]
	if user.Role != RoleUser || !strings.Contains(user.Content, "role: explore") ||
		!strings.Contains(user.Content, "Tools you may call: Read, Grep.") ||
		!strings.HasSuffix(user.Content, "List the package entry points.") {
		t.Fatalf("role brief missing from the child user turn: %+v", got.messages)
	}
	for _, m := range got.messages[1:] {
		if m.Role == RoleSystem {
			t.Fatalf("child added a system message that would fold into the shared block: %+v", got.messages)
		}
	}
}

// fak-test:runtime fast est=1s
func TestNativeChildPrefixStableKeepsCapabilityFloor(t *testing.T) {
	child := &prefixStableChildPlanner{reply: Message{Role: RoleAssistant, ToolCalls: []ToolCall{{
		ID: "child-write", Type: "function",
		Function: Func{Name: codetools.ToolWrite, Arguments: `{"path":"x"}`},
	}}}}
	_, item := runPrefixStableParent(t, child)
	if !strings.Contains(item.Error, "outside the inherited child capability floor") {
		t.Fatalf("read-only child reached Write through the shared declaration: %+v", item)
	}
}
