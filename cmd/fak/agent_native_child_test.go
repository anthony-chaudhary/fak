package main

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/adjudicator"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/codetools"
)

type nativeChildCapturePlanner struct {
	model string
	tools []agent.ToolDef
	calls int
}

func (p *nativeChildCapturePlanner) Model() string { return p.model }

func (p *nativeChildCapturePlanner) Complete(_ context.Context, _ []agent.Message, tools []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	p.tools = append([]agent.ToolDef(nil), tools...)
	return &agent.Completion{Message: agent.Message{Role: agent.RoleAssistant, Content: "child complete"}}, nil
}

func nativeChildTestTool(name string) agent.ToolDef {
	return agent.ToolDef{Type: "function", Function: agent.ToolDefFunction{Name: name}}
}

func nativeChildToolNames(defs []agent.ToolDef) []string {
	names := make([]string, 0, len(defs))
	for _, def := range defs {
		names = append(names, def.Function.Name)
	}
	return names
}

func TestNativeChildToolCatalogBoundsNestingAndReadOnly(t *testing.T) {
	parent := []agent.ToolDef{
		nativeChildTestTool(codetools.ToolRead),
		nativeChildTestTool(codetools.ToolWrite),
		nativeChildTestTool(codetools.ToolEdit),
		nativeChildTestTool(codetools.ToolBash),
		nativeChildTestTool(codetools.ToolGrep),
		nativeChildTestTool(codetools.ToolGlob),
		nativeChildTestTool(agent.ToolTaskSpawn),
		nativeChildTestTool(agent.ToolTaskWait),
		nativeChildTestTool(agent.ToolTaskStatus),
		nativeChildTestTool(agent.ToolTaskCancel),
		nativeChildTestTool("web_search"),
	}

	readWrite := nativeChildToolNames(nativeChildToolCatalog(parent, false))
	if strings.Contains(strings.Join(readWrite, ","), "task_") {
		t.Fatalf("read-write child retained a task tool: %v", readWrite)
	}
	if !containsNativeChildTool(readWrite, codetools.ToolWrite) || !containsNativeChildTool(readWrite, "web_search") {
		t.Fatalf("read-write child lost inherited non-task tools: %v", readWrite)
	}

	readOnly := nativeChildToolNames(nativeChildToolCatalog(parent, true))
	want := []string{codetools.ToolRead, codetools.ToolGrep, codetools.ToolGlob}
	if strings.Join(readOnly, ",") != strings.Join(want, ",") {
		t.Fatalf("read-only child tools = %v, want %v", readOnly, want)
	}
}

func containsNativeChildTool(names []string, want string) bool {
	for _, name := range names {
		if name == want {
			return true
		}
	}
	return false
}

func TestNativeChildRunnerReusesPlannerModelAndPolicy(t *testing.T) {
	t.Cleanup(agent.Configure)
	planner := &nativeChildCapturePlanner{model: "qwen3.8-27b-q4km"}
	var boundary agent.ModelRequestBoundary
	parentCatalog := []agent.ToolDef{
		nativeChildTestTool(codetools.ToolRead),
		nativeChildTestTool(codetools.ToolWrite),
		nativeChildTestTool(agent.ToolTaskSpawn),
	}
	policy := adjudicator.Policy{
		Posture: adjudicator.PostureFailClosed,
		Allow:   map[string]bool{codetools.ToolRead: true},
	}
	runner := newNativeChildTaskRunner(
		planner,
		1,
		[]agent.RunOption{
			agent.WithProvider("openai"),
			agent.WithBaseURL("http://127.0.0.1:8080/v1"),
			agent.WithModelRequestObserver(func(got agent.ModelRequestBoundary) error {
				boundary = got
				return nil
			}),
		},
		parentCatalog,
		policy,
	)

	result, err := runner(context.Background(), agent.ChildTaskRunRequest{Prompt: "inspect", ReadOnly: true})
	if err != nil {
		t.Fatalf("native child runner: %v", err)
	}
	if result != "child complete" || planner.calls != 1 {
		t.Fatalf("result=%v planner calls=%d", result, planner.calls)
	}
	if boundary.Model != planner.model {
		t.Fatalf("child model = %q, want inherited planner model %q", boundary.Model, planner.model)
	}
	if got := nativeChildToolNames(boundary.Tools); strings.Join(got, ",") != codetools.ToolRead {
		t.Fatalf("model-bound child tools = %v, want only Read", got)
	}
	snapshot := adjudicator.Default.PolicySnapshot()
	if snapshot.Posture != policy.Posture || !snapshot.Allow[codetools.ToolRead] || snapshot.Allow[codetools.ToolWrite] {
		t.Fatalf("child policy snapshot = %+v, want caller floor", snapshot)
	}
}

func TestNativeChildPlannerRejectsMutationBeforeDispatch(t *testing.T) {
	underlying := &nativeChildToolCallPlanner{tool: codetools.ToolWrite}
	planner := nativeChildPlanner{
		Planner: underlying,
		allowed: map[string]struct{}{codetools.ToolRead: {}},
	}
	_, err := planner.Complete(context.Background(), nil, []agent.ToolDef{nativeChildTestTool(codetools.ToolRead)})
	if err == nil || !strings.Contains(err.Error(), "outside the inherited child capability floor") {
		t.Fatalf("mutation error = %v, want inherited-floor refusal", err)
	}
	if underlying.calls != 1 {
		t.Fatalf("underlying planner calls = %d, want 1", underlying.calls)
	}
}

type nativeChildToolCallPlanner struct {
	tool  string
	calls int
}

func (p *nativeChildToolCallPlanner) Model() string { return "native-child-tool-call" }

func (p *nativeChildToolCallPlanner) Complete(_ context.Context, _ []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	p.calls++
	return &agent.Completion{Message: agent.Message{
		Role: agent.RoleAssistant,
		ToolCalls: []agent.ToolCall{{
			ID: "mutation",
			Function: agent.Func{
				Name:      p.tool,
				Arguments: `{"file_path":"blocked.txt","content":"blocked"}`,
			},
		}},
	}}, nil
}

var _ agent.Planner = (*nativeChildCapturePlanner)(nil)
var _ agent.Planner = (*nativeChildToolCallPlanner)(nil)
