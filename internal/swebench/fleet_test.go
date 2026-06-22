package swebench

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// fleet_test.go is the no-model / no-GPU / no-network witness for the fleet coding
// agent: a scripted in-process planner stands in for the gateway model, and a temp
// git repo stands in for the cloned instance. It proves the MECHANICS — tool
// dispatch, edits applied to a real worktree, the unified-diff capture in the exact
// harness shape, the step cap, path-escape refusal, and the honest "no gateway"
// error. Whether a REAL model resolves instances is the DGX/GPU residual, not this.

// fakePlanner is an in-process agent.Planner whose per-turn completion is supplied
// by fn(call, messages). It records the call count so a test can assert turn budget.
type fakePlanner struct {
	model string
	fn    func(call int, messages []agent.Message) *agent.Completion
	calls int
}

func (f *fakePlanner) Model() string { return f.model }

func (f *fakePlanner) Complete(_ context.Context, messages []agent.Message, _ []agent.ToolDef, _ ...agent.SampleOpt) (*agent.Completion, error) {
	c := f.fn(f.calls, messages)
	f.calls++
	return c, nil
}

func toolCallCompletion(id, name, args string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{
		Role:      "assistant",
		ToolCalls: []agent.ToolCall{{ID: id, Type: "function", Function: agent.Func{Name: name, Arguments: args}}},
	}}
}

func finalCompletion(text string) *agent.Completion {
	return &agent.Completion{Message: agent.Message{Role: "assistant", Content: text}}
}

func requireGit(t *testing.T) {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not available")
	}
}

// gitFixtureWith returns a workspace preparer that initializes dir as a git repo
// holding one committed file (the "base commit" state the agent edits from).
func gitFixtureWith(path, content string) func(ctx context.Context, in Instance, dir string) error {
	return func(ctx context.Context, in Instance, dir string) error {
		if _, err := runGit(ctx, dir, "init", "--quiet"); err != nil {
			return err
		}
		if err := os.WriteFile(filepath.Join(dir, path), []byte(content), 0o644); err != nil {
			return err
		}
		if _, err := runGit(ctx, dir, "add", "-A"); err != nil {
			return err
		}
		_, err := runGit(ctx, dir, "-c", "user.email=t@fak", "-c", "user.name=fak", "commit", "--quiet", "-m", "base")
		return err
	}
}

// TestFleetRunnerProducesPatch is the end-to-end mechanics witness: a scripted
// planner reads the buggy file, writes the fix, then finishes; the runner returns
// a real unified-diff prediction carrying the fix.
func TestFleetRunnerProducesPatch(t *testing.T) {
	requireGit(t)
	const buggy = "def add(a, b):\n    return a - b\n"
	const fixed = "def add(a, b):\n    return a + b\n"

	planner := &fakePlanner{model: "test-model", fn: func(call int, _ []agent.Message) *agent.Completion {
		switch call {
		case 0:
			return toolCallCompletion("c1", "read_file", `{"path":"calc.py"}`)
		case 1:
			return toolCallCompletion("c2", "write_file", `{"path":"calc.py","content":`+strconv.Quote(fixed)+`}`)
		default:
			return toolCallCompletion("c3", "finish", `{}`)
		}
	}}

	fr := &fleetRunner{
		cfg:     RunConfig{MaxSteps: 10},
		planner: planner,
		prepare: gitFixtureWith("calc.py", buggy),
	}
	in := Instance{
		InstanceID:       "demo__calc-1",
		Repo:             "demo/calc",
		ProblemStatement: "add() subtracts instead of adding",
	}

	pred, err := fr.RunInstance(context.Background(), in)
	if err != nil {
		t.Fatalf("RunInstance: %v", err)
	}
	if pred.InstanceID != in.InstanceID {
		t.Errorf("InstanceID = %q, want %q", pred.InstanceID, in.InstanceID)
	}
	if pred.ModelNameOrPath != "test-model" {
		t.Errorf("ModelNameOrPath = %q, want test-model", pred.ModelNameOrPath)
	}
	if !strings.Contains(pred.ModelPatch, "diff --git") {
		t.Errorf("ModelPatch is not a unified diff:\n%s", pred.ModelPatch)
	}
	if !strings.Contains(pred.ModelPatch, "+    return a + b") || !strings.Contains(pred.ModelPatch, "-    return a - b") {
		t.Errorf("ModelPatch missing the fix delta:\n%s", pred.ModelPatch)
	}
	if planner.calls != 3 {
		t.Errorf("planner.calls = %d, want 3 (read, write, finish)", planner.calls)
	}
}

// TestRunCodingAgentRespectsStepCap: a planner that never finishes stops exactly
// at MaxSteps with no error, so the partial diff is still captured downstream.
func TestRunCodingAgentRespectsStepCap(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "f.txt"), []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	planner := &fakePlanner{fn: func(_ int, _ []agent.Message) *agent.Completion {
		return toolCallCompletion("r", "read_file", `{"path":"f.txt"}`) // never calls finish
	}}
	steps, err := runCodingAgent(context.Background(), planner, Instance{InstanceID: "x"}, dir, 4, false)
	if err != nil {
		t.Fatalf("runCodingAgent: %v", err)
	}
	if steps != 4 {
		t.Errorf("steps = %d, want 4 (the cap)", steps)
	}
	if planner.calls != 4 {
		t.Errorf("planner.calls = %d, want 4", planner.calls)
	}
}

// TestRunCodingAgentStopsOnFinalAnswer: a completion with no tool calls ends the
// loop, and edits issued before it are applied.
func TestRunCodingAgentStopsOnFinalAnswer(t *testing.T) {
	dir := t.TempDir()
	planner := &fakePlanner{fn: func(call int, _ []agent.Message) *agent.Completion {
		if call == 0 {
			return toolCallCompletion("c", "write_file", `{"path":"a.txt","content":"hi"}`)
		}
		return finalCompletion("done")
	}}
	steps, err := runCodingAgent(context.Background(), planner, Instance{}, dir, 10, false)
	if err != nil {
		t.Fatalf("runCodingAgent: %v", err)
	}
	if steps != 2 {
		t.Errorf("steps = %d, want 2", steps)
	}
	if b, _ := os.ReadFile(filepath.Join(dir, "a.txt")); string(b) != "hi" {
		t.Errorf("write_file not applied: %q", b)
	}
}

// TestFleetRunnerErrorsWithoutGateway: no injected planner + no gateway address is
// an honest error, not a placeholder patch.
func TestFleetRunnerErrorsWithoutGateway(t *testing.T) {
	fr := &fleetRunner{cfg: RunConfig{GatewayAddr: ""}}
	_, err := fr.RunInstance(context.Background(), Instance{InstanceID: "x"})
	if err == nil {
		t.Fatal("expected an error when no gateway is configured")
	}
	if !strings.Contains(err.Error(), "no gateway") {
		t.Errorf("err = %v, want a 'no gateway' message", err)
	}
}

// TestExecToolRejectsEscape: a poisoned ../ path is refused and nothing is written
// outside the worktree.
func TestExecToolRejectsEscape(t *testing.T) {
	dir := t.TempDir()
	res, fin := execTool(context.Background(), dir,
		agent.ToolCall{Function: agent.Func{Name: "write_file", Arguments: `{"path":"../evil.txt","content":"x"}`}}, false)
	if fin {
		t.Error("write_file must not signal finish")
	}
	if !strings.Contains(res, "escapes workspace") {
		t.Errorf("result = %q, want an escape rejection", res)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "evil.txt")); err == nil {
		t.Fatal("a file escaped the workspace")
	}
}

// TestExecToolRunGatedByAllowExec: the shell tool is refused unless allowExec.
func TestExecToolRunGatedByAllowExec(t *testing.T) {
	dir := t.TempDir()
	res, _ := execTool(context.Background(), dir,
		agent.ToolCall{Function: agent.Func{Name: "run", Arguments: `{"cmd":"echo hi"}`}}, false)
	if !strings.Contains(res, "disabled") {
		t.Errorf("run tool should be disabled without allowExec, got %q", res)
	}
}

func TestGatewayBaseURL(t *testing.T) {
	cases := map[string]string{
		"localhost:8080":        "http://localhost:8080/v1",
		"http://h:9/v1":         "http://h:9/v1",
		"https://api.x.com/v1/": "https://api.x.com/v1",
		"":                      "",
		"host:1/v1":             "http://host:1/v1",
	}
	for in, want := range cases {
		if got := gatewayBaseURL(in); got != want {
			t.Errorf("gatewayBaseURL(%q) = %q, want %q", in, got, want)
		}
	}
}
