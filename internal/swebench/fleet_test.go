package swebench

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codelint"
)

// fleet_test.go is the no-model / no-GPU / no-network witness for the fleet coding
// agent: a scripted in-process CodePlanner stands in for the gateway model, and a
// temp git repo stands in for the cloned instance. It proves the MECHANICS — tool
// dispatch, edits applied to a real worktree, the unified-diff capture in the exact
// harness shape, the step cap, path-escape refusal, and the honest "no planner"
// error. Whether a REAL model resolves instances is the GPU-server residual, not this.

// fakePlanner is an in-process CodePlanner whose per-turn response is supplied by
// fn(call, messages). It records the call count so a test can assert turn budget.
type fakePlanner struct {
	model string
	fn    func(call int, messages []ChatMessage) ChatTurn
	calls int
}

func (f *fakePlanner) Model() string { return f.model }

func (f *fakePlanner) Complete(_ context.Context, messages []ChatMessage, _ []ChatTool) (ChatTurn, error) {
	t := f.fn(f.calls, messages)
	f.calls++
	return t, nil
}

func toolCallTurn(id, name, args string) ChatTurn {
	return ChatTurn{Message: ChatMessage{
		Role:      "assistant",
		ToolCalls: []ChatToolCall{{ID: id, Name: name, Args: args}},
	}}
}

func finalTurn(text string) ChatTurn {
	return ChatTurn{Message: ChatMessage{Role: "assistant", Content: text}}
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

	planner := &fakePlanner{model: "test-model", fn: func(call int, _ []ChatMessage) ChatTurn {
		switch call {
		case 0:
			return toolCallTurn("c1", "read_file", `{"path":"calc.py"}`)
		case 1:
			return toolCallTurn("c2", "write_file", `{"path":"calc.py","content":`+strconv.Quote(fixed)+`}`)
		default:
			return toolCallTurn("c3", "finish", `{}`)
		}
	}}

	fr := &fleetRunner{
		cfg:     RunConfig{MaxSteps: 10, Planner: planner},
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
	planner := &fakePlanner{fn: func(_ int, _ []ChatMessage) ChatTurn {
		return toolCallTurn("r", "read_file", `{"path":"f.txt"}`) // never calls finish
	}}
	steps, err := runCodingAgent(context.Background(), planner, Instance{InstanceID: "x"}, dir, 4, false, nil)
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

// TestRunCodingAgentStopsOnFinalAnswer: a turn with no tool calls ends the loop,
// and edits issued before it are applied.
func TestRunCodingAgentStopsOnFinalAnswer(t *testing.T) {
	dir := t.TempDir()
	planner := &fakePlanner{fn: func(call int, _ []ChatMessage) ChatTurn {
		if call == 0 {
			return toolCallTurn("c", "write_file", `{"path":"a.txt","content":"hi"}`)
		}
		return finalTurn("done")
	}}
	steps, err := runCodingAgent(context.Background(), planner, Instance{}, dir, 10, false, nil)
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

// TestFleetRunnerErrorsWithoutPlanner: no injected planner is an honest error, not
// a placeholder patch.
func TestFleetRunnerErrorsWithoutPlanner(t *testing.T) {
	fr := &fleetRunner{cfg: RunConfig{}}
	_, err := fr.RunInstance(context.Background(), Instance{InstanceID: "x"})
	if err == nil {
		t.Fatal("expected an error when no planner is configured")
	}
	if !strings.Contains(err.Error(), "no planner") {
		t.Errorf("err = %v, want a 'no planner' message", err)
	}
}

// TestExecToolRejectsEscape: a poisoned ../ path is refused and nothing is written
// outside the worktree.
func TestExecToolRejectsEscape(t *testing.T) {
	dir := t.TempDir()
	res, fin := execTool(context.Background(), dir,
		ChatToolCall{Name: "write_file", Args: `{"path":"../evil.txt","content":"x"}`}, false, nil)
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

// TestExecToolLintsWriteWhenEnabled: with a linter (LintWrites on), a write of
// broken code gets the kernel's language-server diagnostics appended to the
// model-facing result so the agent self-corrects; with a nil linter (off) it does
// not; a clean write is silent either way. Uses the in-process Go pack, so it needs
// no external toolchain. The write itself always lands — the lint is advisory.
func TestExecToolLintsWriteWhenEnabled(t *testing.T) {
	dir := t.TempDir()
	reg := codelint.DefaultRegistry()

	broken := ChatToolCall{Name: "write_file", Args: `{"path":"bad.go","content":"package x\nfunc ("}`}
	res, fin := execTool(context.Background(), dir, broken, false, reg)
	if fin {
		t.Error("write_file must not signal finish")
	}
	if !strings.Contains(res, "codelint:") || !strings.Contains(res, "GO_PARSE") {
		t.Fatalf("want a GO_PARSE diagnostic appended to the write result, got %q", res)
	}
	if _, err := os.Stat(filepath.Join(dir, "bad.go")); err != nil {
		t.Errorf("the write is advisory — the broken file should still exist: %v", err)
	}

	resOff, _ := execTool(context.Background(), dir, broken, false, nil)
	if strings.Contains(resOff, "codelint:") {
		t.Errorf("LintWrites off (nil linter) must not append diagnostics, got %q", resOff)
	}

	clean := ChatToolCall{Name: "write_file", Args: `{"path":"ok.go","content":"package x\n\nfunc F() int { return 1 }\n"}`}
	resClean, _ := execTool(context.Background(), dir, clean, false, reg)
	if strings.Contains(resClean, "codelint:") {
		t.Errorf("a clean write must be silent, got %q", resClean)
	}
}

// TestExecToolBashGatedByAllowExec: the shell (bash) tool is refused unless allowExec.
func TestExecToolBashGatedByAllowExec(t *testing.T) {
	dir := t.TempDir()
	res, _ := execTool(context.Background(), dir,
		ChatToolCall{Name: "bash", Args: `{"command":"echo hi"}`}, false, nil)
	if !strings.Contains(res, "disabled") {
		t.Errorf("bash tool should be disabled without allowExec, got %q", res)
	}
}
