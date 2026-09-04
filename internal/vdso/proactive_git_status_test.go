package vdso

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestProactive_GitStatusServedInline(t *testing.T) {
	v := New(DefaultCacheSize)
	interceptor := NewProactiveInterceptor(WithVDSO(v))

	// Pre-populate vDSO with cached git status result
	cmdArgs := `{"command":"git status"}`
	call := &abi.ToolCall{
		Tool: ToolClaudeBash,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(cmdArgs), Len: int64(len(cmdArgs))},
		Meta: map[string]string{
			"readOnlyHint":   "true",
			"idempotentHint": "true",
		},
	}
	res := &abi.Result{
		Call:    call,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("On branch main\nnothing to commit, working tree clean"), Len: 46, Taint: abi.TaintTrusted},
		Status:  abi.StatusOK,
	}
	v.Emit(abi.Event{
		Kind:   abi.EvComplete,
		Call:   call,
		Result: res,
	})

	// Test 1: PlanStep with git status
	turn := TurnState{
		TurnIndex: 0,
		PlanStep:  "Check repository state using `git status`",
	}

	inlineRes, hit := interceptor.Evaluate(context.Background(), turn)
	if !hit || inlineRes == nil {
		t.Fatalf("expected git status to be intercepted proactively")
	}
	if inlineRes.Tool != ToolClaudeBash {
		t.Errorf("expected tool %q, got %q", ToolClaudeBash, inlineRes.Tool)
	}
	if string(inlineRes.Content) != "On branch main\nnothing to commit, working tree clean" {
		t.Errorf("unexpected content: %s", string(inlineRes.Content))
	}
	if inlineRes.ModelLatency != 0 || inlineRes.RemoteTokens != 0 {
		t.Errorf("expected 0 latency and 0 remote tokens, got %v / %d", inlineRes.ModelLatency, inlineRes.RemoteTokens)
	}

	// Test 2: TargetTool = "bash", TargetPath = "git status"
	turn2 := TurnState{
		TurnIndex:  1,
		TargetTool: "bash",
		TargetPath: "git status",
	}
	inlineRes2, hit2 := interceptor.Evaluate(context.Background(), turn2)
	if !hit2 || inlineRes2 == nil {
		t.Fatalf("expected direct bash git status target to be intercepted")
	}
	if inlineRes2.Tool != ToolClaudeBash {
		t.Errorf("expected ToolClaudeBash, got %s", inlineRes2.Tool)
	}
}
