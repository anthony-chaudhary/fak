package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// floor_2088_test.go — adversarial tests for #2088, CONTRACT A: the production
// capability floor must affirmatively allow the native harness's OWN built-in
// tool names, which the native loop emits lowercase. A missing lowercase entry
// meant the floor's exact-map affirmative-allow lookup fell through to
// default-deny, killing every opencode agentic LAN turn with an opaque
// "Unexpected server error".

// nativeBuiltinToolNames are the EXACT lowercase spellings the native harness
// emits (the contract's allowed set), paired with the capital Claude-lineage
// spelling that must also remain admitted.
var nativeBuiltinToolNames = []string{
	"skill",
	"question",
	"context_control",
	"todowrite",
	"todoread",
	"task_spawn",
	"task_wait",
	"task_status",
	"task_cancel",
	"read",
	"grep",
	"glob",
	"write",
	"edit",
	"bash",
}

// TestFloor2088NativeBuiltinsAdmitted is the positive half: every lowercase
// native built-in name adjudicates to VerdictAllow through the real package
// adjudication entrypoint (New(DefaultPolicy()).Adjudicate), with the
// capital Claude-lineage spellings also still admitted.
func TestFloor2088NativeBuiltinsAdmitted(t *testing.T) {
	a := New(DefaultPolicy())
	ctx := context.Background()

	// Lowercase native spellings: the #2088 regression set.
	for _, tool := range nativeBuiltinToolNames {
		t.Run("lower_"+tool, func(t *testing.T) {
			v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("native built-in %q: got %v/%s, want ALLOW",
					tool, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}

	// Capital Claude-lineage spellings that must not regress.
	for _, tool := range []string{"Skill", "Read", "Glob", "Grep", "Write", "Edit", "Bash", "TodoWrite"} {
		t.Run("capital_"+tool, func(t *testing.T) {
			v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
			if v.Kind != abi.VerdictAllow {
				t.Fatalf("Claude-lineage spelling %q: got %v/%s, want ALLOW",
					tool, v.Kind, abi.ReasonName(v.Reason))
			}
		})
	}
}

// TestFloor2088UnknownStillDefaultDeny is the negative control: the floor MUST
// stay default-deny for genuinely unknown names. A blanket allowance of the
// native set must not have widened the floor to admit arbitrary tools.
func TestFloor2088UnknownStillDefaultDeny(t *testing.T) {
	a := New(DefaultPolicy())
	ctx := context.Background()

	unknowns := []string{
		"frobnicate_widget",
		"totally_unknown_custom_tool",
		"rm_rf_everything",
	}
	for _, tool := range unknowns {
		v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
		if v.Kind != abi.VerdictDeny {
			t.Errorf("unknown tool %q: got Kind=%v, want VerdictDeny", tool, v.Kind)
		}
		if v.Reason != abi.ReasonDefaultDeny {
			t.Errorf("unknown tool %q: got Reason=%s, want DEFAULT_DENY", tool, abi.ReasonName(v.Reason))
		}
	}
}

// TestFloor2088UnknownDenialDispositionTerminal pins the deny-loopback contract:
// a DEFAULT_DENY denial must be TERMINAL per kernel.VerdictDisposition, so a
// budget/loop driver does not retry an unreachable tool forever.
func TestFloor2088UnknownDenialDispositionTerminal(t *testing.T) {
	a := New(DefaultPolicy())
	v := a.Adjudicate(context.Background(), inlineCall("frobnicate_widget", `{}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unknown tool: got %v/%s, want Deny/DEFAULT_DENY",
			v.Kind, abi.ReasonName(v.Reason))
	}
	if got := kernel.VerdictDisposition(v); got != "TERMINAL" {
		t.Fatalf("kernel.VerdictDisposition(DEFAULT_DENY) = %q, want TERMINAL", got)
	}
	// And MALFORMED must remain RETRYABLE, the other half of the closed contract.
	malformed := abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMalformed}
	if got := kernel.VerdictDisposition(malformed); got != "RETRYABLE" {
		t.Fatalf("kernel.VerdictDisposition(MALFORMED) = %q, want RETRYABLE", got)
	}
}

// TestFloor2088NeverAdmitsNativeBuiltins is the promptmmu compaction guard: a
// native built-in name that is now affirmatively allowed must NOT be reported as
// "never admittable" (which would let the inbound tool-def compactor DROP the
// tool definition and re-break the turn). An unknown name stays never-admittable.
//
// The package DOES expose Policy.NeverAdmits (decide.go), so this sub-assertion
// is exercised rather than skipped.
func TestFloor2088NeverAdmitsNativeBuiltins(t *testing.T) {
	p := DefaultPolicy()

	for _, tool := range nativeBuiltinToolNames {
		t.Run("lower_"+tool, func(t *testing.T) {
			if p.NeverAdmits(tool) {
				t.Fatalf("NeverAdmits(%q) = true; an admitted native built-in must never be pruned", tool)
			}
		})
	}
	for _, tool := range []string{"Skill", "Read", "Glob", "Grep", "Write", "Edit", "Bash", "TodoWrite"} {
		t.Run("capital_"+tool, func(t *testing.T) {
			if p.NeverAdmits(tool) {
				t.Fatalf("NeverAdmits(%q) = true; an admitted Claude-lineage spelling must never be pruned", tool)
			}
		})
	}

	// Negative control: an unknown name is never admittable.
	if !p.NeverAdmits("frobnicate_widget") {
		t.Fatalf("NeverAdmits(%q) = false; an unknown name must remain never-admittable", "frobnicate_widget")
	}
}
