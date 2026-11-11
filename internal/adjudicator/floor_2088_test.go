package adjudicator

import (
	"context"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	_ "github.com/anthony-chaudhary/fak/internal/blob"
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

// TestFloor2088UnknownNowDefaultOpen is the posture control: the floor now ships
// default-open, so genuinely unknown names are admitted by default. The
// provable-refusal rungs (the Deny map, SelfModifyGlobs, the dangerous-gotcha
// floor) still hold for the specific named dangers — see
// TestDefaultPolicyAdjudicationVerdicts for those.
func TestFloor2088UnknownNowDefaultOpen(t *testing.T) {
	a := New(DefaultPolicy())
	ctx := context.Background()

	unknowns := []string{
		"frobnicate_widget",
		"totally_unknown_custom_tool",
		"rm_rf_everything",
	}
	for _, tool := range unknowns {
		v := a.Adjudicate(ctx, inlineCall(tool, `{}`))
		if v.Kind != abi.VerdictAllow {
			t.Errorf("unknown tool %q: got Kind=%v, want VerdictAllow (default-open)", tool, v.Kind)
		}
		if v.Meta["posture"] != "default_open" {
			t.Errorf("unknown tool %q: got Meta[posture]=%q, want default_open", tool, v.Meta["posture"])
		}
	}
}

// verdictDisposition mirrors internal/kernel's VerdictDisposition/Disposition
// mapping so this pin carries no adjudicator->kernel census edge.
func verdictDisposition(v abi.Verdict) string {
	if v.Disposition != "" {
		return v.Disposition
	}
	if v.Meta != nil && v.Meta["disposition"] != "" {
		return v.Meta["disposition"]
	}
	switch v.Reason {
	case abi.ReasonMisroute, abi.ReasonMalformed, abi.ReasonShellDialect, abi.ReasonPolicyBlock, abi.ReasonTaintEgress:
		return "RETRYABLE"
	case abi.ReasonRateLimited, abi.ReasonLeaseHeld:
		return "WAIT"
	case abi.ReasonSelfModify, abi.ReasonTrustViolation, abi.ReasonIntegrityRefuted:
		return "ESCALATE"
	default:
		return "TERMINAL"
	}
}

// TestFloor2088UnknownDenialDispositionTerminal pins the deny-loopback contract:
// a DEFAULT_DENY denial must be TERMINAL per the loopback disposition mapping
// (verdictDisposition above), so a budget/loop driver does not retry an
// unreachable tool forever. The mapping is posture-independent, so the
// DEFAULT_DENY is triggered through a fail-closed floor (the production floor
// now ships default-open and no longer emits DEFAULT_DENY for unknown tools).
func TestFloor2088UnknownDenialDispositionTerminal(t *testing.T) {
	a := New(Policy{Posture: PostureFailClosed})
	v := a.Adjudicate(context.Background(), inlineCall("frobnicate_widget", `{}`))
	if v.Kind != abi.VerdictDeny || v.Reason != abi.ReasonDefaultDeny {
		t.Fatalf("unknown tool under fail-closed: got %v/%s, want Deny/DEFAULT_DENY",
			v.Kind, abi.ReasonName(v.Reason))
	}
	if got := verdictDisposition(v); got != "TERMINAL" {
		t.Fatalf("verdictDisposition(DEFAULT_DENY) = %q, want TERMINAL", got)
	}
	// And MALFORMED must remain RETRYABLE, the other half of the closed contract.
	malformed := abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonMalformed}
	if got := verdictDisposition(malformed); got != "RETRYABLE" {
		t.Fatalf("verdictDisposition(MALFORMED) = %q, want RETRYABLE", got)
	}
}

// TestFloor2088NeverAdmitsNativeBuiltins is the promptmmu compaction guard: a
// native built-in name that is now affirmatively allowed must NOT be reported as
// "never admittable" (which would let the inbound tool-def compactor DROP the
// tool definition and re-break the turn). Under the default-open posture every
// name except a hard name-deny is admittable, so the negative control is a hard
// name-deny (the Deny map holds regardless of posture).
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

	// Negative control: a hard name-deny is still never admittable even under
	// default-open (the Deny map holds regardless of posture).
	if !p.NeverAdmits("exfiltrate") {
		t.Fatalf("NeverAdmits(%q) = false; a hard name-deny must remain never-admittable", "exfiltrate")
	}
}
