package adjudicator

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestAdjudicate_CircuitBreakerDoomLoop(t *testing.T) {
	ctx := context.Background()

	t.Run("consecutive identical refusals trip circuit breaker at threshold K=3", func(t *testing.T) {
		p := Policy{
			Allow: map[string]bool{"calculate": true},
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessionID := "sess-doomloop-1"

		failingCall := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}

		// Turn 1: First refusal -> POLICY_BLOCK
		v1 := a.Adjudicate(ctx, failingCall)
		if v1.Kind != abi.VerdictDeny {
			t.Fatalf("turn 1: got kind %v, want %v", v1.Kind, abi.VerdictDeny)
		}
		if v1.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("turn 1: got reason %v (%s), want %v (POLICY_BLOCK)", v1.Reason, abi.ReasonName(v1.Reason), abi.ReasonPolicyBlock)
		}

		// Turn 2: Second refusal -> POLICY_BLOCK
		v2 := a.Adjudicate(ctx, failingCall)
		if v2.Kind != abi.VerdictDeny {
			t.Fatalf("turn 2: got kind %v, want %v", v2.Kind, abi.VerdictDeny)
		}
		if v2.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("turn 2: got reason %v (%s), want %v (POLICY_BLOCK)", v2.Reason, abi.ReasonName(v2.Reason), abi.ReasonPolicyBlock)
		}

		// Turn 3: Third consecutive identical refusal -> trips circuit breaker (DOOM_LOOP)
		v3 := a.Adjudicate(ctx, failingCall)
		if v3.Kind != abi.VerdictDeny {
			t.Fatalf("turn 3: got kind %v, want %v", v3.Kind, abi.VerdictDeny)
		}
		if v3.Reason != ReasonDoomLoop {
			t.Fatalf("turn 3: got reason %v (%s), want %v (DOOM_LOOP)", v3.Reason, abi.ReasonName(v3.Reason), ReasonDoomLoop)
		}
		if abi.ReasonName(v3.Reason) != "DOOM_LOOP" {
			t.Fatalf("turn 3: ReasonName = %q, want %q", abi.ReasonName(v3.Reason), "DOOM_LOOP")
		}
		if v3.Disposition != "DOOM_LOOP" {
			t.Fatalf("turn 3: Disposition = %q, want %q", v3.Disposition, "DOOM_LOOP")
		}
		if v3.Meta == nil {
			t.Fatal("turn 3: expected non-nil Meta on circuit breaker trip")
		}
		if v3.Meta["circuit_breaker"] != "tripped" {
			t.Fatalf("turn 3: Meta[circuit_breaker] = %q, want 'tripped'", v3.Meta["circuit_breaker"])
		}
		if !strings.Contains(v3.Meta["remedy"], "pivot") {
			t.Fatalf("turn 3: Meta[remedy] = %q, expected guidance to pivot", v3.Meta["remedy"])
		}
		if !strings.Contains(v3.Meta["fix"], "consecutive identical refusals") {
			t.Fatalf("turn 3: Meta[fix] = %q, expected fix guidance", v3.Meta["fix"])
		}
		if v3.Meta["streak"] != "3" {
			t.Fatalf("turn 3: Meta[streak] = %q, want '3'", v3.Meta["streak"])
		}
		if v3.Meta["original_reason"] != "POLICY_BLOCK" {
			t.Fatalf("turn 3: Meta[original_reason] = %q, want 'POLICY_BLOCK'", v3.Meta["original_reason"])
		}

		// Turn 4: Continuing identical refusal stays tripped
		v4 := a.Adjudicate(ctx, failingCall)
		if v4.Kind != abi.VerdictDeny || v4.Reason != ReasonDoomLoop {
			t.Fatalf("turn 4: got kind %v reason %v, want Deny DOOM_LOOP", v4.Kind, v4.Reason)
		}
		if v4.Meta["streak"] != "4" {
			t.Fatalf("turn 4: Meta[streak] = %q, want '4'", v4.Meta["streak"])
		}
	})

	t.Run("permitted tool call resets circuit breaker counter", func(t *testing.T) {
		p := Policy{
			Allow: map[string]bool{"calculate": true},
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessionID := "sess-reset-permitted"

		failingCall := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}

		// 2 consecutive denials
		a.Adjudicate(ctx, failingCall)
		a.Adjudicate(ctx, failingCall)

		_, _, count, tripped := a.CircuitBreakerState(sessionID)
		if count != 2 || tripped {
			t.Fatalf("expected count 2, tripped false; got count=%d, tripped=%v", count, tripped)
		}

		// Permitted tool call
		allowedCall := &abi.ToolCall{
			Tool:    "calculate",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"expr":"2+2"}`)},
			TraceID: sessionID,
		}
		vAllowed := a.Adjudicate(ctx, allowedCall)
		if vAllowed.Kind != abi.VerdictAllow {
			t.Fatalf("allowedCall: got kind %v, want %v", vAllowed.Kind, abi.VerdictAllow)
		}

		// State should be completely cleared
		_, _, countAfter, _ := a.CircuitBreakerState(sessionID)
		if countAfter != 0 {
			t.Fatalf("expected count 0 after permitted call, got %d", countAfter)
		}

		// Next denial starts at count 1, does not trip
		vAfter := a.Adjudicate(ctx, failingCall)
		if vAfter.Kind != abi.VerdictDeny || vAfter.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("vAfter: got %v (%s), want POLICY_BLOCK (not DOOM_LOOP)", vAfter.Reason, abi.ReasonName(vAfter.Reason))
		}
	})

	t.Run("intervening distinct tool call resets counter", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash":       abi.ReasonPolicyBlock,
				"write_file": abi.ReasonSelfModify,
			},
		}
		a := New(p)
		sessionID := "sess-reset-distinct-tool"

		bashCall := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}

		// 2 Bash denials
		a.Adjudicate(ctx, bashCall)
		a.Adjudicate(ctx, bashCall)

		// Distinct tool call: write_file denied with ReasonSelfModify
		writeCall := &abi.ToolCall{
			Tool:    "write_file",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"internal/abi/types.go"}`)},
			TraceID: sessionID,
		}
		vWrite := a.Adjudicate(ctx, writeCall)
		if vWrite.Kind != abi.VerdictDeny {
			t.Fatalf("writeCall: got kind %v, want Deny", vWrite.Kind)
		}
		if vWrite.Reason != abi.ReasonSelfModify {
			t.Fatalf("writeCall: got reason %v (%s), want SELF_MODIFY", vWrite.Reason, abi.ReasonName(vWrite.Reason))
		}

		tool, _, count, _ := a.CircuitBreakerState(sessionID)
		if tool != "write_file" || count != 1 {
			t.Fatalf("expected write_file with count 1; got tool=%q, count=%d", tool, count)
		}

		// Returning to Bash starts at 1, does NOT trip
		vBashAgain := a.Adjudicate(ctx, bashCall)
		if vBashAgain.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("vBashAgain: got reason %v (%s), want POLICY_BLOCK", vBashAgain.Reason, abi.ReasonName(vBashAgain.Reason))
		}
	})

	t.Run("intervening different reason code resets counter", func(t *testing.T) {
		pPolicyBlock := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(pPolicyBlock)
		sessionID := "sess-reset-diff-reason"

		call := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"echo test"}`)},
			TraceID: sessionID,
		}

		// 2 PolicyBlock denials
		a.Adjudicate(ctx, call)
		a.Adjudicate(ctx, call)

		_, _, count2, _ := a.CircuitBreakerState(sessionID)
		if count2 != 2 {
			t.Fatalf("expected count 2, got %d", count2)
		}

		// Same tool, but policy updated to deny with ReasonMalformed
		a.SetPolicy(Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonMalformed,
			},
		})
		vMalformed := a.Adjudicate(ctx, call)
		if vMalformed.Reason != abi.ReasonMalformed {
			t.Fatalf("vMalformed: got reason %v (%s), want MALFORMED", vMalformed.Reason, abi.ReasonName(vMalformed.Reason))
		}

		// Different reason code reset streak to 1
		_, _, countMalformed, _ := a.CircuitBreakerState(sessionID)
		if countMalformed != 1 {
			t.Fatalf("expected streak reset to 1 on different reason, got %d", countMalformed)
		}

		// Revert to PolicyBlock denial, starts at streak 1
		a.SetPolicy(pPolicyBlock)
		vPolicyAgain := a.Adjudicate(ctx, call)
		if vPolicyAgain.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("vPolicyAgain: got reason %v (%s), want POLICY_BLOCK", vPolicyAgain.Reason, abi.ReasonName(vPolicyAgain.Reason))
		}
		_, _, countPolicyAgain, _ := a.CircuitBreakerState(sessionID)
		if countPolicyAgain != 1 {
			t.Fatalf("expected streak 1, got %d", countPolicyAgain)
		}
	})

	t.Run("sessions are isolated", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)

		callA := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: "session-A",
		}
		callB := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: "session-B",
		}

		// Session A has 2 denials
		a.Adjudicate(ctx, callA)
		a.Adjudicate(ctx, callA)

		// Session B has 1 denial
		vB1 := a.Adjudicate(ctx, callB)
		if vB1.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("vB1: got reason %v (%s), want POLICY_BLOCK", vB1.Reason, abi.ReasonName(vB1.Reason))
		}

		// Session A's 3rd denial trips DOOM_LOOP
		vA3 := a.Adjudicate(ctx, callA)
		if vA3.Reason != ReasonDoomLoop {
			t.Fatalf("vA3: got reason %v (%s), want DOOM_LOOP", vA3.Reason, abi.ReasonName(vA3.Reason))
		}

		// Session B's 2nd denial does NOT trip
		vB2 := a.Adjudicate(ctx, callB)
		if vB2.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("vB2: got reason %v (%s), want POLICY_BLOCK", vB2.Reason, abi.ReasonName(vB2.Reason))
		}

		// Session B's 3rd denial trips
		vB3 := a.Adjudicate(ctx, callB)
		if vB3.Reason != ReasonDoomLoop {
			t.Fatalf("vB3: got reason %v (%s), want DOOM_LOOP", vB3.Reason, abi.ReasonName(vB3.Reason))
		}
	})

	t.Run("context with session ID", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessCtx := ContextWithSessionID(ctx, "ctx-session-123")

		callWithoutTraceID := &abi.ToolCall{
			Tool: "Bash",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
		}

		a.Adjudicate(sessCtx, callWithoutTraceID)
		a.Adjudicate(sessCtx, callWithoutTraceID)
		v3 := a.Adjudicate(sessCtx, callWithoutTraceID)

		if v3.Reason != ReasonDoomLoop {
			t.Fatalf("v3: got reason %v (%s), want DOOM_LOOP", v3.Reason, abi.ReasonName(v3.Reason))
		}
	})

	t.Run("custom threshold", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		a.SetCircuitBreakerThreshold(2)
		sessionID := "sess-custom-thresh"

		call := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}

		v1 := a.Adjudicate(ctx, call)
		if v1.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("v1: got %v (%s), want POLICY_BLOCK", v1.Reason, abi.ReasonName(v1.Reason))
		}

		// Turn 2 trips because threshold is 2
		v2 := a.Adjudicate(ctx, call)
		if v2.Reason != ReasonDoomLoop {
			t.Fatalf("v2: got %v (%s), want DOOM_LOOP at threshold 2", v2.Reason, abi.ReasonName(v2.Reason))
		}
	})

	t.Run("manual ResetCircuitBreaker", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessionID := "sess-manual-reset"

		call := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}

		a.Adjudicate(ctx, call)
		a.Adjudicate(ctx, call)
		a.ResetCircuitBreaker(sessionID)

		// After reset, 3rd call is effectively 1st call again
		v := a.Adjudicate(ctx, call)
		if v.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("after ResetCircuitBreaker, got %v (%s), want POLICY_BLOCK", v.Reason, abi.ReasonName(v.Reason))
		}
	})

	t.Run("adapted arguments reset counter", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessionID := "sess-adapt-args"

		call1 := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /"}`)},
			TraceID: sessionID,
		}
		call2Adapted := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /tmp/other"}`)},
			TraceID: sessionID,
		}

		a.Adjudicate(ctx, call1)
		a.Adjudicate(ctx, call1)
		// Turn 3 adapts argument to /tmp/other
		vAdapted := a.Adjudicate(ctx, call2Adapted)
		if vAdapted.Reason != abi.ReasonPolicyBlock {
			t.Fatalf("adapted call: got %v (%s), want POLICY_BLOCK (not DOOM_LOOP)", vAdapted.Reason, abi.ReasonName(vAdapted.Reason))
		}
		_, _, countAdapted, _ := a.CircuitBreakerState(sessionID)
		if countAdapted != 1 {
			t.Fatalf("expected count 1 after adapting arguments, got %d", countAdapted)
		}
	})

	t.Run("minor apologetic variations still trip circuit breaker", func(t *testing.T) {
		p := Policy{
			Deny: map[string]abi.ReasonCode{
				"Bash": abi.ReasonPolicyBlock,
			},
		}
		a := New(p)
		sessionID := "sess-apologies"

		call1 := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /","description":"initial delete"}`)},
			TraceID: sessionID,
		}
		call2 := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /","description":"apologies, retrying delete"}`)},
			TraceID: sessionID,
		}
		call3 := &abi.ToolCall{
			Tool:    "Bash",
			Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"command":"rm -rf /","description":"sorry, trying again"}`)},
			TraceID: sessionID,
		}

		a.Adjudicate(ctx, call1)
		a.Adjudicate(ctx, call2)
		v3 := a.Adjudicate(ctx, call3)

		if v3.Reason != ReasonDoomLoop {
			t.Fatalf("v3 with apologetic description variation: got %v (%s), want DOOM_LOOP", v3.Reason, abi.ReasonName(v3.Reason))
		}
	})
}
