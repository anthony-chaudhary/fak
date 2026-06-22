package journal

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRowCallSeqCorrelation is the witness that one call's DECIDE and its later
// QUARANTINE can be tied together: both rows carry the same CallSeq (the kernel's
// ToolCall.SeqNo), so "show me everything the kernel did for call N" is a filter,
// not a guess from tool+digest.
func TestRowCallSeqCorrelation(t *testing.T) {
	j := OpenMemory()

	call := &abi.ToolCall{Tool: "fetch_policy", TraceID: "t1", SeqNo: 7,
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"x"}`)}}

	j.Emit(abi.Event{Kind: abi.EvDecide, Call: call,
		Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "monitor"}})
	j.Emit(abi.Event{Kind: abi.EvQuarantine, Call: call,
		Verdict: &abi.Verdict{Kind: abi.VerdictQuarantine, Reason: abi.ReasonTrustViolation, By: "ifc-sink"},
		Result:  &abi.Result{Call: call, Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte("poison")}}})

	rows := j.Recent(0)
	if len(rows) != 2 {
		t.Fatalf("rows = %d, want 2", len(rows))
	}
	if rows[0].CallSeq != 7 || rows[1].CallSeq != 7 {
		t.Errorf("both rows should share CallSeq 7, got %d / %d", rows[0].CallSeq, rows[1].CallSeq)
	}
	if rows[0].Kind != "DECIDE" || rows[1].Kind != "QUARANTINE" {
		t.Errorf("kinds = %s / %s, want DECIDE / QUARANTINE", rows[0].Kind, rows[1].Kind)
	}
	// The correlation/disclosure fields must NOT break tamper-evidence.
	if n, err := VerifyRows(rows); err != nil || n != 2 {
		t.Fatalf("VerifyRows = n=%d err=%v, want 2 nil (chain must survive the new fields)", n, err)
	}
}

// TestRowPersistsWitness is the witness that the bounded-disclosure claim — the
// offending glob a self-modify deny names — now survives into the durable record,
// where before it lived only in the synchronous in-band response.
func TestRowPersistsWitness(t *testing.T) {
	j := OpenMemory()
	j.Emit(abi.Event{
		Kind: abi.EvDeny,
		Call: &abi.ToolCall{Tool: "write_file", SeqNo: 3,
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"path":"internal/abi/x.go"}`)}},
		Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonSelfModify, By: "monitor",
			Payload: abi.WitnessPayload{Claim: "internal/abi/"}},
	})
	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("rows = %d, want 1", len(rows))
	}
	if rows[0].Witness != "internal/abi/" {
		t.Errorf("Witness = %q, want internal/abi/", rows[0].Witness)
	}
	if rows[0].Reason != "SELF_MODIFY" || rows[0].CallSeq != 3 {
		t.Errorf("row = %+v, want SELF_MODIFY / CallSeq 3", rows[0])
	}
}
