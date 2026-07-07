package journal

import (
	"context"
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

// TestAppendAgentEventChainsAndVerifies pins #2011 acceptance bullet 1 at the
// journal seam: N in-process microagent lifecycle rows appended to ONE journal
// form a single hash chain that verifies end-to-end, each row tagged with its
// agent id and stamped by the host.
func TestAppendAgentEventChainsAndVerifies(t *testing.T) {
	j := OpenMemory()
	const agents = 120
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("ma-%03d", i)
		j.AppendAgentEvent(KindAgentSpawn, id, "")
		j.AppendAgentEvent(KindAgentDone, id, "")
	}
	rows := j.Recent(0)
	if len(rows) != agents*2 {
		t.Fatalf("rows = %d, want %d", len(rows), agents*2)
	}
	if n, err := VerifyRows(rows); err != nil {
		t.Fatalf("VerifyRows: %v (checked %d)", err, n)
	}
	r0 := rows[0]
	if r0.Kind != KindAgentSpawn || r0.TraceID != "ma-000" || r0.Tool != "ma-000" || r0.By != "microagent-host" {
		t.Fatalf("row0 = %+v, want AGENT_SPAWN tagged ma-000 by microagent-host", r0)
	}
	// An AGENT_ERROR row carries the terminal reason on Reason.
	last := j.AppendAgentEvent(KindAgentError, "ma-999", "error: boom")
	if last.Reason != "error: boom" || last.Kind != KindAgentError {
		t.Fatalf("error row = %+v, want AGENT_ERROR reason 'error: boom'", last)
	}
	if n, err := VerifyRows(j.Recent(0)); err != nil {
		t.Fatalf("VerifyRows after error row: %v (checked %d)", err, n)
	}
}

// TestAppendAgentEventNilReceiverIsNoop mirrors AppendCrash: a caller that guarded
// the host journal on may call AppendAgentEvent unconditionally.
func TestAppendAgentEventNilReceiverIsNoop(t *testing.T) {
	var j *Journal
	if got := j.AppendAgentEvent(KindAgentSpawn, "ma-0", ""); got != (Row{}) {
		t.Fatalf("nil-receiver AppendAgentEvent = %+v, want zero Row", got)
	}
}

// TestAgentLifecycleAndQuarantineShareOneChain pins #2011 acceptance bullet 2:
// with lifecycle rows mixed into the host journal, a QUARANTINE the shared gateway
// raises (the journal is the kernel's ABI emitter) lands in the SAME chain and
// keeps its witness + call_seq — no forensic drop — and the mixed chain verifies
// end-to-end.
func TestAgentLifecycleAndQuarantineShareOneChain(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	j := OpenMemory()
	abi.RegisterEmitter(j)

	const agents = 5
	for i := 0; i < agents; i++ {
		id := fmt.Sprintf("ma-%d", i)
		j.AppendAgentEvent(KindAgentSpawn, id, "")
		j.AppendAgentEvent(KindAgentDone, id, "")
	}

	// The shared gateway quarantines a secret-exfil result; because the host
	// journal is the registered emitter, the QUARANTINE lands in the SAME chain.
	call := &abi.ToolCall{
		Tool:    "read_webpage",
		TraceID: "ma-3",
		SeqNo:   4242,
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.test"}`)},
	}
	result := &abi.Result{
		Call:    call,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"page":"api_key=sk-abcdef0123456789abcdef0123 leaked"}`)},
	}
	if v := kernel.New("").AdmitResult(context.Background(), call, result); v.Kind != abi.VerdictQuarantine {
		t.Fatalf("AdmitResult verdict = %v, want Quarantine", v.Kind)
	}

	rows := j.Recent(0)
	if len(rows) != agents*2+1 {
		t.Fatalf("rows = %d, want %d (lifecycle + one quarantine)", len(rows), agents*2+1)
	}
	if n, err := VerifyRows(rows); err != nil {
		t.Fatalf("mixed chain VerifyRows: %v (checked %d)", err, n)
	}
	q := rows[len(rows)-1]
	if q.Kind != "QUARANTINE" || q.Verdict != "QUARANTINE" {
		t.Fatalf("last row = %+v, want a QUARANTINE decision row", q)
	}
	if q.CallSeq != 4242 {
		t.Fatalf("QUARANTINE CallSeq = %d, want originating call 4242 (forensic drop)", q.CallSeq)
	}
	if q.Witness == "" {
		t.Fatalf("QUARANTINE Witness empty — forensic drop when lifecycle rows share the chain")
	}
}
