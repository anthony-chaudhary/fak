package journal

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ctxmmu"
	"github.com/anthony-chaudhary/fak/internal/kernel"
)

func TestResultQuarantineRowCarriesWitnessAndCallSeq(t *testing.T) {
	abi.ResetForTest()
	abi.RegisterResultAdmitter(10, ctxmmu.New())

	j := OpenMemory()
	abi.RegisterEmitter(j)

	call := &abi.ToolCall{
		Tool:    "read_webpage",
		TraceID: "trace-1958",
		SeqNo:   1958,
		Args:    abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"url":"https://example.test"}`)},
	}
	result := &abi.Result{
		Call:    call,
		Status:  abi.StatusOK,
		Payload: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"page":"api_key=sk-abcdef0123456789abcdef0123 leaked"}`)},
	}

	v := kernel.New("").AdmitResult(context.Background(), call, result)
	if v.Kind != abi.VerdictQuarantine {
		t.Fatalf("AdmitResult verdict = %v, want Quarantine", v.Kind)
	}

	rows := j.Recent(0)
	if len(rows) != 1 {
		t.Fatalf("journal rows = %d, want 1", len(rows))
	}
	row := rows[0]
	if row.Kind != "QUARANTINE" || row.Verdict != "QUARANTINE" || row.Reason != "SECRET_EXFIL" {
		t.Fatalf("row classification = %+v, want QUARANTINE/QUARANTINE/SECRET_EXFIL", row)
	}
	if row.CallSeq != 1958 {
		t.Fatalf("row CallSeq = %d, want originating call sequence 1958", row.CallSeq)
	}
	if row.Witness == "" {
		t.Fatalf("row Witness is empty; result quarantine rows must carry the detector claim")
	}
	for _, want := range []string{"ctxmmu", "secret_pattern", "quarantine_id=q1"} {
		if !strings.Contains(row.Witness, want) {
			t.Fatalf("row Witness = %q, want it to contain %q", row.Witness, want)
		}
	}
}
