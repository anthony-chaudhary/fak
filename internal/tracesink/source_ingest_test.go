package tracesink

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/ifc"
)

// TestPayloadCaptureDeniedSourceLeavesZeroBytesAtRest is the #5910
// payload-bearing witness. The source predicate runs before TraceSink copies the
// args into its replay corpus; persistence sees one digest-only refusal and zero
// bytes from either the denied source or the rest of that call's payload.
func TestPayloadCaptureDeniedSourceLeavesZeroBytesAtRest(t *testing.T) {
	sink := NewTraceSink(Options{Ledger: ifc.NewLedger(), Clock: fixedClock()})
	deniedSource := `C:\lab\fak-private\tools\dgxbridge\LAB-ACCESS-SOURCE-5910`
	deniedPayload := "DENIED-TRACE-PAYLOAD-5910"
	args, err := json.Marshal(map[string]string{
		"path":    deniedSource,
		"payload": deniedPayload,
	})
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	sink.Emit(abi.Event{Kind: abi.EvSubmit, Call: &abi.ToolCall{
		TraceID: "trace-5910",
		Tool:    "Read",
		Args:    abi.Ref{Kind: abi.RefInline, Inline: args},
	}})

	stored, err := json.Marshal(sink.Trace())
	if err != nil {
		t.Fatalf("Marshal trace: %v", err)
	}
	for _, forbidden := range []string{deniedSource, deniedPayload} {
		if bytes.Contains(stored, []byte(forbidden)) {
			t.Fatalf("denied bytes reached trace at rest: %s", stored)
		}
	}

	trace := sink.Trace()
	if len(trace.Calls) != 1 {
		t.Fatalf("calls = %d, want exactly one refusal", len(trace.Calls))
	}
	refusal := trace.Calls[0]
	if refusal.Meta[MetaCaptureRefused] != "true" || refusal.Meta[MetaCaptureReason] != "SECRET_EXFIL" {
		t.Fatalf("refusal meta = %#v, want capture_refused + SECRET_EXFIL", refusal.Meta)
	}
	if refusal.Meta[MetaCaptureSourceDigest] == "" {
		t.Fatalf("refusal lost source digest: %#v", refusal.Meta)
	}
	if sink.Refused() != 1 {
		t.Fatalf("refused = %d, want 1", sink.Refused())
	}
}
