package trajectory

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestRecorderDeniedSourceStoresOnlyRefusal is the #5910 trajectory witness:
// source admission runs before the query-bearing Turn enters the Recorder, and
// the exported corpus contains one auditable refusal but none of the denied bytes.
func TestRecorderDeniedSourceStoresOnlyRefusal(t *testing.T) {
	r := New()
	deniedSource := `../fak-private/tools/dgxbridge/LAB-ACCESS-SOURCE-5910`
	deniedPayload := "DENIED-TRAJECTORY-PAYLOAD-5910"
	r.Emit(abi.Event{Kind: abi.EvDecide, Call: &abi.ToolCall{
		TraceID: "trace-5910",
		Tool:    "Read",
		Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(
			`{"path":"` + deniedSource + `"}`,
		)},
		Meta: map[string]string{"query": deniedPayload},
	}, Verdict: &abi.Verdict{Kind: abi.VerdictAllow}})

	var stored bytes.Buffer
	if n, err := r.ExportTo(&stored); err != nil || n != 1 {
		t.Fatalf("ExportTo = %d, %v; want one refusal row", n, err)
	}
	for _, forbidden := range []string{deniedSource, deniedPayload} {
		if bytes.Contains(stored.Bytes(), []byte(forbidden)) {
			t.Fatalf("denied bytes reached trajectory at rest: %s", stored.Bytes())
		}
	}

	turns := r.Turns()
	if len(turns) != 1 {
		t.Fatalf("turns = %d, want exactly one refusal", len(turns))
	}
	got := turns[0]
	if got.Verdict != "DENY" || got.Reason != "SECRET_EXFIL" || got.Query != "" {
		t.Fatalf("refusal turn = %+v, want content-free DENY/SECRET_EXFIL", got)
	}
	if got.Labels["capture_refused"] != "true" || got.Labels["capture_source_digest"] == "" {
		t.Fatalf("refusal audit labels = %#v, want refusal marker + source digest", got.Labels)
	}
}
