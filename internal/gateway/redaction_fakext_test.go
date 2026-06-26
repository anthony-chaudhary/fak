package gateway

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/wirescreen"
)

// TestWireRedactionsFrom (#882): the FakExt projection of the agent's reversible
// redaction records preserves the audit fields — including the CAS digest of the
// unredacted original (the reverse-on-audit handle) and the span COUNT — and an
// empty record set projects to nil so the `redactions` key is omitted on the wire.
func TestWireRedactionsFrom(t *testing.T) {
	if got := wireRedactionsFrom(nil); got != nil {
		t.Errorf("empty records must project to nil, got %v", got)
	}

	recs := []agent.TranscriptRedaction{
		{
			Index: 0, Tool: "fetch_url", By: "pii",
			Original: abi.Ref{Digest: "deadbeefcafef00d"}, Len: 42,
			Spans: []wirescreen.Span{{Start: 5, End: 9, Kind: "credit_card"}, {Start: 20, End: 24, Kind: "email"}},
		},
		{Index: 2, By: "model", Original: abi.Ref{Digest: "0011223344"}, Len: 7},
	}
	got := wireRedactionsFrom(recs)
	if len(got) != 2 {
		t.Fatalf("projected %d records, want 2", len(got))
	}
	if got[0] != (WireRedaction{Index: 0, Tool: "fetch_url", By: "pii", Original: "deadbeefcafef00d", Len: 42, Spans: 2}) {
		t.Errorf("record 0 projection wrong: %+v", got[0])
	}
	if got[1] != (WireRedaction{Index: 2, By: "model", Original: "0011223344", Len: 7, Spans: 0}) {
		t.Errorf("record 1 projection wrong: %+v", got[1])
	}
}
