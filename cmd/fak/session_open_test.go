package main

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

func TestRenderSessionOpenCapturesFullSharedDescriptor(t *testing.T) {
	var out strings.Builder
	renderSessionOpen(&out, gateway.SessionClientAttachResponse{
		Descriptor:   gateway.SessionClientDescriptor{SessionID: "logical-7", ExecutionEpoch: "epoch-a", EventHead: 9, CapabilityDigest: "sha256:abc", Capabilities: []string{"detach", "observe", "replay", "text_input"}, Endpoint: "http://gateway/v1/fak/session/logical-7", State: gateway.SessionState{Run: "RUNNING", Rev: 5}, Terminal: gateway.SessionTerminalView{Transcript: "same bytes\r\n", ByteLength: 12, Digest: "sha256:def"}, Effects: []gateway.SessionEffect{{ID: "effect-1", Verdict: gateway.SessionEffectUncertain, Check: "query receipt"}}},
		AttachmentID: "att-3", InputLease: true, Cursor: 9,
		Events: []gateway.SessionChangeEvent{{Seq: 9, SessionState: gateway.SessionState{Run: "RUNNING", Rev: 5}}},
	})
	got := out.String()
	for _, want := range []string{"fak session logical-7", "execution_epoch=epoch-a", "event_head=9", "input_lease=true", "detach,observe,replay,text_input", "#9 state=RUNNING rev=5", "browser=http://gateway/v1/fak/session/logical-7/open", "terminal_bytes=12 terminal_digest=sha256:def", "same bytes", "effect=effect-1 verdict=UNCERTAIN check=query receipt"} {
		if !strings.Contains(got, want) {
			t.Fatalf("terminal capture missing %q\n%s", want, got)
		}
	}
}
