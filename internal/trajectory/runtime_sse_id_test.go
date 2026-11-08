package trajectory

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1ms lane=default
func TestRuntimeSSERejectsUnsafeIDBeforeWriting(t *testing.T) {
	for _, id := range []string{"event\nretry: 0", "event\rdata: injected", "event\r\n\r\ndata: extra", "event\x00suffix"} {
		event := safeRuntimeEvent(t)
		event.EventID = id
		frame, err := EncodeRuntimeEvent(event, RuntimeSSE)
		if err == nil || frame != nil {
			t.Fatalf("unsafe ID %q produced frame %q: %v", id, frame, err)
		}
		var out bytes.Buffer
		if err := WriteRuntimeEvent(&out, event, RuntimeSSE); err == nil || out.Len() != 0 {
			t.Fatalf("unsafe ID %q wrote %d bytes: %v", id, out.Len(), err)
		}
		// NDJSON can represent the same identifier safely with JSON escapes.
		frame, err = EncodeRuntimeEvent(event, RuntimeNDJSON)
		if err != nil {
			t.Fatal(err)
		}
		var wire RuntimeWireEvent
		if err := json.Unmarshal(frame, &wire); err != nil || wire.RuntimeEvent.EventID != id || bytes.Count(frame, []byte{'\n'}) != 1 {
			t.Fatalf("NDJSON identity/framing changed for %q: %v", id, err)
		}
	}
	// Punctuation and Unicode are valid field values and must not be stripped.
	event := safeRuntimeEvent(t)
	event.EventID = "session: café / 1"
	frame, err := EncodeRuntimeEvent(event, RuntimeSSE)
	if err != nil || !strings.HasPrefix(string(frame), "id: "+event.EventID+"\n") {
		t.Fatalf("valid SSE ID changed: %q %v", frame, err)
	}
}
