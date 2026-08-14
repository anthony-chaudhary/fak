package harnesskit

import "testing"

func TestProtocolValidationAndUnknownAdditiveEvent(t *testing.T) {
	e := Envelope{Version: ProtocolVersion, RunID: "r", Sequence: 1, EventID: "r:1", Type: EventType("future.additive")}
	if err := e.Validate(); err != nil {
		t.Fatal(err)
	}
	if e.Known() {
		t.Fatal("future event must remain unknown")
	}
	cancel := Input{Version: ProtocolVersion, RunID: "r", InputID: "i", Type: InputCancel, Reason: "operator"}
	if err := cancel.Validate(); err != nil {
		t.Fatal(err)
	}
}
