package fakclient

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestGeneratedRuntimeSDKDoesNotDrift(t *testing.T) {
	want, err := trajectory.GenerateGoRuntimeSDK("fakclient")
	if err != nil {
		t.Fatal(err)
	}
	got, err := os.ReadFile("runtime_event_generated.go")
	if err != nil {
		t.Fatal(err)
	}
	gotNorm := bytes.ReplaceAll(got, []byte("\r\n"), []byte("\n"))
	wantNorm := bytes.ReplaceAll(want, []byte("\r\n"), []byte("\n"))
	if !bytes.Equal(gotNorm, wantNorm) {
		t.Fatal("runtime SDK generated source drifted")
	}
}
func TestRuntimeSDKDecodesCanonicalAllKinds(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "internal", "trajectory", "testdata", "runtime-events-all-kinds.ndjson"))
	if err != nil {
		t.Fatal(err)
	}
	for _, line := range bytes.Split(bytes.TrimSpace(data), []byte("\n")) {
		var event RuntimeEvent
		if err := json.Unmarshal(line, &event); err != nil {
			t.Fatal(err)
		}
		wire := RuntimeWireEvent{Schema: RuntimeEventWireSchema, Event: event, Admission: RuntimeAdmission{Screened: true, Taint: "tainted", Screen: "ctxmmu/1"}}
		body, _ := json.Marshal(wire)
		got, err := DecodeRuntimeNDJSON(body)
		if err != nil {
			t.Fatal(err)
		}
		sse := append([]byte("id: x\nevent: x\ndata: "), body...)
		sse = append(sse, []byte("\n\n")...)
		got2, err := DecodeRuntimeSSE(sse)
		if err != nil || got2.Event.Kind != got.Event.Kind {
			t.Fatalf("sse=%+v err=%v", got2, err)
		}
	}
}
func TestRuntimeSDKRejectsSchemaAndKindDrift(t *testing.T) {
	wire := RuntimeWireEvent{Schema: "wrong", Event: RuntimeEvent{Schema: RuntimeEventSchema, EventID: "e", TraceID: "t", Sequence: 1, Kind: RuntimeKindError}, Admission: RuntimeAdmission{Screened: true}}
	b, _ := json.Marshal(wire)
	if _, err := DecodeRuntimeNDJSON(b); err == nil {
		t.Fatal("accepted schema drift")
	}
	wire.Schema = RuntimeEventWireSchema
	wire.Event.Kind = "new_kind"
	b, _ = json.Marshal(wire)
	if _, err := DecodeRuntimeNDJSON(b); err == nil {
		t.Fatal("accepted kind drift")
	}
}
