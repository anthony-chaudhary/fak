package trajectory

import (
	"bufio"
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"testing"
	"time"
)

func TestRuntimeEventDescriptorAndAllKindsGolden(t *testing.T) {
	d := RuntimeEventSchemaDescriptor()
	if d.Schema != RuntimeEventDescriptorSchema || d.EventSchema != RuntimeEventSchema || !reflect.DeepEqual(d.Kinds, RuntimeEventKinds()) {
		t.Fatalf("descriptor=%+v", d)
	}
	data, err := os.ReadFile("testdata/runtime-events-all-kinds.ndjson")
	if err != nil {
		t.Fatal(err)
	}
	s := bufio.NewScanner(bytes.NewReader(data))
	var runtimeEvents []RuntimeEvent
	for s.Scan() {
		var event RuntimeEvent
		if err := json.Unmarshal(s.Bytes(), &event); err != nil {
			t.Fatal(err)
		}
		if err := ValidateRuntimeEvent(event); err != nil {
			t.Fatal(err)
		}
		runtimeEvents = append(runtimeEvents, event)
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, e := range runtimeEvents {
		seen[e.Kind] = true
	}
	for _, kind := range RuntimeEventKinds() {
		if !seen[kind] {
			t.Fatalf("missing kind %s", kind)
		}
	}
	adapted, err := AsTrajectoryEvents(runtimeEvents)
	if err != nil {
		t.Fatal(err)
	}
	var ndjson bytes.Buffer
	enc := json.NewEncoder(&ndjson)
	for _, e := range adapted {
		if err := enc.Encode(e); err != nil {
			t.Fatal(err)
		}
	}
	decoded, err := DecodeEvents(ndjson.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != len(runtimeEvents) {
		t.Fatalf("decoded=%d", len(decoded))
	}
}

func TestValidateRuntimeEventFailsClosed(t *testing.T) {
	source := RuntimeSource{Component: "loop", Instance: "one", Runtime: "fak"}
	payload := json.RawMessage(`{}`)
	base, _ := NewRuntimeEvent("e", "s", "t", "trace", 1, time.Now().UTC(), RuntimeTurnStarted, source, payload)
	cases := []RuntimeEvent{base, base, base, base}
	cases[0].Kind = "unknown"
	cases[1].Sequence = 0
	cases[2].TraceID = ""
	cases[3].Payload = json.RawMessage(`{`)
	for _, event := range cases {
		if err := ValidateRuntimeEvent(event); err == nil {
			t.Fatalf("accepted %+v", event)
		}
	}
}
