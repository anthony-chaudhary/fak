package trajectory

import (
	"encoding/json"
	"testing"
	"time"
)

func TestToolCallEventsRoundTripAndAdmitExactlyOnce(t *testing.T) {
	src := RuntimeSource{Component: "agent-loop", Instance: "worker-7", Runtime: "fak"}
	events, err := ToolCallEvents("sess-1", "turn-2", "trace-3", "call-4", "search", true, json.RawMessage(`{"ok":true}`), time.Unix(10, 0).UTC(), src)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	admitted := 0
	for i, e := range events {
		if seen[e.EventID] {
			t.Fatalf("duplicate id %q", e.EventID)
		}
		seen[e.EventID] = true
		if e.Sequence != uint64(i+1) {
			t.Fatalf("sequence %d", e.Sequence)
		}
		if e.Kind == "tool_result_admitted" {
			admitted++
		}
	}
	if admitted != 1 {
		t.Fatalf("admitted results=%d", admitted)
	}
	canonical, err := AsTrajectoryEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	wire, err := EncodeEvents(canonical)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEvents(wire)
	if err != nil {
		t.Fatal(err)
	}
	if len(decoded) != 5 || decoded[2].Action != "tool_verdict" {
		t.Fatalf("round trip %#v", decoded)
	}
}

func TestDeniedToolCallHasVerdictWithoutResult(t *testing.T) {
	events, err := ToolCallEvents("s", "t", "x", "c", "delete", false, nil, time.Unix(10, 0).UTC(), RuntimeSource{Component: "agent-loop", Instance: "w", Runtime: "fak"})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range events {
		if e.Kind == "tool_result_admitted" {
			t.Fatal("denied call admitted a result")
		}
	}
	if len(events) != 4 || events[2].Kind != "tool_verdict" {
		t.Fatalf("events %#v", events)
	}
}
