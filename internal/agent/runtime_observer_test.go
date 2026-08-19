package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestRuntimeObserverOwnedLoopSequenceAndExactlyOnceResult(t *testing.T) {
	var out bytes.Buffer
	o, err := NewRuntimeObserver(&out, trajectory.RuntimeNDJSON, "session-1", "trace-1", trajectory.RuntimeSource{Component: "agent-loop", Instance: "test", Runtime: "fak"})
	if err != nil {
		t.Fatal(err)
	}
	o.now = func() time.Time { return time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC) }
	m, err := RunArm(context.Background(), &scriptedPlanner{turns: []*Completion{toolCallTurn("get_user", `{"user_id":"mia_li_3668"}`), {Message: Message{Content: "done"}}}}, "lookup", false, 4, nil, WithProgressObserver(o.Observe))
	if err != nil {
		t.Fatal(err)
	}
	if o.Err() != nil {
		t.Fatal(o.Err())
	}
	if m.ToolCalls != 1 {
		t.Fatalf("metrics=%+v", m)
	}
	lines := bytes.Split(bytes.TrimSpace(out.Bytes()), []byte("\n"))
	kinds := []string{}
	results := 0
	for i, line := range lines {
		var wire trajectory.RuntimeWireEvent
		if err := json.Unmarshal(line, &wire); err != nil {
			t.Fatal(err)
		}
		if wire.RuntimeEvent.Sequence != uint64(i+1) || wire.RuntimeEvent.TraceID != "trace-1" || !wire.Admission.Screened {
			t.Fatalf("wire=%+v", wire)
		}
		kinds = append(kinds, wire.RuntimeEvent.Kind)
		if wire.RuntimeEvent.Kind == trajectory.RuntimeToolResult {
			results++
		}
	}
	want := []string{trajectory.RuntimeTurnStarted, trajectory.RuntimeToolProposed, trajectory.RuntimeVerdict, trajectory.RuntimeToolResult, trajectory.RuntimeTerminalWitness, trajectory.RuntimeTurnStarted, trajectory.RuntimeTerminalWitness}
	if !reflect.DeepEqual(kinds, want) {
		t.Fatalf("kinds=%v want=%v", kinds, want)
	}
	if results != 1 {
		t.Fatalf("results=%d", results)
	}
}

func TestRuntimeObserverRejectsDuplicateAdmission(t *testing.T) {
	var out bytes.Buffer
	o, _ := NewRuntimeObserver(&out, trajectory.RuntimeNDJSON, "s", "tr", trajectory.RuntimeSource{Component: "loop", Instance: "test", Runtime: "fak"})
	event := ProgressEvent{Kind: ProgressResultAdmitted, Turn: 1, CallID: "c", Tool: "x", Taint: "clean"}
	o.Observe(event)
	n := out.Len()
	o.Observe(event)
	if o.Err() == nil {
		t.Fatal("accepted duplicate")
	}
	if out.Len() != n {
		t.Fatal("duplicate emitted bytes")
	}
}
