package trajectory

import (
	"bytes"
	"encoding/json"
	"reflect"
	"strings"
	"testing"
	"time"
)

// These witnesses intentionally use only pre-envelope APIs and raw JSON so the
// same test file compiles against the parent during native symptom verification.

// fak-test:runtime fast est=10ms
func TestRuntimeEventCausalProjection(t *testing.T) {
	t.Parallel()
	r, err := NewRuntimeEvent("event:7", " session opaque ", "turn/7", " Trace/Native ", 7,
		time.Unix(10, 0).UTC(), RuntimeToolResult,
		RuntimeSource{Rung: "agent", Component: "agent-loop", Instance: "worker", Runtime: "fak"}, json.RawMessage(`{"ok":true}`))
	if err != nil {
		t.Fatal(err)
	}
	events, err := AsTrajectoryEvents([]RuntimeEvent{r})
	if err != nil {
		t.Fatal(err)
	}
	first, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	var wire map[string]json.RawMessage
	if err := json.Unmarshal(first, &wire); err != nil {
		t.Fatal(err)
	}
	if len(wire["causal"]) == 0 {
		t.Fatal("AsTrajectoryEvents lost first-class causal identity; only payload carries the native trace")
	}
	var causal map[string]json.RawMessage
	if err := json.Unmarshal(wire["causal"], &causal); err != nil {
		t.Fatal(err)
	}
	for key, want := range map[string]string{
		"schema": "fak-causal-envelope/1", "trace_id": r.TraceID, "trace_format": "opaque",
		"logical_id": r.EventID, "operation": r.Source.Component, "phase": r.Kind,
		"source_schema": RuntimeEventSchema, "source_version": "1", "mapping_status": "unknown", "parent_status": "unknown",
	} {
		var got string
		if err := json.Unmarshal(causal[key], &got); err != nil || got != want {
			t.Fatalf("causal.%s=%s, want %q (error %v)", key, causal[key], want, err)
		}
	}
	if string(causal["sequence"]) != "7" || len(causal["span_id"]) != 0 || len(causal["parent_span_id"]) != 0 || len(causal["links"]) != 0 || len(events[0].ParentIDs) != 0 {
		t.Fatalf("projection invented causality or lost sequence: %s", wire["causal"])
	}
	if got, want := string(causal["correlations"]), `[{"kind":"session","value":" session opaque "},{"kind":"turn","value":"turn/7"}]`; got != want {
		t.Fatalf("correlations=%s, want %s", got, want)
	}
	var native RuntimeEvent
	if err := json.Unmarshal(events[0].Payload, &native); err != nil || !reflect.DeepEqual(native, r) {
		t.Fatalf("native payload changed: %+v, %v", native, err)
	}
	// Both the existing strict stream reader and direct strict Event decoding
	// must retain the new typed metadata rather than silently losing it.
	decoded, err := DecodeEvents(first)
	if err != nil {
		t.Fatal(err)
	}
	var direct Event
	dec := json.NewDecoder(bytes.NewReader(first))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&direct); err != nil {
		t.Fatal(err)
	}
	for _, roundtrip := range [][]Event{decoded, {direct}} {
		second, err := EncodeEvents(roundtrip)
		if err != nil || !bytes.Equal(first, second) {
			t.Fatalf("causal round trip changed bytes: %s, %v", second, err)
		}
	}
	// A source trace that resembles W3C is still explicitly producer-native.
	r.TraceID = "4bf92f3577b34da6a3ce929d0e0e4736"
	events, err = AsTrajectoryEvents([]RuntimeEvent{r})
	if err != nil {
		t.Fatal(err)
	}
	encoded, err := EncodeEvents(events)
	if err != nil || !bytes.Contains(encoded, []byte(`"trace_format":"opaque"`)) {
		t.Fatalf("guessed W3C mapping: %s, %v", encoded, err)
	}
}

// fak-test:runtime fast est=10ms
func TestCausalEnvelopeForkJoinTrajectory(t *testing.T) {
	t.Parallel()
	const causal = `{"schema":"fak-causal-envelope/1","trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","trace_format":"w3c","span_id":"0000000000000004","parent_span_id":"0000000000000001","parent_status":"known","sequence":4,"logical_id":"join","operation":"merge","phase":"completed","correlations":[{"kind":"turn","value":"turn/native"},{"kind":"run","value":"run/native"}],"links":[{"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","span_id":"0000000000000003","trace_format":"w3c"},{"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","span_id":"0000000000000002","trace_format":"w3c"}],"source_schema":"fixture/1","source_version":"1","mapping_status":"exact"}`
	e := Event{Schema: EventSchema, ID: "join-event", ConversationID: "conversation", Kind: EventTool, Action: "joined", Timestamp: time.Unix(10, 0).UTC(), Sequence: 4, ParentIDs: []string{"right-event", "left-event"}, Visibility: VisibilityOperator, Source: EventSource{Type: "fixture", Adapter: "fixture", AdapterVersion: "1"}, Payload: json.RawMessage(`{}`)}
	base, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(base, &fields); err != nil {
		t.Fatal(err)
	}
	fields["causal"] = json.RawMessage(causal)
	wire, err := json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := DecodeEvents(wire)
	if err != nil {
		t.Fatal(err)
	}
	first, err := EncodeEvents(decoded)
	if err != nil {
		t.Fatal(err)
	}
	again, err := DecodeEvents(first)
	if err != nil {
		t.Fatal(err)
	}
	second, err := EncodeEvents(again)
	if err != nil || !bytes.Equal(first, second) {
		t.Fatalf("fork/join changed canonical bytes: %s, %v", second, err)
	}
	if !reflect.DeepEqual(again[0].ParentIDs, e.ParentIDs) {
		t.Fatalf("event parent order or identity changed: %v", again[0].ParentIDs)
	}
	if bytes.Index(first, []byte(`"span_id":"0000000000000002"`)) > bytes.Index(first, []byte(`"span_id":"0000000000000003"`)) || !bytes.Contains(first, []byte(`"parent_span_id":"0000000000000001"`)) {
		t.Fatalf("single parent and fan-in links were conflated: %s", first)
	}
	for _, bad := range []string{
		strings.Replace(causal, `"mapping_status":"exact"`, `"mapping_status":"guessed"`, 1),
		strings.Replace(causal, `"parent_span_id":"0000000000000001"`, `"parent_span_id":"0000000000000004"`, 1),
		strings.Replace(causal, `"source_version":"1"`, `"source_version":"1","extra":true`, 1),
	} {
		fields["causal"] = json.RawMessage(bad)
		invalid, err := json.Marshal(fields)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := DecodeEvents(invalid); err == nil {
			t.Fatalf("strict Event reader accepted malformed causal metadata: %s", bad)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestCausalEnvelopeViewRedactionAndLegacy(t *testing.T) {
	t.Parallel()
	r, err := NewRuntimeEvent("event", "session-private", "turn-private", "trace-private", 1,
		time.Unix(10, 0).UTC(), RuntimeToolResult,
		RuntimeSource{Rung: "agent", Component: "agent-loop", Instance: "worker", Runtime: "fak"}, json.RawMessage(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	events, err := AsTrajectoryEvents([]RuntimeEvent{r})
	if err != nil {
		t.Fatal(err)
	}
	before, err := EncodeEvents(events)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(before, []byte(`"causal":`)) {
		t.Fatal("runtime adapter omitted causal metadata before view projection")
	}
	for _, fields := range [][]string{{"trace_id"}, {"turn_id"}, {"session_id"}, {"unrelated"}, nil} {
		spec := ViewSpec{Schema: ViewSpecSchema, Audience: "fixture", Include: []string{"tool.*"}, RedactionProfile: "fixture/1", RedactFields: fields}
		view, receipt, err := CompileView(events, spec)
		if err != nil {
			t.Fatal(err)
		}
		wire, err := json.Marshal(view)
		if err != nil {
			t.Fatal(err)
		}
		if len(fields) != 0 {
			if bytes.Contains(wire, []byte(`"causal":`)) || !containsCausalField(view[0].RedactedFields) || receipt.RedactedFields != len(view[0].RedactedFields) {
				t.Fatalf("redacting view retained aliased causal data or lost accounting: %s, %+v", wire, receipt)
			}
		} else {
			if !bytes.Contains(wire, []byte(`"causal":`)) || receipt.RedactedFields != 0 {
				t.Fatalf("unredacted view lost causal data: %s", wire)
			}
			// Mutate the returned envelope through reflection to keep this file
			// parent-compatible while proving pointer and slice isolation.
			causal := reflect.ValueOf(&view[0].Event).Elem().FieldByName("Causal").Elem()
			causal.FieldByName("TraceID").SetString("changed")
			causal.FieldByName("Correlations").Index(0).FieldByName("Value").SetString("changed")
		}
		after, err := EncodeEvents(events)
		if err != nil || !bytes.Equal(before, after) {
			t.Fatalf("view mutated canonical evidence: %s, %v", after, err)
		}
	}
	// A pre-envelope row retains its bytes, digest and optional-field absence.
	legacy := events[0]
	legacyWire, err := json.Marshal(legacy)
	if err != nil {
		t.Fatal(err)
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(legacyWire, &fields); err != nil {
		t.Fatal(err)
	}
	delete(fields, "causal")
	legacyWire, err = json.Marshal(fields)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := DecodeEvents(legacyWire)
	if err != nil {
		t.Fatal(err)
	}
	canonical, err := EncodeEvents(rows)
	if err != nil || bytes.Contains(canonical, []byte(`"causal":`)) {
		t.Fatalf("legacy row acquired fabricated metadata: %s, %v", canonical, err)
	}
	status := reflect.ValueOf(rows[0]).MethodByName("CausalMappingStatus")
	if !status.IsValid() || status.Call(nil)[0].String() != "unknown" {
		t.Fatal("legacy row must explicitly report unknown causal mapping")
	}
}

func containsCausalField(fields []string) bool {
	for _, field := range fields {
		if field == "causal" {
			return true
		}
	}
	return false
}
