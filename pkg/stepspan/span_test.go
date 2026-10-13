package stepspan

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

// fak-test:runtime fast est=1s
func fixture() Span {
	return Span{Schema: Schema, TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", SpanID: "00f067aa0ba902b7", Plane: Engine, Kind: DecodeStep, Name: "batched decode", StartUnixNano: 1, EndUnixNano: 2, Status: OK, Service: "fak-serve"}
}
func TestKindsContract(t *testing.T) {
	want := map[Plane]string{Ops: "run wave spawn invocation", Agent: "turn tool_call llm_call adjudicate", Planner: "plan plan_step observe replan", Router: "request admission route_decision attempt", Gateway: "request admission_wait stream", Sched: "tick", Engine: "prefix_lookup prefill_chunk cohort decode_step spec_round sample kv_alloc kv_evict preempt", Kernel: "layer op"}
	for p, names := range want {
		var got []string
		for _, k := range Kinds(p) {
			got = append(got, string(k))
		}
		if !reflect.DeepEqual(got, strings.Fields(names)) {
			t.Fatalf("%s kinds = %v", p, got)
		}
	}
	if Kinds("future") != nil {
		t.Fatal("unknown plane accepted")
	}
}
func TestSchemaRoundTrip(t *testing.T) {
	s := fixture()
	s.ParentSpanID = "a3ce929d0e0e4736"
	s.Links = []Link{{s.TraceID, s.SpanID}}
	s.IDs = IDs{RunID: "run", SessionID: "session", Turn: 3, CallID: "call", RequestID: "request", Attempt: 1, SeqIDs: []string{"a", "b"}, CohortID: 9912, SchedIter: 88123}
	s.Attrs = Attrs{Path: "batched", Lanes: 2, Tokens: 2, Layer: 12, Op: "matmul"}
	if err := s.Validate(); err != nil {
		t.Fatal(err)
	}
	b, err := json.Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	var got Span
	if err = json.Unmarshal(b, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(s, got) {
		t.Fatalf("roundtrip = %#v", got)
	}
	var fields map[string]any
	_ = json.Unmarshal(b, &fields)
	for _, k := range []string{"schema", "trace_id", "span_id", "parent_span_id", "links", "plane", "kind", "name", "start_unix_nano", "end_unix_nano", "status", "host", "pid", "service", "ids", "attrs"} {
		if _, ok := fields[k]; !ok {
			t.Errorf("missing %s", k)
		}
	}
	for _, bad := range []Span{{}, func() Span { v := s; v.Kind = Tick; return v }(), func() Span { v := s; v.EndUnixNano = 0; return v }(), func() Span { v := s; v.Status = "future"; return v }()} {
		if !errors.Is(bad.Validate(), ErrInvalidSpan) {
			t.Fatal("invalid span accepted")
		}
	}
}
func TestIDsW3C(t *testing.T) {
	trace, err := NewTraceID()
	if err != nil || !ValidTraceID(trace) {
		t.Fatalf("trace id: %s %v", trace, err)
	}
	span, err := NewSpanID()
	if err != nil || !ValidSpanID(span) {
		t.Fatalf("span id: %s %v", span, err)
	}
	c := TraceContext{trace, span, 1}
	header, err := c.Traceparent()
	if err != nil {
		t.Fatal(err)
	}
	got, err := ParseTraceparent(header)
	if err != nil || got != c {
		t.Fatalf("traceparent roundtrip: %v %v", got, err)
	}
	for _, bad := range []string{"", strings.Replace(header, trace, strings.Repeat("0", 32), 1), strings.Replace(header, "00-", "ff-", 1), header + "-extra", header[:len(header)-2] + "GG", strings.ToUpper(header)} {
		if _, err := ParseTraceparent(bad); !errors.Is(err, ErrInvalidTraceparent) {
			t.Fatalf("accepted %q", bad)
		}
	}
}
