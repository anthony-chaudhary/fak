package traceabi

import (
	"bytes"
	"encoding/json"
	"os"
	"reflect"
	"slices"
	"strings"
	"testing"
)

// fak-test:runtime fast est=10ms
func TestCausalEnvelopeValidation(t *testing.T) {
	t.Parallel()
	base := testEnvelope()
	for _, tc := range []struct {
		name string
		edit func(*CausalEnvelope)
	}{
		{"schema", func(e *CausalEnvelope) { e.Schema = "future" }},
		{"empty trace", func(e *CausalEnvelope) { e.TraceID = "" }},
		{"empty logical", func(e *CausalEnvelope) { e.LogicalID = "" }},
		{"empty operation", func(e *CausalEnvelope) { e.Operation = "" }},
		{"empty phase", func(e *CausalEnvelope) { e.Phase = "" }},
		{"empty source schema", func(e *CausalEnvelope) { e.SourceSchema = "" }},
		{"empty source version", func(e *CausalEnvelope) { e.SourceVersion = "" }},
		{"zero sequence", func(e *CausalEnvelope) { e.Sequence = 0 }},
		{"unknown format", func(e *CausalEnvelope) { e.TraceFormat = "guessed" }},
		{"unknown mapping", func(e *CausalEnvelope) { e.MappingStatus = "guessed" }},
		{"unknown parent status", func(e *CausalEnvelope) { e.ParentStatus = "guessed" }},
		{"short W3C trace", func(e *CausalEnvelope) { e.TraceID = "trace" }},
		{"zero W3C trace", func(e *CausalEnvelope) { e.TraceID = strings.Repeat("0", 32) }},
		{"uppercase W3C trace", func(e *CausalEnvelope) { e.TraceID = strings.ToUpper(e.TraceID) }},
		{"short W3C span", func(e *CausalEnvelope) { e.SpanID = "span" }},
		{"zero W3C span", func(e *CausalEnvelope) { e.SpanID = strings.Repeat("0", 16) }},
		{"missing span exact", func(e *CausalEnvelope) { e.SpanID = ""; e.ParentStatus = ParentUnknown }},
		{"missing span root", func(e *CausalEnvelope) { e.SpanID = ""; e.MappingStatus = MappingUnknown }},
		{"missing known parent", func(e *CausalEnvelope) { e.ParentStatus = ParentKnown }},
		{"root with parent", func(e *CausalEnvelope) { e.ParentSpanID = "0000000000000002" }},
		{"unknown with parent", func(e *CausalEnvelope) { e.ParentStatus = ParentUnknown; e.ParentSpanID = "0000000000000002" }},
		{"self parent", func(e *CausalEnvelope) { e.ParentStatus = ParentKnown; e.ParentSpanID = e.SpanID }},
		{"zero W3C parent", func(e *CausalEnvelope) { e.ParentStatus = ParentKnown; e.ParentSpanID = strings.Repeat("0", 16) }},
		{"unknown correlation", func(e *CausalEnvelope) { e.Correlations = []Correlation{{"guess", "value"}} }},
		{"empty correlation", func(e *CausalEnvelope) { e.Correlations = []Correlation{{CorrelationTurn, ""}} }},
		{"duplicate correlation", func(e *CausalEnvelope) { e.Correlations = []Correlation{{CorrelationRun, "x"}, {CorrelationRun, "x"}} }},
		{"empty link span", func(e *CausalEnvelope) { e.Links = []SpanLink{{TraceFormat: TraceOpaque, TraceID: "x"}} }},
		{"unknown link format", func(e *CausalEnvelope) { e.Links = []SpanLink{{TraceFormat: "guess", TraceID: "x", SpanID: "y"}} }},
		{"invalid W3C link", func(e *CausalEnvelope) { e.Links = []SpanLink{{TraceFormat: TraceW3C, TraceID: "x", SpanID: "y"}} }},
		{"duplicate link", func(e *CausalEnvelope) { e.Links = []SpanLink{{"x", "y", TraceOpaque}, {"x", "y", TraceOpaque}} }},
		{"invalid UTF-8", func(e *CausalEnvelope) { e.TraceFormat = TraceOpaque; e.TraceID = string([]byte{0xff}) }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := base
			tc.edit(&e)
			first, second := e.Validate(), e.Validate()
			if first == nil || second == nil || first.Error() != second.Error() {
				t.Fatalf("validation must reject deterministically: %v, %v", first, second)
			}
			if _, err := json.Marshal(e); err == nil {
				t.Fatal("marshal admitted invalid envelope")
			}
		})
	}
	unknown := base
	unknown.TraceFormat, unknown.TraceID, unknown.SpanID = TraceOpaque, " opaque trace ", ""
	unknown.MappingStatus, unknown.ParentStatus = MappingUnknown, ParentUnknown
	unknown.Correlations = []Correlation{{CorrelationTurn, " x "}, {CorrelationTurn, "x"}}
	if err := unknown.Validate(); err != nil {
		t.Fatalf("explicitly unknown native identity must remain valid: %v", err)
	}
	derived := base
	derived.MappingStatus = MappingDerived
	derived.ParentStatus = ParentUnknown
	derived.Links = []SpanLink{{TraceID: "other trace", SpanID: "opaque span", TraceFormat: TraceOpaque}}
	if err := derived.Validate(); err != nil {
		t.Fatalf("explicit derived mapping/cross-trace link rejected: %v", err)
	}
}

// fak-test:runtime fast est=10ms
func TestCausalEnvelopeStrictJSON(t *testing.T) {
	t.Parallel()
	encoded, err := json.Marshal(testEnvelope())
	if err != nil {
		t.Fatal(err)
	}
	for _, malformed := range []string{
		`null`, `[]`, string(encoded) + `{}`,
		strings.Replace(string(encoded), `"sequence":1`, `"sequence":0`, 1),
		strings.Replace(string(encoded), `"sequence":1`, `"sequence":1,"sequence":2`, 1),
		strings.Replace(string(encoded), `"sequence":1`, `"Sequence":1`, 1),
		strings.Replace(string(encoded), `"sequence":1`, `"sequence":1,"extra":true`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"guessed"`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"exact","correlations":[{"kind":"run","value":"x","extra":true}]`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"exact","correlations":[{"kind":"run","value":"x","value":"y"}]`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"exact","links":[{"trace_id":"x","span_id":"y","Trace_Format":"opaque"}]`, 1),
		strings.Replace(string(encoded), `"logical_id":"root"`, `"logical_id":"\ud800"`, 1),
		strings.Replace(string(encoded), `"trace_id":"4bf92f3577b34da6a3ce929d0e0e4736","trace_format":"w3c"`, `"trace_id":"\udc00","trace_format":"opaque"`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"exact","correlations":[{"kind":"run","value":"\ud800"}]`, 1),
		strings.Replace(string(encoded), `"mapping_status":"exact"`, `"mapping_status":"exact","links":[{"trace_id":"x","span_id":"\ud800\u0001","trace_format":"opaque"}]`, 1),
	} {
		e := testEnvelope()
		before := e
		if err := e.UnmarshalJSON([]byte(malformed)); err == nil {
			t.Fatalf("accepted malformed causal JSON: %s", malformed)
		}
		if !reflect.DeepEqual(e, before) {
			t.Fatal("failed decode mutated receiver")
		}
	}
	for _, valid := range []struct{ escaped, want string }{
		{`\ud83d\ude00`, "😀"}, {`\\ud800`, `\ud800`}, {`\ufffd`, "�"},
	} {
		wire := strings.Replace(string(encoded), `"logical_id":"root"`, `"logical_id":"`+valid.escaped+`"`, 1)
		var e CausalEnvelope
		if err := json.Unmarshal([]byte(wire), &e); err != nil || e.LogicalID != valid.want {
			t.Fatalf("lossless Unicode value rejected or changed: %q, %v", e.LogicalID, err)
		}
	}
}

// fak-test:runtime fast est=10ms
func TestCausalEnvelopeForkJoin(t *testing.T) {
	t.Parallel()
	golden, err := os.ReadFile("testdata/fork_join.golden.json")
	if err != nil {
		t.Fatal(err)
	}
	var graph []CausalEnvelope
	if err := json.Unmarshal(golden, &graph); err != nil {
		t.Fatal(err)
	}
	if len(graph) != 4 {
		t.Fatalf("fixture must contain root, two asynchronous children and join: %d", len(graph))
	}
	trace := graph[0].TraceID
	for i, e := range graph {
		if e.TraceID != trace || e.Sequence != uint64(i+1) {
			t.Fatalf("fixture lost trace/ordering: %+v", e)
		}
	}
	root, left, right, join := graph[0], graph[1], graph[2], graph[3]
	if root.ParentStatus != ParentRoot || left.ParentSpanID != root.SpanID || right.ParentSpanID != root.SpanID || join.ParentSpanID != root.SpanID || len(join.Links) != 2 || join.Links[0].SpanID != left.SpanID || join.Links[1].SpanID != right.SpanID {
		t.Fatalf("fork/join topology changed: %+v", graph)
	}
	// Deliberately reverse producer ordering. Canonical bytes must recover the
	// golden without sorting or modifying the caller-owned arrays in place.
	slices.Reverse(graph[3].Correlations)
	slices.Reverse(graph[3].Links)
	originalCorrelations := slices.Clone(graph[3].Correlations)
	originalLinks := slices.Clone(graph[3].Links)
	first, err := json.Marshal(graph)
	if err != nil || !bytes.Equal(first, bytes.TrimSpace(golden)) {
		t.Fatalf("canonical golden changed: %s, %v", first, err)
	}
	if !reflect.DeepEqual(originalCorrelations, graph[3].Correlations) || !reflect.DeepEqual(originalLinks, graph[3].Links) {
		t.Fatal("canonical marshal reordered source arrays")
	}
	var again []CausalEnvelope
	if err := json.Unmarshal(first, &again); err != nil {
		t.Fatal(err)
	}
	second, err := json.Marshal(again)
	if err != nil || !bytes.Equal(first, second) {
		t.Fatalf("round trip changed canonical bytes: %s, %v", second, err)
	}
	clone, err := graph[3].Canonical()
	if err != nil {
		t.Fatal(err)
	}
	clone.Correlations[0].Value = "changed"
	clone.Links[0].SpanID = "changed"
	if !reflect.DeepEqual(originalCorrelations, graph[3].Correlations) || !reflect.DeepEqual(originalLinks, graph[3].Links) {
		t.Fatal("canonical copy shares source arrays")
	}
}

func testEnvelope() CausalEnvelope {
	return CausalEnvelope{Schema: Schema, TraceID: "4bf92f3577b34da6a3ce929d0e0e4736", TraceFormat: TraceW3C,
		SpanID: "0000000000000001", ParentStatus: ParentRoot, Sequence: 1, LogicalID: "root",
		Operation: "fixture", Phase: "started", SourceSchema: "fixture/1", SourceVersion: "1", MappingStatus: MappingExact}
}
