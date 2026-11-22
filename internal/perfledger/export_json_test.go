package perfledger

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestJSONScrubTable(t *testing.T) {
	t.Parallel()
	nan, inf := math.NaN(), math.Inf(1)
	cases := []struct {
		name string
		in   any
		want string
	}{
		// Upstream golden: a metric whose stats came out NaN/inf exports as null.
		{"upstream_metric_stats", map[string]any{"ttft": map[string]any{"unit": "ms", "avg": nan, "p99": inf, "min": 1.5}},
			`{"ttft":{"avg":null,"min":1.5,"p99":null,"unit":"ms"}}`},
		{"nil", nil, `null`},
		{"bare_nan", nan, `null`},
		{"bare_neg_inf", math.Inf(-1), `null`},
		{"finite_float", 2.25, `2.25`},
		{"float32_inf", float32(math.Inf(1)), `null`},
		{"float32_finite", float32(0.5), `0.5`},
		{"zero_and_neg_zero", []float64{0, math.Copysign(0, -1)}, `[0,-0]`},
		{"max_float", math.MaxFloat64, `1.7976931348623157e+308`},
		{"typed_float_map", map[string]float64{"a": nan, "b": 3}, `{"a":null,"b":3}`},
		{"nested_slices", []any{[]float64{1, inf}, []any{nan, "x", true}}, `[[1,null],[null,"x",true]]`},
		{"array", [3]float64{nan, 1, math.Inf(-1)}, `[null,1,null]`},
		{"pointer", &inf, `null`},
		{"nil_pointer", (*float64)(nil), `null`},
		{"int_keys", map[int]any{2: nan, 1: 7}, `{"1":7,"2":null}`},
		{"empty_map", map[string]any{}, `{}`},
		{"empty_slice", []any{}, `[]`},
		{"nil_slice", []float64(nil), `null`},
		{"bytes", []byte("hi"), `"aGk="`},
		{"strings_ints", map[string]any{"s": "NaN", "i": int64(5)}, `{"i":5,"s":"NaN"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got, err := json.Marshal(ScrubNonFiniteJSON(tc.in))
			if err != nil {
				t.Fatalf("marshal scrubbed: %v", err)
			}
			if string(got) != tc.want {
				t.Fatalf("got %s want %s", got, tc.want)
			}
		})
	}
}

func TestJSONScrubDoesNotMutateInput(t *testing.T) {
	t.Parallel()
	inner := []float64{math.NaN(), 1}
	in := map[string]any{"v": inner, "f": math.Inf(1)}
	_ = ScrubNonFiniteJSON(in)
	if !math.IsNaN(inner[0]) || !math.IsInf(in["f"].(float64), 1) {
		t.Fatalf("input mutated: %v", in)
	}
}

func TestJSONScrubUnscrubbedMarshalFails(t *testing.T) {
	t.Parallel()
	payload := map[string]any{"avg": math.NaN()}
	if _, err := json.Marshal(payload); err == nil {
		t.Fatal("encoding/json accepted NaN; scrub would be unnecessary")
	}
	if _, err := MarshalScrubbedJSON(payload); err != nil {
		t.Fatalf("scrubbed marshal failed: %v", err)
	}
}

func TestJSONScrubExportFileRoundTrip(t *testing.T) {
	t.Parallel()
	payload := map[string]any{
		"records": []any{
			map[string]any{"tps": math.Inf(1), "n": 3},
			map[string]any{"tps": 41.5, "n": 4},
		},
		"summary": map[string]float64{"cv": math.NaN()},
	}
	out, err := MarshalScrubbedJSON(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	path := filepath.Join(t.TempDir(), "export.json")
	if err := os.WriteFile(path, out, 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read: %v", err)
	}
	if !json.Valid(raw) {
		t.Fatalf("export not valid JSON: %s", raw)
	}
	var back map[string]any
	if err := json.Unmarshal(raw, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	want := map[string]any{
		"records": []any{
			map[string]any{"tps": nil, "n": 3.0},
			map[string]any{"tps": 41.5, "n": 4.0},
		},
		"summary": map[string]any{"cv": nil},
	}
	if !reflect.DeepEqual(back, want) {
		t.Fatalf("round trip got %#v want %#v", back, want)
	}
	wantIndented := "{\n  \"records\": [\n    {\n      \"n\": 3,\n      \"tps\": null\n    },\n    {\n      \"n\": 4,\n      \"tps\": 41.5\n    }\n  ],\n  \"summary\": {\n    \"cv\": null\n  }\n}"
	if string(raw) != wantIndented {
		t.Fatalf("indented layout got\n%s\nwant\n%s", raw, wantIndented)
	}
}

type jscrubTestPtrMarshaler struct{ X float64 }

func (p *jscrubTestPtrMarshaler) MarshalJSON() ([]byte, error) { return []byte(`"custom"`), nil }

type jscrubTestSeqMarshaler []float64

func (s jscrubTestSeqMarshaler) MarshalJSON() ([]byte, error) { return []byte(`"seq"`), nil }

func TestJSONScrubPreservesCustomMarshalers(t *testing.T) {
	t.Parallel()
	in := map[string]any{"p": &jscrubTestPtrMarshaler{X: 1}, "s": jscrubTestSeqMarshaler{1}, "f": math.NaN()}
	got, err := json.Marshal(ScrubNonFiniteJSON(in))
	if err != nil {
		t.Fatalf("marshal scrubbed: %v", err)
	}
	if want := `{"f":null,"p":"custom","s":"seq"}`; string(got) != want {
		t.Fatalf("got %s want %s", got, want)
	}
}
