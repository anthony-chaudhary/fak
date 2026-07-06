package toon

import (
	"encoding/json"
	"reflect"
	"strings"
	"testing"
)

// fromJSON decodes a JSON literal into the encoding/json native shape (float64 numbers,
// map[string]any objects) — the exact domain toon guarantees a round-trip over.
func fromJSON(t *testing.T, s string) any {
	t.Helper()
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		t.Fatalf("corpus literal is not valid JSON: %v\n%s", err, s)
	}
	return v
}

// roundTripCorpus is the property-test corpus: every shape the DoD names. Each entry is a
// JSON literal; the test asserts Decode(Encode(v)) deep-equals v for it under every
// Options variant.
var roundTripCorpus = []string{
	// scalars
	`"hello"`,
	`"a plain sentence with spaces"`,
	`42`,
	`3.14159`,
	`-17`,
	`1000000`,
	`true`,
	`false`,
	`null`,
	`""`,
	// numeric- / bool-looking strings MUST stay strings, not coerce
	`"123"`,
	`"007"`,
	`"3.14"`,
	`"true"`,
	`"false"`,
	`"null"`,
	`"-5"`,
	`"1e9"`,
	// unicode
	`"héllo 世界 — καλημέρα"`,
	`{"name":"世界","emoji":"🚀"}`,
	// embedded delimiters / quotes / newlines
	`{"a":"x,y,z","b":"he said \"hi\"","c":"line1\nline2","d":"tab\there","e":"pipe|bar"}`,
	// flat object
	`{"id":"n1","verdict":"fresh","score":0.9}`,
	// nested object
	`{"id":"x","meta":{"a":"1","b":2,"c":{"deep":true}}}`,
	// uniform array of flat objects (root)
	`[{"id":"n1","verdict":"fresh"},{"id":"n2","verdict":"withheld"}]`,
	// uniform array wrapped in an object (the memview Surface shape)
	`{"notes":[{"id":"n1","verdict":"fresh"},{"id":"n2","verdict":"withheld"}]}`,
	// uniform array with mixed scalar cell types + nulls
	`[{"a":1,"b":"x","c":null,"d":true},{"a":2,"b":"y","c":null,"d":false}]`,
	// ragged array (differing key sets) -> list fallback
	`[{"a":"1"},{"a":"1","b":"2"}]`,
	// array of scalars -> list fallback
	`["a","b","c"]`,
	`[1,"2",true,null,3.5]`,
	// array whose elements carry nested objects -> list fallback
	`[{"a":{"nested":1}},{"a":{"nested":2}}]`,
	// empty containers
	`{}`,
	`[]`,
	`{"empty_obj":{},"empty_arr":[],"id":"z"}`,
	// numeric-string cells inside a uniform table (must round-trip as strings)
	`[{"zip":"00501","name":"a"},{"zip":"90210","name":"b"}]`,
}

func TestRoundTrip(t *testing.T) {
	variants := []Options{
		{},                   // default: comma, no marker
		{Delimiter: '|'},     // pipe
		{Delimiter: '\t'},    // tab
		{LengthMarker: true}, // comma + '#'
		{Delimiter: '|', LengthMarker: true},
	}
	for _, lit := range roundTripCorpus {
		v := fromJSON(t, lit)
		for _, o := range variants {
			enc, err := Encode(v, o)
			if err != nil {
				t.Fatalf("Encode(%s, %+v) error: %v", lit, o, err)
			}
			got, err := Decode(enc)
			if err != nil {
				t.Fatalf("Decode error for %s (%+v): %v\n---encoded---\n%s", lit, o, err, enc)
			}
			if !reflect.DeepEqual(got, v) {
				t.Fatalf("round-trip mismatch for %s (%+v)\n want: %#v\n got:  %#v\n---encoded---\n%s", lit, o, v, got, enc)
			}
		}
	}
}

// TestDeterministic proves object-key order never leaks map iteration order: the same
// value encodes byte-identically across repeated calls (Go randomizes map ranging).
func TestDeterministic(t *testing.T) {
	v := fromJSON(t, `{"z":"1","a":"2","m":{"y":"3","b":"4"},"rows":[{"k":"1","j":"2"},{"k":"3","j":"4"}]}`)
	first, err := Encode(v, Options{})
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		again, err := Encode(v, Options{})
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(first, again) {
			t.Fatalf("non-deterministic encode on iteration %d:\n%s\n---vs---\n%s", i, first, again)
		}
	}
	// keys must be sorted: 'a' before 'm' before 'rows' before 'z'.
	s := string(first)
	if !(strings.Index(s, "\na:") == -1 && strings.HasPrefix(s, "a:")) {
		// 'a' is the first top-level key line
		if !strings.HasPrefix(s, "a: ") {
			t.Fatalf("top-level keys not sorted (want 'a' first):\n%s", s)
		}
	}
}

func TestTabularEligibilityAnchors(t *testing.T) {
	cases := []struct {
		lit  string
		want float64
	}{
		{`[{"a":1,"b":2},{"a":3,"b":4}]`, 1.0}, // uniform flat array -> fully tabular
		{`{"a":{"b":{"c":{"d":1}}}}`, 0.0},     // deeply nested object -> not tabular
		{`{"a":1,"b":2}`, 0.0},                 // flat object (not an array)
		{`"scalar"`, 0.0},                      // bare scalar
		{`["a","b","c"]`, 0.0},                 // scalar array
		{`[{"a":{"n":1}},{"a":{"n":2}}]`, 0.0}, // nested-in-cell -> demoted
	}
	for _, c := range cases {
		got := TabularEligibility(fromJSON(t, c.lit))
		if got != c.want {
			t.Errorf("TabularEligibility(%s) = %v, want %v", c.lit, got, c.want)
		}
	}
	// A mix of one uniform array (4 leaves) and one nested scalar (1 leaf) lands strictly
	// between the anchors.
	mix := TabularEligibility(fromJSON(t, `{"rows":[{"a":1,"b":2},{"a":3,"b":4}],"meta":{"deep":{"x":1}}}`))
	if !(mix > 0 && mix < 1) {
		t.Errorf("mixed eligibility = %v, want strictly between 0 and 1", mix)
	}
}

// TestTabularBeatsJSON is the token-delta witness: on the shape TOON targets (a uniform
// array of flat objects), the tabular encoding is materially smaller than the JSON one it
// replaces, because field names are declared once instead of per row.
func TestTabularBeatsJSON(t *testing.T) {
	v := fromJSON(t, `[
		{"id":"n1","verdict":"fresh","kind":"snippet","score":0.91},
		{"id":"n2","verdict":"withheld","kind":"summary","score":0.42},
		{"id":"n3","verdict":"fresh","kind":"fact","score":0.77},
		{"id":"n4","verdict":"stale","kind":"qa","score":0.13}
	]`)
	toonBytes, err := Encode(v, Options{})
	if err != nil {
		t.Fatal(err)
	}
	jsonBytes, _ := json.Marshal(v)
	if len(toonBytes) >= len(jsonBytes) {
		t.Fatalf("expected TOON (%d bytes) smaller than JSON (%d bytes)\nTOON:\n%s", len(toonBytes), len(jsonBytes), toonBytes)
	}
	if TabularEligibility(v) != 1.0 {
		t.Fatalf("uniform table should be fully tabular-eligible, got %v", TabularEligibility(v))
	}
}

// TestFallbackIsSafe proves a non-uniform array never emits a tabular field header (which
// would carry a wrong field count) — it degrades to the per-item list form and still
// round-trips.
func TestFallbackIsSafe(t *testing.T) {
	v := fromJSON(t, `[{"a":"1"},{"a":"1","b":"2","c":"3"}]`)
	enc, err := Encode(v, Options{})
	if err != nil {
		t.Fatal(err)
	}
	// The header (first line) must be the list form `[N]:`, never a tabular `[N]{...}:`.
	// The per-item JSON lines below legitimately contain braces; only the header matters.
	header := strings.SplitN(string(enc), "\n", 2)[0]
	if strings.ContainsAny(header, "{}") {
		t.Fatalf("ragged array must not emit a tabular {field} header, got header %q:\n%s", header, enc)
	}
	got, err := Decode(enc)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, v) {
		t.Fatalf("fallback round-trip mismatch\n want %#v\n got  %#v", v, got)
	}
}

func TestEncodeErrors(t *testing.T) {
	if _, err := Encode(map[string]any{"a": "b"}, Options{Delimiter: ';'}); err == nil {
		t.Error("expected error for unsupported delimiter ';'")
	}
	if _, err := Encode(map[string]any{"bad": make(chan int)}, Options{}); err == nil {
		t.Error("expected error for an unsupported (non-JSON) scalar type")
	}
}

func TestDecodeErrors(t *testing.T) {
	if _, err := Decode(nil); err == nil {
		t.Error("expected error for empty input")
	}
	// a header that promises 2 rows but supplies 1
	if _, err := Decode([]byte("rows[2]{a}:\n  1\n")); err == nil {
		t.Error("expected error for a short row count")
	}
	// a row whose cell count disagrees with the header field count
	if _, err := Decode([]byte("rows[1]{a,b}:\n  1\n")); err == nil {
		t.Error("expected error for a header/row field-count mismatch")
	}
}

// TestLengthMarkerShape confirms the '#' marker actually appears when requested and that
// Decode reads it back (round-trip already covered above; this pins the surface syntax).
func TestLengthMarkerShape(t *testing.T) {
	v := fromJSON(t, `[{"a":1,"b":2}]`)
	enc, err := Encode(v, Options{LengthMarker: true})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(enc), "[#1]") {
		t.Fatalf("LengthMarker should emit '[#1]':\n%s", enc)
	}
}
