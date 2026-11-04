package agent

import (
	"encoding/json"
	"strconv"
	"testing"
)

// TestToolDefStrictPreservedAndAutoWantsGrammar pins the adapted TensorFold
// strict-tool grammar (zig/src/server/tool_grammar.zig
// @9e8c368b0b4c2c7892a0c9f0f302d7c164016209): a bool function.strict survives
// wire decode and tool-schema normalization, a string "true" does not mark, and
// AutoWantsGrammar decides auto-grammar wants from the marks alone.
// fak-test:runtime fast est=10ms
func TestToolDefStrictPreservedAndAutoWantsGrammar(t *testing.T) {
	const schema = `{"type":"object"}`

	decode := func(name string, raw string) ToolDef {
		t.Helper()
		var td ToolDef
		if err := json.Unmarshal([]byte(raw), &td); err != nil {
			t.Fatalf("%s: decode rejected %q: %v", name, raw, err)
		}
		return td
	}

	assertStrict := func(name string, f ToolDefFunction, want *bool) {
		t.Helper()
		got := "<absent>"
		if f.Strict != nil {
			got = strconv.FormatBool(*f.Strict)
		}
		wantStr := "<absent>"
		if want != nil {
			wantStr = strconv.FormatBool(*want)
		}
		if got != wantStr {
			t.Errorf("%s: strict = %s, want %s", name, got, wantStr)
		}
	}

	boolTrue := true
	boolFalse := false

	// 1. Decode: only a JSON boolean captures the mark; every other shape is
	//    accepted without marking (upstream: s == .bool and s.bool).
	cases := []struct {
		name string
		raw  string
		want *bool
	}{
		{"strict-true", `{"name":"lookup","description":"find","parameters":` + schema + `,"strict":true}`, &boolTrue},
		{"strict-false", `{"name":"lookup","description":"find","parameters":` + schema + `,"strict":false}`, &boolFalse},
		{"strict-string-true", `{"name":"lookup","description":"find","parameters":` + schema + `,"strict":"true"}`, nil},
		{"strict-null", `{"name":"lookup","description":"find","parameters":` + schema + `,"strict":null}`, nil},
		{"strict-absent", `{"name":"lookup","description":"find","parameters":` + schema + `}`, nil},
	}
	for _, c := range cases {
		td := decode(c.name, `{"type":"function","function":`+c.raw+`}`)
		if td.Function.Name != "lookup" || td.Function.Description != "find" {
			t.Errorf("%s: name/description = %q/%q, want lookup/find", c.name, td.Function.Name, td.Function.Description)
		}
		assertStrict(c.name+" decode", td.Function, c.want)
	}

	// 2. Normalization keeps the mark AND the parameter schema byte-exact.
	marked := decode("norm-marked", `{"type":"function","function":{"name":"lookup","description":"find","parameters":`+schema+`,"strict":true}}`)
	unmarked := decode("norm-unmarked", `{"type":"function","function":{"name":"grep","description":"search","parameters":`+schema+`}}`)
	norm := openAICompatibleTools([]ToolDef{marked, unmarked})
	if len(norm) != 2 {
		t.Fatalf("normalization returned %d tools, want 2", len(norm))
	}
	assertStrict("norm-marked", norm[0].Function, &boolTrue)
	assertStrict("norm-unmarked", norm[1].Function, nil)
	if string(norm[0].Function.Parameters) != schema {
		t.Errorf("normalized parameters = %s, want %s", norm[0].Function.Parameters, schema)
	}

	// 3. Wire re-marshal: an explicit mark rides back out; an absent one does not.
	b, err := json.Marshal(norm[0].Function)
	if err != nil {
		t.Fatalf("marshal marked tool: %v", err)
	}
	wantMarked := `{"name":"lookup","description":"find","parameters":{"type":"object"},"strict":true}`
	if string(b) != wantMarked {
		t.Errorf("marked wire bytes = %s, want %s", b, wantMarked)
	}
	b, err = json.Marshal(norm[1].Function)
	if err != nil {
		t.Fatalf("marshal unmarked tool: %v", err)
	}
	wantUnmarked := `{"name":"grep","description":"search","parameters":{"type":"object"}}`
	if string(b) != wantUnmarked {
		t.Errorf("unmarked wire bytes = %s, want %s", b, wantUnmarked)
	}

	// 4. AutoWantsGrammar: strict:true wants; strict:false / string "true" do
	//    not; mixed catalogs are decided by their marked tools; empty wants none.
	stringTrue := decode("auto-string", `{"type":"function","function":{"name":"soft","description":"s","parameters":`+schema+`,"strict":"true"}}`)
	explicitFalse := decode("auto-false", `{"type":"function","function":{"name":"plain","description":"p","parameters":`+schema+`,"strict":false}}`)
	autoCases := []struct {
		name  string
		tools []ToolDef
		want  bool
	}{
		{"empty", nil, false},
		{"unmarked-only", []ToolDef{unmarked}, false},
		{"explicit-false", []ToolDef{explicitFalse}, false},
		{"string-true", []ToolDef{stringTrue}, false},
		{"single-true", []ToolDef{marked}, true},
		{"mixed-marked-wins", []ToolDef{unmarked, explicitFalse, marked, stringTrue}, true},
		{"true-first", []ToolDef{marked, unmarked}, true},
	}
	for _, c := range autoCases {
		if got := AutoWantsGrammar(c.tools); got != c.want {
			t.Errorf("AutoWantsGrammar(%s) = %t, want %t", c.name, got, c.want)
		}
	}
}
