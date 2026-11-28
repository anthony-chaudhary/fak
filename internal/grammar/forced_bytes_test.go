package grammar

import (
	"slices"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestJSONSchemaMaskForcedBytesMultiPropertyTransition(t *testing.T) {
	const eos = 0
	dec := jsonSchemaTokenDecoder{
		nil,                 // 0: EOS
		[]byte(`{"first":`), // 1
		[]byte(`"val"`),     // 2
		[]byte(`,"`),        // 3
		[]byte(`nextkey`),   // 4
		[]byte(`":`),        // 5
		[]byte(`"`),         // 6
		[]byte(`,`),         // 7
		[]byte(`n`),         // 8
		[]byte(`:`),         // 9
		[]byte(`}`),         // 10
	}
	schema := []byte(`{
		"type": "object",
		"properties": {
			"first": {"type": "string"},
			"nextkey": {"type": "string"}
		},
		"required": ["first", "nextkey"],
		"additionalProperties": false
	}`)
	mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}

	history := []int{1, 2} // `{"first":"val"` — closing quote of first string value
	got := mask.ForcedBytes(history, 0)
	want := `,"nextkey":"`
	if string(got) != want {
		t.Fatalf("ForcedBytes = %q, want %q", got, want)
	}

	capped := mask.ForcedBytes(history, 4)
	if string(capped) != `,"ne` {
		t.Fatalf("ForcedBytes(maxBytes=4) = %q, want %q", capped, `,"ne`)
	}

	gotTokens := mask.ForcedTokens(history, len(dec), 8)
	wantTokens := []int{3, 4, 5, 6} // ," + nextkey + ": + "
	if !slices.Equal(gotTokens, wantTokens) {
		t.Fatalf("ForcedTokens = %v, want %v", gotTokens, wantTokens)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestJSONSchemaMaskForcedBytesSingleVariantEnum(t *testing.T) {
	const eos = 0
	dec := jsonSchemaTokenDecoder{
		nil,                  // 0: EOS
		[]byte(`{"status":`), // 1
		[]byte(`"ok"`),       // 2
		[]byte(`}`),          // 3
		[]byte(`"`),          // 4
		[]byte(`o`),          // 5
		[]byte(`k`),          // 6
	}
	schema := []byte(`{
		"type": "object",
		"properties": {
			"status": {"type": "string", "enum": ["ok"]}
		},
		"required": ["status"],
		"additionalProperties": false
	}`)
	mask, err := CompileJSONSchema(schema, dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}

	got := mask.ForcedBytes(nil, 64)
	want := `{"status":"ok"}`
	if string(got) != want {
		t.Fatalf("ForcedBytes(nil) = %q, want %q", got, want)
	}

	gotTokens := mask.ForcedTokens(nil, len(dec), 16)
	wantTokens := []int{1, 2, 3}
	if !slices.Equal(gotTokens, wantTokens) {
		t.Fatalf("ForcedTokens(nil) = %v, want %v", gotTokens, wantTokens)
	}

	if after := mask.ForcedBytes(wantTokens, 64); len(after) != 0 {
		t.Fatalf("ForcedBytes after complete object = %q, want empty", after)
	}
}
