package grammar

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

var _ model.LogitMask = (*JSONSchemaMask)(nil)

type jsonSchemaTokenDecoder [][]byte

func (d jsonSchemaTokenDecoder) TokenBytes(id int) []byte {
	if id < 0 || id >= len(d) {
		return nil
	}
	return d[id]
}

func jsonSchemaSurvivors(logits []float32) map[int]bool {
	out := map[int]bool{}
	for id, value := range logits {
		if !math.IsInf(float64(value), -1) {
			out[id] = true
		}
	}
	return out
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskChecksWholeTokenAndAdmitsEOSAfterObject(t *testing.T) {
	const eos = 1
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"a":"x"}`), // complete object
		nil,                 // EOS
		[]byte(`{"a":"x"}tail`),
		[]byte(`{"a":"`),
		[]byte(`x"}`),
		nil, // zero-byte ordinary token
		[]byte(`{"a":"x`),
		[]byte(`garbage`),
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"additionalProperties":false}`), dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}

	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	got := jsonSchemaSurvivors(logits)
	if !got[0] || !got[3] || !got[6] || got[eos] || got[2] || got[5] || got[7] {
		t.Fatalf("initial survivors = %v; want complete/partial object tokens only", got)
	}
	dec[0][0] = 'x' // cached token bytes must not alias mutable decoder storage

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{3}, logits)
	got = jsonSchemaSurvivors(logits)
	if !got[4] || got[eos] {
		t.Fatalf("split completion survivors = %v; want token 4 and no EOS", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{0}, logits)
	got = jsonSchemaSurvivors(logits)
	if len(got) != 1 || !got[eos] {
		t.Fatalf("complete object survivors = %v; want EOS only", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskDeadEndsFailClosed(t *testing.T) {
	dec := jsonSchemaTokenDecoder{[]byte(`garbage`), []byte(`{"a":"x"}`), nil}
	mask, err := CompileJSONSchema([]byte(`{"properties":{"a":{"type":"string"}},"required":["a"]}`), dec, CompileOptions{EOS: 2})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}
	for _, history := range [][]int{{0}, {2}} {
		logits := make([]float32, len(dec))
		mask.MaskLogits(history, logits)
		if got := jsonSchemaSurvivors(logits); len(got) != 0 {
			t.Fatalf("history %v survivors = %v, want fail-closed empty set", history, got)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskOptionalScalarBranches(t *testing.T) {
	const eos = 1
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"a":"x"}`),
		nil,
		[]byte(`{"a":"x","b":true}`),
		[]byte(`{"a":"x"}tail`),
		[]byte(`{"b":true}`),
		[]byte(`{"a":"x"`),
		[]byte(`}`),
		[]byte(`,"b":true}`),
		[]byte(`,"b":false}`),
		[]byte(`{"b":true,"a":"x"}`),
		[]byte(`{"a":"x","b":true}tail`),
		[]byte(`{"a":"x","b":true,"b":false}`),
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"boolean"}},"required":["a"],"additionalProperties":false}`), dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema optional scalar: %v", err)
	}

	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); !got[0] || !got[2] || !got[5] || got[eos] || got[3] || got[4] || got[9] || got[10] || got[11] {
		t.Fatalf("initial survivors = %v; want required field with omitted/present optional branches", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{5}, logits)
	if got := jsonSchemaSurvivors(logits); !got[6] || !got[7] || !got[8] || got[eos] {
		t.Fatalf("branch-point survivors = %v; want close or optional boolean", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{2}, logits)
	if got := jsonSchemaSurvivors(logits); len(got) != 1 || !got[eos] {
		t.Fatalf("complete optional object survivors = %v; want EOS only", got)
	}

	emptyMask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"a":{"type":"string"}},"required":[],"additionalProperties":false}`), jsonSchemaTokenDecoder{[]byte(`{}`), []byte(`{"a":"x"}`)}, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatalf("CompileJSONSchema optional-only object: %v", err)
	}
	logits = make([]float32, 2)
	emptyMask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); !got[0] || !got[1] {
		t.Fatalf("optional-only survivors = %v; want empty and populated objects", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskIntegerEnumBranches(t *testing.T) {
	const eos = 1
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"count":1}`),
		nil,
		[]byte(`{"count":10}`),
		[]byte(`{"count":100}`),
		[]byte(`{"count":2}`),
		[]byte(`{"count":`),
		[]byte(`1`),
		[]byte(`0`),
		[]byte(`}`),
		[]byte(`00}`),
		[]byte(`01}`),
		[]byte(`1}`),
		[]byte(`10}`),
		[]byte(`100}`),
		[]byte(`100}tail`),
		nil,
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"count":{"type":"integer","enum":[1,10,100]}},"required":["count"],"additionalProperties":false}`), dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema integer enum: %v", err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); !got[0] || !got[2] || !got[3] || !got[5] || got[eos] || got[4] || got[14] || got[15] {
		t.Fatalf("initial survivors = %v, want all integer enum objects and no EOS", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{5}, logits)
	if got := jsonSchemaSurvivors(logits); !got[6] || !got[11] || !got[12] || !got[13] || got[eos] || got[4] || got[9] || got[10] || got[14] {
		t.Fatalf("value-start survivors = %v, want exact enum prefixes/whole values", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{5, 6}, logits)
	if got := jsonSchemaSurvivors(logits); !got[7] || !got[8] || !got[9] || got[eos] || got[10] || got[11] {
		t.Fatalf("prefix 1 survivors = %v, want delimiter and continuation through 10/100", got)
	}

	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{0}, logits)
	if got := jsonSchemaSurvivors(logits); len(got) != 1 || !got[eos] {
		t.Fatalf("complete enum object survivors = %v, want EOS only", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskIntegerEnumOptionalComposition(t *testing.T) {
	const eos = 2
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"color":"red"}`),
		[]byte(`{"color":"red","count":10}`),
		nil,
		[]byte(`{"color":"red","count":11}`),
		[]byte(`{"count":10,"color":"red"}`),
		[]byte(`{"color":"blue"}`),
		[]byte(`{"color":"red","count":10}tail`),
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"color":{"type":"string","const":"red"},"count":{"type":"integer","enum":[1,10,100]}},"required":["color"],"additionalProperties":false}`), dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema optional integer enum: %v", err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); !got[0] || !got[1] || got[eos] || got[3] || got[4] || got[5] || got[6] {
		t.Fatalf("optional enum survivors = %v, want canonical omission/inclusion only", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskIntegerEnumSignedBoundaries(t *testing.T) {
	const eos = 1
	dec := jsonSchemaTokenDecoder{
		[]byte(`{"n":-9223372036854775808}`),
		nil,
		[]byte(`{"n":9223372036854775807}`),
		[]byte(`{"n":0}`),
		[]byte(`{"n":-0}`),
		[]byte(`{"n":-922337203685`),
		[]byte(`4775808}`),
		[]byte(`4775809}`),
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"n":{"type":"integer","enum":[-9223372036854775808,-0,9223372036854775807]}},"required":["n"],"additionalProperties":false}`), dec, CompileOptions{EOS: eos})
	if err != nil {
		t.Fatalf("CompileJSONSchema signed boundaries: %v", err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); !got[0] || !got[2] || !got[3] || !got[5] || got[eos] || got[4] {
		t.Fatalf("signed boundary survivors = %v", got)
	}
	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{5}, logits)
	if got := jsonSchemaSurvivors(logits); !got[6] || got[7] || got[eos] {
		t.Fatalf("split minimum suffix survivors = %v", got)
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCompileJSONSchemaIntegerEnumBoundsAndRefusals(t *testing.T) {
	compile := func(t *testing.T, property string) *JSONSchemaMask {
		t.Helper()
		schema := []byte(`{"type":"object","properties":{"n":` + property + `},"required":["n"],"additionalProperties":false}`)
		mask, err := CompileJSONSchema(schema, jsonSchemaTokenDecoder{[]byte(`{"n":0}`)}, CompileOptions{EOS: -1})
		if err != nil {
			t.Fatalf("CompileJSONSchema %s: %v", property, err)
		}
		return mask
	}

	var thirtyTwo strings.Builder
	thirtyTwo.WriteString(`{"type":"integer","enum":[`)
	for i := range 32 {
		if i > 0 {
			thirtyTwo.WriteByte(',')
		}
		fmt.Fprint(&thirtyTwo, i)
	}
	thirtyTwo.WriteString(`]}`)
	mask := compile(t, thirtyTwo.String())
	if got := len(mask.objects[0].fields[0].integerEnum); got != 32 {
		t.Fatalf("32-member enum normalized to %d members", got)
	}

	duplicates := compile(t, `{"type":"integer","enum":[1,1,10,-0,0]}`)
	wantCanonical := []string{"0", "1", "10"}
	gotCanonical := make([]string, len(duplicates.objects[0].fields[0].integerEnum))
	for i, literal := range duplicates.objects[0].fields[0].integerEnum {
		gotCanonical[i] = string(literal)
	}
	if !slices.Equal(gotCanonical, wantCanonical) {
		t.Fatalf("canonical integer enum = %v, want %v", gotCanonical, wantCanonical)
	}

	for _, property := range []string{
		`{"type":"integer","enum":[-9223372036854775808,9223372036854775807]}`,
		`{"type":"integer","const":-9223372036854775808}`,
		`{"type":"integer","const":9223372036854775807}`,
	} {
		compile(t, property)
	}

	thirtyThreeDuplicates := `{"type":"integer","enum":[` + strings.TrimSuffix(strings.Repeat(`1,`, 33), ",") + `]}`
	refusals := []string{
		thirtyThreeDuplicates,
		`{"type":"integer","enum":[]}`,
		`{"type":"integer","enum":null}`,
		`{"type":"integer","enum":{}}`,
		`{"type":"integer","enum":[1.0]}`,
		`{"type":"integer","enum":[1e0]}`,
		`{"type":"integer","enum":[9223372036854775808]}`,
		`{"type":"integer","enum":[-9223372036854775809]}`,
		`{"type":"integer","enum":[1,"2"]}`,
		`{"type":"integer","enum":[1,null]}`,
		`{"type":"integer","enum":[true]}`,
		`{"type":"integer","const":1.0}`,
		`{"type":"integer","const":1e0}`,
		`{"type":"integer","const":null}`,
		`{"type":"integer","enum":[1],"const":1}`,
	}
	for _, property := range refusals {
		t.Run(property, func(t *testing.T) {
			schema := []byte(`{"type":"object","properties":{"n":` + property + `},"required":["n"],"additionalProperties":false}`)
			_, err := CompileJSONSchema(schema, jsonSchemaTokenDecoder{[]byte(`{"n":0}`)}, CompileOptions{EOS: -1})
			if !errors.Is(err, ErrUnsupportedSchema) {
				t.Fatalf("err = %v, want ErrUnsupportedSchema", err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCompileJSONSchemaRejectsUnsupportedSchemaWithoutWeakening(t *testing.T) {
	cases := []string{
		`{"properties":{"nested":{"type":"object"}},"required":["nested"]}`,
		`{"properties":{"required":{"type":"string"},"hidden":{"type":"object"}},"required":["required"]}`,
		`{"properties":{"a":{"type":"string","pattern":"^x$"}},"required":["a"]}`,
		`{"properties":{"a":{"type":"array"}},"required":["a"]}`,
	}
	for _, schema := range cases {
		if _, err := CompileJSONSchema([]byte(schema), jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1}); !errors.Is(err, ErrUnsupportedSchema) {
			t.Fatalf("schema %s: err = %v, want ErrUnsupportedSchema", schema, err)
		}
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestCompileJSONSchemaOptionalBoundsAndPreservedRefusals(t *testing.T) {
	schemaWithProperties := func(total, required int) []byte {
		var schema strings.Builder
		schema.WriteString(`{"type":"object","properties":{`)
		for i := 0; i < total; i++ {
			if i > 0 {
				schema.WriteByte(',')
			}
			fmt.Fprintf(&schema, `"p%02d":{"type":"string"}`, i)
		}
		schema.WriteString(`},"required":[`)
		for i := 0; i < required; i++ {
			if i > 0 {
				schema.WriteByte(',')
			}
			fmt.Fprintf(&schema, `"p%02d"`, i)
		}
		schema.WriteString(`],"additionalProperties":false}`)
		return []byte(schema.String())
	}

	mask, err := CompileJSONSchema(schemaWithProperties(4, 0), jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatalf("four optional properties: %v", err)
	}
	if got := len(mask.objects); got != 16 {
		t.Fatalf("four optional properties compiled %d shapes, want 16", got)
	}
	if _, err := CompileJSONSchema(schemaWithProperties(16, 16), jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1}); err != nil {
		t.Fatalf("16 total properties: %v", err)
	}
	maxCombined, err := CompileJSONSchema(schemaWithProperties(16, 12), jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatalf("16 total and four optional properties: %v", err)
	}
	if got := len(maxCombined.objects); got != 16 {
		t.Fatalf("combined maximum compiled %d shapes, want 16", got)
	}
	duplicateRequiredSchema := strings.Replace(string(schemaWithProperties(5, 1)), `"required":["p00"]`, `"required":["p00","p00"]`, 1)
	duplicateRequired, err := CompileJSONSchema([]byte(duplicateRequiredSchema), jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatalf("duplicate required membership with four unique optionals: %v", err)
	}
	if got := len(duplicateRequired.objects); got != 16 {
		t.Fatalf("duplicate required membership compiled %d shapes, want 16", got)
	}
	base := []byte(`{"type":"object","properties":{},"required":[]}`)
	atByteLimit := append(append([]byte(nil), base...), []byte(strings.Repeat(" ", maxJSONSchemaBytes-len(base)))...)
	if _, err := CompileJSONSchema(atByteLimit, jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1}); err != nil {
		t.Fatalf("schema at byte limit: %v", err)
	}

	cases := map[string][]byte{
		"schema byte limit":         append(atByteLimit, ' '),
		"property count limit":      schemaWithProperties(17, 17),
		"optional count limit":      schemaWithProperties(5, 0),
		"null schema":               []byte(`null`),
		"array schema":              []byte(`[]`),
		"null properties":           []byte(`{"type":"object","properties":null,"required":[]}`),
		"array properties":          []byte(`{"type":"object","properties":[],"required":[]}`),
		"null required":             []byte(`{"type":"object","properties":{},"required":null}`),
		"object required":           []byte(`{"type":"object","properties":{},"required":{}}`),
		"null additionalProperties": []byte(`{"type":"object","properties":{},"required":[],"additionalProperties":null}`),
	}
	for name, schema := range cases {
		t.Run(name, func(t *testing.T) {
			_, err := CompileJSONSchema(schema, jsonSchemaTokenDecoder{[]byte(`{}`)}, CompileOptions{EOS: -1})
			if !errors.Is(err, ErrUnsupportedSchema) {
				t.Fatalf("err = %v, want ErrUnsupportedSchema", err)
			}
		})
	}
}

// fak-test:runtime fast est=10ms lane=default
func TestJSONSchemaMaskEmptyObjectAndMissingEOSFailClosed(t *testing.T) {
	dec := jsonSchemaTokenDecoder{[]byte(`{}`), nil}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{},"required":[]}`), dec, CompileOptions{EOS: -1})
	if err != nil {
		t.Fatalf("CompileJSONSchema: %v", err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if got := jsonSchemaSurvivors(logits); len(got) != 1 || !got[0] {
		t.Fatalf("empty-object initial survivors = %v, want token 0", got)
	}
	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{0}, logits)
	if got := jsonSchemaSurvivors(logits); len(got) != 0 {
		t.Fatalf("complete object without EOS survivors = %v, want fail-closed empty set", got)
	}
}
