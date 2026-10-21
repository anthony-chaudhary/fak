package grammar

import (
	"errors"
	"math"
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
func TestCompileJSONSchemaRejectsUnsupportedSchemaWithoutWeakening(t *testing.T) {
	cases := []string{
		`{"properties":{"nested":{"type":"object"}},"required":["nested"]}`,
		`{"properties":{"required":{"type":"string"},"optional":{"type":"string"}},"required":["required"]}`,
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
