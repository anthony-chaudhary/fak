package grammar

import (
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
)

// arrayNextBytes reports which single bytes the mask admits after prefix, and
// whether EOS (id 0) is admitted. It drives the byte-level vocabulary used by
// the cursor differential test.
func arrayNextBytes(t *testing.T, mask *JSONSchemaMask, dec jsonSchemaTokenDecoder, prefix string) (string, bool) {
	t.Helper()
	history := make([]int, len(prefix))
	for i := range prefix {
		history[i] = cursorSingleByteBase + int(prefix[i])
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(history, logits)
	var allowed []byte
	for b := 0; b < 256; b++ {
		if !math.IsInf(float64(logits[cursorSingleByteBase+b]), -1) {
			allowed = append(allowed, byte(b))
		}
	}
	return string(allowed), !math.IsInf(float64(logits[0]), -1)
}

func compileArraySchema(t *testing.T, dec jsonSchemaTokenDecoder, property string, required bool) *JSONSchemaMask {
	t.Helper()
	req := `[]`
	if required {
		req = `["v"]`
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"v":`+property+`},"required":`+req+`,"additionalProperties":false}`), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatalf("CompileJSONSchema %s: %v", property, err)
	}
	return mask
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaArrayBoundsAndDelimiters(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(21)))
	mask := compileArraySchema(t, dec, `{"type":"array","items":{"type":"string","enum":["a","b"]},"minItems":1,"maxItems":2}`, true)
	for _, tc := range []struct {
		prefix string
		want   string
		eos    bool
	}{
		{`{"v":`, `[`, false},
		{`{"v":[`, `"`, false},
		{`{"v":["`, `ab`, false},
		{`{"v":["a"`, `,]`, false},
		{`{"v":["a",`, `"`, false},
		{`{"v":["a","b"`, `]`, false},
		{`{"v":["a","b"]`, `}`, false},
		{`{"v":["a"]}`, ``, true},
	} {
		got, eos := arrayNextBytes(t, mask, dec, tc.prefix)
		if got != tc.want || eos != tc.eos {
			t.Errorf("after %s: bytes %q eos=%v, want %q eos=%v", tc.prefix, got, eos, tc.want, tc.eos)
		}
	}
	for _, dead := range []string{`{"v":[]`, `{"v":["a","b",`, `{"v":["c"`, `{"v":["a"]]`} {
		if got, eos := arrayNextBytes(t, mask, dec, dead); got != "" || eos {
			t.Errorf("dead prefix %s admitted %q eos=%v", dead, got, eos)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaArrayScalarItemKinds(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(22)))
	for _, tc := range []struct {
		items string
		valid []string
		dead  []string
	}{
		{`{"type":"integer"}`, []string{`[]`, `[1,23,-4]`, `[0]`}, []string{`[01`, `[1.`, `[-` + `]`, `[,`}},
		{`{"type":"number"}`, []string{`[0.5,1]`, `[-0.25]`}, []string{`[1.]`, `[.5`}},
		{`{"type":"boolean"}`, []string{`[true,false]`}, []string{`[tru]`, `[null`}},
		{`{"type":"integer","enum":[1,10,100]}`, []string{`[1,10,100,1]`}, []string{`[2`, `[1000`}},
		{`{"type":"string"}`, []string{`[""]`, `["a b","c"]`}, []string{`["\`, `[a`}},
	} {
		mask := compileArraySchema(t, dec, `{"type":"array","items":`+tc.items+`}`, true)
		for _, value := range tc.valid {
			doc := `{"v":` + value + `}`
			for i := range doc {
				got, _ := arrayNextBytes(t, mask, dec, doc[:i])
				if !strings.ContainsRune(got, rune(doc[i])) {
					t.Fatalf("items %s: %q rejected byte %q after %q (allowed %q)", tc.items, doc, doc[i], doc[:i], got)
				}
			}
			if _, eos := arrayNextBytes(t, mask, dec, doc); !eos {
				t.Fatalf("items %s: complete %q does not admit EOS", tc.items, doc)
			}
		}
		for _, value := range tc.dead {
			prefix := `{"v":` + value
			last := prefix[len(prefix)-1]
			if got, _ := arrayNextBytes(t, mask, dec, prefix[:len(prefix)-1]); strings.ContainsRune(got, rune(last)) {
				t.Fatalf("items %s: admitted %q after %q", tc.items, last, prefix[:len(prefix)-1])
			}
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaArrayWholeTokensAndOptionalShapes(t *testing.T) {
	dec := jsonSchemaTokenDecoder{
		nil,
		[]byte(`{"v":[`),
		[]byte(`"a","b"]}`),
		[]byte(`"a"]]`),
		[]byte(`"a"],"w":1}`),
		[]byte(`{}`),
		[]byte(`"a"]}`),
	}
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"v":{"type":"array","items":{"type":"string","enum":["a","b"]},"maxItems":2},"w":{"type":"integer"}},"additionalProperties":false}`), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	logits := make([]float32, len(dec))
	mask.MaskLogits(nil, logits)
	if math.IsInf(float64(logits[1]), -1) || math.IsInf(float64(logits[5]), -1) || !math.IsInf(float64(logits[0]), -1) {
		t.Fatalf("initial survivors %v: want the array opener and the empty object", logits)
	}
	logits = make([]float32, len(dec))
	mask.MaskLogits([]int{1}, logits)
	for id, want := range []bool{false, false, true, false, true, false, true} {
		if got := !math.IsInf(float64(logits[id]), -1); got != want {
			t.Fatalf("after array opener token %d %q admitted=%v, want %v", id, dec[id], got, want)
		}
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestJSONSchemaArrayStateStaysFinite(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(23)))
	mask := compileArraySchema(t, dec, `{"type":"array","items":{"type":"string"},"minItems":2}`, true)
	prefix := `{"v":[`
	for i := 0; i < 20; i++ {
		prefix += `"x",`
	}
	prefix += `"x"`
	got, _ := arrayNextBytes(t, mask, dec, prefix)
	before := len(mask.cache)
	again, _ := arrayNextBytes(t, mask, dec, prefix+`,"y"`)
	if got != `,]` || again != got || len(mask.cache) != before {
		t.Fatalf("unbounded array state grew: %q -> %q, cache %d -> %d", got, again, before, len(mask.cache))
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestCompileJSONSchemaArrayRefusals(t *testing.T) {
	dec := jsonSchemaTokenDecoder{[]byte(`{}`)}
	for _, property := range []string{
		`{"type":"array"}`,
		`{"type":"array","items":[{"type":"string"}]}`,
		`{"type":"array","items":{"type":"array","items":{"type":"string"}}}`,
		`{"type":"array","items":{"type":"object","properties":{"a":{"type":"string"}}}}`,
		`{"type":"array","items":{"type":"string"},"uniqueItems":true}`,
		`{"type":"array","items":{"type":"string"},"minItems":3,"maxItems":2}`,
		`{"type":"array","items":{"type":"string"},"maxItems":65}`,
		`{"type":"array","items":{"type":"string"},"minItems":-1}`,
		`{"type":"array","items":{"type":"string"},"minItems":1.5}`,
		`{"type":"array","items":{"type":"string","pattern":"a"}}`,
	} {
		_, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"v":`+property+`},"required":["v"],"additionalProperties":false}`), dec, CompileOptions{EOS: -1})
		if !errors.Is(err, ErrUnsupportedSchema) {
			t.Errorf("%s: err = %v, want ErrUnsupportedSchema", property, err)
		}
	}
	_, err := Compile([]ToolSpec{{Name: "f", Schema: []byte(`{"type":"object","properties":{"v":{"type":"array","items":{"type":"string"}}},"required":["v"]}`)}}, dec, CompileOptions{EOS: -1})
	if !errors.Is(err, ErrUnsupportedSchema) {
		t.Errorf("tool array schema: err = %v, want ErrUnsupportedSchema", err)
	}
	if maxResponseArrayItems != 64 {
		t.Fatalf("array bound %d is not the documented 64", maxResponseArrayItems)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestJSONSchemaArrayMaxItemsZeroAdmitsOnlyEmpty(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(24)))
	for _, items := range []string{`{"type":"integer"}`, `{"type":"boolean"}`, `{"type":"string"}`} {
		mask := compileArraySchema(t, dec, `{"type":"array","items":`+items+`,"maxItems":0}`, true)
		if got, _ := arrayNextBytes(t, mask, dec, `{"v":[`); got != `]` {
			t.Errorf("items %s maxItems 0: after [ admitted %q, want only ]", items, got)
		}
	}
}
