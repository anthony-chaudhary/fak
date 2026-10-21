package grammar

import (
	"errors"
	"fmt"
	"math/rand"
	"strings"
	"testing"
)

func checkValidDocuments(t *testing.T, mask *JSONSchemaMask, dec jsonSchemaTokenDecoder, docs ...string) {
	t.Helper()
	for _, doc := range docs {
		for i := range doc {
			got, eos := arrayNextBytes(t, mask, dec, doc[:i])
			if eos || !strings.ContainsRune(got, rune(doc[i])) {
				t.Fatalf("%q: byte %q after %q not admitted (allowed %q eos=%v)", doc, doc[i], doc[:i], got, eos)
			}
		}
		if _, eos := arrayNextBytes(t, mask, dec, doc); !eos {
			t.Fatalf("%q: complete document does not admit EOS", doc)
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaNestedObjectFlattensIntoSegments(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(31)))
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"addr":{"type":"object","properties":{"city":{"type":"string"},"zip":{"type":"integer"}},"required":["zip","city"],"additionalProperties":false},"name":{"type":"string","enum":["x"]}},"required":["addr","name"],"additionalProperties":false}`), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	if len(mask.objects) != 1 {
		t.Fatalf("required-only nested schema expanded to %d shapes", len(mask.objects))
	}
	checkValidDocuments(t, mask, dec, `{"addr":{"city":"a b","zip":10},"name":"x"}`, `{"addr":{"city":"","zip":-3},"name":"x"}`)
	for _, dead := range []string{`{"addr":{"zip"`, `{"addr":{}`, `{"addr":{"city":"a"}`, `{"name"`, `{"addr":{"city":"a","zip":1},"name":"x"}}`} {
		if got, eos := arrayNextBytes(t, mask, dec, dead[:len(dead)-1]); strings.ContainsRune(got, rune(dead[len(dead)-1])) && !eos {
			t.Errorf("admitted %q after %q", dead[len(dead)-1], dead[:len(dead)-1])
		}
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestJSONSchemaNestedOptionalShapesDeduplicate(t *testing.T) {
	dec := cursorTestVocab(rand.New(rand.NewSource(32)))
	mask, err := CompileJSONSchema([]byte(`{"type":"object","properties":{"id":{"type":"integer"},"meta":{"type":"object","properties":{"tag":{"type":"string","enum":["t"]},"xs":{"type":"array","items":{"type":"boolean"},"maxItems":2}},"required":["xs"]}},"required":["id"]}`), dec, CompileOptions{EOS: 0})
	if err != nil {
		t.Fatal(err)
	}
	// meta and meta.tag are optional: four selections, but omitting meta makes
	// tag irrelevant, leaving three distinct shapes.
	if len(mask.objects) != 3 {
		t.Fatalf("shapes = %d, want 3", len(mask.objects))
	}
	checkValidDocuments(t, mask, dec, `{"id":1}`, `{"id":1,"meta":{"xs":[]}}`, `{"id":7,"meta":{"tag":"t","xs":[true,false]}}`)
	if got, _ := arrayNextBytes(t, mask, dec, `{"id":1,"meta":{"xs":[true,false`); got != `]` {
		t.Fatalf("nested array maxItems: after two items admitted %q, want ]", got)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestCompileJSONSchemaNestedBudgets(t *testing.T) {
	dec := jsonSchemaTokenDecoder{[]byte(`{}`)}
	compile := func(schema string) error {
		_, err := CompileJSONSchema([]byte(schema), dec, CompileOptions{EOS: -1})
		return err
	}
	inner := make([]string, 0, 15)
	for i := range 15 {
		inner = append(inner, fmt.Sprintf(`"p%02d":{"type":"integer"}`, i))
	}
	innerRequired := `"p00"`
	for i := 1; i < 15; i++ {
		innerRequired += fmt.Sprintf(`,"p%02d"`, i)
	}
	// 1 outer + 15 inner properties is exactly the 16-property budget.
	atLimit := `{"type":"object","properties":{"o":{"type":"object","properties":{` + strings.Join(inner, ",") + `},"required":[` + innerRequired + `]}},"required":["o"]}`
	if err := compile(atLimit); err != nil {
		t.Fatalf("16 properties across the tree: %v", err)
	}
	overLimit := strings.Replace(atLimit, `"properties":{"o"`, `"properties":{"extra":{"type":"integer"},"o"`, 1)
	overLimit = strings.Replace(overLimit, `"required":["o"]`, `"required":["extra","o"]`, 1)
	if err := compile(overLimit); !errors.Is(err, ErrUnsupportedSchema) {
		t.Fatalf("17 properties across the tree: err = %v, want ErrUnsupportedSchema", err)
	}
	// Optional members count across the whole tree.
	optional := `{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"},"c":{"type":"integer"},"o":{"type":"object","properties":{"d":{"type":"integer"},"e":{"type":"integer"}},"required":["d"]}},"required":[]}`
	if err := compile(optional); !errors.Is(err, ErrUnsupportedSchema) {
		t.Fatalf("5 optional properties across the tree: err = %v, want ErrUnsupportedSchema", err)
	}
	depthThree := `{"type":"object","properties":{"a":{"type":"object","properties":{"b":{"type":"object","properties":{"c":{"type":"integer"}},"required":["c"]}},"required":["b"]}},"required":["a"]}`
	if err := compile(depthThree); err != nil {
		t.Fatalf("depth 3: %v", err)
	}
}
