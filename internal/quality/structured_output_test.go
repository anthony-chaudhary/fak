package quality

import (
	"strings"
	"testing"
)

// soInvoiceSpec is the hermetic schema contract for the structured-output
// tests: four required keys with types, two of them pinned to exact values.
const soInvoiceSpec = `{
  "required": {"invoice_id": "string", "total": "number", "paid": "boolean", "items": "array"},
  "expected": {"invoice_id": "INV-7", "total": 1250.5}
}`

// soFaithfulOutput conforms to the spec and matches both pinned values.
const soFaithfulOutput = `{"invoice_id":"INV-7","total":1250.5,"paid":false,"items":[{"sku":"A1","qty":2}]}`

// soStructuredCase builds a valid case whose Reference.Text carries the
// schema spec the structured-output oracle judges engine output against.
func soStructuredCase(spec string, minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "structured-output-invoice",
		Version:   1,
		Prompt:    "Emit the invoice as a JSON object matching the declared schema.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 64},
		Reference: Trace{Text: spec},
		Oracles:   []string{"structured-output-validity"},
		Rubric:    RubricSpec{MinScore: minScore},
	}
}

// soJudge runs the oracle directly against an engine output text.
func soJudge(c QualityCase, output string) Verdict {
	return soStructuredOutput{}.Judge(Trace{}, Trace{Text: output}, c)
}

// TestSoStructuredOutputRegistered proves the oracle registered under its
// stable name and kind, so cases can reference it by name.
func TestSoStructuredOutputRegistered(t *testing.T) {
	os, err := Lookup([]string{"structured-output-validity"})
	if err != nil {
		t.Fatalf("Lookup(structured-output-validity): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestSoStructuredOutputFaithfulPasses is the happy path: output that parses,
// conforms to every required key/type, and matches both pinned values passes
// at score 1.0.
func TestSoStructuredOutputFaithfulPasses(t *testing.T) {
	v := soJudge(soStructuredCase(soInvoiceSpec, 1), soFaithfulOutput)
	if !v.Pass {
		t.Fatalf("faithful structured output must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
}

// TestSoStructuredOutputMissingKeyFails is a defect witness: dropping the
// required "paid" key fails the oracle, and Detail names that exact key.
func TestSoStructuredOutputMissingKeyFails(t *testing.T) {
	missingPaid := `{"invoice_id":"INV-7","total":1250.5,"items":[]}`
	v := soJudge(soStructuredCase(soInvoiceSpec, 1), missingPaid)
	if v.Pass {
		t.Fatalf("output missing a required key must not pass; got %+v", v)
	}
	if want := 5.0 / 6.0; v.Score != want {
		t.Errorf("score = %v, want %v (5 of 6 checks passed)", v.Score, want)
	}
	if !strings.Contains(v.Detail, `missing required key "paid"`) {
		t.Errorf("Detail must name the missing key; got %q", v.Detail)
	}
}

// TestSoStructuredOutputWrongTypeFails is a defect witness: "total" carried
// as a JSON string instead of a number fails, and Detail names the key with
// its actual vs declared type.
func TestSoStructuredOutputWrongTypeFails(t *testing.T) {
	stringTotal := `{"invoice_id":"INV-7","total":"1250.5","paid":false,"items":[]}`
	v := soJudge(soStructuredCase(soInvoiceSpec, 1), stringTotal)
	if v.Pass {
		t.Fatalf("output with a wrong-typed key must not pass; got %+v", v)
	}
	if !strings.Contains(v.Detail, `key "total" has type string, want number`) {
		t.Errorf("Detail must name the type violation; got %q", v.Detail)
	}
}

// TestSoStructuredOutputWrongValueFails is the semantic defect witness: output
// that is grammar/schema-valid but carries the wrong pinned value fails, and
// Detail names the key with got vs want.
func TestSoStructuredOutputWrongValueFails(t *testing.T) {
	wrongID := `{"invoice_id":"INV-8","total":1250.5,"paid":false,"items":[]}`
	v := soJudge(soStructuredCase(soInvoiceSpec, 1), wrongID)
	if v.Pass {
		t.Fatalf("schema-valid output with a wrong value must not pass; got %+v", v)
	}
	if want := 5.0 / 6.0; v.Score != want {
		t.Errorf("score = %v, want %v (5 of 6 checks passed)", v.Score, want)
	}
	if !strings.Contains(v.Detail, `key "invoice_id" = "INV-8", want "INV-7"`) {
		t.Errorf("Detail must name the value violation; got %q", v.Detail)
	}
}

// TestSoStructuredOutputUnparseableFails is the grammar defect witness:
// truncated JSON fails closed at score 0, and a non-object top level fails
// naming the actual top-level type.
func TestSoStructuredOutputUnparseableFails(t *testing.T) {
	c := soStructuredCase(soInvoiceSpec, 1)
	v := soJudge(c, `{"invoice_id":"INV-7","total":`)
	if v.Pass || v.Score != 0 {
		t.Fatalf("unparseable output must fail closed at score 0; got %+v", v)
	}
	if !strings.Contains(v.Detail, "does not parse as JSON") {
		t.Errorf("Detail must name the parse violation; got %q", v.Detail)
	}
	v = soJudge(c, `[1, 2, 3]`)
	if v.Pass || v.Score != 0 {
		t.Fatalf("non-object top level must fail closed at score 0; got %+v", v)
	}
	if !strings.Contains(v.Detail, "top-level JSON value is array, want object") {
		t.Errorf("Detail must name the top-level type; got %q", v.Detail)
	}
}

// TestSoStructuredOutputIntegerType proves the "integer" type name checks
// integrality of the decoded number, not the Go type: 3 conforms, 3.5 fails.
func TestSoStructuredOutputIntegerType(t *testing.T) {
	spec := `{"required": {"qty": "integer"}}`
	if v := soJudge(soStructuredCase(spec, 1), `{"qty": 3}`); !v.Pass {
		t.Fatalf("integral number must satisfy integer; got %+v", v)
	}
	v := soJudge(soStructuredCase(spec, 1), `{"qty": 3.5}`)
	if v.Pass {
		t.Fatalf("fractional number must not satisfy integer; got %+v", v)
	}
	if !strings.Contains(v.Detail, `key "qty" has type number, want integer`) {
		t.Errorf("Detail must name the integer violation; got %q", v.Detail)
	}
}

// TestSoStructuredOutputUnusableSpecFailsClosed defines the spec edges: an
// empty spec, a zero-check spec, and an unknown type name each fail closed —
// a structured-output case that checks nothing is not green.
func TestSoStructuredOutputUnusableSpecFailsClosed(t *testing.T) {
	for name, spec := range map[string]string{
		"empty":        "",
		"zero-check":   `{}`,
		"unknown-type": `{"required": {"total": "decimal"}}`,
	} {
		v := soJudge(soStructuredCase(spec, 1), soFaithfulOutput)
		if v.Pass || v.Score != 0 {
			t.Errorf("%s spec must fail closed at score 0; got %+v", name, v)
		}
		if !strings.Contains(v.Detail, "schema spec unusable") {
			t.Errorf("%s spec Detail must name the spec problem; got %q", name, v.Detail)
		}
	}
}

// TestSoStructuredOutputMinScoreTolerance proves the threshold gate: the same
// missing-key output (5/6 checks) passes when the case tolerates MinScore 0.8,
// and the tolerated Detail still names the violation.
func TestSoStructuredOutputMinScoreTolerance(t *testing.T) {
	missingPaid := `{"invoice_id":"INV-7","total":1250.5,"items":[]}`
	v := soJudge(soStructuredCase(soInvoiceSpec, 0.8), missingPaid)
	if !v.Pass {
		t.Fatalf("5/6 checks must pass at MinScore 0.8; got %+v", v)
	}
	if !strings.Contains(v.Detail, `missing required key "paid"`) {
		t.Errorf("tolerated Detail should still name the violation; got %q", v.Detail)
	}
}

// TestSoStructuredOutputSpineIntegration runs a defective engine through the
// full spine: the failure bundle names structured-output-validity as the
// failing oracle and carries the violation in its detail.
func TestSoStructuredOutputSpineIntegration(t *testing.T) {
	c := soStructuredCase(soInvoiceSpec, 1)
	eng := ScriptedRunner{
		Label: "engine-structured-defect",
		Trace: Trace{Text: `{"invoice_id":"INV-7","total":"1250.5","paid":false,"items":[]}`},
	}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("wrong-typed structured output must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "structured-output-validity" {
		t.Errorf("failing oracle = %q, want structured-output-validity", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, `key "total" has type string, want number`) {
		t.Errorf("bundle detail must name the type violation; got %q", fb.Detail)
	}
}
