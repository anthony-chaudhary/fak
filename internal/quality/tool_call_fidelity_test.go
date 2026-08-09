package quality

import (
	"strings"
	"testing"
)

// toolWeatherSpec is the case-carried tool contract for these tests: the
// engine must call get_weather with a required string city, a required
// bounded-integer days, and an optional enum-constrained units.
const toolWeatherSpec = `{"tool":"get_weather","args":{` +
	`"city":{"type":"string","required":true},` +
	`"days":{"type":"integer","required":true,"min":1,"max":10},` +
	`"units":{"type":"string","enum":["celsius","fahrenheit"]}}}`

// toolFaithfulCall selects the right tool and conforms to every argument
// constraint of toolWeatherSpec.
const toolFaithfulCall = `{"tool":"get_weather","args":{"city":"Oslo","days":3,"units":"celsius"}}`

// toolFidelityCase builds a valid case whose Reference.Text carries the tool
// contract the tool-call-fidelity oracle judges the engine's call against.
func toolFidelityCase(minScore float64) QualityCase {
	return QualityCase{
		Schema:    CaseSchema,
		ID:        "tool-call-fidelity-weather",
		Version:   1,
		Prompt:    "Fetch the 3-day Oslo forecast in celsius.",
		Params:    SamplingParams{Temperature: 0, MaxTokens: 32},
		Reference: Trace{Text: toolWeatherSpec},
		Oracles:   []string{"tool-call-fidelity"},
		Rubric:    RubricSpec{MinScore: minScore},
	}
}

// TestToolCallFidelityRegistered proves the oracle registered under its stable
// name and kind, so cases can reference it by name.
func TestToolCallFidelityRegistered(t *testing.T) {
	os, err := Lookup([]string{"tool-call-fidelity"})
	if err != nil {
		t.Fatalf("Lookup(tool-call-fidelity): %v", err)
	}
	if got := os[0].Kind(); got != "rubric" {
		t.Errorf("Kind() = %q, want rubric", got)
	}
}

// TestToolCallFidelityFaithfulCallPasses is the happy path: the right tool
// with fully conformant arguments passes at score 1.0.
func TestToolCallFidelityFaithfulCallPasses(t *testing.T) {
	c := toolFidelityCase(1)
	v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: toolFaithfulCall}, c)
	if !v.Pass {
		t.Fatalf("faithful tool call must pass; got %+v", v)
	}
	if v.Score != 1 {
		t.Errorf("score = %v, want 1.0", v.Score)
	}
}

// TestToolCallFidelityWrongToolFails is the selection-defect witness: a call
// naming a different tool fails closed at score 0 and Detail names both the
// selected and the expected tool.
func TestToolCallFidelityRejectsSchemaLegalInventedLiteral(t *testing.T) {
	c := toolFidelityCase(1)
	eng := Trace{Text: `{"tool":"get_weather","args":{"city":"Oslo","days":7,"units":"celsius"}}`}
	got := toolCallFidelity{}.Judge(Trace{}, eng, c)
	if got.Pass || got.Score != 2.0/3.0 {
		t.Fatalf("schema-legal invented days must fail at 2/3, got %+v", got)
	}
	for _, want := range []string{`argument "days"`, "invented literal 7", "task instruction"} {
		if !strings.Contains(got.Detail, want) {
			t.Fatalf("detail %q missing %q", got.Detail, want)
		}
	}
}

func TestToolCallFidelityInstructionLiteralNormalization(t *testing.T) {
	c := toolFidelityCase(1)
	c.Prompt = "Fetch the 3-day OSLO forecast in degrees C."
	eng := Trace{Text: `{"tool":"get_weather","args":{"city":"oslo","days":3,"units":"celsius"}}`}
	got := toolCallFidelity{}.Judge(Trace{}, eng, c)
	if !got.Pass || got.Score != 1 {
		t.Fatalf("normalized instruction literals should pass, got %+v", got)
	}
}

func TestToolCallFidelityWrongToolFails(t *testing.T) {
	c := toolFidelityCase(1)
	wrong := `{"tool":"send_email","args":{"city":"Oslo","days":3,"units":"celsius"}}`
	v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: wrong}, c)
	if v.Pass {
		t.Fatalf("wrong tool selection must not pass; got %+v", v)
	}
	if v.Score != 0 {
		t.Errorf("score = %v, want 0 (selection is not gradable)", v.Score)
	}
	for _, want := range []string{"wrong tool", `"send_email"`, `"get_weather"`} {
		if !strings.Contains(v.Detail, want) {
			t.Errorf("Detail must contain %q; got %q", want, v.Detail)
		}
	}
}

// TestToolCallFidelityMissingRequiredArgFails is the missing-argument witness:
// dropping the required "city" fails the oracle and Detail names the argument.
func TestToolCallFidelityMissingRequiredArgFails(t *testing.T) {
	c := toolFidelityCase(1)
	v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: `{"tool":"get_weather","args":{"days":3,"units":"celsius"}}`}, c)
	if v.Pass {
		t.Fatalf("missing required argument must not pass; got %+v", v)
	}
	if want := 2.0 / 3.0; v.Score != want {
		t.Errorf("score = %v, want %v (2 of 3 argument checks passed)", v.Score, want)
	}
	if !strings.Contains(v.Detail, `missing required argument "city"`) {
		t.Errorf("Detail must name the missing argument; got %q", v.Detail)
	}
}

// TestToolCallFidelityMistypedArgFails is the malformed-argument witness: a
// string where an integer is required, and a fractional number where an
// integer is required, each fail with Detail naming the type fault.
func TestToolCallFidelityMistypedArgFails(t *testing.T) {
	c := toolFidelityCase(1)
	for call, wantDetail := range map[string]string{
		`{"tool":"get_weather","args":{"city":"Oslo","days":"three","units":"celsius"}}`: `argument "days" has type string, want integer`,
		`{"tool":"get_weather","args":{"city":"Oslo","days":2.5,"units":"celsius"}}`:     `argument "days" has type number, want integer`,
	} {
		v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: call}, c)
		if v.Pass {
			t.Fatalf("mistyped argument must not pass; call %s got %+v", call, v)
		}
		if !strings.Contains(v.Detail, wantDetail) {
			t.Errorf("Detail must contain %q; got %q", wantDetail, v.Detail)
		}
	}
}

// TestToolCallFidelityConstraintViolationsFail is the constraint witness: an
// enum-violating string and an out-of-range integer each fail with Detail
// naming the offending argument and constraint.
func TestToolCallFidelityConstraintViolationsFail(t *testing.T) {
	c := toolFidelityCase(1)
	for call, wantDetail := range map[string]string{
		`{"tool":"get_weather","args":{"city":"Oslo","days":3,"units":"kelvin"}}`:   `argument "units" = "kelvin", not one of [celsius, fahrenheit]`,
		`{"tool":"get_weather","args":{"city":"Oslo","days":14,"units":"celsius"}}`: `argument "days" = 14, above maximum 10`,
		`{"tool":"get_weather","args":{"city":"Oslo","days":0,"units":"celsius"}}`:  `argument "days" = 0, below minimum 1`,
	} {
		v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: call}, c)
		if v.Pass {
			t.Fatalf("constraint violation must not pass; call %s got %+v", call, v)
		}
		if !strings.Contains(v.Detail, wantDetail) {
			t.Errorf("Detail must contain %q; got %q", wantDetail, v.Detail)
		}
	}
}

// TestToolCallFidelityUnknownArgFails is the undeclared-argument witness: an
// argument outside the tool's schema is one failed check, and Detail names it.
func TestToolCallFidelityUnknownArgFails(t *testing.T) {
	c := toolFidelityCase(1)
	call := `{"tool":"get_weather","args":{"city":"Oslo","days":3,"units":"celsius","zip":"90210"}}`
	v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: call}, c)
	if v.Pass {
		t.Fatalf("undeclared argument must not pass; got %+v", v)
	}
	if want := 3.0 / 4.0; v.Score != want {
		t.Errorf("score = %v, want %v (3 of 4 argument checks passed)", v.Score, want)
	}
	if !strings.Contains(v.Detail, `argument "zip" is not in tool "get_weather"'s schema`) {
		t.Errorf("Detail must name the undeclared argument; got %q", v.Detail)
	}
}

// TestToolCallFidelityMalformedCallFailsClosed defines the malformed-call
// edge: an empty, unparseable, or tool-less call fails closed at score 0.
func TestToolCallFidelityMalformedCallFailsClosed(t *testing.T) {
	c := toolFidelityCase(1)
	for _, text := range []string{"", "   ", `{"tool":"get_weather","args":`, `{"args":{"city":"Oslo"}}`, `"get_weather"`} {
		v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: text}, c)
		if v.Pass || v.Score != 0 {
			t.Errorf("malformed call %q: got %+v, want fail closed at score 0", text, v)
		}
		if !strings.Contains(v.Detail, "malformed") {
			t.Errorf("Detail for %q must name the call malformed; got %q", text, v.Detail)
		}
	}
}

// TestToolCallFidelityUnusableSpecFailsClosed defines the bad-spec edge: an
// empty spec, one naming no tool, and one declaring an unknown type each fail
// closed — a tool-call case that checks nothing is not green.
func TestToolCallFidelityUnusableSpecFailsClosed(t *testing.T) {
	for _, spec := range []string{"", `{"args":{"city":{"type":"string"}}}`, `{"tool":"get_weather","args":{"days":{"type":"int"}}}`} {
		c := toolFidelityCase(1)
		c.Reference.Text = spec
		v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: toolFaithfulCall}, c)
		if v.Pass || v.Score != 0 {
			t.Errorf("spec %q: got %+v, want fail closed at score 0", spec, v)
		}
		if !strings.Contains(v.Detail, "tool spec unusable") {
			t.Errorf("Detail for spec %q must name the spec unusable; got %q", spec, v.Detail)
		}
	}
}

// TestToolCallFidelityMinScoreTolerance proves the threshold gate: the same
// undeclared-argument call (3/4 checks passed) passes when the case tolerates
// MinScore 0.6, with the tolerated fault still named — but a wrong tool stays
// a failure at any threshold.
func TestToolCallFidelityMinScoreTolerance(t *testing.T) {
	c := toolFidelityCase(0.6)
	call := `{"tool":"get_weather","args":{"city":"Oslo","days":3,"units":"celsius","zip":"90210"}}`
	v := toolCallFidelity{}.Judge(Trace{}, Trace{Text: call}, c)
	if !v.Pass {
		t.Fatalf("3/4 checks must pass at MinScore 0.6; got %+v", v)
	}
	if !strings.Contains(v.Detail, `"zip"`) {
		t.Errorf("tolerated-fault detail should still name the argument; got %q", v.Detail)
	}
	wrong := `{"tool":"send_email","args":{"city":"Oslo","days":3,"units":"celsius"}}`
	if v := (toolCallFidelity{}).Judge(Trace{}, Trace{Text: wrong}, c); v.Pass {
		t.Fatalf("wrong tool must fail at any threshold; got %+v", v)
	}
}

// TestToolCallFidelitySpineIntegration runs a defective call through the full
// spine: the failure bundle names tool-call-fidelity as the failing oracle and
// carries the offending argument in its detail.
func TestToolCallFidelitySpineIntegration(t *testing.T) {
	c := toolFidelityCase(1)
	eng := ScriptedRunner{
		Label: "engine-missing-arg",
		Trace: Trace{Text: `{"tool":"get_weather","args":{"days":3,"units":"celsius"}}`},
	}
	res, err := RunCase(c, ReferenceRunner{}, eng, oraclesFor(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("defective tool call must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "tool-call-fidelity" {
		t.Errorf("failing oracle = %q, want tool-call-fidelity", fb.FailingOracle)
	}
	if !strings.Contains(fb.Detail, `missing required argument "city"`) {
		t.Errorf("bundle detail must name the missing argument; got %q", fb.Detail)
	}
}
