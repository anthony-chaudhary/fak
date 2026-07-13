package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"reflect"
	"sort"
	"strings"
)

// soStructuredOutput is the constrained/structured-output oracle (#4548): an
// engine asked for schema-shaped output must emit text that PARSES as a JSON
// object, CONFORMS to the case's declared schema (every required key present
// with the right JSON type), and MATCHES any pinned expected values.
// Grammar-valid but semantically wrong structured output — the right keys
// carrying the wrong figures — is exactly the fluent-but-wrong failure class
// the spine exists to catch, so conformance and value fidelity are judged
// together in one rubric.
//
// The contract travels IN the case, additively: Reference.Text carries one
// soSchemaSpec JSON document ({"required": {key: type}, "expected": {key:
// value}}) and the engine's structured output is eng.Text. Known type names
// are string, number, integer, boolean, object, array, null.
//
// Score = passed checks / declared checks, where each required key is one
// presence+type check and each expected entry is one value check; Pass iff
// Score >= Rubric.MinScore (default 1: full conformance). On failure Detail
// names the FIRST violation in deterministic (sorted-key) order — "missing
// required key", "key has type X, want Y", or "key = got, want want" — so a
// schema defect localizes to the offending key, per the spine contract.
//
// Edge behavior (defined and tested): unparseable output, or a top-level JSON
// value that is not an object, fails closed at score 0 naming the parse
// violation; a case with no usable spec (empty, unparseable, zero checks, or
// an unknown type name) fails closed too — a structured-output case that
// checks nothing is not green.
type soStructuredOutput struct{}

func (soStructuredOutput) Name() string { return "structured-output-validity" }
func (soStructuredOutput) Kind() string { return "rubric" }

func init() { Register(soStructuredOutput{}) }

// soSchemaSpec is the case-carried structured-output contract: the required
// keys with their JSON type names, plus optional exact expected values.
type soSchemaSpec struct {
	Required map[string]string `json:"required"`
	Expected map[string]any    `json:"expected,omitempty"`
}

func (soStructuredOutput) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "structured-output-validity", Kind: "rubric", Pass: true, Score: 1}
	spec, err := soParseSpec(c.Reference.Text)
	if err != nil {
		v.Pass = false
		v.Score = 0
		v.Detail = "schema spec unusable: " + err.Error()
		return v
	}
	obj, perr := soParseObject(eng.Text)
	if perr != nil {
		v.Pass = false
		v.Score = 0
		v.Detail = "output violation: " + perr.Error()
		return v
	}
	total := len(spec.Required) + len(spec.Expected)
	passed := 0
	var violations []string
	for _, k := range soSortedKeys(spec.Required) {
		want := spec.Required[k]
		got, ok := obj[k]
		switch {
		case !ok:
			violations = append(violations, fmt.Sprintf("missing required key %q", k))
		case !soTypeMatches(want, got):
			violations = append(violations, fmt.Sprintf("key %q has type %s, want %s", k, soJSONType(got), want))
		default:
			passed++
		}
	}
	for _, k := range soSortedKeys(spec.Expected) {
		want := spec.Expected[k]
		got, ok := obj[k]
		switch {
		case !ok:
			violations = append(violations, fmt.Sprintf("missing expected key %q", k))
		case !reflect.DeepEqual(got, want):
			violations = append(violations, fmt.Sprintf("key %q = %s, want %s", k, soJSON(got), soJSON(want)))
		default:
			passed++
		}
	}
	v.Score = float64(passed) / float64(total)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: full schema conformance and value fidelity
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("structured-output score %.2f < %.2f (%d/%d checks passed); first violation: %s",
			v.Score, min, passed, total, violations[0])
		return v
	}
	if len(violations) > 0 {
		v.Detail = fmt.Sprintf("structured-output score %.2f >= %.2f (%d/%d checks passed; tolerated violation: %s)",
			v.Score, min, passed, total, violations[0])
		return v
	}
	v.Detail = fmt.Sprintf("output parsed and conformed: all %d schema check(s) passed", total)
	return v
}

// soTypeNames is the closed vocabulary of JSON type names a spec may declare,
// sorted for deterministic error messages.
var soTypeNames = []string{"array", "boolean", "integer", "null", "number", "object", "string"}

// soParseSpec parses and admission-checks the case-carried schema spec. It
// refuses (with a reason) an empty, unparseable, or zero-check spec, and a
// required entry declaring a type name outside the closed vocabulary — a spec
// bug must fail the case loudly, not silently skip checks.
func soParseSpec(text string) (soSchemaSpec, error) {
	var s soSchemaSpec
	if strings.TrimSpace(text) == "" {
		return s, fmt.Errorf("case Reference.Text carries no schema spec (a structured-output case that checks nothing is not green)")
	}
	if err := json.Unmarshal([]byte(text), &s); err != nil {
		return s, fmt.Errorf("schema spec does not parse: %v", err)
	}
	if len(s.Required)+len(s.Expected) == 0 {
		return s, fmt.Errorf("schema spec declares no required keys and no expected values")
	}
	for _, k := range soSortedKeys(s.Required) {
		if !soKnownType(s.Required[k]) {
			return s, fmt.Errorf("schema spec key %q declares unknown type %q (known: %s)",
				k, s.Required[k], strings.Join(soTypeNames, ", "))
		}
	}
	return s, nil
}

// soParseObject parses the engine's structured output, requiring a JSON
// object at the top level.
func soParseObject(text string) (map[string]any, error) {
	var v any
	if err := json.Unmarshal([]byte(text), &v); err != nil {
		return nil, fmt.Errorf("output does not parse as JSON: %v", err)
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("top-level JSON value is %s, want object", soJSONType(v))
	}
	return obj, nil
}

// soKnownType reports whether t is in the closed type-name vocabulary.
func soKnownType(t string) bool {
	for _, n := range soTypeNames {
		if t == n {
			return true
		}
	}
	return false
}

// soTypeMatches reports whether a decoded JSON value satisfies a declared
// type name. "integer" is a number with no fractional part — encoding/json
// decodes every JSON number to float64, so integrality is checked, not the
// Go type.
func soTypeMatches(want string, got any) bool {
	switch want {
	case "string":
		_, ok := got.(string)
		return ok
	case "boolean":
		_, ok := got.(bool)
		return ok
	case "number":
		_, ok := got.(float64)
		return ok
	case "integer":
		f, ok := got.(float64)
		return ok && f == math.Trunc(f)
	case "object":
		_, ok := got.(map[string]any)
		return ok
	case "array":
		_, ok := got.([]any)
		return ok
	case "null":
		return got == nil
	}
	return false
}

// soJSONType names the JSON type of a decoded value for violation messages.
func soJSONType(v any) string {
	switch v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case float64:
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	}
	return fmt.Sprintf("%T", v)
}

// soJSON renders a decoded value back to compact JSON for violation messages.
func soJSON(v any) string {
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprintf("%v", v)
	}
	return string(b)
}

// soSortedKeys returns m's keys sorted, so violation order — and therefore
// the FIRST violation a Detail names — is deterministic across runs.
func soSortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
