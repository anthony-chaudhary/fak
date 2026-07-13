package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
)

// toolCallFidelity is the tool-call selection + argument-fidelity oracle
// (#4565): an engine turn that calls a tool must select the RIGHT tool for the
// task and pass WELL-FORMED arguments that conform to that tool's declared
// schema. A fluent turn that reaches for the wrong tool, or the right tool
// with a missing / mistyped / out-of-constraint argument, executes the wrong
// action downstream — exactly the fluent-but-wrong failure class the spine
// exists to catch — so selection and argument conformance are judged together
// in one rubric.
//
// The contract travels IN the case, additively: Reference.Text carries one
// toolCallSpec JSON document ({"tool": name, "args": {arg: {"type": t,
// "required": bool, "enum": [...], "min": x, "max": y}}}) and the engine's
// emitted tool call is eng.Text ({"tool": name, "args": {...}}). Known type
// names are string, number, integer, boolean, object, array, null.
//
// Selection is not gradable: a call naming the wrong tool fails closed at
// score 0, naming both tools. With the right tool selected, Score = passed
// argument checks / total argument checks, where each declared argument is one
// check (presence if required, then type, then enum/range constraints) and
// each undeclared argument the engine passed is one failed check; Pass iff
// Score >= Rubric.MinScore (default 1: full conformance). On failure Detail
// names the FIRST fault in deterministic (sorted-argument) order, so an
// argument defect localizes to the offending key, per the spine contract.
//
// Edge behavior (defined and tested): an empty or unparseable call, or one
// naming no tool, fails closed at score 0 as malformed; a case with no usable
// spec (empty, unparseable, no tool name, or an unknown type name) fails
// closed too — a tool-call case that checks nothing is not green.
type toolCallFidelity struct{}

func (toolCallFidelity) Name() string { return "tool-call-fidelity" }
func (toolCallFidelity) Kind() string { return "rubric" }

func init() { Register(toolCallFidelity{}) }

// toolCallSpec is the case-carried tool contract: the tool the case expects
// the engine to select, and the schema of that tool's arguments.
type toolCallSpec struct {
	Tool string                 `json:"tool"`
	Args map[string]toolArgSpec `json:"args,omitempty"`
}

// toolArgSpec is one argument's schema: its JSON type name, whether it must be
// present, and optional value constraints (a closed string vocabulary and/or a
// numeric range).
type toolArgSpec struct {
	Type     string   `json:"type"`
	Required bool     `json:"required,omitempty"`
	Enum     []string `json:"enum,omitempty"`
	Min      *float64 `json:"min,omitempty"`
	Max      *float64 `json:"max,omitempty"`
}

// toolCallRecord is the engine's emitted tool call as carried in eng.Text: the
// tool it selected and the arguments it passed.
type toolCallRecord struct {
	Tool string         `json:"tool"`
	Args map[string]any `json:"args,omitempty"`
}

func (toolCallFidelity) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "tool-call-fidelity", Kind: "rubric", Pass: true, Score: 1}
	spec, err := toolParseSpec(c.Reference.Text)
	if err != nil {
		v.Pass = false
		v.Score = 0
		v.Detail = "tool spec unusable: " + err.Error()
		return v
	}
	call, cerr := toolParseCall(eng.Text)
	if cerr != nil {
		v.Pass = false
		v.Score = 0
		v.Detail = "tool call malformed: " + cerr.Error()
		return v
	}
	if call.Tool != spec.Tool {
		v.Pass = false
		v.Score = 0
		v.Detail = fmt.Sprintf("wrong tool selected: engine called %q, case expects %q", call.Tool, spec.Tool)
		return v
	}
	passed := 0
	var faults []string
	for _, name := range toolSortedKeys(spec.Args) {
		as := spec.Args[name]
		got, ok := call.Args[name]
		switch {
		case !ok && as.Required:
			faults = append(faults, fmt.Sprintf("missing required argument %q", name))
		case !ok:
			passed++ // optional and absent: vacuously conformant
		case !toolTypeMatches(as.Type, got):
			faults = append(faults, fmt.Sprintf("argument %q has type %s, want %s", name, toolJSONType(got), as.Type))
		default:
			if fault := toolConstraintFault(name, as, got); fault != "" {
				faults = append(faults, fault)
			} else {
				passed++
			}
		}
	}
	extra := 0
	for _, name := range toolSortedKeys(call.Args) {
		if _, declared := spec.Args[name]; !declared {
			extra++
			faults = append(faults, fmt.Sprintf("argument %q is not in tool %q's schema", name, spec.Tool))
		}
	}
	total := len(spec.Args) + extra
	if total == 0 {
		v.Detail = fmt.Sprintf("tool %q selected correctly; no arguments declared or passed", spec.Tool)
		return v
	}
	v.Score = float64(passed) / float64(total)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: full argument conformance
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("tool-call fidelity %.2f < %.2f (%d/%d argument checks passed); first fault: %s",
			v.Score, min, passed, total, faults[0])
		return v
	}
	if len(faults) > 0 {
		v.Detail = fmt.Sprintf("tool-call fidelity %.2f >= %.2f (%d/%d argument checks passed; tolerated fault: %s)",
			v.Score, min, passed, total, faults[0])
		return v
	}
	v.Detail = fmt.Sprintf("tool %q selected correctly; all %d argument check(s) passed", spec.Tool, total)
	return v
}

// toolTypeNames is the closed vocabulary of JSON type names an argument spec
// may declare, sorted for deterministic error messages.
var toolTypeNames = []string{"array", "boolean", "integer", "null", "number", "object", "string"}

// toolParseSpec parses and admission-checks the case-carried tool contract. It
// refuses (with a reason) an empty or unparseable spec, one naming no expected
// tool, and an argument declaring a type name outside the closed vocabulary —
// a spec bug must fail the case loudly, not silently skip checks.
func toolParseSpec(text string) (toolCallSpec, error) {
	var s toolCallSpec
	if strings.TrimSpace(text) == "" {
		return s, fmt.Errorf("case Reference.Text carries no tool spec (a tool-call case that checks nothing is not green)")
	}
	if err := json.Unmarshal([]byte(text), &s); err != nil {
		return s, fmt.Errorf("tool spec does not parse: %v", err)
	}
	if s.Tool == "" {
		return s, fmt.Errorf("tool spec names no expected tool")
	}
	for _, name := range toolSortedKeys(s.Args) {
		if !toolKnownType(s.Args[name].Type) {
			return s, fmt.Errorf("tool spec argument %q declares unknown type %q (known: %s)",
				name, s.Args[name].Type, strings.Join(toolTypeNames, ", "))
		}
	}
	return s, nil
}

// toolParseCall parses the engine's emitted tool call, requiring a JSON object
// carrying a non-empty tool name. A nil args map normalizes to empty so a call
// passing no arguments is judged against the schema, not treated specially.
func toolParseCall(text string) (toolCallRecord, error) {
	var r toolCallRecord
	if strings.TrimSpace(text) == "" {
		return r, fmt.Errorf("engine emitted no tool call")
	}
	if err := json.Unmarshal([]byte(text), &r); err != nil {
		return r, fmt.Errorf("tool call does not parse as JSON: %v", err)
	}
	if r.Tool == "" {
		return r, fmt.Errorf("tool call names no tool")
	}
	if r.Args == nil {
		r.Args = map[string]any{}
	}
	return r, nil
}

// toolConstraintFault checks a type-conformant argument value against the
// spec's value constraints and returns the first fault, or "" if none. Enum
// constrains string values; min/max constrain numeric values.
func toolConstraintFault(name string, as toolArgSpec, got any) string {
	if s, ok := got.(string); ok && len(as.Enum) > 0 {
		for _, e := range as.Enum {
			if s == e {
				return ""
			}
		}
		return fmt.Sprintf("argument %q = %q, not one of [%s]", name, s, strings.Join(as.Enum, ", "))
	}
	if f, ok := got.(float64); ok {
		if as.Min != nil && f < *as.Min {
			return fmt.Sprintf("argument %q = %s, below minimum %s", name, toolNum(f), toolNum(*as.Min))
		}
		if as.Max != nil && f > *as.Max {
			return fmt.Sprintf("argument %q = %s, above maximum %s", name, toolNum(f), toolNum(*as.Max))
		}
	}
	return ""
}

// toolKnownType reports whether t is in the closed type-name vocabulary.
func toolKnownType(t string) bool {
	for _, n := range toolTypeNames {
		if t == n {
			return true
		}
	}
	return false
}

// toolTypeMatches reports whether a decoded JSON argument value satisfies a
// declared type name. "integer" is a number with no fractional part —
// encoding/json decodes every JSON number to float64, so integrality is
// checked, not the Go type.
func toolTypeMatches(want string, got any) bool {
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

// toolJSONType names the JSON type of a decoded value for fault messages.
func toolJSONType(v any) string {
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

// toolNum renders a numeric bound/value compactly for fault messages.
func toolNum(f float64) string {
	return strconv.FormatFloat(f, 'g', -1, 64)
}

// toolSortedKeys returns m's keys sorted, so fault order — and therefore the
// FIRST fault a Detail names — is deterministic across runs.
func toolSortedKeys[V any](m map[string]V) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
