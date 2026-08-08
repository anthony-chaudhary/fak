package quality

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
)

// toolCallFidelity is the tool-call selection + argument-fidelity oracle
// (#4565): an engine turn that calls a tool must select the RIGHT tool for the
// task and pass WELL-FORMED arguments that conform to that tool's declared
// schema. A fluent turn that reaches for the wrong tool, or the right tool
// with a missing / mistyped / out-of-constraint argument, executes the wrong
// action downstream â€” exactly the fluent-but-wrong failure class the spine
// exists to catch â€” so selection and argument conformance are judged together
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
// closed too â€” a tool-call case that checks nothing is not green.
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
		return rubricFail(v, "tool spec unusable: "+err.Error())
	}
	call, cerr := toolParseCall(eng.Text)
	if cerr != nil {
		return rubricFail(v, "tool call malformed: "+cerr.Error())
	}
	if call.Tool != spec.Tool {
		return rubricFail(v, fmt.Sprintf("wrong tool selected: engine called %q, case expects %q",
			call.Tool, spec.Tool))
	}
	passed := 0
	var faults []string
	for _, name := range sortedKeys(spec.Args) {
		as := spec.Args[name]
		got, ok := call.Args[name]
		switch {
		case !ok && as.Required:
			faults = append(faults, fmt.Sprintf("missing required argument %q", name))
		case !ok:
			passed++ // optional and absent: vacuously conformant
		case !jsonTypeMatches(as.Type, got):
			faults = append(faults, fmt.Sprintf("argument %q has type %s, want %s", name, jsonTypeName(got), as.Type))
		default:
			if fault := toolConstraintFault(name, as, got); fault != "" {
				faults = append(faults, fault)
			} else if fault := toolInstructionFault(name, got, c.Prompt); fault != "" {
				faults = append(faults, fault)
			} else {
				passed++
			}
		}
	}
	extra := 0
	for _, name := range sortedKeys(call.Args) {
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
	min, short := rubricScore(&v, c, passed, total)
	if short {
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

// toolParseSpec parses and admission-checks the case-carried tool contract. It
// refuses (with a reason) an empty or unparseable spec, one naming no expected
// tool, and an argument declaring a type name outside the closed vocabulary â€”
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
	for _, name := range sortedKeys(s.Args) {
		if !knownJSONType(s.Args[name].Type) {
			return s, fmt.Errorf("tool spec argument %q declares unknown type %q (known: %s)",
				name, s.Args[name].Type, strings.Join(jsonTypeNames, ", "))
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

// toolNum renders a numeric bound/value compactly for fault messages.
func toolNum(f float64) string {
	return strconv.FormatFloat(f, 'g', -1, 64)
}

var toolNumberLiteral = regexp.MustCompile(`[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?`)

// toolInstructionFault rejects schema-legal values invented by the model. A
// schema proves legality; only the task instruction establishes intent.
func toolInstructionFault(name string, got any, prompt string) string {
	return toolInstructionValueFault(name, got, prompt)
}

func toolInstructionValueFault(path string, got any, prompt string) string {
	switch x := got.(type) {
	case map[string]any:
		for _, key := range sortedKeys(x) {
			if fault := toolInstructionValueFault(path+"."+key, x[key], prompt); fault != "" {
				return fault
			}
		}
		return ""
	case []any:
		for i, value := range x {
			if fault := toolInstructionValueFault(fmt.Sprintf("%s[%d]", path, i), value, prompt); fault != "" {
				return fault
			}
		}
		return ""
	case string:
		if toolInstructionContainsString(prompt, x) {
			return ""
		}
		return fmt.Sprintf("argument %q has invented literal %q absent from the task instruction", path, x)
	case float64:
		for _, literal := range toolNumberLiteral.FindAllString(prompt, -1) {
			want, err := strconv.ParseFloat(literal, 64)
			if err == nil && want == x {
				return ""
			}
		}
		return fmt.Sprintf("argument %q has invented literal %s absent from the task instruction", path, toolNum(x))
	case bool:
		if toolInstructionHasToken(prompt, strconv.FormatBool(x)) {
			return ""
		}
		return fmt.Sprintf("argument %q has invented literal %t absent from the task instruction", path, x)
	case nil:
		if toolInstructionHasToken(prompt, "null") {
			return ""
		}
		return fmt.Sprintf("argument %q has invented literal null absent from the task instruction", path)
	default:
		return fmt.Sprintf("argument %q has an unsupported instruction literal", path)
	}
}

func toolInstructionContainsString(prompt, value string) bool {
	needle := toolInstructionText(value)
	if needle == "" {
		return false
	}
	haystack := " " + toolInstructionText(prompt) + " "
	if strings.Contains(haystack, " "+needle+" ") {
		return true
	}
	aliases := map[string][]string{
		"celsius":    {"c", "deg c", "degree c", "degrees c", "centigrade"},
		"fahrenheit": {"f", "deg f", "degree f", "degrees f"},
	}
	for canonical, variants := range aliases {
		all := append([]string{canonical}, variants...)
		matched := false
		for _, variant := range all {
			if needle == variant {
				matched = true
				break
			}
		}
		if !matched {
			continue
		}
		for _, variant := range all {
			if strings.Contains(haystack, " "+variant+" ") {
				return true
			}
		}
	}
	return false
}

func toolInstructionHasToken(prompt, token string) bool {
	return strings.Contains(" "+toolInstructionText(prompt)+" ", " "+token+" ")
}

func toolInstructionText(s string) string {
	s = strings.ToLower(strings.ReplaceAll(s, "°", " deg "))
	var b strings.Builder
	space := true
	for _, r := range s {
		if (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9') {
			b.WriteRune(r)
			space = false
			continue
		}
		if !space {
			b.WriteByte(' ')
			space = true
		}
	}
	return strings.TrimSpace(b.String())
}
