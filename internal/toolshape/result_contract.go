package toolshape

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// ResultContractExtension is the versioned JSON-Schema extension understood by
// ParseResultContract. It lives on the tool's parameter schema so it survives
// every provider adapter without changing the public tool-definition wire.
const ResultContractExtension = "x-fak-result-contract"

const resultContractVersion = "fak-result-contract/1"

// ResultContract declares the complete output set for one tool result. Outputs
// are a set of named JSON values: each name occurs exactly once unless an
// explicitly versioned future extension changes that rule.
type ResultContract struct {
	Schema  string       `json:"schema"`
	Outputs []ResultSpec `json:"outputs"`
}

// ResultSpec declares one required output and its JSON value type.
type ResultSpec struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// ResultShapeReceipt is a content-free receipt for one validation. It exposes
// names, types, counts, and the typed failure only; result values never enter it.
type ResultShapeReceipt struct {
	Schema        string           `json:"schema"`
	Expected      []ResultShape    `json:"expected"`
	Observed      []ResultShape    `json:"observed"`
	ExpectedCount int              `json:"expected_count"`
	ObservedCount int              `json:"observed_count"`
	Failure       ResultShapeError `json:"failure,omitempty"`
}

// ResultShape names one observed or expected output without retaining its value.
type ResultShape struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// ResultShapeError is a stable, machine-readable refusal class.
type ResultShapeError string

const (
	ResultShapeOK         ResultShapeError = ""
	ResultShapeMalformed  ResultShapeError = "malformed"
	ResultShapeMissing    ResultShapeError = "missing"
	ResultShapeAdditional ResultShapeError = "additional"
	ResultShapeDuplicate  ResultShapeError = "duplicate"
	ResultShapeFused      ResultShapeError = "fused"
	ResultShapeSplit      ResultShapeError = "split"
	ResultShapeType       ResultShapeError = "type_mismatch"
	ResultShapeAmbiguous  ResultShapeError = "ambiguous_pairing"
)

// ResultShapeFailure is returned when the expected and observed outputs do not
// form the contract's exact bijection.
type ResultShapeFailure struct {
	Kind    ResultShapeError
	Receipt ResultShapeReceipt
}

func (e *ResultShapeFailure) Error() string {
	return "tool result shape refused: " + string(e.Kind)
}

// ParseResultContract reads a result contract from a tool's JSON parameter
// schema. A missing extension means the tool has no result-shape contract.
func ParseResultContract(parameters json.RawMessage) (ResultContract, bool, error) {
	if len(bytes.TrimSpace(parameters)) == 0 {
		return ResultContract{}, false, nil
	}
	var schema map[string]json.RawMessage
	if err := json.Unmarshal(parameters, &schema); err != nil {
		return ResultContract{}, false, fmt.Errorf("tool parameter schema: %w", err)
	}
	raw, ok := schema[ResultContractExtension]
	if !ok {
		return ResultContract{}, false, nil
	}
	var c ResultContract
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&c); err != nil {
		return ResultContract{}, true, fmt.Errorf("%s: %w", ResultContractExtension, err)
	}
	if c.Schema != resultContractVersion {
		return ResultContract{}, true, fmt.Errorf("%s: unsupported schema %q", ResultContractExtension, c.Schema)
	}
	if len(c.Outputs) == 0 {
		return ResultContract{}, true, fmt.Errorf("%s: outputs must not be empty", ResultContractExtension)
	}
	seen := make(map[string]struct{}, len(c.Outputs))
	for i := range c.Outputs {
		o := &c.Outputs[i]
		o.Name = strings.TrimSpace(o.Name)
		if o.Name == "" {
			return ResultContract{}, true, fmt.Errorf("%s: output %d has no name", ResultContractExtension, i)
		}
		if _, exists := seen[o.Name]; exists {
			return ResultContract{}, true, fmt.Errorf("%s: duplicate output name %q", ResultContractExtension, o.Name)
		}
		seen[o.Name] = struct{}{}
		if !validJSONType(o.Type) {
			return ResultContract{}, true, fmt.Errorf("%s: output %q has unsupported type %q", ResultContractExtension, o.Name, o.Type)
		}
	}
	sort.Slice(c.Outputs, func(i, j int) bool { return c.Outputs[i].Name < c.Outputs[j].Name })
	return c, true, nil
}

// ValidateResult enforces an exact name/type bijection. The result encoding is a
// JSON object whose members are the declared outputs. An array at the top level
// is classified as split; a scalar is fused because it collapses the named set.
func (c ResultContract) ValidateResult(raw string) (ResultShapeReceipt, error) {
	receipt := ResultShapeReceipt{Schema: resultContractVersion, ExpectedCount: len(c.Outputs)}
	for _, o := range c.Outputs {
		receipt.Expected = append(receipt.Expected, ResultShape{Name: o.Name, Type: o.Type})
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return failReceipt(receipt, ResultShapeMalformed)
	}
	if trimmed[0] != '{' {
		var top any
		dec := json.NewDecoder(strings.NewReader(raw))
		dec.UseNumber()
		if err := dec.Decode(&top); err != nil {
			return failReceipt(receipt, ResultShapeMalformed)
		}
		if err := ensureEOF(dec); err != nil {
			return failReceipt(receipt, ResultShapeAmbiguous)
		}
		if _, split := top.([]any); split {
			return failReceipt(receipt, ResultShapeSplit)
		}
		return failReceipt(receipt, ResultShapeFused)
	}
	obj, duplicate, err := decodeJSONObjectExact(raw)
	if err != nil {
		if err == errMultipleJSONValues {
			return failReceipt(receipt, ResultShapeAmbiguous)
		}
		return failReceipt(receipt, ResultShapeMalformed)
	}
	if duplicate {
		return failReceipt(receipt, ResultShapeDuplicate)
	}

	receipt.ObservedCount = len(obj)
	observed := make(map[string]string, len(obj))
	for name, value := range obj {
		typ := jsonType(value)
		observed[name] = typ
		receipt.Observed = append(receipt.Observed, ResultShape{Name: name, Type: typ})
	}
	sort.Slice(receipt.Observed, func(i, j int) bool { return receipt.Observed[i].Name < receipt.Observed[j].Name })

	expected := make(map[string]string, len(c.Outputs))
	for _, o := range c.Outputs {
		if _, duplicate := expected[o.Name]; duplicate {
			return failReceipt(receipt, ResultShapeDuplicate)
		}
		expected[o.Name] = o.Type
	}
	for name := range expected {
		if _, ok := observed[name]; !ok {
			return failReceipt(receipt, ResultShapeMissing)
		}
	}
	for name := range observed {
		if _, ok := expected[name]; !ok {
			return failReceipt(receipt, ResultShapeAdditional)
		}
	}
	for name, want := range expected {
		if observed[name] != want {
			return failReceipt(receipt, ResultShapeType)
		}
	}
	return receipt, nil
}

func failReceipt(r ResultShapeReceipt, kind ResultShapeError) (ResultShapeReceipt, error) {
	r.Failure = kind
	return r, &ResultShapeFailure{Kind: kind, Receipt: r}
}

var errMultipleJSONValues = fmt.Errorf("multiple JSON values")

// decodeJSONObjectExact preserves member multiplicity. Unmarshalling directly
// into a map would silently keep the last duplicate and make a malformed result
// look bijective.
func decodeJSONObjectExact(raw string) (map[string]any, bool, error) {
	dec := json.NewDecoder(strings.NewReader(raw))
	dec.UseNumber()
	tok, err := dec.Token()
	if err != nil {
		return nil, false, err
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '{' {
		return nil, false, fmt.Errorf("result is not an object")
	}
	obj := make(map[string]any)
	duplicate := false
	for dec.More() {
		tok, err = dec.Token()
		if err != nil {
			return nil, false, err
		}
		name, ok := tok.(string)
		if !ok {
			return nil, false, fmt.Errorf("object member name is not a string")
		}
		if _, exists := obj[name]; exists {
			duplicate = true
		}
		var value any
		if err := dec.Decode(&value); err != nil {
			return nil, false, err
		}
		obj[name] = value
	}
	if _, err := dec.Token(); err != nil {
		return nil, false, err
	}
	if err := ensureEOF(dec); err != nil {
		return nil, false, errMultipleJSONValues
	}
	return obj, duplicate, nil
}

func ensureEOF(dec *json.Decoder) error {
	var extra any
	if err := dec.Decode(&extra); err == nil {
		return fmt.Errorf("multiple JSON values")
	} else if err.Error() != "EOF" {
		return err
	}
	return nil
}

func validJSONType(t string) bool {
	switch t {
	case "null", "boolean", "number", "integer", "string", "array", "object":
		return true
	default:
		return false
	}
}

func jsonType(v any) string {
	switch x := v.(type) {
	case nil:
		return "null"
	case bool:
		return "boolean"
	case json.Number:
		if !strings.ContainsAny(string(x), ".eE") {
			return "integer"
		}
		return "number"
	case string:
		return "string"
	case []any:
		return "array"
	case map[string]any:
		return "object"
	default:
		return "unknown"
	}
}
