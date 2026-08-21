package toolplugin

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
)

// BudgetContract binds one tool to explicit JSON-pointer aliases for one bounded integer argument.
// Pointers are tried in order; names alone never activate a contract.
type BudgetContract struct {
	Tool     string
	Pointers []string
	Ceiling  int64
}

// BudgetResolution is the deterministic result of applying a budget contract.
type BudgetResolution struct {
	Decision  string          `json:"decision"`
	Pointer   string          `json:"pointer,omitempty"`
	Requested int64           `json:"requested,omitempty"`
	Effective int64           `json:"effective,omitempty"`
	Arguments json.RawMessage `json:"arguments"`
}

// ResolveBudget clamps the first explicitly configured integer pointer for tool.
// Invalid inputs and unsupported values fail closed without returning mutated bytes.
func ResolveBudget(tool string, arguments []byte, contracts []BudgetContract) (BudgetResolution, error) {
	original := append(json.RawMessage(nil), arguments...)
	contract, ok := budgetContract(tool, contracts)
	if !ok {
		return BudgetResolution{Decision: "pass", Arguments: original}, nil
	}
	if contract.Ceiling < 0 || len(contract.Pointers) == 0 {
		return BudgetResolution{Decision: "error", Arguments: original}, errors.New("invalid budget contract")
	}
	dec := json.NewDecoder(bytes.NewReader(arguments))
	dec.UseNumber()
	var root any
	if err := dec.Decode(&root); err != nil || dec.More() {
		if err == nil {
			err = errors.New("multiple JSON values")
		}
		return BudgetResolution{Decision: "error", Arguments: original}, fmt.Errorf("decode arguments: %w", err)
	}
	for _, pointer := range contract.Pointers {
		parent, key, value, found, err := resolvePointer(root, pointer)
		if err != nil {
			return BudgetResolution{Decision: "error", Arguments: original}, err
		}
		if !found {
			continue
		}
		n, ok := value.(json.Number)
		if !ok {
			return BudgetResolution{Decision: "error", Arguments: original}, fmt.Errorf("budget value at %s is not an integer", pointer)
		}
		requested, err := strconv.ParseInt(n.String(), 10, 64)
		if err != nil || requested < 0 {
			return BudgetResolution{Decision: "error", Arguments: original}, fmt.Errorf("budget value at %s is not a non-negative int64", pointer)
		}
		if requested <= contract.Ceiling {
			return BudgetResolution{Decision: "pass", Pointer: pointer, Requested: requested, Effective: requested, Arguments: original}, nil
		}
		parent[key] = json.Number(strconv.FormatInt(contract.Ceiling, 10))
		effective, err := json.Marshal(root)
		if err != nil {
			return BudgetResolution{Decision: "error", Arguments: original}, err
		}
		return BudgetResolution{Decision: "clamp", Pointer: pointer, Requested: requested, Effective: contract.Ceiling, Arguments: effective}, nil
	}
	return BudgetResolution{Decision: "pass", Arguments: original}, nil
}

func budgetContract(tool string, contracts []BudgetContract) (BudgetContract, bool) {
	for _, c := range contracts {
		if c.Tool == tool {
			return c, true
		}
	}
	return BudgetContract{}, false
}

func resolvePointer(root any, pointer string) (map[string]any, string, any, bool, error) {
	if pointer == "" || !strings.HasPrefix(pointer, "/") {
		return nil, "", nil, false, fmt.Errorf("invalid JSON pointer %q", pointer)
	}
	parts := strings.Split(pointer[1:], "/")
	current := root
	for i, raw := range parts {
		part, err := decodePointerToken(raw)
		if err != nil {
			return nil, "", nil, false, err
		}
		obj, ok := current.(map[string]any)
		if !ok {
			return nil, "", nil, false, nil
		}
		value, found := obj[part]
		if !found {
			return nil, "", nil, false, nil
		}
		if i == len(parts)-1 {
			return obj, part, value, true, nil
		}
		current = value
	}
	return nil, "", nil, false, nil
}

func decodePointerToken(s string) (string, error) {
	var b strings.Builder
	for i := 0; i < len(s); i++ {
		if s[i] != '~' {
			b.WriteByte(s[i])
			continue
		}
		if i+1 >= len(s) {
			return "", errors.New("invalid JSON pointer escape")
		}
		i++
		switch s[i] {
		case '0':
			b.WriteByte('~')
		case '1':
			b.WriteByte('/')
		default:
			return "", errors.New("invalid JSON pointer escape")
		}
	}
	return b.String(), nil
}
