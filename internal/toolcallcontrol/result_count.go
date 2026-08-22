package toolcallcontrol

import (
	"bytes"
	"encoding/json"
	"io"
	"strconv"
	"strings"
)

// ResultCountContract declares one argument that is safe to reduce for a read tool.
type ResultCountContract struct {
	Tool            string `json:"tool"`
	ArgumentPointer string `json:"argument_pointer"`
	Minimum         int64  `json:"minimum"`
	Maximum         int64  `json:"maximum"`
	ReductionSafe   bool   `json:"reduction_safe"`
}

// ResultCountChange records the deterministic mutation applied to a tool call.
type ResultCountChange struct {
	Path      string            `json:"path"`
	From      int64             `json:"from"`
	To        int64             `json:"to"`
	Dimension ResponseDimension `json:"dimension"`
}

// ResultCountVerdict is the result of applying one count contract.
type ResultCountVerdict struct {
	Decision        string              `json:"decision"`
	Reason          string              `json:"reason"`
	EffectiveArgs   json.RawMessage     `json:"effective_args"`
	Changes         []ResultCountChange `json:"changes,omitempty"`
	ModelRoundTrips int64               `json:"model_round_trips"`
}

// ClampResultCount lowers an explicitly present integer count at a contracted JSON pointer.
// Optimization fails open: unsupported or malformed calls retain their original arguments.
func ClampResultCount(mode Mode, tool string, args json.RawMessage, contract ResultCountContract, proposed int64, exempt bool) ResultCountVerdict {
	pass := func(reason string) ResultCountVerdict {
		return ResultCountVerdict{Decision: "pass", Reason: reason, EffectiveArgs: append(json.RawMessage(nil), args...)}
	}
	if mode == ModeOff {
		return pass("mode_off")
	}
	if tool != contract.Tool || contract.Tool == "" {
		return pass("contract_mismatch")
	}
	if !contract.ReductionSafe {
		return pass("reduction_not_contracted")
	}
	if exempt {
		return pass("exempt")
	}
	if contract.Minimum < 0 || contract.Maximum < contract.Minimum || proposed < contract.Minimum || proposed > contract.Maximum {
		return pass("invalid_contract")
	}

	var root any
	decoder := json.NewDecoder(bytes.NewReader(args))
	decoder.UseNumber()
	if err := decoder.Decode(&root); err != nil {
		return pass("invalid_json")
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return pass("invalid_json")
	}
	parent, key, found := jsonPointerParent(root, contract.ArgumentPointer)
	if !found {
		return pass("argument_missing")
	}
	requested, ok := exactNonnegativeInt(parent[key])
	if !ok || requested < contract.Minimum || requested > contract.Maximum {
		return pass("argument_outside_contract")
	}
	if proposed >= requested {
		return pass("within_budget")
	}
	if mode == ModeShadow {
		return pass("observe_only")
	}
	parent[key] = proposed
	effective, err := json.Marshal(root)
	if err != nil {
		return pass("rewrite_failed")
	}
	return ResultCountVerdict{
		Decision:      "clamp",
		Reason:        "requested_items_above_policy_maximum",
		EffectiveArgs: effective,
		Changes: []ResultCountChange{{
			Path: contract.ArgumentPointer, From: requested, To: proposed, Dimension: ResponseItems,
		}},
	}
}

func jsonPointerParent(root any, pointer string) (map[string]any, string, bool) {
	if pointer == "" || !strings.HasPrefix(pointer, "/") {
		return nil, "", false
	}
	parts := strings.Split(pointer[1:], "/")
	current := root
	for _, encoded := range parts[:len(parts)-1] {
		key, ok := decodePointerToken(encoded)
		if !ok {
			return nil, "", false
		}
		object, ok := current.(map[string]any)
		if !ok {
			return nil, "", false
		}
		current, ok = object[key]
		if !ok {
			return nil, "", false
		}
	}
	key, ok := decodePointerToken(parts[len(parts)-1])
	if !ok {
		return nil, "", false
	}
	parent, ok := current.(map[string]any)
	if !ok {
		return nil, "", false
	}
	_, found := parent[key]
	return parent, key, found
}

func decodePointerToken(token string) (string, bool) {
	var decoded strings.Builder
	for i := 0; i < len(token); i++ {
		if token[i] != '~' {
			decoded.WriteByte(token[i])
			continue
		}
		if i+1 >= len(token) {
			return "", false
		}
		i++
		switch token[i] {
		case '0':
			decoded.WriteByte('~')
		case '1':
			decoded.WriteByte('/')
		default:
			return "", false
		}
	}
	return decoded.String(), true
}

func exactNonnegativeInt(value any) (int64, bool) {
	number, ok := value.(json.Number)
	if !ok || strings.ContainsAny(number.String(), ".eE+") {
		return 0, false
	}
	parsed, err := strconv.ParseInt(number.String(), 10, 64)
	if err != nil || parsed < 0 {
		return 0, false
	}
	return parsed, true
}
