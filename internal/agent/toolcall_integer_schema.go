// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from vLLM's schema-guided integer coercion; see third_party/vllm-integer-args/NOTICE.

package agent

import (
	"encoding/json"
	"math/big"
	"strconv"
	"strings"
)

func normalizeIntegerSchemaArguments(comp *Completion, tools []ToolDef) {
	for i := range comp.Message.ToolCalls {
		call := &comp.Message.ToolCalls[i]
		for _, tool := range tools {
			if tool.Function.Name != call.Function.Name {
				continue
			}
			var schema struct {
				Properties map[string]json.RawMessage `json:"properties"`
			}
			if json.Unmarshal(tool.Function.Parameters, &schema) != nil || len(schema.Properties) == 0 {
				break
			}
			var args map[string]json.RawMessage
			if json.Unmarshal([]byte(call.Function.Arguments), &args) != nil {
				break
			}
			changed := false
			for name, property := range schema.Properties {
				var field struct {
					Type string `json:"type"`
				}
				if json.Unmarshal(property, &field) != nil || field.Type != "integer" {
					continue
				}
				var value string
				if json.Unmarshal(args[name], &value) != nil {
					continue
				}
				if integer, ok := exactIntegerArgument(value); ok {
					args[name] = json.RawMessage(integer)
					changed = true
				}
			}
			if changed {
				if encoded, err := json.Marshal(args); err == nil {
					call.Function.Arguments = string(encoded)
				}
			}
			break
		}
	}
}

func exactIntegerArgument(value string) (string, bool) {
	value = strings.TrimSpace(value)
	if len(value) == 0 || len(value) > 256 || (value[0] != '-' && (value[0] < '0' || value[0] > '9')) || !json.Valid([]byte(value)) {
		return "", false
	}
	// Bound exponent expansion before exact arithmetic; float64 would silently
	// round large whole numbers and an unbounded exponent could exhaust memory.
	if at := strings.IndexAny(value, "eE"); at >= 0 {
		exponent, err := strconv.ParseInt(value[at+1:], 10, 16)
		if err != nil || exponent < -256 || exponent > 256 {
			return "", false
		}
	}
	number, ok := new(big.Rat).SetString(value)
	if !ok || !number.IsInt() {
		return "", false
	}
	return number.Num().String(), true
}
