// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from vLLM's named tool_choice structured-output schema; see third_party/vllm-tool-choice-schema/NOTICE.

package agent

import "encoding/json"

// newNativeForcedToolConstraint masks a forced tool call to its envelope
// schema, as vLLM's get_json_schema_from_tools constrains a named tool_choice.
// It returns nil, keeping the unmasked forced-tool path, for the Write artifact
// mode, a tool without a parameters object, a schema outside the native mask
// subset, or a request whose decode the mask cannot serve.
func (p *InKernelPlanner) newNativeForcedToolConstraint(sp SampleParams, tools []ToolDef) *nativeStructuredConstraint {
	if sp.NativeInferenceReceipt || sp.RawPrompt || p.dflash2 != nil {
		return nil
	}
	name := inKernelEffectiveToolName(sp.ToolChoice, tools)
	if name == "" || name == "Write" {
		return nil
	}
	schema, ok := inKernelForcedToolEnvelopeSchema(name, tools)
	if !ok {
		return nil
	}
	constraint, err := p.newNativeJSONSchemaConstraint(schema, nil)
	if err != nil {
		return nil
	}
	constraint.forcedTool = true
	return constraint
}

// inKernelForcedToolEnvelopeSchema is the object the forced-tool instruction
// asks for: the tool name as a const beside the tool's parameters schema.
func inKernelForcedToolEnvelopeSchema(name string, tools []ToolDef) (json.RawMessage, bool) {
	for _, tool := range tools {
		if tool.Function.Name != name {
			continue
		}
		var parameters map[string]json.RawMessage
		if json.Unmarshal(tool.Function.Parameters, &parameters) != nil || parameters == nil {
			return nil, false
		}
		// $schema only names the dialect; nested in the envelope the mask refuses it.
		delete(parameters, "$schema")
		schema, err := json.Marshal(map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":      map[string]string{"type": "string", "const": name},
				"arguments": parameters,
			},
			"required":             []string{"name", "arguments"},
			"additionalProperties": false,
		})
		return schema, err == nil
	}
	return nil, false
}
