package agent

import (
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// positiveTopK normalizes a per-request top-k for the wire: nil (unset) and any
// non-positive value (the planner's "0 => no truncation" convention) both omit the
// field, because the native top_k providers (Anthropic, Gemini) require a positive
// integer and reject 0/negative. A positive k is forwarded as-is. Returning a
// *int keeps the request struct's `omitempty` working — a nil result drops the key.
func positiveTopK(k *int) *int {
	if k == nil || *k <= 0 {
		return nil
	}
	v := *k
	return &v
}

// responsesText maps the chat-style `response_format` carrier onto the Responses
// API's `text.format` shape. The two OpenAI surfaces spell structured output
// differently: chat nests the schema under `response_format.json_schema.{name,
// strict,schema}`, while Responses flattens it to `text.format.{type:"json_schema",
// name,strict,schema}`. So a `json_schema` carrier is rewritten with its inner
// `json_schema` object's members hoisted up alongside `type`; a `json_object` /
// `text` carrier (no inner wrapper) passes through verbatim. An unset or
// unparseable carrier returns nil so the `omitempty` drops `text` entirely —
// the body is then byte-for-byte the pre-seam request.
func responsesText(rf json.RawMessage) *openAIResponsesText {
	if len(rf) == 0 {
		return nil
	}
	var carrier map[string]json.RawMessage
	if err := json.Unmarshal(rf, &carrier); err != nil {
		return nil
	}
	typ := carrier["type"]
	if len(typ) == 0 {
		return nil
	}
	// json_schema: hoist the inner json_schema members up beside `type`.
	if string(typ) == `"json_schema"` {
		var inner map[string]json.RawMessage
		if err := json.Unmarshal(carrier["json_schema"], &inner); err != nil || inner == nil {
			// Malformed/absent inner wrapper — forward the carrier as-is rather
			// than drop it, so an odd shape still reaches the provider's validator.
			return &openAIResponsesText{Format: rf}
		}
		flat := make(map[string]json.RawMessage, len(inner)+1)
		flat["type"] = typ
		for k, v := range inner {
			flat[k] = v
		}
		format, err := json.Marshal(flat)
		if err != nil {
			return &openAIResponsesText{Format: rf}
		}
		return &openAIResponsesText{Format: format}
	}
	// json_object / text and any other typed shape: pass the carrier through verbatim.
	return &openAIResponsesText{Format: rf}
}

// jsonObjectOrFallback decodes s and returns it only when it is a JSON object; otherwise
// it returns fallback. Shared by the carriers that must hand the gateway a map (a decoded
// object when the model emitted one, a labeled wrapper otherwise).
func jsonObjectOrFallback(s string, fallback any) any {
	var v any
	if err := json.Unmarshal([]byte(s), &v); err == nil {
		if _, ok := v.(map[string]any); ok {
			return v
		}
	}
	return fallback
}

func jsonObjectOrRaw(raw string) any {
	if strings.TrimSpace(raw) == "" {
		return map[string]any{}
	}
	return jsonObjectOrFallback(raw, map[string]any{"_raw": raw})
}

func responseObject(content string) any {
	return jsonObjectOrFallback(content, map[string]any{"content": content})
}

func serviceTierWire(provider Provider, mode modelroute.ServiceMode) string {
	if mode != modelroute.ServiceModeFast {
		return ""
	}
	if provider == ProviderAnthropic {
		return "auto"
	}
	return "priority"
}

func parseServiceTier(provider Provider, wire string) modelroute.ServiceMode {
	switch wire {
	case "priority", "fast":
		return modelroute.ServiceModeFast
	case "auto":
		if provider == ProviderAnthropic {
			return modelroute.ServiceModeFast
		}
	case "default", "standard":
		return modelroute.ServiceModeStandard
	}
	return modelroute.ServiceModeUnknown
}

// mapSchemaTypes walks a decoded JSON Schema value and rewrites every "type" field's
// string value through conv (in place), recursing into nested maps and arrays.
func mapSchemaTypes(v any, conv func(string) string) any {
	switch x := v.(type) {
	case map[string]any:
		for k, val := range x {
			if k == "type" {
				if s, ok := val.(string); ok {
					x[k] = conv(s)
					continue
				}
			}
			x[k] = mapSchemaTypes(val, conv)
		}
	case []any:
		for i, val := range x {
			x[i] = mapSchemaTypes(val, conv)
		}
	}
	return v
}

func uppercaseSchemaTypes(v any) any { return mapSchemaTypes(v, strings.ToUpper) }
