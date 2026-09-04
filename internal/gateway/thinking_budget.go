package gateway

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/agentopt"
)

// DynamicThinkingConfig configures dynamic turn-level thinking budget modulation (#11185).
// When enabled, it modulates thinking token budgets or reasoning_effort without modifying
// the system prompt, tool definitions, or message prefixes, preserving provider KV caches.
type DynamicThinkingConfig struct {
	Enabled        bool
	DefaultTier    agentopt.EffortTier
	Provider       string // "anthropic", "openai", "gemini"
	PreservePrefix bool
}

// ModulateAnthropicThinking modulates the thinking block in an Anthropic request body (raw bytes)
// according to the effort tier, preserving the prompt prefix.
// If tier is EffortNone or EffortLow: clamps/injects budget_tokens or disables thinking.
// Returns the modified body and whether it was modified.
func ModulateAnthropicThinking(raw []byte, tier agentopt.EffortTier) ([]byte, bool) {
	if len(raw) == 0 {
		return raw, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, false
	}

	budget := tier.ThinkingBudget()
	var thinkingVal map[string]any

	if rawThinking, ok := obj["thinking"]; ok && len(rawThinking) > 0 {
		_ = json.Unmarshal(rawThinking, &thinkingVal)
	}

	if thinkingVal == nil {
		thinkingVal = make(map[string]any)
	}

	if tier == agentopt.EffortNone {
		// When reasoning effort is none: disable or clamp budget_tokens: 0
		thinkingVal["type"] = "disabled"
		delete(thinkingVal, "budget_tokens")
	} else {
		thinkingVal["type"] = "enabled"
		thinkingVal["budget_tokens"] = budget
	}

	thBytes, err := json.Marshal(thinkingVal)
	if err != nil {
		return raw, false
	}

	// Splice or update "thinking" in the raw JSON
	if _, ok := obj["thinking"]; ok {
		// Replace "thinking" value in place
		out, ok := spliceTopLevelKey(raw, "thinking", thBytes)
		if ok {
			return out, true
		}
	}

	// Insert "thinking" at top level
	obj["thinking"] = thBytes
	newBytes, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return newBytes, true
}

// ModulateOpenAIThinking modulates reasoning_effort in an OpenAI chat completion body or map.
func ModulateOpenAIThinking(raw []byte, tier agentopt.EffortTier) ([]byte, bool) {
	if len(raw) == 0 {
		return raw, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, false
	}

	effortStr := string(tier)
	if tier == agentopt.EffortNone {
		effortStr = "low" // OpenAI o1/o3-mini supports low, medium, high
	}

	valBytes, err := json.Marshal(effortStr)
	if err != nil {
		return raw, false
	}

	if _, ok := obj["reasoning_effort"]; ok {
		if out, ok := spliceTopLevelKey(raw, "reasoning_effort", valBytes); ok {
			return out, true
		}
	}

	obj["reasoning_effort"] = valBytes
	newBytes, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return newBytes, true
}

// ModulateGeminiThinking modulates thinkingConfig in a Gemini generateContent body.
func ModulateGeminiThinking(raw []byte, tier agentopt.EffortTier) ([]byte, bool) {
	if len(raw) == 0 {
		return raw, false
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(raw, &obj); err != nil {
		return raw, false
	}

	budget := tier.ThinkingBudget()
	var genConfig map[string]any
	if rawGC, ok := obj["generationConfig"]; ok && len(rawGC) > 0 {
		_ = json.Unmarshal(rawGC, &genConfig)
	}
	if genConfig == nil {
		genConfig = make(map[string]any)
	}

	var tc map[string]any
	if rawTC, ok := genConfig["thinkingConfig"].(map[string]any); ok {
		tc = rawTC
	} else {
		tc = make(map[string]any)
	}

	if tier == agentopt.EffortNone {
		tc["thinkingBudget"] = 0
	} else {
		tc["thinkingBudget"] = budget
	}
	genConfig["thinkingConfig"] = tc

	gcBytes, err := json.Marshal(genConfig)
	if err != nil {
		return raw, false
	}

	obj["generationConfig"] = gcBytes
	newBytes, err := json.Marshal(obj)
	if err != nil {
		return raw, false
	}
	return newBytes, true
}

// spliceTopLevelKey replaces a top-level key's value in raw JSON bytes, preserving prefix bytes.
func spliceTopLevelKey(raw []byte, keyName string, newValue []byte) ([]byte, bool) {
	key := []byte(fmt.Sprintf("%q", keyName))
	ki := bytes.Index(raw, key)
	if ki < 0 {
		return nil, false
	}
	i := ki + len(key)
	// Skip whitespace and ':'
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) || raw[i] != ':' {
		return nil, false
	}
	i++
	for i < len(raw) && (raw[i] == ' ' || raw[i] == '\t' || raw[i] == '\n' || raw[i] == '\r') {
		i++
	}
	if i >= len(raw) {
		return nil, false
	}

	start := i
	// Determine end of JSON value
	end := findJSONValueEnd(raw, start)
	if end < 0 {
		return nil, false
	}

	var b bytes.Buffer
	b.Grow(len(raw) - (end - start) + len(newValue))
	b.Write(raw[:start])
	b.Write(newValue)
	b.Write(raw[end:])
	return b.Bytes(), true
}

func findJSONValueEnd(raw []byte, start int) int {
	if start >= len(raw) {
		return -1
	}
	switch raw[start] {
	case '{', '[':
		open := raw[start]
		close := byte('}')
		if open == '[' {
			close = ']'
		}
		depth := 0
		inString := false
		var esc bool
		for i := start; i < len(raw); i++ {
			c := raw[i]
			if inString {
				if esc {
					esc = false
				} else if c == '\\' {
					esc = true
				} else if c == '"' {
					inString = false
				}
				continue
			}
			if c == '"' {
				inString = true
				continue
			}
			if c == open {
				depth++
			} else if c == close {
				depth--
				if depth == 0 {
					return i + 1
				}
			}
		}
		return -1
	case '"':
		var esc bool
		for i := start + 1; i < len(raw); i++ {
			c := raw[i]
			if esc {
				esc = false
			} else if c == '\\' {
				esc = true
			} else if c == '"' {
				return i + 1
			}
		}
		return -1
	default:
		// Number, boolean, null: terminates on comma, closing brace, bracket, or whitespace
		for i := start; i < len(raw); i++ {
			c := raw[i]
			if c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n' || c == '\r' {
				return i
			}
		}
		return len(raw)
	}
}
