package projectassets

import "strings"

// PiSamplingParams returns the samplingParams the fak Pi writers own for a model,
// or nil when fak leaves the model's sampling to Pi and the server. Qwen 3.8 is
// served without thinking, so it takes the vendor's non-thinking coding profile;
// a request that names no sampling lets a server default decide, and a greedy
// default sends a quantized Qwen into repeated identical tool calls.
func PiSamplingParams(modelID string) map[string]interface{} {
	leaf := strings.ToLower(modelID)
	if i := strings.LastIndex(leaf, "/"); i >= 0 {
		leaf = leaf[i+1:]
	}
	if !strings.HasPrefix(leaf, "qwen3.8") && !strings.HasPrefix(leaf, "qwen38") {
		return nil
	}
	return map[string]interface{}{
		"temperature":      0.7,
		"top_p":            0.8,
		"top_k":            20,
		"min_p":            0,
		"presence_penalty": 1.5,
	}
}
