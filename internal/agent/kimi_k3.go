package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// isKimiK3Model recognizes Moonshot's native API ID and the short Kimi Code ID.
// Provider-prefixed catalog IDs are accepted so an OpenAI-compatible router can use
// the same contract without a second model registry.
func isKimiK3Model(modelID string) bool {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	if i := strings.LastIndexByte(modelID, '/'); i >= 0 {
		modelID = modelID[i+1:]
	}
	return modelID == "kimi-k3" || modelID == "k3"
}

func normalizeKimiK3Request(raw json.RawMessage) (json.RawMessage, error) {
	body := make(map[string]any)
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, fmt.Errorf("kimi k3 request body is empty")
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("kimi k3 request body: %w", err)
	}
	if effort, ok := body["reasoning_effort"]; ok && effort != "max" {
		return nil, fmt.Errorf("kimi k3 reasoning_effort must be %q, got %v", "max", effort)
	}
	body["reasoning_effort"] = "max"
	delete(body, "temperature")
	delete(body, "top_p")
	delete(body, "thinking")
	return json.Marshal(body)
}
