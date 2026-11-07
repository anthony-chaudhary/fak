package agent

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// canonicalKimiModelID reduces a model ID to its bare form: lowercased, trimmed,
// and stripped of any provider/catalog prefix ("moonshotai/kimi-k3" -> "kimi-k3").
func canonicalKimiModelID(modelID string) string {
	modelID = strings.ToLower(strings.TrimSpace(modelID))
	if i := strings.LastIndexByte(modelID, '/'); i >= 0 {
		modelID = modelID[i+1:]
	}
	return modelID
}

// isKimiK3Model recognizes Moonshot's native API ID and the short Kimi Code ID.
// Provider-prefixed catalog IDs are accepted so an OpenAI-compatible router can use
// the same contract without a second model registry. Matching is exact against this
// closed set: K3.1 spellings are deliberately NOT K3 and belong to isKimiK31Model.
func isKimiK3Model(modelID string) bool {
	id := canonicalKimiModelID(modelID)
	return id == "kimi-k3" || id == "k3"
}

// isKimiK31Model recognizes the Kimi K3.1 Pro API family: Moonshot's native API
// IDs and the short Kimi Code IDs. "-highspeed" is a serving mode of the same
// weights (faster, pricier), not a different model, so it rides the same wire
// contract. Matching is exact against this closed set; a bare "kimi-k3.1" is not
// a documented API ID and is intentionally not recognized.
func isKimiK31Model(modelID string) bool {
	switch id := canonicalKimiModelID(modelID); id {
	case "kimi-k3.1-pro", "kimi-k3.1-pro-highspeed", "k3.1-pro", "k3.1-pro-highspeed":
		return true
	}
	return false
}

// normalizeKimiK3Request applies the Kimi K3/K3.1 wire contract to a marshaled
// OpenAI-compatible chat request. Moonshot fixes sampling server-side for both
// families, so temperature/top_p/thinking are stripped. Effort differs per family:
// K3 accepts only reasoning_effort=max; K3.1 accepts low|high|max and defaults to
// max when the field is absent.
func normalizeKimiK3Request(modelID string, raw json.RawMessage) (json.RawMessage, error) {
	body := make(map[string]any)
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, fmt.Errorf("kimi k3 request body is empty")
	}
	if err := json.Unmarshal(raw, &body); err != nil {
		return nil, fmt.Errorf("kimi k3 request body: %w", err)
	}
	if isKimiK31Model(modelID) {
		effort, ok := body["reasoning_effort"]
		if ok && effort != "low" && effort != "high" && effort != "max" {
			return nil, fmt.Errorf("kimi k3.1 reasoning_effort must be \"low\", \"high\" or \"max\", got %v", effort)
		}
		if !ok {
			body["reasoning_effort"] = "max"
		}
	} else {
		if effort, ok := body["reasoning_effort"]; ok && effort != "max" {
			return nil, fmt.Errorf("kimi k3 reasoning_effort must be %q, got %v", "max", effort)
		}
		body["reasoning_effort"] = "max"
	}
	delete(body, "temperature")
	delete(body, "top_p")
	delete(body, "thinking")
	return json.Marshal(body)
}
