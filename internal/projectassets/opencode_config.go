package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultOpenCodeBaseURL is the standard local gateway address used by fak serve.
const DefaultOpenCodeBaseURL = "http://127.0.0.1:8080/v1"

// DefaultOpenCodeModelID is the default served model identifier for local inference.
const DefaultOpenCodeModelID = "fak-local"

// OpenCodeProviderConfig defines the OpenAI-compatible provider structure for OpenCode.
type OpenCodeProviderConfig struct {
	NPM     string                 `json:"npm"`
	Name    string                 `json:"name"`
	Options map[string]interface{} `json:"options"`
	Models  map[string]interface{} `json:"models"`
}

// GenerateOpenCodeConfig creates a standalone opencode.json configuration targeting a fak serve gateway.
func GenerateOpenCodeConfig(baseURL, modelID string) ([]byte, error) {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = DefaultOpenCodeBaseURL
	}
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = DefaultOpenCodeModelID
	}

	cfg := map[string]interface{}{
		"$schema":  "https://opencode.ai/config.json",
		"snapshot": false,
		"provider": map[string]interface{}{
			"fak": map[string]interface{}{
				"npm":  "@ai-sdk/openai-compatible",
				"name": "fak (kernel-adjudicated)",
				"options": map[string]interface{}{
					"baseURL": baseURL,
				},
				"models": map[string]interface{}{
					modelID: map[string]interface{}{
						"name": modelID,
					},
				},
			},
		},
	}
	return json.MarshalIndent(cfg, "", "  ")
}

// EnsureOpenCodeProviderConfig ensures opencode.json in root contains the "fak" provider pointing
// to baseURL with modelID, while preserving all existing instructions, permissions, and agent configurations.
// It also enforces "snapshot": false to avoid workspace bloat.
func EnsureOpenCodeProviderConfig(root, baseURL, modelID string) (bool, error) {
	if root == "" {
		root = "."
	}
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = DefaultOpenCodeBaseURL
	}
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = DefaultOpenCodeModelID
	}

	configPath := filepath.Join(root, "opencode.json")
	var raw map[string]interface{}

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			// Create fresh config
			out, genErr := GenerateOpenCodeConfig(baseURL, modelID)
			if genErr != nil {
				return false, fmt.Errorf("generate opencode.json: %w", genErr)
			}
			if writeErr := os.WriteFile(configPath, append(out, '\n'), 0644); writeErr != nil {
				return false, fmt.Errorf("write opencode.json: %w", writeErr)
			}
			return true, nil
		}
		return false, fmt.Errorf("read opencode.json: %w", err)
	}

	if unmarshalErr := json.Unmarshal(data, &raw); unmarshalErr != nil {
		return false, fmt.Errorf("parse existing opencode.json: %w", unmarshalErr)
	}
	if raw == nil {
		raw = make(map[string]interface{})
	}

	modified := false

	// Enforce snapshot: false
	if snap, ok := raw["snapshot"]; !ok || snap != false {
		raw["snapshot"] = false
		modified = true
	}

	// Ensure $schema is set if missing
	if _, ok := raw["$schema"]; !ok {
		raw["$schema"] = "https://opencode.ai/config.json"
		modified = true
	}

	// Retrieve or initialize provider map
	var providerMap map[string]interface{}
	if existingProv, ok := raw["provider"]; ok {
		if pMap, ok := existingProv.(map[string]interface{}); ok {
			providerMap = pMap
		}
	}
	if providerMap == nil {
		providerMap = make(map[string]interface{})
		raw["provider"] = providerMap
		modified = true
	}

	// Build target fak provider map
	targetFak := map[string]interface{}{
		"npm":  "@ai-sdk/openai-compatible",
		"name": "fak (kernel-adjudicated)",
		"options": map[string]interface{}{
			"baseURL": baseURL,
		},
		"models": map[string]interface{}{
			modelID: map[string]interface{}{
				"name": modelID,
			},
		},
	}

	// Check if fak provider needs updating
	existingFak, hasFak := providerMap["fak"]
	if !hasFak {
		providerMap["fak"] = targetFak
		modified = true
	} else {
		existingBytes, _ := json.Marshal(existingFak)
		targetBytes, _ := json.Marshal(targetFak)
		if string(existingBytes) != string(targetBytes) {
			providerMap["fak"] = targetFak
			modified = true
		}
	}

	if modified {
		out, marshalErr := json.MarshalIndent(raw, "", "  ")
		if marshalErr != nil {
			return false, fmt.Errorf("serialize opencode.json: %w", marshalErr)
		}
		if writeErr := os.WriteFile(configPath, append(out, '\n'), 0644); writeErr != nil {
			return false, fmt.Errorf("write opencode.json: %w", writeErr)
		}
	}

	return modified, nil
}
