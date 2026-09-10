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

// DefaultOpenCodeHaloModelID is the primary served model identifier for AMD Strix Halo local inference.
const DefaultOpenCodeHaloModelID = "qwen-2.5-coder-32b-instruct"

// ResolveDynamicHaloModel dynamically resolves the default Halo model identifier.
// It checks FAK_HALO_MODEL, FAK_MODEL, opencode.json in root (inspecting top-level model,
// provider.fak models, and model tier profiles), falling back to DefaultOpenCodeHaloModelID.
func ResolveDynamicHaloModel(root string) string {
	if m := os.Getenv("FAK_HALO_MODEL"); strings.TrimSpace(m) != "" {
		return CleanModelPrefix(m)
	}
	if m := os.Getenv("FAK_MODEL"); strings.TrimSpace(m) != "" {
		return CleanModelPrefix(m)
	}
	if root == "" {
		root = "."
	}
	cfgPath := filepath.Join(root, "opencode.json")
	if data, err := os.ReadFile(cfgPath); err == nil {
		var raw struct {
			Model             string `json:"model"`
			ModelTierProfiles struct {
				Fast struct {
					ModelID string `json:"model_id"`
				} `json:"fast"`
				Balanced struct {
					ModelID string `json:"model_id"`
				} `json:"balanced"`
			} `json:"model_tier_profiles"`
			Provider map[string]struct {
				Models map[string]interface{} `json:"models"`
			} `json:"provider"`
		}
		if json.Unmarshal(data, &raw) == nil {
			if raw.ModelTierProfiles.Fast.ModelID != "" {
				return CleanModelPrefix(raw.ModelTierProfiles.Fast.ModelID)
			}
			if raw.ModelTierProfiles.Balanced.ModelID != "" {
				return CleanModelPrefix(raw.ModelTierProfiles.Balanced.ModelID)
			}
			if raw.Model != "" {
				return CleanModelPrefix(raw.Model)
			}
			if fakProv, ok := raw.Provider["fak"]; ok && len(fakProv.Models) > 0 {
				for k := range fakProv.Models {
					if strings.Contains(strings.ToLower(k), "coder") || strings.Contains(strings.ToLower(k), "qwen") {
						return k
					}
				}
				for k := range fakProv.Models {
					return k
				}
			}
		}
	}
	return DefaultOpenCodeHaloModelID
}

// CleanModelPrefix strips provider prefix (e.g. "fak/qwen-2.5-coder-32b-instruct" -> "qwen-2.5-coder-32b-instruct").
func CleanModelPrefix(m string) string {
	m = strings.TrimSpace(m)
	if idx := strings.Index(m, "/"); idx != -1 {
		return m[idx+1:]
	}
	return m
}

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
	if modelID == "" || modelID == DefaultOpenCodeModelID {
		modelID = ResolveDynamicHaloModel(".")
	}

	modelsMap := map[string]interface{}{
		modelID: map[string]interface{}{
			"name": modelID,
		},
	}
	if modelID != DefaultOpenCodeHaloModelID {
		modelsMap[DefaultOpenCodeHaloModelID] = map[string]interface{}{
			"name": "Qwen 2.5 Coder 32B Instruct (Local Halo)",
			"limit": map[string]interface{}{
				"context": float64(131072),
				"output":  float64(8192),
			},
		}
	}

	cfg := map[string]interface{}{
		"$schema":  "https://opencode.ai/config.json",
		"snapshot": false,
		"model":    "fak/" + modelID,
		"provider": map[string]interface{}{
			"fak": map[string]interface{}{
				"npm":  "@ai-sdk/openai-compatible",
				"name": "fak (kernel-adjudicated)",
				"options": map[string]interface{}{
					"baseURL": baseURL,
				},
				"models": modelsMap,
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
	if modelID == "" || modelID == DefaultOpenCodeModelID {
		modelID = ResolveDynamicHaloModel(root)
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

	// Ensure top-level model is set to the main tier model if missing or pointing to fak provider
	if curModel, ok := raw["model"].(string); !ok || curModel == "" || strings.HasPrefix(curModel, "fak/") {
		if curModel != "fak/"+modelID {
			raw["model"] = "fak/" + modelID
			modified = true
		}
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

	// Retrieve existing models in provider "fak" if present
	existingModels := make(map[string]interface{})
	if existingFak, ok := providerMap["fak"].(map[string]interface{}); ok {
		if em, ok := existingFak["models"].(map[string]interface{}); ok {
			for k, v := range em {
				existingModels[k] = v
			}
		}
	}

	existingModels[modelID] = map[string]interface{}{
		"name": modelID,
	}
	if _, ok := existingModels[DefaultOpenCodeHaloModelID]; !ok {
		existingModels[DefaultOpenCodeHaloModelID] = map[string]interface{}{
			"name": "Qwen 2.5 Coder 32B Instruct (Local Halo)",
			"limit": map[string]interface{}{
				"context": float64(131072),
				"output":  float64(8192),
			},
		}
	}

	// Build target fak provider map
	targetFak := map[string]interface{}{
		"npm":  "@ai-sdk/openai-compatible",
		"name": "fak (kernel-adjudicated)",
		"options": map[string]interface{}{
			"baseURL": baseURL,
		},
		"models": existingModels,
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

	if existingAgents, ok := raw["agent"].(map[string]interface{}); ok {
		targetModel := "fak/" + modelID
		if fastAgent, ok := existingAgents["agent.tier.fast"].(map[string]interface{}); ok {
			curModel, _ := fastAgent["model"].(string)
			if curModel == "" || curModel == DefaultOpenCodeHaloModelID || curModel == modelID || curModel == "fak/"+DefaultOpenCodeHaloModelID {
				fastAgent["model"] = targetModel
				modified = true
			}
		}
		if balancedAgent, ok := existingAgents["agent.tier.balanced"].(map[string]interface{}); ok {
			curModel, _ := balancedAgent["model"].(string)
			if curModel == "" || curModel == DefaultOpenCodeHaloModelID || curModel == modelID || curModel == "fak/"+DefaultOpenCodeHaloModelID {
				balancedAgent["model"] = targetModel
				modified = true
			}
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
