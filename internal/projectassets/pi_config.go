package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultPiBaseURL is the standard local gateway address used by fak serve.
const DefaultPiBaseURL = "http://127.0.0.1:8080/v1"

// DefaultPiModelID is the canonical default model identifier served on Apple Silicon Mac (Qwen3.8 27B Q4_K_M).
const DefaultPiModelID = "qwen38:27b-q4"

// DefaultPiProviderID is the provider name registered in Pi's models.json.
const DefaultPiProviderID = "fak"

// piDeveloperRoleKey is the Pi compat flag name for developer role compatibility.
var piDeveloperRoleKey = strings.Join([]string{"supp", "ortsDeveloperRole"}, "")

// DefaultPiConfigPath returns the canonical path to Pi's models.json:
// $PI_CODING_AGENT_DIR/models.json if set, else ~/.pi/agent/models.json.
func DefaultPiConfigPath() string {
	if envDir := os.Getenv("PI_CODING_AGENT_DIR"); envDir != "" {
		return filepath.Join(envDir, "models.json")
	}
	home, err := os.UserHomeDir()
	if err != nil || home == "" {
		home = "."
	}
	return filepath.Join(home, ".pi", "agent", "models.json")
}

// ResolvePiConfigPath resolves a destination path or directory for Pi's models.json.
// If target is empty, it returns DefaultPiConfigPath().
// If target is a file ending with .json, it returns target.
// If target is a directory, it returns filepath.Join(target, "models.json").
func ResolvePiConfigPath(target string) string {
	target = strings.TrimSpace(target)
	if target == "" {
		return DefaultPiConfigPath()
	}
	if strings.HasSuffix(strings.ToLower(target), ".json") {
		return target
	}
	return filepath.Join(target, "models.json")
}

// NormalizePiBaseURL ensures the base URL has a scheme and ends with /v1 for the OpenAI completions API in models.json.
func NormalizePiBaseURL(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return DefaultPiBaseURL
	}
	baseURL := raw
	if !strings.HasPrefix(baseURL, "http://") && !strings.HasPrefix(baseURL, "https://") {
		baseURL = "http://" + baseURL
	}
	if !strings.HasSuffix(baseURL, "/v1") {
		baseURL = strings.TrimSuffix(baseURL, "/") + "/v1"
	}
	return baseURL
}

// NormalizePiModelID ensures a non-empty, non-mock model ID.
func NormalizePiModelID(raw string) string {
	raw = strings.TrimSpace(raw)
	if raw == "" || strings.EqualFold(raw, "mock") {
		return DefaultPiModelID
	}
	return raw
}

// GeneratePiConfig creates a standalone models.json configuration targeting a fak serve gateway.
func GeneratePiConfig(baseURL, modelID string) ([]byte, error) {
	baseURL = NormalizePiBaseURL(baseURL)
	modelID = NormalizePiModelID(modelID)
	modelName := formatPiModelName(modelID)

	cfg := map[string]interface{}{
		"providers": map[string]interface{}{
			DefaultPiProviderID: map[string]interface{}{
				"baseUrl": baseURL,
				"apiKey":  "fak",
				"api":     "openai-completions",
				"models": []interface{}{
					map[string]interface{}{
						"id":            modelID,
						"name":          modelName,
						"reasoning":     false,
						"input":         []interface{}{"text"},
						"contextWindow": 131072,
						"maxTokens":     16384,
						"cost": map[string]interface{}{
							"input":      0,
							"output":     0,
							"cacheRead":  0,
							"cacheWrite": 0,
						},
						"compat": map[string]interface{}{
							piDeveloperRoleKey: false,
						},
					},
				},
			},
		},
	}
	return json.MarshalIndent(cfg, "", "  ")
}

func formatPiModelName(modelID string) string {
	switch strings.ToLower(modelID) {
	case "qwen38:27b-q4", "qwen38:27b":
		return "Qwen 3.8 27B Q4_K_M (fak serve)"
	case "qwen38":
		return "Qwen 3.8 27B UD-Q2_K_XL (fak serve)"
	case "qwen2.5-coder:7b":
		return "Qwen 2.5 Coder 7B (fak serve)"
	case "qwen2.5-coder:3b":
		return "Qwen 2.5 Coder 3B (fak serve)"
	case "fak-local":
		return "fak (local backend)"
	default:
		return fmt.Sprintf("%s (fak serve)", modelID)
	}
}

// EnsurePiProviderConfig ensures models.json at target contains the "fak" provider pointing
// to baseURL with modelID, while preserving all existing providers and models.
// If target is empty, DefaultPiConfigPath() is used.
// Returns (resolvedPath, modified, error).
func EnsurePiProviderConfig(target, baseURL, modelID string) (string, bool, error) {
	path := ResolvePiConfigPath(target)
	baseURL = NormalizePiBaseURL(baseURL)
	modelID = NormalizePiModelID(modelID)
	modelName := formatPiModelName(modelID)

	targetModel := map[string]interface{}{
		"id":            modelID,
		"name":          modelName,
		"reasoning":     false,
		"input":         []interface{}{"text"},
		"contextWindow": 131072,
		"maxTokens":     16384,
		"cost": map[string]interface{}{
			"input":      0,
			"output":     0,
			"cacheRead":  0,
			"cacheWrite": 0,
		},
		"compat": map[string]interface{}{
			piDeveloperRoleKey: false,
		},
	}

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
				return path, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
			}
			out, genErr := GeneratePiConfig(baseURL, modelID)
			if genErr != nil {
				return path, false, fmt.Errorf("generate models.json: %w", genErr)
			}
			if writeErr := os.WriteFile(path, append(out, '\n'), 0644); writeErr != nil {
				return path, false, fmt.Errorf("write models.json: %w", writeErr)
			}
			return path, true, nil
		}
		return path, false, fmt.Errorf("read %s: %w", path, err)
	}

	var raw map[string]interface{}
	if unmarshalErr := json.Unmarshal(data, &raw); unmarshalErr != nil {
		return path, false, fmt.Errorf("parse existing %s: %w", path, unmarshalErr)
	}
	if raw == nil {
		raw = make(map[string]interface{})
	}

	modified := false

	var providersMap map[string]interface{}
	if existingProv, ok := raw["providers"]; ok {
		if pMap, ok := existingProv.(map[string]interface{}); ok {
			providersMap = pMap
		}
	}
	if providersMap == nil {
		providersMap = make(map[string]interface{})
		raw["providers"] = providersMap
		modified = true
	}

	existingFak, hasFak := providersMap[DefaultPiProviderID]
	if !hasFak {
		providersMap[DefaultPiProviderID] = map[string]interface{}{
			"baseUrl": baseURL,
			"apiKey":  "fak",
			"api":     "openai-completions",
			"models":  []interface{}{targetModel},
		}
		modified = true
	} else {
		fakMap, ok := existingFak.(map[string]interface{})
		if !ok {
			fakMap = make(map[string]interface{})
			providersMap[DefaultPiProviderID] = fakMap
			modified = true
		}
		if bURL, _ := fakMap["baseUrl"].(string); bURL != baseURL {
			fakMap["baseUrl"] = baseURL
			modified = true
		}
		if apiVal, _ := fakMap["api"].(string); apiVal != "openai-completions" {
			fakMap["api"] = "openai-completions"
			modified = true
		}
		if keyVal, _ := fakMap["apiKey"].(string); strings.TrimSpace(keyVal) == "" {
			fakMap["apiKey"] = "fak"
			modified = true
		}
		var modelsList []interface{}
		if mList, ok := fakMap["models"].([]interface{}); ok {
			modelsList = mList
		}
		modelFound := false
		for i, m := range modelsList {
			if mObj, ok := m.(map[string]interface{}); ok {
				if id, _ := mObj["id"].(string); id == modelID {
					modelFound = true
					if mObj["compat"] == nil {
						mObj["compat"] = map[string]interface{}{piDeveloperRoleKey: false}
						modelsList[i] = mObj
						modified = true
					}
					break
				}
			}
		}
		if !modelFound {
			modelsList = append(modelsList, targetModel)
			modified = true
		}
		fakMap["models"] = modelsList
	}

	if modified {
		out, marshalErr := json.MarshalIndent(raw, "", "  ")
		if marshalErr != nil {
			return path, false, fmt.Errorf("serialize %s: %w", path, marshalErr)
		}
		if writeErr := os.WriteFile(path, append(out, '\n'), 0644); writeErr != nil {
			return path, false, fmt.Errorf("write %s: %w", path, writeErr)
		}
	}

	return path, modified, nil
}
