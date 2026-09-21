package projectassets

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// stripUTF8BOM removes a leading UTF-8 byte-order mark (EF BB BF) if present, so
// config files written by Windows editors still parse as standard JSON.
func stripUTF8BOM(data []byte) []byte {
	return bytes.TrimPrefix(data, []byte{0xEF, 0xBB, 0xBF})
}

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
//
// It uses the default served-window prior for the safe context budget; callers that know the
// real served window should use GeneratePiConfigForWindow so the resident target is derived
// from it (see pi_context_budget.go and docs/long-context-defaults.md: cap is not target).
func GeneratePiConfig(baseURL, modelID string) ([]byte, error) {
	return GeneratePiConfigForWindow(baseURL, modelID, DefaultPiServedWindow)
}

// GeneratePiConfigForWindow creates a standalone models.json whose "fak" provider advertises a
// SAFE resident context target derived from servedWindow (at most half the window) instead of the
// raw hard cap. The written contextWindow is a Pi auto-compaction tripwire
// (`contextTokens > contextWindow - reserveTokens`), so a smaller value makes Pi compact inside
// the safe envelope rather than at the ceiling.
func GeneratePiConfigForWindow(baseURL, modelID string, servedWindow int) ([]byte, error) {
	baseURL = NormalizePiBaseURL(baseURL)
	modelID = NormalizePiModelID(modelID)
	modelName := formatPiModelName(modelID)
	// Per-model budget: the registry names the served window for a known id (DeepSeek
	// V4.1 Flash -> 500k resident), and servedWindow is the caller's override for an
	// unnamed one. See pi_model_windows.go.
	budget := PiModelContextBudget(modelID, servedWindow)

	cfg := map[string]interface{}{
		"providers": map[string]interface{}{
			DefaultPiProviderID: map[string]interface{}{
				"baseUrl": baseURL,
				"apiKey":  "fak",
				"api":     "openai-completions",
				"models": []interface{}{
					piModelEntry(modelID, modelName, budget),
				},
			},
		},
	}
	return json.MarshalIndent(cfg, "", "  ")
}

// piModelEntry builds one Pi model entry from the derived safe budget. The contextWindow is the
// RESIDENT TARGET (at most half the served window), and maxTokens carries the output reserve, so
// the single JSON object encodes both halves of the cap-vs-target split. See pi_context_budget.go.
func piModelEntry(modelID, modelName string, budget PiContextBudget) map[string]interface{} {
	return map[string]interface{}{
		"id":            modelID,
		"name":          modelName,
		"reasoning":     false,
		"input":         []interface{}{"text"},
		"contextWindow": budget.ResidentTarget,
		"maxTokens":     budget.OutputReserve,
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
//
// It uses the default served-window prior; callers that know the real served window should use
// EnsurePiProviderConfigForWindow so the safe resident target is derived from it.
func EnsurePiProviderConfig(target, baseURL, modelID string) (string, bool, error) {
	return EnsurePiProviderConfigForWindow(target, baseURL, modelID, DefaultPiServedWindow)
}

// EnsurePiProviderConfigForWindow is EnsurePiProviderConfig with an explicit served window, so
// the written contextWindow is a SAFE resident target (at most half the window) rather than the
// raw cap. It also REPAIRS an existing fak model entry whose contextWindow still advertises the
// raw window (the pre-doctrine value), so an upgrade path exists for configs fak itself wrote.
func EnsurePiProviderConfigForWindow(target, baseURL, modelID string, servedWindow int) (string, bool, error) {
	path := ResolvePiConfigPath(target)
	baseURL = NormalizePiBaseURL(baseURL)
	modelID = NormalizePiModelID(modelID)
	modelName := formatPiModelName(modelID)

	// Per-model budget: a known id (DeepSeek V4.1 Flash) uses the registry window, an
	// unknown one falls back to the caller's servedWindow. servedWindow is therefore an
	// override for an unnamed model, not the single source it used to be.
	budget := PiModelContextBudget(modelID, servedWindow)
	targetModel := piModelEntry(modelID, modelName, budget)

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
				return path, false, fmt.Errorf("mkdir %s: %w", filepath.Dir(path), err)
			}
			out, genErr := GeneratePiConfigForWindow(baseURL, modelID, servedWindow)
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
	if unmarshalErr := json.Unmarshal(stripUTF8BOM(data), &raw); unmarshalErr != nil {
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
					// Repair the pre-doctrine context budget for THIS model: fak previously
					// wrote the raw window (131072) as contextWindow, and before per-model
					// resolution it wrote whatever the backend last advertised — so a
					// DeepSeek entry inherited the Qwen 65536. Recompute against this
					// entry's own model id so the fix also REPAIRS an existing catalog.
					entryBudget := PiModelContextBudget(modelID, servedWindow)
					if repairPiModelBudget(mObj, entryBudget) {
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
