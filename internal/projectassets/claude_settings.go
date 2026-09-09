package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// DefaultClaudeBaseURL is the standard local gateway address used by fak serve for Claude Code.
// Claude Code appends /v1/messages, so this URL excludes any /v1 suffix.
const DefaultClaudeBaseURL = "http://127.0.0.1:8080"

// DefaultClaudeModelID is the default served model identifier for local inference on Apple Silicon Mac.
const DefaultClaudeModelID = "qwen38:27b-q4"

// DefaultClaudeAPIKey is the default non-empty API key placeholder passed to Claude Code.
// Claude Code requires a non-empty key when pointing at custom ANTHROPIC_BASE_URL.
const DefaultClaudeAPIKey = "fak-local-dogfood"

// DefaultClaudeAPITimeoutMS is the default Claude Code API timeout in milliseconds (30 minutes).
const DefaultClaudeAPITimeoutMS = "1800000"

// NormalizeClaudeBaseURL cleans a raw gateway address for Claude Code consumption,
// ensuring the scheme is present and stripping trailing slashes and /v1.
func NormalizeClaudeBaseURL(raw string) string {
	u := strings.TrimSpace(raw)
	if u == "" {
		return DefaultClaudeBaseURL
	}
	if !strings.HasPrefix(u, "http://") && !strings.HasPrefix(u, "https://") {
		u = "http://" + u
	}
	u = strings.TrimRight(u, "/")
	if strings.HasSuffix(u, "/v1") {
		u = strings.TrimSuffix(u, "/v1")
		u = strings.TrimRight(u, "/")
	}
	return u
}

// NormalizeClaudeModelID returns the model ID or the Mac default if empty.
func NormalizeClaudeModelID(modelID string) string {
	m := strings.TrimSpace(modelID)
	if m == "" {
		return DefaultClaudeModelID
	}
	return m
}

// NormalizeClaudeAPIKey returns the API key or the local placeholder if empty.
func NormalizeClaudeAPIKey(key string) string {
	k := strings.TrimSpace(key)
	if k == "" {
		return DefaultClaudeAPIKey
	}
	return k
}

// ClaudeEnvMap constructs the complete map of environment variables that Claude Code needs
// to speak directly to a fak serve backend on Mac without guard.
func ClaudeEnvMap(baseURL, modelID, apiKey string) map[string]string {
	baseURL = NormalizeClaudeBaseURL(baseURL)
	modelID = NormalizeClaudeModelID(modelID)
	apiKey = NormalizeClaudeAPIKey(apiKey)

	return map[string]string{
		"ANTHROPIC_BASE_URL":                       baseURL,
		"ANTHROPIC_API_KEY":                        apiKey,
		"ANTHROPIC_MODEL":                          modelID,
		"ANTHROPIC_DEFAULT_OPUS_MODEL":             modelID,
		"ANTHROPIC_DEFAULT_SONNET_MODEL":           modelID,
		"ANTHROPIC_DEFAULT_HAIKU_MODEL":            modelID,
		"ANTHROPIC_SMALL_FAST_MODEL":               modelID,
		"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
		"API_TIMEOUT_MS":                           DefaultClaudeAPITimeoutMS,
	}
}

// GenerateClaudeSettings creates a standalone .claude/settings.json configuration targeting a fak serve gateway.
func GenerateClaudeSettings(baseURL, modelID, apiKey string) ([]byte, error) {
	envMap := ClaudeEnvMap(baseURL, modelID, apiKey)
	envObj := make(map[string]interface{}, len(envMap))
	for k, v := range envMap {
		envObj[k] = v
	}

	cfg := map[string]interface{}{
		"env": envObj,
	}
	return json.MarshalIndent(cfg, "", "  ")
}

// EnsureClaudeSettingsConfig ensures .claude/settings.json in root contains the "env" section
// pointing to baseURL with modelID and apiKey, while preserving existing hooks, permissions,
// and other settings.
func EnsureClaudeSettingsConfig(root, baseURL, modelID, apiKey string) (bool, error) {
	if root == "" {
		root = "."
	}
	settingsDir := filepath.Join(root, ".claude")
	settingsPath := filepath.Join(settingsDir, "settings.json")

	envMap := ClaudeEnvMap(baseURL, modelID, apiKey)

	data, err := os.ReadFile(settingsPath)
	if err != nil {
		if os.IsNotExist(err) {
			if mkdirErr := os.MkdirAll(settingsDir, 0o755); mkdirErr != nil {
				return false, fmt.Errorf("create .claude dir: %w", mkdirErr)
			}
			out, genErr := GenerateClaudeSettings(baseURL, modelID, apiKey)
			if genErr != nil {
				return false, fmt.Errorf("generate settings.json: %w", genErr)
			}
			if writeErr := os.WriteFile(settingsPath, append(out, '\n'), 0o644); writeErr != nil {
				return false, fmt.Errorf("write settings.json: %w", writeErr)
			}
			return true, nil
		}
		return false, fmt.Errorf("read settings.json: %w", err)
	}

	var raw map[string]interface{}
	if unmarshalErr := json.Unmarshal(data, &raw); unmarshalErr != nil {
		return false, fmt.Errorf("parse existing settings.json: %w", unmarshalErr)
	}
	if raw == nil {
		raw = make(map[string]interface{})
	}

	modified := false
	existingEnvRaw, hasEnv := raw["env"]
	var existingEnv map[string]interface{}
	if hasEnv {
		if m, ok := existingEnvRaw.(map[string]interface{}); ok {
			existingEnv = m
		}
	}
	if existingEnv == nil {
		existingEnv = make(map[string]interface{})
		raw["env"] = existingEnv
		modified = true
	}

	for k, v := range envMap {
		curr, ok := existingEnv[k]
		if !ok || curr != v {
			existingEnv[k] = v
			modified = true
		}
	}

	if !modified {
		return false, nil
	}

	out, err := json.MarshalIndent(raw, "", "  ")
	if err != nil {
		return false, fmt.Errorf("encode updated settings.json: %w", err)
	}
	if writeErr := os.WriteFile(settingsPath, append(out, '\n'), 0o644); writeErr != nil {
		return false, fmt.Errorf("write updated settings.json: %w", writeErr)
	}
	return true, nil
}
