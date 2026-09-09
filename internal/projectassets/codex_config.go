package projectassets

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

const (
	// DefaultCodexBaseURL is the standard local gateway address used by fak serve.
	DefaultCodexBaseURL = "http://127.0.0.1:8080/v1"

	// DefaultCodexModelID is the canonical Mac Metal served model identifier.
	DefaultCodexModelID = "qwen38:27b-q4"

	// DefaultCodexWireAPI is the preferred OpenAI Responses wire API for Codex.
	DefaultCodexWireAPI = "responses"

	// DefaultCodexEnvKey is the default environment variable for auth (e.g. OPENAI_API_KEY).
	DefaultCodexEnvKey = "OPENAI_API_KEY"

	// DefaultCodexProviderID is the provider identifier registered in Codex config.
	DefaultCodexProviderID = "fak"
)

// GenerateCodexConfig generates the TOML configuration block for Codex model_providers.fak.
func GenerateCodexConfig(baseURL, modelID, wireAPI, envKey string) string {
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = DefaultCodexBaseURL
	}
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = DefaultCodexModelID
	}
	wireAPI = strings.TrimSpace(wireAPI)
	if wireAPI == "" {
		wireAPI = DefaultCodexWireAPI
	}
	envKey = strings.TrimSpace(envKey)
	if envKey == "" {
		envKey = DefaultCodexEnvKey
	}

	var sb strings.Builder
	sb.WriteString(fmt.Sprintf("model_provider = %q\n", DefaultCodexProviderID))
	sb.WriteString(fmt.Sprintf("model = %q\n\n", modelID))
	sb.WriteString(fmt.Sprintf("[model_providers.%s]\n", DefaultCodexProviderID))
	sb.WriteString("name = \"fak serve\"\n")
	sb.WriteString(fmt.Sprintf("base_url = %q\n", baseURL))
	sb.WriteString(fmt.Sprintf("wire_api = %q\n", wireAPI))
	sb.WriteString(fmt.Sprintf("env_key = %q\n", envKey))
	return sb.String()
}

// ResolveCodexConfigFile resolves the path to Codex's config.toml in priority order:
// 1. explicit codexHome or config file path
// 2. $CODEX_HOME/config.toml
// 3. workspace directory containing .codex/config.toml or config.toml
// 4. ~/.codex/config.toml (standard default)
func ResolveCodexConfigFile(codexHome, dir string) string {
	if h := strings.TrimSpace(codexHome); h != "" {
		h = expandTilde(h)
		if strings.HasSuffix(strings.ToLower(h), ".toml") {
			return h
		}
		return filepath.Join(h, "config.toml")
	}
	if envHome := strings.TrimSpace(os.Getenv("CODEX_HOME")); envHome != "" {
		envHome = expandTilde(envHome)
		if strings.HasSuffix(strings.ToLower(envHome), ".toml") {
			return envHome
		}
		return filepath.Join(envHome, "config.toml")
	}
	if d := strings.TrimSpace(dir); d != "" {
		d = expandTilde(d)
		if strings.HasSuffix(strings.ToLower(d), ".toml") {
			return d
		}
		cand1 := filepath.Join(d, ".codex", "config.toml")
		if _, err := os.Stat(cand1); err == nil {
			return cand1
		}
		cand2 := filepath.Join(d, "config.toml")
		if _, err := os.Stat(cand2); err == nil {
			return cand2
		}
		if _, err := os.Stat(filepath.Join(d, ".codex")); err == nil {
			return cand1
		}
		return cand2
	}
	if uHome, err := os.UserHomeDir(); err == nil && uHome != "" {
		return filepath.Join(uHome, ".codex", "config.toml")
	}
	return filepath.Join(".codex", "config.toml")
}

func expandTilde(p string) string {
	if strings.HasPrefix(p, "~/") || strings.HasPrefix(p, "~\\") {
		if home, err := os.UserHomeDir(); err == nil && home != "" {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}

// EnsureCodexProviderConfig ensures config.toml contains the "fak" provider table pointing
// to baseURL with modelID, wireAPI, and envKey, setting model_provider = "fak", while preserving
// all existing tables and comments.
func EnsureCodexProviderConfig(targetPath, baseURL, modelID, wireAPI, envKey string) (bool, error) {
	targetPath = strings.TrimSpace(targetPath)
	if targetPath == "" {
		targetPath = ResolveCodexConfigFile("", "")
	}
	baseURL = strings.TrimSpace(baseURL)
	if baseURL == "" {
		baseURL = DefaultCodexBaseURL
	}
	modelID = strings.TrimSpace(modelID)
	if modelID == "" {
		modelID = DefaultCodexModelID
	}
	wireAPI = strings.TrimSpace(wireAPI)
	if wireAPI == "" {
		wireAPI = DefaultCodexWireAPI
	}
	envKey = strings.TrimSpace(envKey)
	if envKey == "" {
		envKey = DefaultCodexEnvKey
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		if os.IsNotExist(err) {
			if mkErr := os.MkdirAll(filepath.Dir(targetPath), 0755); mkErr != nil {
				return false, fmt.Errorf("create directory for %s: %w", targetPath, mkErr)
			}
			content := GenerateCodexConfig(baseURL, modelID, wireAPI, envKey)
			if wErr := os.WriteFile(targetPath, []byte(content), 0644); wErr != nil {
				return false, fmt.Errorf("write %s: %w", targetPath, wErr)
			}
			return true, nil
		}
		return false, fmt.Errorf("read %s: %w", targetPath, err)
	}

	existing := string(data)
	lines := strings.Split(existing, "\n")

	// Target provider section
	sectionHeader := fmt.Sprintf("[model_providers.%s]", DefaultCodexProviderID)
	providerLines := []string{
		sectionHeader,
		"name = \"fak serve\"",
		fmt.Sprintf("base_url = %q", baseURL),
		fmt.Sprintf("wire_api = %q", wireAPI),
		fmt.Sprintf("env_key = %q", envKey),
	}

	// Locate existing section if any
	startIdx, endIdx := findCodexTOMLSection(lines, "model_providers."+DefaultCodexProviderID)

	modified := false

	// Check / update top-level model_provider and model
	topEnd := findFirstTOMLSectionHeader(lines)
	if topEnd < 0 {
		topEnd = len(lines)
	}

	modelProvLine := fmt.Sprintf("model_provider = %q", DefaultCodexProviderID)
	modelLine := fmt.Sprintf("model = %q", modelID)

	mpIdx := findKeyInLines(lines[:topEnd], "model_provider")
	if mpIdx >= 0 {
		if strings.TrimSpace(lines[mpIdx]) != modelProvLine {
			lines[mpIdx] = modelProvLine
			modified = true
		}
	} else {
		// Insert at top
		lines = append([]string{modelProvLine}, lines...)
		modified = true
		topEnd++
		if startIdx >= 0 {
			startIdx++
			endIdx++
		}
	}

	mIdx := findKeyInLines(lines[:topEnd], "model")
	if mIdx >= 0 {
		if strings.TrimSpace(lines[mIdx]) != modelLine {
			lines[mIdx] = modelLine
			modified = true
		}
	} else {
		// Insert after model_provider
		insertAt := 1
		if insertAt > topEnd {
			insertAt = topEnd
		}
		lines = append(lines[:insertAt], append([]string{modelLine}, lines[insertAt:]...)...)
		modified = true
		if startIdx >= 0 {
			startIdx++
			endIdx++
		}
	}

	// Now handle the [model_providers.fak] section
	if startIdx >= 0 {
		// Section exists: compare lines
		existingSec := strings.Join(lines[startIdx:endIdx], "\n")
		newSec := strings.Join(providerLines, "\n")
		if strings.TrimSpace(existingSec) != strings.TrimSpace(newSec) {
			lines = append(lines[:startIdx], append(providerLines, lines[endIdx:]...)...)
			modified = true
		}
	} else {
		// Append section at end
		if len(lines) > 0 && strings.TrimSpace(lines[len(lines)-1]) != "" {
			lines = append(lines, "")
		}
		lines = append(lines, providerLines...)
		modified = true
	}

	if !modified {
		return false, nil
	}

	out := strings.Join(lines, "\n")
	if !strings.HasSuffix(out, "\n") {
		out += "\n"
	}

	if wErr := os.WriteFile(targetPath, []byte(out), 0644); wErr != nil {
		return false, fmt.Errorf("write %s: %w", targetPath, wErr)
	}
	return true, nil
}

func findFirstTOMLSectionHeader(lines []string) int {
	for i, raw := range lines {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "[") {
			return i
		}
	}
	return -1
}

func findKeyInLines(lines []string, key string) int {
	for i, raw := range lines {
		trimmed := strings.TrimSpace(raw)
		if strings.HasPrefix(trimmed, "#") {
			continue
		}
		parts := strings.SplitN(trimmed, "=", 2)
		if len(parts) == 2 && strings.TrimSpace(parts[0]) == key {
			return i
		}
	}
	return -1
}

func findCodexTOMLSection(lines []string, section string) (start, end int) {
	start = -1
	for i, raw := range lines {
		line := strings.TrimSpace(raw)
		if !strings.HasPrefix(line, "[") {
			continue
		}
		header := strings.Trim(line, "[]")
		header = strings.TrimSpace(header)
		if start < 0 {
			if header == section {
				start = i
			}
			continue
		}
		// In section: next header marks end
		return start, i
	}
	if start < 0 {
		return -1, -1
	}
	return start, len(lines)
}
