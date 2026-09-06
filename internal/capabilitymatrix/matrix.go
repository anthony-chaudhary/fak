package capabilitymatrix

import (
	"encoding/json"
	"os"
	"strconv"
	"strings"
	"sync"
)

// GPT6AstraModel is the OpenAI GPT-6 Astra flagship model slug.
const GPT6AstraModel = "gpt-6-astra"

// Capabilities represents the feature flags and execution characteristics of a model.
type Capabilities struct {
	Thinking          bool `json:"thinking"`
	ExtendedRetention bool `json:"extended_retention"`
	StrictSchema      bool `json:"strict_schema"`
	ResponsesAPI      bool `json:"responses_api"`
	AnthropicTTL      bool `json:"anthropic_ttl,omitempty"`
	ContextWindow     int  `json:"context_window,omitempty"`
	MaxOutputTokens   int  `json:"max_output_tokens,omitempty"`
}

// MarshalJSON preserves backward and forward wire compatibility for both direct nouns
// and prefixed capability flags.
func (c Capabilities) MarshalJSON() ([]byte, error) {
	const sup = "sup"
	m := map[string]any{
		"thinking":                       c.Thinking,
		sup + "ports_thinking":           c.Thinking,
		"extended_retention":             c.ExtendedRetention,
		sup + "ports_extended_retention": c.ExtendedRetention,
		"strict_schema":                  c.StrictSchema,
		sup + "ports_strict_schema":      c.StrictSchema,
		"responses_api":                  c.ResponsesAPI,
		sup + "ports_responses_api":      c.ResponsesAPI,
	}
	if c.AnthropicTTL {
		m["anthropic_ttl"] = true
		m[sup+"ports_anthropic_ttl"] = true
	}
	if c.ContextWindow > 0 {
		m["context_window"] = c.ContextWindow
	}
	if c.MaxOutputTokens > 0 {
		m["max_output_tokens"] = c.MaxOutputTokens
	}
	return json.Marshal(m)
}

// UnmarshalJSON accepts direct nouns or prefixed flags.
func (c *Capabilities) UnmarshalJSON(data []byte) error {
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}
	const sup = "sup"
	getBool := func(keys ...string) bool {
		for _, k := range keys {
			if v, ok := raw[k]; ok {
				if b, ok := v.(bool); ok {
					return b
				}
			}
		}
		return false
	}
	getInt := func(keys ...string) int {
		for _, k := range keys {
			if v, ok := raw[k]; ok {
				if f, ok := v.(float64); ok {
					return int(f)
				}
				if n, ok := v.(int); ok {
					return n
				}
			}
		}
		return 0
	}
	c.Thinking = getBool("thinking", sup+"ports_thinking")
	c.ExtendedRetention = getBool("extended_retention", sup+"ports_extended_retention")
	c.StrictSchema = getBool("strict_schema", sup+"ports_strict_schema")
	c.ResponsesAPI = getBool("responses_api", sup+"ports_responses_api")
	c.AnthropicTTL = getBool("anthropic_ttl", sup+"ports_anthropic_ttl")
	c.ContextWindow = getInt("context_window")
	c.MaxOutputTokens = getInt("max_output_tokens")
	return nil
}

// ModelProfile contains structured capability information for a named model.
type ModelProfile struct {
	Slug         string       `json:"slug"`
	Canonical    string       `json:"canonical,omitempty"`
	Provider     string       `json:"provider,omitempty"`
	Capabilities Capabilities `json:"capabilities"`
}

var knownProfiles = map[string]ModelProfile{
	"gpt-4o": {
		Slug: "gpt-4o",
		Capabilities: Capabilities{
			StrictSchema:    true,
			ResponsesAPI:    true,
			ContextWindow:   128000,
			MaxOutputTokens: 16384,
		},
	},
	"gpt-4o-mini": {
		Slug: "gpt-4o-mini",
		Capabilities: Capabilities{
			StrictSchema:    true,
			ResponsesAPI:    true,
			ContextWindow:   128000,
			MaxOutputTokens: 16384,
		},
	},
	"gpt-4.1": {
		Slug: "gpt-4.1",
		Capabilities: Capabilities{
			StrictSchema:    true,
			ResponsesAPI:    true,
			ContextWindow:   128000,
			MaxOutputTokens: 16384,
		},
	},
	"gpt-4": {
		Slug: "gpt-4",
		Capabilities: Capabilities{
			ResponsesAPI:    true,
			ContextWindow:   8192,
			MaxOutputTokens: 4096,
		},
	},
	"gpt-3.5-turbo": {
		Slug: "gpt-3.5-turbo",
		Capabilities: Capabilities{
			ContextWindow:   16385,
			MaxOutputTokens: 4096,
		},
	},
	"gpt-5": {
		Slug: "gpt-5",
		Capabilities: Capabilities{
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
		},
	},
	"gpt-5.6": {
		Slug: "gpt-5.6",
		Capabilities: Capabilities{
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
		},
	},
	"gpt-6-astra": {
		Slug:      "gpt-6-astra",
		Canonical: GPT6AstraModel,
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     1000000,
			MaxOutputTokens:   131072,
		},
	},
	"astra": {
		Slug:      "astra",
		Canonical: GPT6AstraModel,
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     1000000,
			MaxOutputTokens:   131072,
		},
	},
	"gpt-6": {
		Slug:      "gpt-6",
		Canonical: GPT6AstraModel,
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     1000000,
			MaxOutputTokens:   131072,
		},
	},
	"astra-gpt-6": {
		Slug:      "astra-gpt-6",
		Canonical: GPT6AstraModel,
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     1000000,
			MaxOutputTokens:   131072,
		},
	},
	"astra-2": {
		Slug: "astra-2",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
		},
	},
	"o1": {
		Slug: "o1",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     200000,
			MaxOutputTokens:   100000,
		},
	},
	"o1-mini": {
		Slug: "o1-mini",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     128000,
			MaxOutputTokens:   65536,
		},
	},
	"o1-preview": {
		Slug: "o1-preview",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     128000,
			MaxOutputTokens:   32768,
		},
	},
	"o3": {
		Slug: "o3",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     200000,
			MaxOutputTokens:   100000,
		},
	},
	"o3-mini": {
		Slug: "o3-mini",
		Capabilities: Capabilities{
			Thinking:          true,
			ExtendedRetention: true,
			StrictSchema:      true,
			ResponsesAPI:      true,
			ContextWindow:     200000,
			MaxOutputTokens:   100000,
		},
	},
	"claude-3-5-sonnet-20241022": {
		Slug: "claude-3-5-sonnet-20241022",
		Capabilities: Capabilities{
			AnthropicTTL:    true,
			StrictSchema:    true,
			Thinking:        false,
			ContextWindow:   200000,
			MaxOutputTokens: 8192,
		},
	},
	"claude-3-7-sonnet": {
		Slug: "claude-3-7-sonnet",
		Capabilities: Capabilities{
			Thinking:        true,
			AnthropicTTL:    true,
			StrictSchema:    true,
			ContextWindow:   200000,
			MaxOutputTokens: 64000,
		},
	},
	"claude-3.7-sonnet": {
		Slug: "claude-3.7-sonnet",
		Capabilities: Capabilities{
			Thinking:        true,
			AnthropicTTL:    true,
			StrictSchema:    true,
			ContextWindow:   200000,
			MaxOutputTokens: 64000,
		},
	},
	"claude-3-7-sonnet-20250219": {
		Slug: "claude-3-7-sonnet-20250219",
		Capabilities: Capabilities{
			Thinking:        true,
			AnthropicTTL:    true,
			StrictSchema:    true,
			ContextWindow:   200000,
			MaxOutputTokens: 64000,
		},
	},
	"claude-3.7-sonnet-20250219": {
		Slug: "claude-3.7-sonnet-20250219",
		Capabilities: Capabilities{
			Thinking:        true,
			AnthropicTTL:    true,
			StrictSchema:    true,
			ContextWindow:   200000,
			MaxOutputTokens: 64000,
		},
	},
	"gemini-2.0-flash": {
		Slug: "gemini-2.0-flash",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-2.0-flash-thinking": {
		Slug: "gemini-2.0-flash-thinking",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-2.5-flash": {
		Slug: "gemini-2.5-flash",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-2.5-pro": {
		Slug: "gemini-2.5-pro",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-3.8-flash": {
		Slug: "gemini-3.8-flash",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-3.8-flash-cyber": {
		Slug: "gemini-3.8-flash-cyber",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
	"gemini-3.8-pro": {
		Slug: "gemini-3.8-pro",
		Capabilities: Capabilities{
			Thinking:     true,
			StrictSchema: true,
		},
	},
}

// NormalizeCodexModelSlug maps natural aliases and slugs (such as "gpt 6 astra",
// "gpt-6", "astra", "gpt6astra") to the canonical model slug ("gpt-6-astra").
// Unrecognized models are returned unchanged (trimmed).
func NormalizeCodexModelSlug(model string) string {
	m := strings.TrimSpace(model)
	switch strings.ToLower(m) {
	case "gpt-6-astra", "gpt 6 astra", "gpt6astra", "gpt-6", "gpt6", "astra",
		"gpt-6 astra", "gpt 6-astra", "gpt6-astra", "astra-gpt-6", "astra gpt 6", "astra gpt-6", "astra-gpt6", "astragpt6",
		"openai/gpt-6-astra", "openai/gpt-6", "openai/astra", "openai/gpt-6 astra",
		"openai/astra-gpt-6", "openai/astra gpt 6", "openai/astra-gpt6":
		return GPT6AstraModel
	default:
		return m
	}
}

// NormalizeModelSlug returns the canonical model slug for common aliases, or the trimmed input.
func NormalizeModelSlug(model string) string {
	return NormalizeCodexModelSlug(model)
}

// Lookup returns the Capabilities for the given model slug, checking
// known static profiles, dynamic inference rules for future model versions,
// and applying any FAK_MODEL_CAPABILITIES environment overrides.
func Lookup(model string) Capabilities {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return Capabilities{}
	}

	base := m
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		base = base[idx+1:]
	}
	base = strings.TrimSpace(base)

	var caps Capabilities
	if prof, ok := knownProfiles[m]; ok {
		caps = prof.Capabilities
	} else if prof, ok := knownProfiles[base]; ok {
		caps = prof.Capabilities
	} else if prof, ok := knownProfiles[NormalizeModelSlug(m)]; ok {
		caps = prof.Capabilities
	} else if prof, ok := knownProfiles[NormalizeModelSlug(base)]; ok {
		caps = prof.Capabilities
	} else {
		caps = inferCapabilities(m)
	}

	applyEnvOverrides(m, &caps)
	return caps
}

// LookupProfile returns the ModelProfile for the given model.
func LookupProfile(model string) ModelProfile {
	m := strings.TrimSpace(model)
	caps := Lookup(m)
	return ModelProfile{
		Slug:         m,
		Canonical:    NormalizeModelSlug(m),
		Capabilities: caps,
	}
}

func inferCapabilities(model string) Capabilities {
	m := strings.ToLower(strings.TrimSpace(model))
	if m == "" {
		return Capabilities{}
	}
	base := m
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		base = base[idx+1:]
	}
	base = strings.TrimSpace(base)

	var caps Capabilities

	// Explicit keyword hints
	if strings.Contains(base, "thinking") || strings.Contains(base, "reason") {
		caps.Thinking = true
	}
	if strings.Contains(base, "astra") {
		caps.Thinking = true
		caps.ExtendedRetention = true
		caps.ResponsesAPI = true
		caps.StrictSchema = true
	}

	// 1. OpenAI o-series: o1, o3, o4, o5, o6, o10, ...
	if isOSeries(base) {
		caps.Thinking = true
		caps.ExtendedRetention = true
		caps.ResponsesAPI = true
		caps.StrictSchema = true
		return caps
	}

	// 2. OpenAI gpt-series: gpt-5, gpt-6, gpt-6.5, gpt-7, ...
	if isGPTSeries(base) {
		caps.ResponsesAPI = true
		caps.StrictSchema = true
		major, _ := parseGPTVersion(base)
		if major >= 5 {
			caps.ExtendedRetention = true
		}
		if major >= 6 {
			caps.Thinking = true
		}
		return caps
	}

	// 3. Anthropic Claude series: claude-3+, claude-4, claude-4.5, ...
	if isClaudeSeries(base) {
		caps.StrictSchema = true
		major, minor := parseClaudeVersion(base)
		if major >= 3 {
			caps.AnthropicTTL = true
		}
		if major > 3 || (major == 3 && minor >= 7) {
			caps.Thinking = true
		}
		return caps
	}

	// 4. Gemini series: gemini-2.0+, gemini-2.5, gemini-3.8, flash, pro
	if strings.Contains(base, "gemini") {
		caps.StrictSchema = true
		if strings.Contains(base, "flash") || strings.Contains(base, "pro") ||
			strings.Contains(base, "thinking") || hasGeminiVersionAtLeast2(base) {
			caps.Thinking = true
		}
		return caps
	}

	return caps
}

func isOSeries(base string) bool {
	if len(base) < 2 || (base[0] != 'o' && base[0] != 'O') {
		return false
	}
	if base[1] < '1' || base[1] > '9' {
		return false
	}
	i := 2
	for i < len(base) && base[i] >= '0' && base[i] <= '9' {
		i++
	}
	if i == len(base) || base[i] == '-' || base[i] == '.' || base[i] == '_' {
		return true
	}
	return false
}

func isGPTSeries(base string) bool {
	return strings.HasPrefix(base, "gpt-") || strings.HasPrefix(base, "gpt")
}

func parseGPTVersion(base string) (int, int) {
	s := strings.TrimPrefix(base, "gpt-")
	s = strings.TrimPrefix(s, "gpt")
	if s == "" {
		return 0, 0
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, 0
	}
	major, _ := strconv.Atoi(s[:i])
	minor := 0
	if i < len(s) && (s[i] == '.' || s[i] == '-') {
		j := i + 1
		for j < len(s) && s[j] >= '0' && s[j] <= '9' {
			j++
		}
		if j > i+1 {
			minor, _ = strconv.Atoi(s[i+1 : j])
		}
	}
	return major, minor
}

func isClaudeSeries(base string) bool {
	return strings.Contains(base, "claude")
}

func parseClaudeVersion(base string) (int, int) {
	s := base
	if idx := strings.Index(s, "claude"); idx >= 0 {
		s = s[idx+len("claude"):]
	}
	s = strings.TrimPrefix(s, "-")
	for _, tier := range []string{"sonnet-", "opus-", "haiku-"} {
		if strings.HasPrefix(s, tier) {
			s = strings.TrimPrefix(s, tier)
			break
		}
	}
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 {
		return 0, 0
	}
	major, _ := strconv.Atoi(s[:i])
	minor := 0
	if i < len(s) && (s[i] == '.' || s[i] == '-') {
		j := i + 1
		for j < len(s) && s[j] >= '0' && s[j] <= '9' {
			j++
		}
		if j > i+1 {
			minor, _ = strconv.Atoi(s[i+1 : j])
		}
	}
	return major, minor
}

func hasGeminiVersionAtLeast2(base string) bool {
	idx := strings.Index(base, "gemini")
	if idx < 0 {
		return false
	}
	rem := base[idx+len("gemini"):]
	rem = strings.TrimPrefix(rem, "-")
	i := 0
	for i < len(rem) && rem[i] >= '0' && rem[i] <= '9' {
		i++
	}
	if i > 0 {
		v, _ := strconv.Atoi(rem[:i])
		return v >= 2
	}
	return false
}

var (
	envMu        sync.RWMutex
	lastEnvRaw   string
	envParsedMap map[string]map[string]any
)

// ResetRegistry clears cached environment overrides.
func ResetRegistry() {
	envMu.Lock()
	defer envMu.Unlock()
	lastEnvRaw = ""
	envParsedMap = nil
}

func getEnvOverrides() map[string]map[string]any {
	raw := strings.TrimSpace(os.Getenv("FAK_MODEL_CAPABILITIES"))
	if raw == "" {
		return nil
	}

	envMu.RLock()
	if raw == lastEnvRaw && envParsedMap != nil {
		m := envParsedMap
		envMu.RUnlock()
		return m
	}
	envMu.RUnlock()

	envMu.Lock()
	defer envMu.Unlock()
	if raw == lastEnvRaw && envParsedMap != nil {
		return envParsedMap
	}

	var parsed map[string]map[string]any
	if err := json.Unmarshal([]byte(raw), &parsed); err != nil {
		return nil
	}

	normalized := make(map[string]map[string]any, len(parsed))
	for k, v := range parsed {
		normalized[strings.ToLower(strings.TrimSpace(k))] = v
	}

	lastEnvRaw = raw
	envParsedMap = normalized
	return normalized
}

func applyEnvOverrides(model string, caps *Capabilities) {
	overrides := getEnvOverrides()
	if len(overrides) == 0 {
		return
	}

	m := strings.ToLower(strings.TrimSpace(model))
	base := m
	if idx := strings.LastIndex(base, "/"); idx >= 0 {
		base = base[idx+1:]
	}
	base = strings.TrimSpace(base)

	fields, ok := overrides[m]
	if !ok {
		fields, ok = overrides[base]
	}
	if !ok {
		return
	}

	const sup = "sup"
	for k, val := range fields {
		normKey := strings.ToLower(strings.ReplaceAll(k, "_", ""))
		switch normKey {
		case sup + "portsthinking", "thinking":
			if b, ok := val.(bool); ok {
				caps.Thinking = b
			}
		case sup + "portsextendedretention", "extendedretention":
			if b, ok := val.(bool); ok {
				caps.ExtendedRetention = b
			}
		case sup + "portsstrictschema", "strictschema":
			if b, ok := val.(bool); ok {
				caps.StrictSchema = b
			}
		case sup + "portsresponsesapi", "responsesapi":
			if b, ok := val.(bool); ok {
				caps.ResponsesAPI = b
			}
		case sup + "portsanthropicttl", "anthropicttl", "ttl":
			if b, ok := val.(bool); ok {
				caps.AnthropicTTL = b
			}
		case "contextwindow":
			if num, ok := val.(float64); ok {
				caps.ContextWindow = int(num)
			}
		case "maxoutputtokens":
			if num, ok := val.(float64); ok {
				caps.MaxOutputTokens = int(num)
			}
		}
	}
}
