package sweepconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

type PriceHint struct {
	Input  float64 `json:"input"`
	Output float64 `json:"output"`
	Source string  `json:"source"`
}

type ModelConfig struct {
	Name      string     `json:"name"`
	Provider  string     `json:"provider"`
	BaseURL   string     `json:"base_url,omitempty"`
	APIKeyEnv string     `json:"api_key_env,omitempty"`
	LocalShim string     `json:"local_shim,omitempty"`
	PriceHint *PriceHint `json:"price_hint,omitempty"`
	Enabled   bool       `json:"enabled"`
}

type WorkloadConfig struct {
	MaxTurns       int    `json:"max_turns"`
	Trials         int    `json:"trials"`
	TimeoutS       int    `json:"timeout_s"`
	TranscriptPath string `json:"transcript_path,omitempty"`
}

type SweepProfile struct {
	Name          string         `json:"name"`
	Description   string         `json:"description"`
	Models        []ModelConfig  `json:"models"`
	Workload      WorkloadConfig `json:"workload"`
	OutputDir     string         `json:"output_dir"`
	SkipAPI       bool           `json:"skip_api"`
	SkipOffline   bool           `json:"skip_offline"`
	SkipLocalShim bool           `json:"skip_local_shim"`
	FailFast      bool           `json:"fail_fast"`
	Tags          []string       `json:"tags"`
	Public        bool           `json:"public"`
}

func DefaultProfile(name string) SweepProfile {
	return SweepProfile{
		Name:      name,
		Models:    []ModelConfig{},
		Workload:  WorkloadConfig{MaxTurns: 12, Trials: 1, TimeoutS: 600},
		OutputDir: "fak/experiments/agent-live/sweep",
		Tags:      []string{},
		Public:    true,
	}
}

func LoadProfile(path string) (SweepProfile, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return SweepProfile{}, err
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err == nil {
		return profileFromMap(raw)
	}
	return parseYAMLProfile(string(b))
}

func SaveProfile(profile SweepProfile, path string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil && filepath.Dir(path) != "." {
		return err
	}
	return os.WriteFile(path, []byte(renderYAML(profile)), 0o644)
}

func ListProfiles(directory string) []SweepProfile {
	if directory == "" {
		directory = filepath.Join("tools", "sweep_profiles")
	}
	var profiles []SweepProfile
	for _, ext := range []string{"*.yaml", "*.yml"} {
		paths, _ := filepath.Glob(filepath.Join(directory, ext))
		sort.Strings(paths)
		for _, path := range paths {
			profile, err := LoadProfile(path)
			if err == nil {
				profiles = append(profiles, profile)
			}
		}
	}
	return profiles
}

func GetProfilePath(name, directory string) string {
	if directory == "" {
		directory = filepath.Join("tools", "sweep_profiles")
	}
	for _, ext := range []string{".yaml", ".yml"} {
		path := filepath.Join(directory, name+ext)
		if _, err := os.Stat(path); err == nil {
			return path
		}
	}
	return filepath.Join(directory, name+".yaml")
}

func profileFromMap(raw map[string]any) (SweepProfile, error) {
	name := str(raw["name"])
	if name == "" {
		return SweepProfile{}, fmt.Errorf("profile name is required")
	}
	p := DefaultProfile(name)
	p.Description = str(raw["description"])
	if v, ok := raw["output_dir"]; ok {
		p.OutputDir = str(v)
	}
	if v, ok := raw["skip_api"]; ok {
		p.SkipAPI = boolv(v)
	}
	if v, ok := raw["skip_offline"]; ok {
		p.SkipOffline = boolv(v)
	}
	if v, ok := raw["skip_local_shim"]; ok {
		p.SkipLocalShim = boolv(v)
	}
	if v, ok := raw["fail_fast"]; ok {
		p.FailFast = boolv(v)
	}
	if v, ok := raw["public"]; ok {
		p.Public = boolv(v)
	}
	p.Tags = stringSlice(raw["tags"])
	if wl, ok := raw["workload"].(map[string]any); ok {
		if v, ok := wl["max_turns"]; ok {
			p.Workload.MaxTurns = intv(v, p.Workload.MaxTurns)
		}
		if v, ok := wl["trials"]; ok {
			p.Workload.Trials = intv(v, p.Workload.Trials)
		}
		if v, ok := wl["timeout_s"]; ok {
			p.Workload.TimeoutS = intv(v, p.Workload.TimeoutS)
		}
		p.Workload.TranscriptPath = str(wl["transcript_path"])
	}
	if models, ok := raw["models"].([]any); ok {
		for _, mv := range models {
			mm, ok := mv.(map[string]any)
			if !ok {
				continue
			}
			m := ModelConfig{
				Name:      str(mm["name"]),
				Provider:  firstNonEmpty(str(mm["provider"]), "unknown"),
				BaseURL:   str(mm["base_url"]),
				APIKeyEnv: str(mm["api_key_env"]),
				LocalShim: str(mm["local_shim"]),
				Enabled:   true,
			}
			if v, ok := mm["enabled"]; ok {
				m.Enabled = boolv(v)
			}
			if ph, ok := mm["price_hint"].(map[string]any); ok {
				m.PriceHint = &PriceHint{
					Input:  floatv(ph["input"]),
					Output: floatv(ph["output"]),
					Source: firstNonEmpty(str(ph["source"]), "manual"),
				}
			}
			if m.Name != "" {
				p.Models = append(p.Models, m)
			}
		}
	}
	return p, nil
}

func parseYAMLProfile(text string) (SweepProfile, error) {
	raw := map[string]any{}
	var section string
	var current map[string]any
	var priceHint map[string]any
	var models []any
	var tags []any
	workload := map[string]any{}
	for _, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		indent := len(line) - len(strings.TrimLeft(line, " "))
		trimmed := strings.TrimSpace(line)
		if indent == 0 {
			key, value, ok := cutYAML(trimmed)
			if !ok {
				continue
			}
			section = key
			current = nil
			priceHint = nil
			if value != "" {
				raw[key] = scalar(value)
				section = ""
			} else if key == "workload" {
				raw[key] = workload
			}
			continue
		}
		switch section {
		case "workload":
			key, value, ok := cutYAML(trimmed)
			if ok {
				workload[key] = scalar(value)
			}
		case "models":
			if strings.HasPrefix(trimmed, "- ") {
				current = map[string]any{}
				models = append(models, current)
				priceHint = nil
				rest := strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))
				if rest != "" {
					key, value, ok := cutYAML(rest)
					if ok {
						current[key] = scalar(value)
					}
				}
				raw["models"] = models
				continue
			}
			if current == nil {
				continue
			}
			key, value, ok := cutYAML(trimmed)
			if !ok {
				continue
			}
			if key == "price_hint" && value == "" {
				priceHint = map[string]any{}
				current["price_hint"] = priceHint
				continue
			}
			if priceHint != nil && indent >= 6 {
				priceHint[key] = scalar(value)
			} else {
				current[key] = scalar(value)
			}
		case "tags":
			if strings.HasPrefix(trimmed, "- ") {
				tags = append(tags, strings.TrimSpace(strings.TrimPrefix(trimmed, "- ")))
				raw["tags"] = tags
			}
		}
	}
	if _, ok := raw["workload"]; !ok && len(workload) > 0 {
		raw["workload"] = workload
	}
	return profileFromMap(raw)
}

func renderYAML(p SweepProfile) string {
	var b strings.Builder
	fmt.Fprintf(&b, "name: %s\n", p.Name)
	if p.Description != "" {
		fmt.Fprintf(&b, "description: %s\n", p.Description)
	}
	b.WriteString("models:\n")
	for _, m := range p.Models {
		fmt.Fprintf(&b, "  - name: %s\n", m.Name)
		fmt.Fprintf(&b, "    provider: %s\n", firstNonEmpty(m.Provider, "unknown"))
		if m.BaseURL != "" {
			fmt.Fprintf(&b, "    base_url: %s\n", m.BaseURL)
		}
		if m.APIKeyEnv != "" {
			fmt.Fprintf(&b, "    api_key_env: %s\n", m.APIKeyEnv)
		}
		if m.LocalShim != "" {
			fmt.Fprintf(&b, "    local_shim: %s\n", m.LocalShim)
		}
		if m.PriceHint != nil {
			b.WriteString("    price_hint:\n")
			fmt.Fprintf(&b, "      input: %g\n", m.PriceHint.Input)
			fmt.Fprintf(&b, "      output: %g\n", m.PriceHint.Output)
			fmt.Fprintf(&b, "      source: %s\n", firstNonEmpty(m.PriceHint.Source, "manual"))
		}
		fmt.Fprintf(&b, "    enabled: %t\n", m.Enabled)
	}
	b.WriteString("workload:\n")
	fmt.Fprintf(&b, "  max_turns: %d\n", p.Workload.MaxTurns)
	fmt.Fprintf(&b, "  trials: %d\n", p.Workload.Trials)
	fmt.Fprintf(&b, "  timeout_s: %d\n", p.Workload.TimeoutS)
	if p.Workload.TranscriptPath != "" {
		fmt.Fprintf(&b, "  transcript_path: %s\n", p.Workload.TranscriptPath)
	}
	fmt.Fprintf(&b, "output_dir: %s\n", p.OutputDir)
	fmt.Fprintf(&b, "skip_api: %t\n", p.SkipAPI)
	fmt.Fprintf(&b, "skip_offline: %t\n", p.SkipOffline)
	fmt.Fprintf(&b, "skip_local_shim: %t\n", p.SkipLocalShim)
	fmt.Fprintf(&b, "fail_fast: %t\n", p.FailFast)
	b.WriteString("tags:\n")
	for _, tag := range p.Tags {
		fmt.Fprintf(&b, "  - %s\n", tag)
	}
	fmt.Fprintf(&b, "public: %t\n", p.Public)
	return b.String()
}

func cutYAML(line string) (string, string, bool) {
	key, value, ok := strings.Cut(line, ":")
	if !ok {
		return "", "", false
	}
	return strings.TrimSpace(key), strings.TrimSpace(value), true
}

func scalar(value string) any {
	value = strings.TrimSpace(value)
	switch strings.ToLower(value) {
	case "true":
		return true
	case "false":
		return false
	}
	if i, err := strconv.Atoi(value); err == nil {
		return i
	}
	if f, err := strconv.ParseFloat(value, 64); err == nil {
		return f
	}
	return strings.Trim(value, `"'`)
}

func str(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	return ""
}

func boolv(v any) bool {
	switch x := v.(type) {
	case bool:
		return x
	case string:
		return strings.EqualFold(x, "true")
	default:
		return false
	}
}

func intv(v any, fallback int) int {
	switch x := v.(type) {
	case int:
		return x
	case float64:
		return int(x)
	case json.Number:
		if i, err := x.Int64(); err == nil {
			return int(i)
		}
	case string:
		if i, err := strconv.Atoi(x); err == nil {
			return i
		}
	}
	return fallback
}

func floatv(v any) float64 {
	switch x := v.(type) {
	case int:
		return float64(x)
	case float64:
		return x
	case json.Number:
		f, _ := x.Float64()
		return f
	case string:
		f, _ := strconv.ParseFloat(x, 64)
		return f
	default:
		return 0
	}
}

func stringSlice(v any) []string {
	var out []string
	for _, item := range anySlice(v) {
		if s := str(item); s != "" {
			out = append(out, s)
		}
	}
	return out
}

func anySlice(v any) []any {
	if a, ok := v.([]any); ok {
		return a
	}
	return nil
}

func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if strings.TrimSpace(v) != "" {
			return strings.TrimSpace(v)
		}
	}
	return ""
}
