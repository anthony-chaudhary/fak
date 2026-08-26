package sweepconfig

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
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
				Provider:  strmatch.FirstNonBlank(str(mm["provider"]), "unknown"),
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
					Source: strmatch.FirstNonBlank(str(ph["source"]), "manual"),
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
	var currentLine int
	var models []any
	var tags []any
	workload := map[string]any{}
	for index, line := range strings.Split(strings.ReplaceAll(text, "\r\n", "\n"), "\n") {
		lineNumber := index + 1
		line = strings.TrimSuffix(line, "\r")
		if strings.TrimSpace(line) == "" || strings.HasPrefix(strings.TrimSpace(line), "#") {
			continue
		}
		indent, err := yamlIndent(line)
		if err != nil {
			return SweepProfile{}, yamlLineError(lineNumber, err)
		}
		trimmed := strings.TrimSpace(line)
		if indent == 0 {
			if err := validateYAMLModel(current, currentLine); err != nil {
				return SweepProfile{}, err
			}
			current = nil
			currentLine = 0
			priceHint = nil
			key, value, err := cutYAML(trimmed)
			if err != nil {
				return SweepProfile{}, yamlLineError(lineNumber, err)
			}
			if _, exists := raw[key]; exists {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("duplicate key %q", key))
			}
			section = key
			switch key {
			case "name", "description", "output_dir":
				if err := setYAMLScalar(raw, key, value, yamlString); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
				if key == "name" && str(raw[key]) == "" {
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("profile name is required"))
				}
				section = ""
			case "skip_api", "skip_offline", "skip_local_shim", "fail_fast", "public":
				if err := setYAMLScalar(raw, key, value, yamlBool); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
				section = ""
			case "workload":
				if value != "" {
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("workload must use an indented mapping"))
				}
				raw[key] = workload
			case "models":
				if value != "" {
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("models must use an indented sequence"))
				}
				raw[key] = models
			case "tags":
				if value != "" {
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("tags must use an indented sequence"))
				}
				raw[key] = tags
			default:
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("unsupported key %q", key))
			}
			continue
		}
		switch section {
		case "workload":
			if indent != 2 {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("workload fields must be indented by 2 spaces"))
			}
			key, value, err := cutYAML(trimmed)
			if err != nil {
				return SweepProfile{}, yamlLineError(lineNumber, err)
			}
			if _, exists := workload[key]; exists {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("duplicate workload key %q", key))
			}
			switch key {
			case "max_turns", "trials", "timeout_s":
				if err := setYAMLScalar(workload, key, value, yamlInt); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
			case "transcript_path":
				if err := setYAMLScalar(workload, key, value, yamlString); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
			default:
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("unsupported workload key %q", key))
			}
		case "models":
			if indent == 2 && (trimmed == "-" || strings.HasPrefix(trimmed, "- ")) {
				if err := validateYAMLModel(current, currentLine); err != nil {
					return SweepProfile{}, err
				}
				current = map[string]any{}
				models = append(models, current)
				raw["models"] = models
				priceHint = nil
				currentLine = lineNumber
				rest := strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))
				if trimmed == "-" {
					rest = ""
				}
				if rest == "" {
					continue
				}
				if err := parseYAMLModelField(current, &priceHint, rest); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
				continue
			}
			if current == nil {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("model fields require a preceding sequence item"))
			}
			if indent == 4 {
				priceHint = nil
				if err := parseYAMLModelField(current, &priceHint, trimmed); err != nil {
					return SweepProfile{}, yamlLineError(lineNumber, err)
				}
				continue
			}
			key, value, err := cutYAML(trimmed)
			if err != nil {
				return SweepProfile{}, yamlLineError(lineNumber, err)
			}
			if indent == 6 && priceHint != nil {
				if _, exists := priceHint[key]; exists {
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("duplicate price_hint key %q", key))
				}
				switch key {
				case "input", "output":
					if err := setYAMLScalar(priceHint, key, value, yamlFloat); err != nil {
						return SweepProfile{}, yamlLineError(lineNumber, err)
					}
				case "source":
					if err := setYAMLScalar(priceHint, key, value, yamlString); err != nil {
						return SweepProfile{}, yamlLineError(lineNumber, err)
					}
				default:
					return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("unsupported price_hint key %q", key))
				}
				continue
			}
			return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("model fields must be indented by 4 spaces, or price_hint fields by 6"))
		case "tags":
			if indent != 2 || !strings.HasPrefix(trimmed, "- ") {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("tags must be sequence items indented by 2 spaces"))
			}
			parsed, err := yamlString(strings.TrimPrefix(trimmed, "- "))
			if err != nil {
				return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("tag: %w", err))
			}
			tags = append(tags, parsed)
			raw["tags"] = tags
		default:
			return SweepProfile{}, yamlLineError(lineNumber, fmt.Errorf("unexpected indented content"))
		}
	}
	if err := validateYAMLModel(current, currentLine); err != nil {
		return SweepProfile{}, err
	}
	return profileFromMap(raw)
}

func renderYAML(p SweepProfile) string {
	var b strings.Builder
	fmt.Fprintf(&b, "name: %s\n", yamlQuote(p.Name))
	if p.Description != "" {
		fmt.Fprintf(&b, "description: %s\n", yamlQuote(p.Description))
	}
	b.WriteString("models:\n")
	for _, m := range p.Models {
		fmt.Fprintf(&b, "  - name: %s\n", yamlQuote(m.Name))
		fmt.Fprintf(&b, "    provider: %s\n", yamlQuote(strmatch.FirstNonBlank(m.Provider, "unknown")))
		if m.BaseURL != "" {
			fmt.Fprintf(&b, "    base_url: %s\n", yamlQuote(m.BaseURL))
		}
		if m.APIKeyEnv != "" {
			fmt.Fprintf(&b, "    api_key_env: %s\n", yamlQuote(m.APIKeyEnv))
		}
		if m.LocalShim != "" {
			fmt.Fprintf(&b, "    local_shim: %s\n", yamlQuote(m.LocalShim))
		}
		if m.PriceHint != nil {
			b.WriteString("    price_hint:\n")
			fmt.Fprintf(&b, "      input: %g\n", m.PriceHint.Input)
			fmt.Fprintf(&b, "      output: %g\n", m.PriceHint.Output)
			fmt.Fprintf(&b, "      source: %s\n", yamlQuote(strmatch.FirstNonBlank(m.PriceHint.Source, "manual")))
		}
		fmt.Fprintf(&b, "    enabled: %t\n", m.Enabled)
	}
	b.WriteString("workload:\n")
	fmt.Fprintf(&b, "  max_turns: %d\n", p.Workload.MaxTurns)
	fmt.Fprintf(&b, "  trials: %d\n", p.Workload.Trials)
	fmt.Fprintf(&b, "  timeout_s: %d\n", p.Workload.TimeoutS)
	if p.Workload.TranscriptPath != "" {
		fmt.Fprintf(&b, "  transcript_path: %s\n", yamlQuote(p.Workload.TranscriptPath))
	}
	fmt.Fprintf(&b, "output_dir: %s\n", yamlQuote(p.OutputDir))
	fmt.Fprintf(&b, "skip_api: %t\n", p.SkipAPI)
	fmt.Fprintf(&b, "skip_offline: %t\n", p.SkipOffline)
	fmt.Fprintf(&b, "skip_local_shim: %t\n", p.SkipLocalShim)
	fmt.Fprintf(&b, "fail_fast: %t\n", p.FailFast)
	b.WriteString("tags:\n")
	for _, tag := range p.Tags {
		fmt.Fprintf(&b, "  - %s\n", yamlQuote(tag))
	}
	fmt.Fprintf(&b, "public: %t\n", p.Public)
	return b.String()
}

func cutYAML(line string) (string, string, error) {
	key, value, ok := strings.Cut(line, ":")
	if !ok {
		return "", "", fmt.Errorf("expected key: value")
	}
	key = strings.TrimSpace(key)
	if key == "" {
		return "", "", fmt.Errorf("empty key")
	}
	value, err := stripYAMLComment(value)
	if err != nil {
		return "", "", err
	}
	return key, strings.TrimSpace(value), nil
}

func yamlIndent(line string) (int, error) {
	for i := 0; i < len(line); i++ {
		switch line[i] {
		case ' ':
			continue
		case '\t':
			return 0, fmt.Errorf("tabs are not supported for indentation")
		default:
			return i, nil
		}
	}
	return len(line), nil
}

func stripYAMLComment(value string) (string, error) {
	start := 0
	for start < len(value) && (value[start] == ' ' || value[start] == '\t') {
		start++
	}
	if start == len(value) {
		return value, nil
	}
	quote := value[start]
	if quote != '\'' && quote != '"' {
		for i := start; i < len(value); i++ {
			if value[i] == '#' && (i == start || value[i-1] == ' ' || value[i-1] == '\t') {
				return value[:i], nil
			}
		}
		return value, nil
	}
	escaped := false
	for i := start + 1; i < len(value); i++ {
		c := value[i]
		if quote == '\'' {
			if c == '\'' {
				if i+1 < len(value) && value[i+1] == '\'' {
					i++
					continue
				}
				return stripYAMLTrailingComment(value, i+1)
			}
			continue
		}
		if escaped {
			escaped = false
			continue
		}
		if c == '\\' {
			escaped = true
		} else if c == '"' {
			return stripYAMLTrailingComment(value, i+1)
		}
	}
	return "", fmt.Errorf("unterminated quoted scalar")
}

func stripYAMLTrailingComment(value string, start int) (string, error) {
	for start < len(value) && (value[start] == ' ' || value[start] == '\t') {
		start++
	}
	if start == len(value) {
		return value, nil
	}
	if value[start] == '#' {
		return value[:start], nil
	}
	return value, nil
}

func scalar(value string) (any, error) {
	var err error
	value, err = stripYAMLComment(value)
	if err != nil {
		return nil, err
	}
	value = strings.TrimSpace(value)
	if value == "" {
		return "", nil
	}
	switch value[0] {
	case '"':
		decoded, err := strconv.Unquote(value)
		if err != nil {
			return nil, fmt.Errorf("invalid double-quoted scalar: %w", err)
		}
		return decoded, nil
	case '\'':
		return unquoteYAMLSingle(value)
	case '[', '{':
		return nil, fmt.Errorf("flow collections are not supported")
	case '|', '>':
		return nil, fmt.Errorf("block scalars are not supported")
	case '&', '*', '!':
		return nil, fmt.Errorf("anchors, aliases, and tags are not supported")
	}
	if strings.Contains(value, ": ") {
		return nil, fmt.Errorf("plain nested mappings are not supported; quote the value")
	}
	switch strings.ToLower(value) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	case "null", "~":
		return nil, fmt.Errorf("null scalars are not supported")
	}
	if i, err := strconv.Atoi(value); err == nil {
		return i, nil
	}
	if f, err := strconv.ParseFloat(value, 64); err == nil {
		return f, nil
	}
	return value, nil
}

func unquoteYAMLSingle(value string) (string, error) {
	if len(value) < 2 || value[len(value)-1] != '\'' {
		return "", fmt.Errorf("unterminated single-quoted scalar")
	}
	var b strings.Builder
	for i := 1; i < len(value)-1; i++ {
		if value[i] != '\'' {
			b.WriteByte(value[i])
			continue
		}
		if i+1 >= len(value)-1 || value[i+1] != '\'' {
			return "", fmt.Errorf("invalid single-quoted scalar")
		}
		b.WriteByte('\'')
		i++
	}
	return b.String(), nil
}

func yamlString(value string) (string, error) {
	parsed, err := scalar(value)
	if err != nil {
		return "", err
	}
	text, ok := parsed.(string)
	if !ok {
		return "", fmt.Errorf("expected a string")
	}
	return text, nil
}

func yamlBool(value string) (bool, error) {
	parsed, err := scalar(value)
	if err != nil {
		return false, err
	}
	boolean, ok := parsed.(bool)
	if !ok {
		return false, fmt.Errorf("expected true or false")
	}
	return boolean, nil
}

func yamlInt(value string) (int, error) {
	parsed, err := scalar(value)
	if err != nil {
		return 0, err
	}
	integer, ok := parsed.(int)
	if !ok {
		return 0, fmt.Errorf("expected an integer")
	}
	return integer, nil
}

func yamlFloat(value string) (float64, error) {
	parsed, err := scalar(value)
	if err != nil {
		return 0, err
	}
	switch number := parsed.(type) {
	case int:
		return float64(number), nil
	case float64:
		return number, nil
	default:
		return 0, fmt.Errorf("expected a number")
	}
}

func setYAMLScalar[T any](target map[string]any, key, value string, parse func(string) (T, error)) error {
	parsed, err := parse(value)
	if err != nil {
		return fmt.Errorf("%s: %w", key, err)
	}
	target[key] = parsed
	return nil
}

func parseYAMLModelField(current map[string]any, priceHint *map[string]any, text string) error {
	key, value, err := cutYAML(text)
	if err != nil {
		return err
	}
	return setYAMLModelField(current, priceHint, key, value)
}

func setYAMLModelField(current map[string]any, priceHint *map[string]any, key, value string) error {
	if _, exists := current[key]; exists {
		return fmt.Errorf("duplicate model key %q", key)
	}
	switch key {
	case "name", "provider", "base_url", "api_key_env", "local_shim":
		return setYAMLScalar(current, key, value, yamlString)
	case "enabled":
		return setYAMLScalar(current, key, value, yamlBool)
	case "price_hint":
		if value != "" {
			return fmt.Errorf("price_hint must use an indented mapping")
		}
		*priceHint = map[string]any{}
		current[key] = *priceHint
	default:
		return fmt.Errorf("unsupported model key %q", key)
	}
	return nil
}

func validateYAMLModel(current map[string]any, line int) error {
	if current == nil {
		return nil
	}
	if str(current["name"]) == "" {
		return yamlLineError(line, fmt.Errorf("model name is required"))
	}
	return nil
}

func yamlLineError(line int, err error) error {
	return fmt.Errorf("line %d: %w", line, err)
}

func yamlQuote(value string) string {
	return strconv.Quote(value)
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
