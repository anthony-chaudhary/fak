package logvault

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// SourceConfig is the declarative configuration schema for one logvault source.
type SourceConfig struct {
	ID       string   `json:"id" toml:"id"`
	Root     string   `json:"root" toml:"root"`
	Includes []string `json:"includes" toml:"includes"`
	Excludes []string `json:"excludes" toml:"excludes"`
	MaxBytes int64    `json:"max_bytes" toml:"max_bytes"`
	Note     string   `json:"note" toml:"note"`
}

// ToSource converts SourceConfig to a runtime Source, resolving relative Root
// against repoRoot when non-empty.
func (c SourceConfig) ToSource(repoRoot string) Source {
	root := c.Root
	if repoRoot != "" && !filepath.IsAbs(root) && root != "" {
		root = filepath.Join(repoRoot, root)
	} else if root != "" {
		root = filepath.Clean(root)
	}
	return Source{
		ID:       c.ID,
		Root:     root,
		Includes: append([]string(nil), c.Includes...),
		Excludes: append([]string(nil), c.Excludes...),
		MaxBytes: c.MaxBytes,
		Note:     c.Note,
	}
}

// DeclarativeConfig wraps a list of declarative source configurations.
type DeclarativeConfig struct {
	Sources []SourceConfig `json:"sources" toml:"sources"`
}

// ToSources converts all SourceConfigs to runtime Sources.
func (c DeclarativeConfig) ToSources(repoRoot string) []Source {
	var out []Source
	for _, sc := range c.Sources {
		out = append(out, sc.ToSource(repoRoot))
	}
	return out
}

// FromConfigs converts a slice of SourceConfigs to runtime Sources.
func FromConfigs(configs []SourceConfig, repoRoot string) []Source {
	var out []Source
	for _, sc := range configs {
		out = append(out, sc.ToSource(repoRoot))
	}
	return out
}

// stripTOMLComment strips comments from line while preserving hash characters inside quotes.
func stripTOMLComment(line string) string {
	inQuote := byte(0)
	escaped := false
	for i := 0; i < len(line); i++ {
		ch := line[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' && inQuote == '"' {
			escaped = true
			continue
		}
		if ch == '\'' || ch == '"' {
			if inQuote == 0 {
				inQuote = ch
			} else if inQuote == ch {
				inQuote = 0
			}
			continue
		}
		if ch == '#' && inQuote == 0 {
			return line[:i]
		}
	}
	return line
}

// parseTOMLString unquotes single or double-quoted TOML strings.
func parseTOMLString(v string) (string, error) {
	v = strings.TrimSpace(v)
	if len(v) < 2 {
		return "", fmt.Errorf("invalid TOML string %q", v)
	}
	if v[0] == '\'' && v[len(v)-1] == '\'' {
		return v[1 : len(v)-1], nil
	}
	if v[0] != '"' || v[len(v)-1] != '"' {
		return "", fmt.Errorf("invalid TOML string %q", v)
	}
	var s string
	if err := json.Unmarshal([]byte(v), &s); err != nil {
		return "", fmt.Errorf("invalid TOML quoted string %q: %w", v, err)
	}
	return s, nil
}

// parseTOMLStringArray extracts string elements from a bracketed array string like `["a", "b"]`.
func parseTOMLStringArray(v string) ([]string, error) {
	v = strings.TrimSpace(v)
	if !strings.HasPrefix(v, "[") || !strings.HasSuffix(v, "]") {
		return nil, fmt.Errorf("invalid TOML array %q", v)
	}
	body := strings.TrimSpace(v[1 : len(v)-1])
	if body == "" {
		return nil, nil
	}

	var items []string
	var cur strings.Builder
	inQuote := byte(0)
	escaped := false

	flushItem := func() error {
		raw := strings.TrimSpace(cur.String())
		cur.Reset()
		if raw == "" {
			return nil
		}
		s, err := parseTOMLString(raw)
		if err != nil {
			return err
		}
		items = append(items, s)
		return nil
	}

	for i := 0; i < len(body); i++ {
		ch := body[i]
		if escaped {
			cur.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' && inQuote == '"' {
			cur.WriteByte(ch)
			escaped = true
			continue
		}
		if ch == '\'' || ch == '"' {
			if inQuote == 0 {
				inQuote = ch
			} else if inQuote == ch {
				inQuote = 0
			}
			cur.WriteByte(ch)
			continue
		}
		if ch == ',' && inQuote == 0 {
			if err := flushItem(); err != nil {
				return nil, err
			}
			continue
		}
		cur.WriteByte(ch)
	}
	if inQuote != 0 {
		return nil, errors.New("unclosed quote in TOML array")
	}
	if err := flushItem(); err != nil {
		return nil, err
	}
	return items, nil
}

// arrayClosed checks whether all opening brackets [ have matching ] outside quotes.
func arrayClosed(s string) bool {
	inQuote := byte(0)
	escaped := false
	openCount := 0
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if escaped {
			escaped = false
			continue
		}
		if ch == '\\' && inQuote == '"' {
			escaped = true
			continue
		}
		if ch == '\'' || ch == '"' {
			if inQuote == 0 {
				inQuote = ch
			} else if inQuote == ch {
				inQuote = 0
			}
			continue
		}
		if inQuote == 0 {
			if ch == '[' {
				openCount++
			} else if ch == ']' {
				openCount--
			}
		}
	}
	return inQuote == 0 && openCount <= 0
}

// parseSourcesTOMLData parses TOML data into a slice of SourceConfigs.
func parseSourcesTOMLData(data []byte) ([]SourceConfig, error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	var sources []SourceConfig
	var current *SourceConfig

	flushCurrent := func() {
		if current != nil {
			if current.ID != "" || current.Root != "" {
				sources = append(sources, *current)
			}
			current = nil
		}
	}

	lineNum := 0
	for scanner.Scan() {
		lineNum++
		rawLine := scanner.Text()
		clean := strings.TrimSpace(stripTOMLComment(rawLine))
		if clean == "" {
			continue
		}

		// [[sources]] or [[source]] (array of tables)
		if strings.HasPrefix(clean, "[[") && strings.HasSuffix(clean, "]]") {
			header := strings.TrimSpace(clean[2 : len(clean)-2])
			if header == "sources" || header == "source" {
				flushCurrent()
				current = &SourceConfig{}
				continue
			}
			return nil, fmt.Errorf("line %d: unrecognized table array header %q", lineNum, clean)
		}

		// [sources.<id>] or [source.<id>] or [sources] or [source]
		if strings.HasPrefix(clean, "[") && strings.HasSuffix(clean, "]") && !strings.Contains(clean, "=") {
			header := strings.TrimSpace(clean[1 : len(clean)-1])
			if header == "sources" || header == "source" {
				flushCurrent()
				current = &SourceConfig{}
				continue
			}
			if strings.HasPrefix(header, "sources.") || strings.HasPrefix(header, "source.") {
				prefix := "sources."
				if strings.HasPrefix(header, "source.") {
					prefix = "source."
				}
				id := strings.TrimSpace(strings.TrimPrefix(header, prefix))
				id = strings.Trim(id, "\"' ")
				if id == "" {
					return nil, fmt.Errorf("line %d: empty source id in header %q", lineNum, clean)
				}
				flushCurrent()
				current = &SourceConfig{ID: id}
				continue
			}
			// Other section headers (e.g. outside logvault sources) - flush and ignore
			flushCurrent()
			current = nil
			continue
		}

		// If we are not inside a valid section and see key-value, treat as top-level source
		if current == nil {
			current = &SourceConfig{}
		}

		key, val, ok := strings.Cut(clean, "=")
		if !ok {
			return nil, fmt.Errorf("line %d: expected key = value, got %q", lineNum, clean)
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)

		// Check for multiline array
		if strings.HasPrefix(val, "[") && !arrayClosed(val) {
			for scanner.Scan() {
				lineNum++
				nextLine := strings.TrimSpace(stripTOMLComment(scanner.Text()))
				if nextLine == "" {
					continue
				}
				val += " " + nextLine
				if arrayClosed(val) {
					break
				}
			}
			if !arrayClosed(val) {
				return nil, fmt.Errorf("line %d: unclosed array for key %q", lineNum, key)
			}
		}

		switch strings.ToLower(key) {
		case "id":
			s, err := parseTOMLString(val)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid id %q: %w", lineNum, val, err)
			}
			current.ID = s
		case "root":
			s, err := parseTOMLString(val)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid root %q: %w", lineNum, val, err)
			}
			current.Root = s
		case "includes":
			arr, err := parseTOMLStringArray(val)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid includes %q: %w", lineNum, val, err)
			}
			current.Includes = arr
		case "excludes":
			arr, err := parseTOMLStringArray(val)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid excludes %q: %w", lineNum, val, err)
			}
			current.Excludes = arr
		case "max_bytes":
			n, err := strconv.ParseInt(val, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid max_bytes %q: %w", lineNum, val, err)
			}
			current.MaxBytes = n
		case "note":
			s, err := parseTOMLString(val)
			if err != nil {
				return nil, fmt.Errorf("line %d: invalid note %q: %w", lineNum, val, err)
			}
			current.Note = s
		default:
			// Ignore unknown keys gracefully for forward compatibility
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	flushCurrent()
	return sources, nil
}

// ParseSourcesTOML parses TOML data and returns a slice of Sources.
func ParseSourcesTOML(data []byte) ([]Source, error) {
	configs, err := parseSourcesTOMLData(data)
	if err != nil {
		return nil, err
	}
	return FromConfigs(configs, ""), nil
}

// ParseSourcesTOMLString parses a TOML string and returns a slice of Sources.
func ParseSourcesTOMLString(s string) ([]Source, error) {
	return ParseSourcesTOML([]byte(s))
}

// ParseSourcesJSON parses sources from JSON data (array of SourceConfig or object with "sources").
func ParseSourcesJSON(data []byte) ([]Source, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}
	if trimmed[0] == '[' {
		var configs []SourceConfig
		if err := json.Unmarshal(trimmed, &configs); err != nil {
			return nil, err
		}
		return FromConfigs(configs, ""), nil
	}
	var wrapper DeclarativeConfig
	if err := json.Unmarshal(trimmed, &wrapper); err != nil {
		return nil, err
	}
	return wrapper.ToSources(""), nil
}

// ParseSources parses declarative source configuration from either TOML or JSON bytes.
func ParseSources(data []byte) ([]Source, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, nil
	}
	if trimmed[0] == '{' || trimmed[0] == '[' {
		if sources, err := ParseSourcesJSON(trimmed); err == nil {
			return sources, nil
		}
	}
	return ParseSourcesTOML(data)
}

// LoadSourcesTOML parses TOML data and resolves relative paths against repoRoot.
func LoadSourcesTOML(data []byte, repoRoot string) ([]Source, error) {
	configs, err := parseSourcesTOMLData(data)
	if err != nil {
		return nil, err
	}
	return FromConfigs(configs, repoRoot), nil
}

// LoadSourcesTOMLFile loads and parses a TOML file from disk.
// If the file does not exist, it returns nil, nil (valid empty source).
func LoadSourcesTOMLFile(path string, repoRoot string) ([]Source, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	return LoadSourcesTOML(data, repoRoot)
}

// MergeSources combines base and overlay sources by ID. Sources in overlays
// replace matching IDs in base (preserving their position in base), while new
// sources from overlays are appended to the result.
func MergeSources(base []Source, overlays []Source) []Source {
	if len(overlays) == 0 {
		return append([]Source(nil), base...)
	}
	if len(base) == 0 {
		return append([]Source(nil), overlays...)
	}

	res := make([]Source, len(base))
	copy(res, base)
	indexByID := make(map[string]int, len(base))
	for i, s := range res {
		indexByID[s.ID] = i
	}

	for _, o := range overlays {
		if idx, exists := indexByID[o.ID]; exists {
			res[idx] = o
		} else {
			indexByID[o.ID] = len(res)
			res = append(res, o)
		}
	}
	return res
}

// LoadSources parses configData as declarative sources (TOML or JSON) if non-empty,
// resolves relative roots against repoRoot, and merges them with DefaultSources(repoRoot, home).
// If configData is empty, it returns DefaultSources(repoRoot, home).
func LoadSources(configData []byte, repoRoot, home string) ([]Source, error) {
	return LoadSourcesMerged(configData, repoRoot, home)
}

// LoadSourcesMerged parses configData and merges the resulting sources over DefaultSources(repoRoot, home).
func LoadSourcesMerged(configData []byte, repoRoot, home string) ([]Source, error) {
	base := DefaultSources(repoRoot, home)
	if len(bytes.TrimSpace(configData)) == 0 {
		return base, nil
	}
	parsed, err := ParseSources(configData)
	if err != nil {
		return nil, err
	}
	for i := range parsed {
		if repoRoot != "" && !filepath.IsAbs(parsed[i].Root) && parsed[i].Root != "" {
			parsed[i].Root = filepath.Join(repoRoot, parsed[i].Root)
		}
	}
	return MergeSources(base, parsed), nil
}

// LoadSourcesFallback parses configData. If configData is empty or defines zero sources,
// it falls back to DefaultSources(repoRoot, home). If configData defines sources,
// it returns those sources (resolved against repoRoot) without merging DefaultSources.
func LoadSourcesFallback(configData []byte, repoRoot, home string) ([]Source, error) {
	if len(bytes.TrimSpace(configData)) == 0 {
		return DefaultSources(repoRoot, home), nil
	}
	parsed, err := ParseSources(configData)
	if err != nil {
		return nil, err
	}
	if len(parsed) == 0 {
		return DefaultSources(repoRoot, home), nil
	}
	for i := range parsed {
		if repoRoot != "" && !filepath.IsAbs(parsed[i].Root) && parsed[i].Root != "" {
			parsed[i].Root = filepath.Join(repoRoot, parsed[i].Root)
		}
	}
	return parsed, nil
}
