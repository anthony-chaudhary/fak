package frontierswe

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

// LoadTask reads a FrontierSWE task directory (tasks/<name>/) into a typed Task.
// It parses task.toml (required) and folds job.yaml and oracle.yaml when those
// files are present. The task's Name is the directory's base name, and
// ScoringCategory is overlaid from the Catalog when the name is one of the 17
// known tasks (an unknown name leaves it empty — not an error, so the loader is
// useful on a task the catalog has not yet learned).
//
// No network, no Docker, no model: this is pure file parsing over the committed
// shape, which is what makes the rest of the epic offline-testable.
func LoadTask(dir string) (*Task, error) {
	tomlPath := filepath.Join(dir, "task.toml")
	b, err := os.ReadFile(tomlPath)
	if err != nil {
		return nil, fmt.Errorf("frontierswe: read task.toml: %w", err)
	}

	t := &Task{Name: filepath.Base(filepath.Clean(dir))}
	if err := parseTaskTOML(b, t); err != nil {
		return nil, fmt.Errorf("frontierswe: %s: %w", tomlPath, err)
	}
	if c, ok := CategoryOf(t.Name); ok {
		t.ScoringCategory = c
	}

	// job.yaml / oracle.yaml are optional; absence is fine, a parse error is not.
	if jb, err := os.ReadFile(filepath.Join(dir, "job.yaml")); err == nil {
		if err := parseJobYAML(jb, &t.Job); err != nil {
			return nil, fmt.Errorf("frontierswe: %s/job.yaml: %w", dir, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("frontierswe: read job.yaml: %w", err)
	}
	if ob, err := os.ReadFile(filepath.Join(dir, "oracle.yaml")); err == nil {
		if err := parseOracleYAML(ob, &t.Oracle); err != nil {
			return nil, fmt.Errorf("frontierswe: %s/oracle.yaml: %w", dir, err)
		}
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("frontierswe: read oracle.yaml: %w", err)
	}

	return t, nil
}

// --- minimal, self-contained TOML reader for the flat task.toml shape ---
//
// task.toml is a tiny, fixed subset of TOML: a top-level `version = "..."` line
// and a sequence of [section] tables (metadata / agent / verifier / environment),
// each with bare `key = value` lines where value is a quoted string, a number, a
// bool, or a ["a","b"] string array. Plus '#' comments and blank lines. Parsing
// it here keeps the package dependency-free (no TOML library is in the module);
// the precedent is internal/corelocks' hand-rolled array-of-tables reader.
// Anything outside this grammar is reported as a parse error.

func parseTaskTOML(data []byte, t *Task) error {
	section := "" // current [section]; "" is the top-level table
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	for n, raw := range lines {
		line := strings.TrimSpace(stripTOMLComment(raw))
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[[") {
			return fmt.Errorf("line %d: array-of-tables [[...]] not supported in task.toml", n+1)
		}
		if strings.HasPrefix(line, "[") {
			if !strings.HasSuffix(line, "]") {
				return fmt.Errorf("line %d: malformed section header %q", n+1, line)
			}
			section = strings.TrimSpace(line[1 : len(line)-1])
			if section == "" {
				return fmt.Errorf("line %d: empty section header", n+1)
			}
			continue
		}
		key, val, ok := strings.Cut(line, "=")
		if !ok {
			return fmt.Errorf("line %d: not a key = value line: %q", n+1, line)
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		if key == "" {
			return fmt.Errorf("line %d: empty key", n+1)
		}
		if err := assignTOML(t, section, key, val, n+1); err != nil {
			return err
		}
	}
	return nil
}

// assignTOML routes a single key=value line into the right Task field by its
// (section, key). An unknown key inside a known section is ignored (forward
// compatibility — upstream may add fields the spine does not model yet); an
// unknown section is ignored for the same reason.
func assignTOML(t *Task, section, key, val string, line int) error {
	switch section {
	case "":
		if key == "version" {
			s, err := tomlString(val)
			if err != nil {
				return fmt.Errorf("line %d: version: %w", line, err)
			}
			t.Version = s
		}
	case "metadata":
		switch key {
		case "difficulty":
			s, err := tomlString(val)
			if err != nil {
				return fmt.Errorf("line %d: metadata.difficulty: %w", line, err)
			}
			t.Metadata.Difficulty = s
		case "category":
			s, err := tomlString(val)
			if err != nil {
				return fmt.Errorf("line %d: metadata.category: %w", line, err)
			}
			t.Metadata.Category = s
		case "tags":
			arr, err := tomlStringArray(val)
			if err != nil {
				return fmt.Errorf("line %d: metadata.tags: %w", line, err)
			}
			t.Metadata.Tags = arr
		}
	case "agent":
		if key == "timeout_sec" {
			f, err := tomlFloat(val)
			if err != nil {
				return fmt.Errorf("line %d: agent.timeout_sec: %w", line, err)
			}
			t.Agent.TimeoutSec = f
		}
	case "verifier":
		if key == "timeout_sec" {
			f, err := tomlFloat(val)
			if err != nil {
				return fmt.Errorf("line %d: verifier.timeout_sec: %w", line, err)
			}
			t.Verifier.TimeoutSec = f
		}
	case "environment":
		return assignEnvironment(&t.Environment, key, val, line)
	}
	return nil
}

func assignEnvironment(e *Environment, key, val string, line int) error {
	switch key {
	case "docker_image":
		s, err := tomlString(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.docker_image: %w", line, err)
		}
		e.DockerImage = s
	case "build_timeout_sec":
		f, err := tomlFloat(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.build_timeout_sec: %w", line, err)
		}
		e.BuildTimeoutSec = f
	case "cpus":
		i, err := tomlInt(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.cpus: %w", line, err)
		}
		e.CPUs = i
	case "memory_mb":
		i, err := tomlInt(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.memory_mb: %w", line, err)
		}
		e.MemoryMB = i
	case "storage_mb":
		i, err := tomlInt(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.storage_mb: %w", line, err)
		}
		e.StorageMB = i
	case "gpus":
		i, err := tomlInt(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.gpus: %w", line, err)
		}
		e.GPUs = i
	case "allow_internet":
		b, err := tomlBool(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.allow_internet: %w", line, err)
		}
		e.AllowInternet = b
	case "mcp_servers":
		arr, err := tomlStringArray(val)
		if err != nil {
			return fmt.Errorf("line %d: environment.mcp_servers: %w", line, err)
		}
		e.MCPServers = arr
	}
	return nil
}

// stripTOMLComment removes a trailing '#' comment that is not inside a string.
// task.toml has no '#' inside its string/number literals, so a first-unquoted-#
// scan is sufficient.
func stripTOMLComment(s string) string {
	inStr := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"':
			inStr = !inStr
		case '#':
			if !inStr {
				return s[:i]
			}
		}
	}
	return s
}

func tomlString(v string) (string, error) {
	if len(v) < 2 || v[0] != '"' || v[len(v)-1] != '"' {
		return "", fmt.Errorf("expected a double-quoted string, got %q", v)
	}
	inner := v[1 : len(v)-1]
	if strings.Contains(inner, `"`) {
		return "", fmt.Errorf("unexpected quote inside string %q", v)
	}
	return inner, nil
}

func tomlFloat(v string) (float64, error) {
	f, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
	if err != nil {
		return 0, fmt.Errorf("expected a number, got %q", v)
	}
	return f, nil
}

func tomlInt(v string) (int, error) {
	// Tolerate a float-shaped integer (e.g. "8.0") as well as a bare int.
	s := strings.TrimSpace(v)
	if i, err := strconv.Atoi(s); err == nil {
		return i, nil
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil || f != float64(int(f)) {
		return 0, fmt.Errorf("expected an integer, got %q", v)
	}
	return int(f), nil
}

func tomlBool(v string) (bool, error) {
	switch strings.TrimSpace(v) {
	case "true":
		return true, nil
	case "false":
		return false, nil
	default:
		return false, fmt.Errorf("expected true/false, got %q", v)
	}
}

func tomlStringArray(v string) ([]string, error) {
	v = strings.TrimSpace(v)
	if len(v) < 2 || v[0] != '[' || v[len(v)-1] != ']' {
		return nil, fmt.Errorf("expected a [\"...\"] array, got %q", v)
	}
	body := strings.TrimSpace(v[1 : len(v)-1])
	if body == "" {
		return []string{}, nil
	}
	var out []string
	for _, part := range splitTOMLArray(body) {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		s, err := tomlString(part)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// splitTOMLArray splits an array body on top-level commas (respecting quotes).
// The grammar has no nested arrays, so a plain quote-aware split is enough.
func splitTOMLArray(body string) []string {
	var parts []string
	var b strings.Builder
	inStr := false
	for i := 0; i < len(body); i++ {
		c := body[i]
		switch c {
		case '"':
			inStr = !inStr
			b.WriteByte(c)
		case ',':
			if inStr {
				b.WriteByte(c)
			} else {
				parts = append(parts, b.String())
				b.Reset()
			}
		default:
			b.WriteByte(c)
		}
	}
	parts = append(parts, b.String())
	return parts
}

// --- tolerant YAML-ish line readers for job.yaml / oracle.yaml ---
//
// Only the named flat fields are read: a `key: scalar` line, a `key: [a, b]`
// inline array, or a `key:` header followed by `  - item` block-list lines. This
// is deliberately a small tolerant reader, not a YAML implementation — it folds
// the spine's fields and ignores everything else.

func parseJobYAML(data []byte, j *Job) error {
	fields := scanYAML(data)
	if v, ok := fields["agents"]; ok {
		j.Agents = v.list
	}
	if v, ok := fields["n_attempts"]; ok {
		if n, err := yamlInt(v.scalar); err == nil {
			j.NAttempts = n
		}
	}
	if v, ok := fields["n_concurrent_trials"]; ok {
		if n, err := yamlInt(v.scalar); err == nil {
			j.NConcurrentTrial = n
		}
	}
	if v, ok := fields["artifacts"]; ok {
		j.Artifacts = v.list
	}
	return nil
}

func parseOracleYAML(data []byte, o *Oracle) error {
	fields := scanYAML(data)
	if v, ok := fields["command"]; ok {
		o.Command = v.scalar
	}
	if v, ok := fields["reward_key"]; ok {
		o.RewardKey = v.scalar
	}
	return nil
}

type yamlValue struct {
	scalar string
	list   []string
}

// scanYAML reads top-level (column-0) `key:` lines and, for each, captures
// either an inline scalar, an inline [a, b] array, or a following indented
// `- item` block list. Nested mappings and anything past the named fields are
// ignored — this is a field folder, not a parser.
func scanYAML(data []byte) map[string]yamlValue {
	out := map[string]yamlValue{}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	for i := 0; i < len(lines); i++ {
		raw := lines[i]
		line := stripYAMLComment(raw)
		if strings.TrimSpace(line) == "" {
			continue
		}
		// Only top-level keys (no leading whitespace) start a field.
		if line[0] == ' ' || line[0] == '\t' || line[0] == '-' {
			continue
		}
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		rest = strings.TrimSpace(rest)
		switch {
		case rest == "":
			// A block list may follow on indented `- item` lines.
			var list []string
			for j := i + 1; j < len(lines); j++ {
				nxt := stripYAMLComment(lines[j])
				if strings.TrimSpace(nxt) == "" {
					continue
				}
				trimmed := strings.TrimSpace(nxt)
				if (nxt[0] == ' ' || nxt[0] == '\t') && strings.HasPrefix(trimmed, "- ") {
					list = append(list, yamlScalar(strings.TrimSpace(trimmed[2:])))
					i = j
					continue
				}
				break
			}
			out[key] = yamlValue{list: list}
		case strings.HasPrefix(rest, "["):
			out[key] = yamlValue{list: yamlInlineArray(rest)}
		default:
			out[key] = yamlValue{scalar: yamlScalar(rest)}
		}
	}
	return out
}

func stripYAMLComment(s string) string {
	inStr := false
	for i := 0; i < len(s); i++ {
		switch s[i] {
		case '"', '\'':
			inStr = !inStr
		case '#':
			if !inStr {
				return s[:i]
			}
		}
	}
	return s
}

// yamlScalar strips a matching pair of single or double quotes from a scalar.
func yamlScalar(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '"' && s[len(s)-1] == '"') || (s[0] == '\'' && s[len(s)-1] == '\'') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func yamlInlineArray(s string) []string {
	s = strings.TrimSpace(s)
	if len(s) < 2 || s[0] != '[' || s[len(s)-1] != ']' {
		return nil
	}
	body := strings.TrimSpace(s[1 : len(s)-1])
	if body == "" {
		return []string{}
	}
	var out []string
	for _, part := range strings.Split(body, ",") {
		if v := yamlScalar(part); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func yamlInt(s string) (int, error) {
	return strconv.Atoi(strings.TrimSpace(s))
}
