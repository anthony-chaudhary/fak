package frontierswe

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/strmatch"
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
	if err := foldOptionalYAML(dir, "job.yaml", func(b []byte) error { return parseJobYAML(b, &t.Job) }); err != nil {
		return nil, err
	}
	if err := foldOptionalYAML(dir, "oracle.yaml", func(b []byte) error { return parseOracleYAML(b, &t.Oracle) }); err != nil {
		return nil, err
	}

	return t, nil
}

// foldOptionalYAML reads dir/name if it exists and applies parse to its bytes. A
// missing file is not an error (the field stays zero); a read error or a parse
// error is wrapped with the same frontierswe: prefix the two call sites shared.
func foldOptionalYAML(dir, name string, parse func([]byte) error) error {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return fmt.Errorf("frontierswe: read %s: %w", name, err)
	}
	if err := parse(b); err != nil {
		return fmt.Errorf("frontierswe: %s/%s: %w", dir, name, err)
	}
	return nil
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
		line := strings.TrimSpace(strmatch.StripUnquotedComment(raw, '#'))
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
			s, err := strmatch.ParseQuotedScalar(val)
			if err != nil {
				return fmt.Errorf("line %d: version: %w", line, err)
			}
			t.Version = s
		}
	case "metadata":
		switch key {
		case "difficulty":
			s, err := strmatch.ParseQuotedScalar(val)
			if err != nil {
				return fmt.Errorf("line %d: metadata.difficulty: %w", line, err)
			}
			t.Metadata.Difficulty = s
		case "category":
			s, err := strmatch.ParseQuotedScalar(val)
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
		return assignTimeoutSec(&t.Agent.TimeoutSec, section, key, val, line)
	case "verifier":
		return assignTimeoutSec(&t.Verifier.TimeoutSec, section, key, val, line)
	case "environment":
		return assignEnvironment(&t.Environment, key, val, line)
	}
	return nil
}

// assignTimeoutSec folds a `timeout_sec = <float>` line into dst. The agent and
// verifier sections carry the identical single float field; section names the
// table in the error message so the two callers stay byte-identical.
func assignTimeoutSec(dst *float64, section, key, val string, line int) error {
	if key == "timeout_sec" {
		f, err := tomlFloat(val)
		if err != nil {
			return fmt.Errorf("line %d: %s.timeout_sec: %w", line, section, err)
		}
		*dst = f
	}
	return nil
}

func assignEnvironment(e *Environment, key, val string, line int) error {
	switch key {
	case "docker_image":
		s, err := strmatch.ParseQuotedScalar(val)
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
	for _, part := range strmatch.SplitQuoted(body, ',') {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		s, err := strmatch.ParseQuotedScalar(part)
		if err != nil {
			return nil, err
		}
		out = append(out, s)
	}
	return out, nil
}

// --- strict, dependency-free YAML subset readers for job.yaml / oracle.yaml ---
//
// Only the documented top-level flat fields are accepted: `key: scalar`,
// `key: [a, b]`, or `key:` followed by exact `  - item` block-list lines.
// Unknown keys, duplicate keys, malformed quoting, malformed lists, indentation
// drift, and integer shape errors are all parse errors with source lines.

func parseJobYAML(data []byte, j *Job) error {
	fields, err := parseYAMLSubset(data, map[string]yamlFieldKind{
		"agents":              yamlFieldList,
		"n_attempts":          yamlFieldInt,
		"n_concurrent_trials": yamlFieldInt,
		"artifacts":           yamlFieldList,
	})
	if err != nil {
		return err
	}
	if v, ok := fields["agents"]; ok {
		j.Agents = append([]string(nil), v.list...)
	}
	if v, ok := fields["n_attempts"]; ok {
		j.NAttempts = v.number
	}
	if v, ok := fields["n_concurrent_trials"]; ok {
		j.NConcurrentTrial = v.number
	}
	if v, ok := fields["artifacts"]; ok {
		j.Artifacts = append([]string(nil), v.list...)
	}
	return nil
}

func parseOracleYAML(data []byte, o *Oracle) error {
	fields, err := parseYAMLSubset(data, map[string]yamlFieldKind{
		"command":    yamlFieldScalar,
		"reward_key": yamlFieldScalar,
	})
	if err != nil {
		return err
	}
	if v, ok := fields["command"]; ok {
		o.Command = v.scalar
	}
	if v, ok := fields["reward_key"]; ok {
		o.RewardKey = v.scalar
	}
	return nil
}

type yamlFieldKind int

const (
	yamlFieldScalar yamlFieldKind = iota + 1
	yamlFieldList
	yamlFieldInt
)

type yamlValue struct {
	scalar string
	list   []string
	number int
}

func parseYAMLSubset(data []byte, allowed map[string]yamlFieldKind) (map[string]yamlValue, error) {
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	out := make(map[string]yamlValue, len(allowed))
	seen := make(map[string]struct{}, len(allowed))

	for i := 0; i < len(lines); i++ {
		lineNo := i + 1
		line, err := stripYAMLCommentStrict(lines[i], lineNo)
		if err != nil {
			return nil, err
		}
		if strings.TrimSpace(line) == "" {
			continue
		}
		if line[0] == ' ' || line[0] == '\t' {
			return nil, fmt.Errorf("line %d: unexpected indentation", lineNo)
		}
		if line[0] == '-' {
			return nil, fmt.Errorf("line %d: malformed line %q", lineNo, strings.TrimSpace(line))
		}

		key, _, ok := strings.Cut(line, ":")
		if !ok {
			return nil, fmt.Errorf("line %d: malformed line %q", lineNo, strings.TrimSpace(line))
		}
		key = strings.TrimSpace(key)
		kind, ok := allowed[key]
		if !ok {
			return nil, fmt.Errorf("line %d: unknown field %q", lineNo, key)
		}
		if _, dup := seen[key]; dup {
			return nil, fmt.Errorf("line %d: duplicate field %q", lineNo, key)
		}
		seen[key] = struct{}{}

		value, next, err := parseYAMLFieldValue(lines, i, kind)
		if err != nil {
			return nil, err
		}
		out[key] = value
		i = next
	}
	return out, nil
}

func parseYAMLFieldValue(lines []string, index int, kind yamlFieldKind) (yamlValue, int, error) {
	lineNo := index + 1
	line, err := stripYAMLCommentStrict(lines[index], lineNo)
	if err != nil {
		return yamlValue{}, index, err
	}
	_, rest, _ := strings.Cut(line, ":")
	rest = strings.TrimSpace(rest)

	switch kind {
	case yamlFieldScalar:
		if rest == "|" || rest == ">" || strings.HasPrefix(rest, "| ") || strings.HasPrefix(rest, "> ") {
			return yamlValue{}, index, fmt.Errorf("line %d: unsupported block scalar", lineNo)
		}
		scalar, err := parseYAMLScalar(rest, lineNo)
		if err != nil {
			return yamlValue{}, index, err
		}
		return yamlValue{scalar: scalar}, index, nil
	case yamlFieldInt:
		scalar, err := parseYAMLScalar(rest, lineNo)
		if err != nil {
			return yamlValue{}, index, err
		}
		n, err := parseYAMLInt(scalar, lineNo)
		if err != nil {
			return yamlValue{}, index, err
		}
		return yamlValue{scalar: scalar, number: n}, index, nil
	case yamlFieldList:
		if rest == "" {
			list, next, err := parseYAMLBlockList(lines, index)
			if err != nil {
				return yamlValue{}, index, err
			}
			return yamlValue{list: list}, next, nil
		}
		if !strings.HasPrefix(rest, "[") {
			return yamlValue{}, index, fmt.Errorf("line %d: expected list", lineNo)
		}
		list, err := parseYAMLInlineList(rest, lineNo)
		if err != nil {
			return yamlValue{}, index, err
		}
		return yamlValue{list: list}, index, nil
	default:
		return yamlValue{}, index, fmt.Errorf("line %d: unsupported field kind", lineNo)
	}
}

func parseYAMLBlockList(lines []string, index int) ([]string, int, error) {
	var out []string
	next := index
	for j := index + 1; j < len(lines); j++ {
		lineNo := j + 1
		line, err := stripYAMLCommentStrict(lines[j], lineNo)
		if err != nil {
			return nil, index, err
		}
		if strings.TrimSpace(line) == "" {
			next = j
			continue
		}
		if line[0] != ' ' && line[0] != '\t' {
			return out, next, nil
		}
		if !strings.HasPrefix(line, "  - ") {
			return nil, index, fmt.Errorf("line %d: invalid list indentation", lineNo)
		}
		item, err := parseYAMLScalar(strings.TrimSpace(line[4:]), lineNo)
		if err != nil {
			return nil, index, err
		}
		out = append(out, item)
		next = j
	}
	return out, next, nil
}

func parseYAMLInlineList(s string, line int) ([]string, error) {
	if !strings.HasSuffix(s, "]") {
		return nil, fmt.Errorf("line %d: missing closing ]", line)
	}
	body := strings.TrimSpace(s[1 : len(s)-1])
	if body == "" {
		return []string{}, nil
	}

	var (
		parts   []string
		buf     strings.Builder
		inQuote byte
	)
	for i := 0; i < len(body); i++ {
		ch := body[i]
		switch inQuote {
		case '"':
			buf.WriteByte(ch)
			if ch == '\\' {
				i++
				if i >= len(body) {
					return nil, fmt.Errorf("line %d: unterminated double-quoted string", line)
				}
				buf.WriteByte(body[i])
				continue
			}
			if ch == '"' {
				inQuote = 0
			}
		case '\'':
			buf.WriteByte(ch)
			if ch == '\'' {
				if i+1 < len(body) && body[i+1] == '\'' {
					buf.WriteByte(body[i+1])
					i++
					continue
				}
				inQuote = 0
			}
		default:
			switch ch {
			case '"', '\'':
				inQuote = ch
				buf.WriteByte(ch)
			case ',':
				part := strings.TrimSpace(buf.String())
				if part == "" {
					return nil, fmt.Errorf("line %d: empty list item", line)
				}
				parts = append(parts, part)
				buf.Reset()
			case '[', ']':
				return nil, fmt.Errorf("line %d: nested or malformed inline list", line)
			default:
				buf.WriteByte(ch)
			}
		}
	}
	if inQuote != 0 {
		return nil, fmt.Errorf("line %d: unterminated quoted string", line)
	}
	part := strings.TrimSpace(buf.String())
	if part == "" {
		return nil, fmt.Errorf("line %d: empty list item", line)
	}
	parts = append(parts, part)

	out := make([]string, 0, len(parts))
	for _, part := range parts {
		item, err := parseYAMLScalar(part, line)
		if err != nil {
			return nil, err
		}
		out = append(out, item)
	}
	return out, nil
}

func stripYAMLCommentStrict(s string, line int) (string, error) {
	var inQuote byte
	for i := 0; i < len(s); i++ {
		ch := s[i]
		switch inQuote {
		case '"':
			if ch == '\\' {
				i++
				continue
			}
			if ch == '"' {
				inQuote = 0
			}
		case '\'':
			if ch == '\'' {
				if i+1 < len(s) && s[i+1] == '\'' {
					i++
					continue
				}
				inQuote = 0
			}
		default:
			switch ch {
			case '"', '\'':
				inQuote = ch
			case '#':
				return s[:i], nil
			}
		}
	}
	if inQuote == '"' {
		return "", fmt.Errorf("line %d: unterminated double-quoted string", line)
	}
	if inQuote == '\'' {
		return "", fmt.Errorf("line %d: unterminated single-quoted string", line)
	}
	return s, nil
}

func parseYAMLScalar(s string, line int) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	if s[0] == '"' {
		if len(s) < 2 || s[len(s)-1] != '"' {
			return "", fmt.Errorf("line %d: unterminated double-quoted string", line)
		}
		v, err := strconv.Unquote(s)
		if err != nil {
			return "", fmt.Errorf("line %d: invalid double-quoted string: %w", line, err)
		}
		return v, nil
	}
	if s[0] == '\'' {
		if len(s) < 2 || s[len(s)-1] != '\'' {
			return "", fmt.Errorf("line %d: unterminated single-quoted string", line)
		}
		body := s[1 : len(s)-1]
		var out strings.Builder
		for i := 0; i < len(body); i++ {
			if body[i] != '\'' {
				out.WriteByte(body[i])
				continue
			}
			if i+1 >= len(body) || body[i+1] != '\'' {
				return "", fmt.Errorf("line %d: invalid single-quoted string", line)
			}
			out.WriteByte('\'')
			i++
		}
		return out.String(), nil
	}
	if strings.ContainsAny(s, "[]") {
		return "", fmt.Errorf("line %d: nested or malformed inline list", line)
	}
	return s, nil
}

func parseYAMLInt(s string, line int) (int, error) {
	n, err := strconv.Atoi(strings.TrimSpace(s))
	if err != nil {
		return 0, fmt.Errorf("line %d: expected integer, got %q", line, s)
	}
	return n, nil
}
