package toolcatalog

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"regexp"
	"sort"
	"strings"

	frontmattercodec "github.com/anthony-chaudhary/fak/internal/frontmatter"
)

const (
	ProgramVersion  = "fak.skill-program/v1"
	MaxProgramBytes = 256 << 10
)

var (
	toolNameRE  = regexp.MustCompile(`^[A-Za-z0-9_-]{1,64}$`)
	aliasNameRE = regexp.MustCompile(`^[A-Za-z0-9_.:/-]{1,128}$`)
)

// Executor is host-only registration data. It is deliberately absent from ModelTool.
type Executor struct {
	Argv    []string          `json:"argv"`
	Dir     string            `json:"dir,omitempty"`
	Adapter *CommandAdapterV1 `json:"adapter,omitempty"`
}

// CommandAdapterV1 is an explicit JSON-object to process contract. Every
// argument, stdin byte, and environment value is declared; nothing is inferred
// from prose, schema order, or a shell command string.
type CommandAdapterV1 struct {
	Version string            `json:"version"`
	Argv    []ArgvBinding     `json:"argv,omitempty"`
	Stdin   *ValueBinding     `json:"stdin,omitempty"`
	Env     map[string]string `json:"env,omitempty"`
	Result  string            `json:"result"`
}

type ArgvBinding struct {
	Literal   string `json:"literal,omitempty"`
	Field     string `json:"field,omitempty"`
	Flag      string `json:"flag,omitempty"`
	OmitEmpty bool   `json:"omit_empty,omitempty"`
}

type ValueBinding struct {
	Field string `json:"field"`
}

// SkillProgram is the machine-authored portion of a skill. Prose is never inferred
// into execution: a SKILL.md must carry exactly one fak-program JSON block.
type SkillProgram struct {
	Version     string            `json:"version"`
	Name        string            `json:"name"`
	Description string            `json:"description,omitempty"`
	InputSchema json.RawMessage   `json:"input_schema"`
	Executor    Executor          `json:"executor"`
	Aliases     map[string]string `json:"aliases,omitempty"`
}

// Registration is the host's executable identity. Digest binds all semantics,
// including the command, schema, aliases, and stable source identity.
type Registration struct {
	Program SkillProgram `json:"program"`
	Source  string       `json:"source"`
	Digest  string       `json:"digest"`
}

// ModelTool is the only tool shape intended for provider requests. Host command
// details never cross this boundary.
type ModelTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
	Canonical   string          `json:"canonical_name"`
	Digest      string          `json:"registration_digest"`
}

// Omission makes the negative half of model availability inspectable.
type Omission struct {
	Name   string `json:"name"`
	Reason string `json:"reason"`
}

// Snapshot is the exact, content-addressed tool view supplied to a model.
type Snapshot struct {
	Dialect string      `json:"dialect"`
	Tools   []ModelTool `json:"tools"`
	Omitted []Omission  `json:"omitted,omitempty"`
	Digest  string      `json:"digest"`
}

// CompileSkill deterministically compiles an explicit fenced program in SKILL.md.
// Natural-language instructions remain model guidance; they are not executable IR.
func CompileSkill(src []byte, source string) (Registration, error) {
	name, description, err := frontmatter(src)
	if err != nil {
		return Registration{}, err
	}
	body, err := programBlock(src)
	if err != nil {
		return Registration{}, err
	}
	if len(body) > MaxProgramBytes {
		return Registration{}, fmt.Errorf("SKILL_PROGRAM_TOO_LARGE: %d > %d bytes", len(body), MaxProgramBytes)
	}
	normalized, err := normalizeJSON(body)
	if err != nil {
		return Registration{}, fmt.Errorf("SKILL_PROGRAM_INVALID: %w", err)
	}
	var p SkillProgram
	dec := json.NewDecoder(bytes.NewReader(normalized))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&p); err != nil {
		return Registration{}, fmt.Errorf("SKILL_PROGRAM_INVALID: %w", err)
	}
	if p.Name == "" {
		p.Name = name
	}
	if p.Description == "" {
		p.Description = description
	}
	p.InputSchema, err = normalizeJSON(p.InputSchema)
	if err != nil {
		return Registration{}, fmt.Errorf("SKILL_PROGRAM_SCHEMA: %w", err)
	}
	if err := validateProgram(p); err != nil {
		return Registration{}, err
	}
	return newRegistration(p, source)
}

// Expose builds the exact model-visible set. Registration alone never implies
// exposure: canonical names must be explicitly selected after policy filtering.
func Expose(regs []Registration, selected []string, dialect string) (Snapshot, error) {
	byName := make(map[string]Registration, len(regs))
	for _, r := range regs {
		if err := ValidateRegistration(r); err != nil {
			return Snapshot{}, err
		}
		if _, exists := byName[r.Program.Name]; exists {
			return Snapshot{}, fmt.Errorf("TOOL_REGISTRATION_COLLISION: %s", r.Program.Name)
		}
		byName[r.Program.Name] = r
	}
	allowed := make(map[string]bool, len(selected))
	for _, name := range selected {
		if _, ok := byName[name]; !ok {
			return Snapshot{}, fmt.Errorf("TOOL_SELECTION_UNKNOWN: %s", name)
		}
		allowed[name] = true
	}
	var out Snapshot
	out.Dialect = dialect
	for name, r := range byName {
		if !allowed[name] {
			out.Omitted = append(out.Omitted, Omission{Name: name, Reason: "NOT_SELECTED"})
			continue
		}
		visible := name
		if alias := r.Program.Aliases[dialect]; alias != "" {
			visible = alias
		}
		if !aliasNameRE.MatchString(visible) {
			return Snapshot{}, fmt.Errorf("TOOL_ALIAS_INVALID: %s=%q", dialect, visible)
		}
		out.Tools = append(out.Tools, ModelTool{Name: visible, Description: r.Program.Description, InputSchema: r.Program.InputSchema, Canonical: name, Digest: r.Digest})
	}
	sort.Slice(out.Tools, func(i, j int) bool { return out.Tools[i].Name < out.Tools[j].Name })
	sort.Slice(out.Omitted, func(i, j int) bool { return out.Omitted[i].Name < out.Omitted[j].Name })
	seen := map[string]string{}
	for _, tool := range out.Tools {
		if prior := seen[tool.Name]; prior != "" {
			return Snapshot{}, fmt.Errorf("TOOL_DIALECT_COLLISION: %s and %s expose as %s", prior, tool.Canonical, tool.Name)
		}
		seen[tool.Name] = tool.Canonical
	}
	canonical, err := json.Marshal(struct {
		Dialect string      `json:"dialect"`
		Tools   []ModelTool `json:"tools"`
		Omitted []Omission  `json:"omitted,omitempty"`
	}{out.Dialect, out.Tools, out.Omitted})
	if err != nil {
		return Snapshot{}, err
	}
	sum := sha256.Sum256(canonical)
	out.Digest = "sha256:" + hex.EncodeToString(sum[:])
	return out, nil
}

// ValidateSnapshot proves that model-facing names, schemas, canonical identities,
// registration digests, and omissions still match the content-addressed selection.
func ValidateSnapshot(snapshot Snapshot) error {
	seenVisible := make(map[string]string, len(snapshot.Tools))
	seenCanonical := make(map[string]bool, len(snapshot.Tools))
	for _, tool := range snapshot.Tools {
		if !aliasNameRE.MatchString(tool.Name) || !aliasNameRE.MatchString(tool.Canonical) {
			return fmt.Errorf("TOOL_SNAPSHOT_NAME_INVALID: %q/%q", tool.Name, tool.Canonical)
		}
		if tool.Digest == "" {
			return fmt.Errorf("TOOL_SNAPSHOT_REGISTRATION_DIGEST_MISSING: %s", tool.Canonical)
		}
		schema, err := normalizeJSON(tool.InputSchema)
		if err != nil || !bytes.Equal(schema, tool.InputSchema) {
			return fmt.Errorf("TOOL_SNAPSHOT_SCHEMA_INVALID: %s", tool.Canonical)
		}
		if prior := seenVisible[tool.Name]; prior != "" {
			return fmt.Errorf("TOOL_DIALECT_COLLISION: %s and %s expose as %s", prior, tool.Canonical, tool.Name)
		}
		if seenCanonical[tool.Canonical] {
			return fmt.Errorf("TOOL_SNAPSHOT_CANONICAL_COLLISION: %s", tool.Canonical)
		}
		seenVisible[tool.Name] = tool.Canonical
		seenCanonical[tool.Canonical] = true
	}
	canonical, err := json.Marshal(struct {
		Dialect string      `json:"dialect"`
		Tools   []ModelTool `json:"tools"`
		Omitted []Omission  `json:"omitted,omitempty"`
	}{snapshot.Dialect, snapshot.Tools, snapshot.Omitted})
	if err != nil {
		return err
	}
	sum := sha256.Sum256(canonical)
	want := "sha256:" + hex.EncodeToString(sum[:])
	if snapshot.Digest == "" || snapshot.Digest != want {
		return fmt.Errorf("TOOL_SNAPSHOT_DIGEST_MISMATCH: got %s want %s", snapshot.Digest, want)
	}
	return nil
}

// ValidateRegistration proves that host execution and model metadata still
// match the digest minted at compilation. Expose refuses hand-built or mutated registrations.
func ValidateRegistration(r Registration) error {
	if err := validateProgram(r.Program); err != nil {
		return err
	}
	want, err := newRegistration(r.Program, r.Source)
	if err != nil {
		return err
	}
	if r.Digest == "" || r.Digest != want.Digest {
		return fmt.Errorf("TOOL_REGISTRATION_DIGEST_MISMATCH: %s", r.Program.Name)
	}
	return nil
}

func newRegistration(p SkillProgram, source string) (Registration, error) {
	canonical, err := canonicalProgram(p)
	if err != nil {
		return Registration{}, err
	}
	h := sha256.New()
	h.Write([]byte(source))
	h.Write([]byte{0})
	h.Write(canonical)
	return Registration{Program: p, Source: source, Digest: "sha256:" + hex.EncodeToString(h.Sum(nil))}, nil
}

func canonicalProgram(p SkillProgram) ([]byte, error) {
	type alias struct {
		Dialect string `json:"dialect"`
		Name    string `json:"name"`
	}
	type wire struct {
		Version     string          `json:"version"`
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		InputSchema json.RawMessage `json:"input_schema"`
		Executor    Executor        `json:"executor"`
		Aliases     []alias         `json:"aliases,omitempty"`
	}
	aliases := make([]alias, 0, len(p.Aliases))
	for dialect, name := range p.Aliases {
		aliases = append(aliases, alias{Dialect: dialect, Name: name})
	}
	sort.Slice(aliases, func(i, j int) bool { return aliases[i].Dialect < aliases[j].Dialect })
	return json.Marshal(wire{p.Version, p.Name, p.Description, p.InputSchema, p.Executor, aliases})
}

func validateProgram(p SkillProgram) error {
	if p.Version != ProgramVersion {
		return fmt.Errorf("SKILL_PROGRAM_VERSION: want %s", ProgramVersion)
	}
	if !toolNameRE.MatchString(p.Name) {
		return fmt.Errorf("SKILL_PROGRAM_NAME: %q is not provider-portable", p.Name)
	}
	if len(p.Executor.Argv) == 0 || strings.TrimSpace(p.Executor.Argv[0]) == "" {
		return fmt.Errorf("SKILL_PROGRAM_EXECUTOR: argv is required")
	}
	for i, arg := range p.Executor.Argv {
		if strings.IndexByte(arg, 0) >= 0 {
			return fmt.Errorf("SKILL_PROGRAM_EXECUTOR: argv[%d] contains NUL", i)
		}
	}
	var schema map[string]any
	if err := json.Unmarshal(p.InputSchema, &schema); err != nil {
		return fmt.Errorf("SKILL_PROGRAM_SCHEMA: %w", err)
	}
	if schema["type"] != "object" {
		return fmt.Errorf("SKILL_PROGRAM_SCHEMA: root type must be object")
	}
	if adapter := p.Executor.Adapter; adapter != nil {
		if adapter.Version != "fak.command-adapter/v1" {
			return fmt.Errorf("SKILL_PROGRAM_ADAPTER_VERSION: want fak.command-adapter/v1")
		}
		if adapter.Result != "json" {
			return fmt.Errorf("SKILL_PROGRAM_ADAPTER_RESULT: want json")
		}
		declared := map[string]bool{}
		if properties, ok := schema["properties"].(map[string]any); ok {
			for name := range properties {
				declared[name] = true
			}
		}
		checkField := func(field string) error {
			if field == "" || !declared[field] {
				return fmt.Errorf("SKILL_PROGRAM_ADAPTER_FIELD: %q is not declared in input_schema.properties", field)
			}
			return nil
		}
		for i, binding := range adapter.Argv {
			if (binding.Literal == "") == (binding.Field == "") {
				return fmt.Errorf("SKILL_PROGRAM_ADAPTER_ARGV: binding %d requires exactly one of literal or field", i)
			}
			if binding.Field != "" {
				if err := checkField(binding.Field); err != nil {
					return err
				}
			}
		}
		if adapter.Stdin != nil {
			if err := checkField(adapter.Stdin.Field); err != nil {
				return err
			}
		}
		for env, field := range adapter.Env {
			if strings.TrimSpace(env) == "" || strings.Contains(env, "=") || strings.IndexByte(env, 0) >= 0 {
				return fmt.Errorf("SKILL_PROGRAM_ADAPTER_ENV: invalid name %q", env)
			}
			if err := checkField(field); err != nil {
				return err
			}
		}
	}
	for dialect, alias := range p.Aliases {
		if strings.TrimSpace(dialect) == "" || !aliasNameRE.MatchString(alias) {
			return fmt.Errorf("SKILL_PROGRAM_ALIAS: %q=%q", dialect, alias)
		}
	}
	return nil
}

// normalizeJSON rejects duplicate object keys and trailing values, then emits
// encoding/json's stable object-key order. JSON numbers retain their source spelling.
func normalizeJSON(raw []byte) ([]byte, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	value, err := readJSONValue(dec)
	if err != nil {
		return nil, err
	}
	if tok, err := dec.Token(); err != io.EOF {
		if err != nil {
			return nil, err
		}
		return nil, fmt.Errorf("trailing JSON value %v", tok)
	}
	return json.Marshal(value)
}

func readJSONValue(dec *json.Decoder) (any, error) {
	tok, err := dec.Token()
	if err != nil {
		return nil, err
	}
	delim, composite := tok.(json.Delim)
	if !composite {
		return tok, nil
	}
	switch delim {
	case '{':
		out := map[string]any{}
		for dec.More() {
			keyToken, err := dec.Token()
			if err != nil {
				return nil, err
			}
			key, ok := keyToken.(string)
			if !ok {
				return nil, fmt.Errorf("object key is %T", keyToken)
			}
			if _, duplicate := out[key]; duplicate {
				return nil, fmt.Errorf("duplicate object key %q", key)
			}
			value, err := readJSONValue(dec)
			if err != nil {
				return nil, err
			}
			out[key] = value
		}
		if end, err := dec.Token(); err != nil || end != json.Delim('}') {
			return nil, fmt.Errorf("object close: %v %v", end, err)
		}
		return out, nil
	case '[':
		var out []any
		for dec.More() {
			value, err := readJSONValue(dec)
			if err != nil {
				return nil, err
			}
			out = append(out, value)
		}
		if end, err := dec.Token(); err != nil || end != json.Delim(']') {
			return nil, fmt.Errorf("array close: %v %v", end, err)
		}
		return out, nil
	default:
		return nil, fmt.Errorf("unexpected delimiter %q", delim)
	}
}

func frontmatter(src []byte) (string, string, error) {
	lines := strings.Split(strings.ReplaceAll(string(src), "\r\n", "\n"), "\n")
	if len(lines) < 3 || lines[0] != "---" {
		return "", "", fmt.Errorf("SKILL_FRONTMATTER_REQUIRED")
	}
	var name, description string
	closed := false
	for _, line := range lines[1:] {
		if line == "---" {
			closed = true
			break
		}
		key, value, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		value, _ = frontmattercodec.DecodeScalar(value)
		switch strings.TrimSpace(key) {
		case "name":
			name = value
		case "description":
			description = value
		}
	}
	if !closed || name == "" {
		return "", "", fmt.Errorf("SKILL_FRONTMATTER_INVALID: name is required")
	}
	return name, description, nil
}

func programBlock(src []byte) ([]byte, error) {
	const open = "```fak-program"
	text := strings.ReplaceAll(string(src), "\r\n", "\n")
	start := strings.Index(text, open)
	if start < 0 {
		return nil, fmt.Errorf("SKILL_PROGRAM_ABSENT: prose is not executable")
	}
	bodyStart := start + len(open)
	if bodyStart >= len(text) || text[bodyStart] != '\n' {
		return nil, fmt.Errorf("SKILL_PROGRAM_INVALID: opening fence must end the line")
	}
	bodyStart++
	endRel := strings.Index(text[bodyStart:], "\n```")
	if endRel < 0 {
		return nil, fmt.Errorf("SKILL_PROGRAM_INVALID: closing fence missing")
	}
	end := bodyStart + endRel
	if strings.Contains(text[end+4:], open) {
		return nil, fmt.Errorf("SKILL_PROGRAM_AMBIGUOUS: exactly one block is allowed")
	}
	return []byte(strings.TrimSpace(text[bodyStart:end])), nil
}

// ResolveCall maps one provider-visible name from this exact snapshot back to
// its canonical registration identity. Dispatch must use the request snapshot,
// never current registry names or model priors.
func ResolveCall(snapshot Snapshot, visibleName string) (ModelTool, error) {
	if err := ValidateSnapshot(snapshot); err != nil {
		return ModelTool{}, err
	}
	for _, tool := range snapshot.Tools {
		if tool.Name == visibleName {
			return tool, nil
		}
	}
	return ModelTool{}, fmt.Errorf("TOOL_CALL_NOT_EXPOSED: %s", visibleName)
}

// ResolveRegistration verifies that the provider-visible call still identifies
// the exact host registration whose digest was pinned into the request snapshot.
func ResolveRegistration(snapshot Snapshot, visibleName string, registrations []Registration) (Registration, error) {
	tool, err := ResolveCall(snapshot, visibleName)
	if err != nil {
		return Registration{}, err
	}
	for _, registration := range registrations {
		if registration.Program.Name != tool.Canonical {
			continue
		}
		if err := ValidateRegistration(registration); err != nil {
			return Registration{}, err
		}
		if registration.Digest != tool.Digest {
			return Registration{}, fmt.Errorf("TOOL_CALL_REGISTRATION_STALE: %s", tool.Canonical)
		}
		return registration, nil
	}
	return Registration{}, fmt.Errorf("TOOL_CALL_REGISTRATION_MISSING: %s", tool.Canonical)
}
