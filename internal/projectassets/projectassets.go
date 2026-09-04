package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	ManifestPath             = ".claude/project-assets.json"
	maxSkillDescriptionChars = 220
)

type Exclusion struct {
	Pattern string `json:"pattern"`
	Reason  string `json:"reason"`
}
type SkillPolicy struct {
	CanonicalRoot string      `json:"canonical_root"`
	CodexRoot     string      `json:"codex_root"`
	Include       []string    `json:"include"`
	Exclude       []Exclusion `json:"exclude"`
}
type Policy struct {
	CanonicalRoot  string      `json:"canonical_root"`
	Include        []string    `json:"include"`
	Exclude        []Exclusion `json:"exclude"`
	StartupCommand string      `json:"startup_command,omitempty"`
}
type Harness struct {
	Skills      string `json:"skills"`
	Memories    string `json:"memories"`
	GoalPrompts string `json:"goal_prompts"`
}
type Manifest struct {
	Schema      string             `json:"schema"`
	Skills      SkillPolicy        `json:"skills"`
	Memories    Policy             `json:"memories"`
	GoalPrompts Policy             `json:"goal_prompts"`
	Harnesses   map[string]Harness `json:"harnesses"`
}
type Excluded struct {
	Kind   string `json:"kind"`
	Path   string `json:"path"`
	Reason string `json:"reason"`
}
type HarnessReceipt struct {
	Canonical []string   `json:"canonical"`
	Imported  []string   `json:"imported"`
	Excluded  []Excluded `json:"excluded"`
	Duplicate []string   `json:"duplicate"`
	Stale     []string   `json:"stale"`
}
type Receipt struct {
	Schema              string                    `json:"schema"`
	Manifest            string                    `json:"manifest"`
	Harnesses           map[string]HarnessReceipt `json:"harnesses"`
	ZeroUnexplainedGaps bool                      `json:"zero_unexplained_gaps"`
}

func Load(root string) (Manifest, error) {
	var m Manifest
	b, e := os.ReadFile(filepath.Join(root, filepath.FromSlash(ManifestPath)))
	if e != nil {
		return m, e
	}
	if e = json.Unmarshal(b, &m); e != nil {
		return m, e
	}
	if m.Schema != "fak-project-assets/1" {
		return m, fmt.Errorf("unsupported schema %q", m.Schema)
	}
	return m, nil
}
func match(path string, patterns []string) bool {
	for _, p := range patterns {
		ok, _ := filepath.Match(p, filepath.Base(path))
		if ok {
			return true
		}
	}
	return false
}
func excluded(path string, rules []Exclusion) (string, bool) {
	for _, x := range rules {
		ok, _ := filepath.Match(x.Pattern, filepath.Base(path))
		if ok {
			return x.Reason, true
		}
	}
	return "", false
}
func flatFiles(root, dir string) ([]string, error) {
	es, e := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
	if os.IsNotExist(e) {
		return nil, nil
	}
	if e != nil {
		return nil, e
	}
	var out []string
	for _, x := range es {
		if !x.IsDir() {
			out = append(out, filepath.ToSlash(filepath.Join(dir, x.Name())))
		}
	}
	sort.Strings(out)
	return out, nil
}
func skillFiles(root, dir string) ([]string, error) {
	es, e := os.ReadDir(filepath.Join(root, filepath.FromSlash(dir)))
	if os.IsNotExist(e) {
		return nil, nil
	}
	if e != nil {
		return nil, e
	}
	var out []string
	for _, x := range es {
		if x.IsDir() {
			p := filepath.ToSlash(filepath.Join(dir, x.Name(), "SKILL.md"))
			if _, e = os.Stat(filepath.Join(root, filepath.FromSlash(p))); e == nil {
				out = append(out, p)
			}
		}
	}
	sort.Strings(out)
	return out, nil
}

var nameRE = regexp.MustCompile(`(?m)^name:\s*([^\r\n]+)\s*$`)

func skillName(root, p string) (string, error) {
	b, e := os.ReadFile(filepath.Join(root, filepath.FromSlash(p)))
	if e != nil {
		return "", e
	}
	m := nameRE.FindSubmatch(b)
	if len(m) != 2 {
		return "", fmt.Errorf("%s has no frontmatter name", p)
	}
	return strings.TrimSpace(string(m[1])), nil
}

// isGeneratedAdapter distinguishes disposable discovery pointers from deliberate native adapters.
// Sync may refresh only the former; overwriting the latter would erase harness-specific behavior.
func isGeneratedAdapter(body []byte) bool {
	return strings.Contains(string(body), "generated-by: fak project-assets sync")
}
func skillDescription(root, path string) (string, error) {
	body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
	if err != nil {
		return "", err
	}
	for _, line := range strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n") {
		if strings.HasPrefix(line, "description:") {
			description := strings.TrimSpace(strings.TrimPrefix(line, "description:"))
			if description != "" {
				return normalizeYAMLScalar(description), nil
			}
		}
	}
	return "", fmt.Errorf("%s has no frontmatter description", path)
}

// normalizeYAMLScalar turns the frontmatter representation into the description
// text that loaders see. Canonical skills use plain, single-quoted, and
// double-quoted scalars. Some older skills also carry a missing closing quote;
// accepting an implied close lets sync repair their generated adapters instead
// of copying the malformed delimiter into another file.
func normalizeYAMLScalar(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return ""
	}

	switch value[0] {
	case '"':
		quoted := value
		if !yamlDoubleQuotedClosed(quoted) {
			quoted += `"`
		}
		if unquoted, err := unquoteYAMLDouble(quoted); err == nil {
			return unquoted
		}
	case '\'':
		end := len(value)
		if end > 1 && value[end-1] == '\'' {
			end--
		}
		return strings.ReplaceAll(value[1:end], "''", "'")
	}

	return value
}

func yamlDoubleQuotedClosed(value string) bool {
	if len(value) < 2 || value[len(value)-1] != '"' {
		return false
	}
	backslashes := 0
	for i := len(value) - 2; i >= 0 && value[i] == '\\'; i-- {
		backslashes++
	}
	return backslashes%2 == 0
}

// unquoteYAMLDouble maps YAML's extra one-character escapes to Go escapes and
// then delegates Unicode and hex validation to strconv.Unquote. YAML and Go
// otherwise share the escape forms used by skill descriptions.
func unquoteYAMLDouble(value string) (string, error) {
	if len(value) < 2 || value[0] != '"' || value[len(value)-1] != '"' {
		return "", fmt.Errorf("not a double-quoted scalar")
	}
	var quoted strings.Builder
	quoted.Grow(len(value) + 8)
	quoted.WriteByte('"')
	for i := 1; i < len(value)-1; i++ {
		if value[i] != '\\' || i+1 >= len(value)-1 {
			quoted.WriteByte(value[i])
			continue
		}
		i++
		switch value[i] {
		case '0':
			quoted.WriteString(`\x00`)
		case 'e':
			quoted.WriteString(`\x1b`)
		case ' ':
			quoted.WriteString(`\x20`)
		case '/':
			quoted.WriteByte('/')
		case 'N':
			quoted.WriteString(`\u0085`)
		case '_':
			quoted.WriteString(`\u00a0`)
		case 'L':
			quoted.WriteString(`\u2028`)
		case 'P':
			quoted.WriteString(`\u2029`)
		case 'x':
			if i+2 < len(value)-1 {
				quoted.WriteString(`\u00`)
				quoted.WriteByte(value[i+1])
				quoted.WriteByte(value[i+2])
				i += 2
			} else {
				quoted.WriteString(`\x`)
			}
		default:
			quoted.WriteByte('\\')
			quoted.WriteByte(value[i])
		}
	}
	quoted.WriteByte('"')
	return strconv.Unquote(quoted.String())
}

// yamlScalar emits a block-mapping-safe YAML string without adding quoting
// churn to descriptions that are already safe as plain scalars. strconv.Quote
// supplies the escaping for quotes, backslashes, newlines, and control runes.
func yamlScalar(value string) string {
	if yamlPlainScalarSafe(value) {
		return value
	}
	return strconv.Quote(value)
}

func yamlPlainScalarSafe(value string) bool {
	if value == "" || strings.TrimSpace(value) != value || value == "---" || value == "..." {
		return false
	}
	first, _ := utf8.DecodeRuneInString(value)
	if unicode.IsDigit(first) || strings.ContainsRune("-+?.:,[]{}#&*!|>'\"%@`", first) {
		return false
	}
	for _, r := range value {
		if unicode.IsControl(r) || (r != ' ' && unicode.IsSpace(r)) || r == ':' || r == '#' {
			return false
		}
	}
	switch strings.ToLower(value) {
	case "null", "true", "false", "yes", "no", "on", "off", "~":
		return false
	}
	return true
}

func decodeYAMLScalar(value string) string {
	value = strings.TrimSpace(value)
	if len(value) < 2 {
		return value
	}
	switch value[0] {
	case '"':
		if value[len(value)-1] == '"' {
			var decoded string
			if json.Unmarshal([]byte(value), &decoded) == nil {
				return decoded
			}
		}
	case '\'':
		if value[len(value)-1] == '\'' {
			return strings.ReplaceAll(value[1:len(value)-1], "''", "'")
		}
	}
	return value
}
func adapterDescription(description string) string {
	runes := []rune(description)
	if len(runes) <= maxSkillDescriptionChars {
		return string(runes)
	}

	// Keep the discovery synopsis resident and page the canonical workflow on selection.
	// Preserve its leading trigger instead of inventing a second summary.
	const suffix = "..."
	limit := maxSkillDescriptionChars - len([]rune(suffix))
	cut := strings.TrimRightFunc(string(runes[:limit]), unicode.IsSpace)
	if boundary := strings.LastIndexAny(cut, " \t\r\n"); boundary > limit/2 {
		cut = strings.TrimRightFunc(cut[:boundary], unicode.IsSpace)
	}
	return cut + suffix
}

func adapter(name, description, rel string) string {
	description = adapterDescription(description)
	return fmt.Sprintf("---\nname: %s\ndescription: %s\nmetadata:\n  generated-by: fak project-assets sync\n  canonical: %s\n---\n\n# Canonical project skill adapter\n\nLoad and follow [`%s`](%s). This generated discovery adapter contains no maintained workflow body.\n\n## Portability contract\n\n- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.\n- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.\n- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.\n", name, yamlScalar(description), rel, rel, rel)
}

func classify(root, kind string, p Policy) ([]string, []Excluded, error) {
	fs, e := flatFiles(root, p.CanonicalRoot)
	if e != nil {
		return nil, nil, e
	}
	var yes []string
	var no []Excluded
	for _, f := range fs {
		if reason, ok := excluded(f, p.Exclude); ok {
			no = append(no, Excluded{kind, f, reason})
			continue
		}
		if !match(f, p.Include) {
			return nil, nil, fmt.Errorf("unclassified %s %s", kind, f)
		}
		yes = append(yes, f)
	}
	return yes, no, nil
}

func Build(root string, write bool) (Receipt, error) {
	m, e := Load(root)
	r := Receipt{Schema: "fak-project-assets-parity/1", Manifest: ManifestPath, Harnesses: map[string]HarnessReceipt{}}
	if e != nil {
		return r, e
	}
	skills, e := skillFiles(root, m.Skills.CanonicalRoot)
	if e != nil {
		return r, e
	}
	names := map[string][]string{}
	expected := map[string]string{}
	var canon, imports []string
	var sx []Excluded
	for _, p := range skills {
		n, e := skillName(root, p)
		if e != nil {
			return r, e
		}
		names[n] = append(names[n], p)
		if reason, ok := excluded(p, m.Skills.Exclude); ok {
			sx = append(sx, Excluded{"skill", p, reason})
			continue
		}
		if !match(p, m.Skills.Include) {
			return r, fmt.Errorf("unclassified skill %s", p)
		}
		description, e := skillDescription(root, p)
		if e != nil {
			return r, e
		}
		target := filepath.ToSlash(filepath.Join(m.Skills.CodexRoot, n, "SKILL.md"))
		canon = append(canon, p)
		imports = append(imports, target)
		expected[target] = p
		if write {
			abs := filepath.Join(root, filepath.FromSlash(target))
			if e = os.MkdirAll(filepath.Dir(abs), 0755); e != nil {
				return r, e
			}
			if existing, readErr := os.ReadFile(abs); readErr == nil && !isGeneratedAdapter(existing) {
				continue
			}
			rel, _ := filepath.Rel(filepath.Dir(abs), filepath.Join(root, filepath.FromSlash(p)))
			e = os.WriteFile(abs, []byte(adapter(n, description, filepath.ToSlash(rel))), 0644)
			if e != nil {
				return r, e
			}
		}
	}
	var dup []string
	for n, ps := range names {
		if len(ps) > 1 {
			dup = append(dup, n+": "+strings.Join(ps, ", "))
		}
	}
	sort.Strings(dup)
	var stale []string
	for target, source := range expected {
		abs := filepath.Join(root, filepath.FromSlash(target))
		rel, _ := filepath.Rel(filepath.Dir(abs), filepath.Join(root, filepath.FromSlash(source)))
		b, e := os.ReadFile(abs)
		n, _ := skillName(root, source)
		description, descriptionErr := skillDescription(root, source)
		if e != nil || descriptionErr != nil || (isGeneratedAdapter(b) && string(b) != adapter(n, description, filepath.ToSlash(rel))) {
			stale = append(stale, target)
		}
	}
	existing, _ := skillFiles(root, m.Skills.CodexRoot)
	for _, p := range existing {
		if _, ok := expected[p]; !ok {
			stale = append(stale, p)
		}
	}
	sort.Strings(stale)
	sort.Strings(canon)
	sort.Strings(imports)
	codexSkillImports := append([]string{}, imports...)
	memories, mx, e := classify(root, "memory", m.Memories)
	if e != nil {
		return r, e
	}
	prompts, px, e := classify(root, "goal_prompt", m.GoalPrompts)
	if e != nil {
		return r, e
	}
	all := append(append(append([]string{}, canon...), memories...), prompts...)
	sort.Strings(all)
	ex := append(append([]Excluded{}, sx...), mx...)
	ex = append(ex, px...)
	sort.Slice(ex, func(i, j int) bool { return ex[i].Path < ex[j].Path })
	r.Harnesses["claude"] = HarnessReceipt{all, all, ex, dup, nil}
	codexImports := append(codexSkillImports, prompts...)
	codexImports = append(codexImports, m.Memories.StartupCommand)
	sort.Strings(codexImports)
	r.Harnesses["codex"] = HarnessReceipt{all, codexImports, ex, dup, stale}
	nativeImports := append(append([]string{}, canon...), prompts...)
	nativeImports = append(nativeImports, m.Memories.StartupCommand)
	sort.Strings(nativeImports)
	r.Harnesses["fak-native"] = HarnessReceipt{all, nativeImports, ex, dup, nil}
	_, cok := m.Harnesses["codex"]
	_, nok := m.Harnesses["fak-native"]
	r.ZeroUnexplainedGaps = len(dup) == 0 && len(stale) == 0 && cok && nok
	return r, nil
}

// VerifyOpenCodeSnapshot asserts that opencode.json exists in root and explicitly sets "snapshot": false
// to prevent disk hammering and multi-GB SQLite snapshot bloat in large repositories.
func VerifyOpenCodeSnapshot(root string) error {
	path := filepath.Join(root, "opencode.json")
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read opencode.json: %w", err)
	}
	var raw map[string]interface{}
	if err := json.Unmarshal(b, &raw); err != nil {
		return fmt.Errorf("parse opencode.json: %w", err)
	}
	val, ok := raw["snapshot"]
	if !ok {
		return fmt.Errorf("opencode.json missing required \"snapshot\": false")
	}
	snap, isBool := val.(bool)
	if !isBool || snap {
		return fmt.Errorf("opencode.json must explicitly set \"snapshot\": false, got %v", val)
	}
	return nil
}
