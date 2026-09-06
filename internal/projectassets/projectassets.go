package projectassets

import (
	"crypto/sha256"
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

type metadataField struct {
	key   string
	value string
}

func skillMetadata(root, path string) ([]metadataField, error) {
	body, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(path)))
	if err != nil {
		return nil, err
	}
	lines := strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n")
	if len(lines) == 0 || strings.TrimSpace(lines[0]) != "---" {
		return nil, nil
	}

	var meta []metadataField
	inFrontmatter := false
	inMetadata := false

	for i, line := range lines {
		if i == 0 {
			inFrontmatter = true
			continue
		}
		trimmed := strings.TrimSpace(line)
		if trimmed == "---" {
			break
		}
		if !inFrontmatter || trimmed == "" {
			continue
		}

		isIndented := len(line) > 0 && (line[0] == ' ' || line[0] == '\t')
		if !isIndented {
			inMetadata = false
			if strings.HasPrefix(trimmed, "metadata:") {
				inMetadata = true
			}
			continue
		}

		if inMetadata {
			if trimmed == "" || strings.HasPrefix(trimmed, "#") {
				continue
			}
			key, val, ok := strings.Cut(trimmed, ":")
			if !ok {
				continue
			}
			k := strings.TrimSpace(key)
			v := strings.TrimSpace(val)
			if k == "" || k == "generated-by" || k == "canonical" || k == "canonical-description-hash" {
				continue
			}
			if len(v) > 0 && (v[0] == '"' || v[0] == '\'') {
				v = normalizeYAMLScalar(v)
			} else {
				if idx := strings.Index(v, " #"); idx != -1 {
					v = strings.TrimSpace(v[:idx])
				}
				v = normalizeYAMLScalar(v)
			}
			meta = append(meta, metadataField{key: k, value: v})
		}
	}
	return meta, nil
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

// AdapterDescription projects the canonical skill description into the adapter limit.
func AdapterDescription(description string) string {
	return adapterDescription(description)
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

func adapter(name, description, rel string, extraMeta ...metadataField) string {
	truncated := adapterDescription(description)
	var metaLines strings.Builder
	for _, m := range extraMeta {
		metaLines.WriteString(fmt.Sprintf("  %s: %s\n", m.key, yamlScalar(m.value)))
	}
	metaLines.WriteString("  generated-by: fak project-assets sync\n")
	metaLines.WriteString(fmt.Sprintf("  canonical: %s\n", rel))
	if len([]rune(description)) > maxSkillDescriptionChars {
		metaLines.WriteString(fmt.Sprintf("  canonical-description-hash: %x\n", sha256.Sum256([]byte(description))))
	}
	return fmt.Sprintf("---\nname: %s\ndescription: %s\nmetadata:\n%s---\n\n# Canonical project skill adapter\n\nLoad and follow [`%s`](%s). This generated discovery adapter contains no maintained workflow body.\n\n## Portability contract\n\n- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.\n- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.\n- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.\n", name, yamlScalar(truncated), metaLines.String(), rel, rel)
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
		meta, e := skillMetadata(root, p)
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
			e = os.WriteFile(abs, []byte(adapter(n, description, filepath.ToSlash(rel), meta...)), 0644)
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
		meta, metaErr := skillMetadata(root, source)
		if e != nil || descriptionErr != nil || metaErr != nil || (isGeneratedAdapter(b) && strings.ReplaceAll(string(b), "\r\n", "\n") != strings.ReplaceAll(adapter(n, description, filepath.ToSlash(rel), meta...), "\r\n", "\n")) {
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
	r.Harnesses["opencode"] = HarnessReceipt{all, codexImports, ex, dup, stale}
	nativeImports := append(append([]string{}, canon...), prompts...)
	nativeImports = append(nativeImports, m.Memories.StartupCommand)
	sort.Strings(nativeImports)
	r.Harnesses["fak-native"] = HarnessReceipt{all, nativeImports, ex, dup, nil}
	_, cok := m.Harnesses["codex"]
	_, nok := m.Harnesses["fak-native"]
	_, ook := m.Harnesses["opencode"]
	r.ZeroUnexplainedGaps = len(dup) == 0 && len(stale) == 0 && cok && nok && ook
	return r, nil
}

// EnsureSync checks parity with Build(root, false) and synchronizes adapters with Build(root, true) if needed.
// When the workspace defines an .opencode directory or plugin asset, it also verifies and synchronizes the
// OpenCode plugin asset enforcing cross-harness lease admission.
// If workspace assets are in parity, returns (r, false, nil).
// Otherwise, synchronizes missing/stale assets and returns (syncedReceipt, true, syncErr).
func EnsureSync(root string) (Receipt, bool, error) {
	r, err := Build(root, false)
	opencodeDir := filepath.Join(root, ".opencode")
	pluginPath := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	hasOpenCode := false
	if info, statErr := os.Stat(opencodeDir); statErr == nil && info.IsDir() {
		hasOpenCode = true
	} else if _, statErr := os.Stat(pluginPath); statErr == nil {
		hasOpenCode = true
	}

	pluginNeedsSync := false
	if hasOpenCode {
		if pluginErr := VerifyOpenCodePlugin(root); pluginErr != nil {
			pluginNeedsSync = true
		}
	}

	if err == nil && r.ZeroUnexplainedGaps && len(r.Harnesses["codex"].Stale) == 0 && len(r.Harnesses["opencode"].Stale) == 0 && !pluginNeedsSync {
		return r, false, nil
	}

	syncedReceipt, syncErr := Build(root, true)
	if syncErr != nil {
		return syncedReceipt, true, syncErr
	}
	if hasOpenCode {
		if pluginErr := EnsureOpenCodePlugin(root, true); pluginErr != nil {
			return syncedReceipt, true, pluginErr
		}
	}
	return syncedReceipt, true, nil
}

// Ensure checks parity with Build(root, false) and synchronizes adapters with Build(root, true) if autoSync is true.
// When autoSync is true and the workspace defines an .opencode directory, it also ensures the OpenCode plugin asset
// is synchronized. When autoSync is false and an .opencode directory or plugin exists, its lease admission invariants are verified.
func Ensure(root string, autoSync bool) (Receipt, error) {
	if autoSync {
		r, _, err := EnsureSync(root)
		return r, err
	}
	r, err := Build(root, false)
	if err != nil {
		return r, err
	}
	opencodeDir := filepath.Join(root, ".opencode")
	pluginPath := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	hasOpenCode := false
	if info, statErr := os.Stat(opencodeDir); statErr == nil && info.IsDir() {
		hasOpenCode = true
	} else if _, statErr := os.Stat(pluginPath); statErr == nil {
		hasOpenCode = true
	}
	if hasOpenCode {
		if pluginErr := VerifyOpenCodePlugin(root); pluginErr != nil {
			return r, pluginErr
		}
	}
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

// OpenCodePluginPath is the workspace-relative path to the OpenCode DOS proof guard plugin.
const OpenCodePluginPath = ".opencode/plugins/dos-proof-guard.js"

// VerifyOpenCodePlugin asserts that .opencode/plugins/dos-proof-guard.js exists in root,
// is non-empty, and enforces cross-harness lease admission before native OpenCode mutations.
func VerifyOpenCodePlugin(root string) error {
	path := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read opencode plugin %s: %w", path, err)
	}
	content := string(b)
	if strings.TrimSpace(content) == "" {
		return fmt.Errorf("opencode plugin %s is empty", path)
	}
	requiredChecks := []struct {
		pattern string
		desc    string
	}{
		{"tool.execute.before", "pre-mutation execution hook"},
		{"write", "write tool mutation interception"},
		{"edit", "edit tool mutation interception"},
		{"apply_patch", "apply_patch tool mutation interception"},
		{"leaseref", "FAK leaseref admission verification"},
		{"loop", "FAK loop region admission check"},
		{"live_leases", "DOS live leases snapshot inspection"},
		{"arbitrate", "DOS lane lease arbitration check"},
	}
	for _, req := range requiredChecks {
		if !strings.Contains(content, req.pattern) {
			return fmt.Errorf("opencode plugin %s missing %s (%q)", path, req.desc, req.pattern)
		}
	}
	return nil
}

// SyncOpenCodePlugin writes the canonical dos-proof-guard.js plugin into .opencode/plugins/
// under root, ensuring cross-harness lease admission is active for OpenCode.
func SyncOpenCodePlugin(root string) error {
	path := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create opencode plugin dir: %w", err)
	}
	if err := os.WriteFile(path, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		return fmt.Errorf("write opencode plugin: %w", err)
	}
	return nil
}

// EnsureOpenCodePlugin verifies that the OpenCode plugin asset exists and enforces cross-harness
// lease admission. When autoSync is true and the plugin is missing or invalid, it synchronizes it first.
func EnsureOpenCodePlugin(root string, autoSync bool) error {
	if autoSync {
		if err := VerifyOpenCodePlugin(root); err != nil {
			if syncErr := SyncOpenCodePlugin(root); syncErr != nil {
				return syncErr
			}
		}
	}
	return VerifyOpenCodePlugin(root)
}

// DefaultOpenCodePlugin is the canonical dos-proof-guard.js script enforcing
// cross-harness FAK reference fences and DOS lane lease admission before OpenCode native mutations.
const DefaultOpenCodePlugin = `import { execFile } from "node:child_process";
import { realpath } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

const execute = promisify(execFile);
const mutations = new Set(["write", "edit", "apply_patch"]);

async function jsonCommand(command, args, cwd) {
  let stdout;
  try {
    ({ stdout } = await execute(command, args, {
      cwd, timeout: 30000, maxBuffer: 4 * 1024 * 1024, windowsHide: true,
    }));
  } catch (error) {
    throw new Error(` + "`" + `[dos-proof-guard] Lease admission unavailable or refused: ${command}: ${error.stdout || error.stderr || error.message}` + "`" + `);
  }
  try {
    return JSON.parse(stdout);
  } catch {
    throw new Error(` + "`" + `[dos-proof-guard] Malformed lease admission response from ${command}` + "`" + `);
  }
}

// Resolve existing ancestors as well as new files so a symlink cannot disguise
// a write to a leased path. External writes need their own workspace admission.
async function canonicalPath(filename) {
  try {
    return await realpath(filename);
  } catch (error) {
    if (error.code !== "ENOENT") throw error;
    const parent = path.dirname(filename);
    if (parent === filename) throw error;
    return path.join(await canonicalPath(parent), path.basename(filename));
  }
}

function mutationPaths(tool, args) {
  if (tool !== "apply_patch") return [args?.filePath];
  if (typeof args?.patchText !== "string") return [];
  return [...args.patchText.matchAll(/^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)\r?$/gm)]
    .map((match) => match[1].trim());
}

/**
 * dos-proof-guard.js — opencode plugin for DOS on-device proof & cross-validation lifecycle.
 *
 * Reminds agents after file modifications to:
 * 1. Run on-device tests (CLAIM_TEST_GREEN)
 * 2. Validate commit with DOS (dos commit-audit / dos verify)
 * 3. Spawn cross-validator subagent for independent verification
 * 4. File follow-on GitHub tickets by default for discovered edge cases
 */

export default async function dosProofGuardPlugin({ client, directory }) {
  let fileModifiedCount = 0;
  // Capture host configuration once; tool arguments never supply ownership.
  const hostOwner = process.env.FAK_LEASE_OWNER;
  const hostSession = process.env.FAK_LEASE_SESSION;
  const hostLease = process.env.FAK_LEASE_ID;
  const root = await realpath(directory);

  return {
    "tool.execute.before": async (input, output) => {
      if (!mutations.has(input?.tool)) return;
      if (!input.sessionID || (!!hostOwner !== !!hostSession)) {
        throw new Error("[dos-proof-guard] A harness session and paired host lease owner/session are required");
      }
      const session = hostSession || input.sessionID;
      const owner = hostOwner || ` + "`" + `opencode:${session}` + "`" + `;
      const filenames = mutationPaths(input.tool, output?.args);
      if (!filenames.length || filenames.some((name) => typeof name !== "string" || !name.trim())) {
        throw new Error("[dos-proof-guard] Explicit mutation paths are required for lease admission");
      }
      const trees = [];
      for (const filename of filenames) {
        const target = await canonicalPath(path.resolve(root, filename));
        const relative = path.relative(root, target);
        if (!relative || relative === ".." || relative.startsWith(` + "`" + `..${path.sep}` + "`" + `) || path.isAbsolute(relative)) {
          throw new Error("[dos-proof-guard] Mutation path is outside this lease workspace");
        }
        trees.push(relative.split(path.sep).join("/"));
      }

      const refs = await jsonCommand("fak", ["leaseref", "liveness", "--dir", root, "--session", session], root);
      if (!Array.isArray(refs)) throw new Error("[dos-proof-guard] Invalid FAK lease snapshot");
      const own = refs.filter((row) => row.holder === owner && row.session_id === session && (!hostLease || row.id === hostLease));
      if (hostLease && own.length !== 1) throw new Error("[dos-proof-guard] Host lease identity is not current");
      if (own.length > 1) throw new Error("[dos-proof-guard] Set host FAK_LEASE_ID to identify the active lease");
      const regionArgs = ["loop", "region", "--dir", root, "--actor", owner, "--no-queue", "--json"];
      for (const tree of trees) regionArgs.push("--tree", tree);
      if (own.length === 1) {
        const lease = own[0];
        if (!lease.id || !Number.isInteger(lease.generation) || lease.generation < 1) {
          throw new Error("[dos-proof-guard] Own FAK lease lacks a fencing generation");
        }
        const fence = await jsonCommand("fak", ["leaseref", "fence", "--dir", root, "--id", lease.id, "--holder", owner, "--generation", String(lease.generation)], root);
        if (fence.ok !== true) throw new Error("[dos-proof-guard] Own FAK lease fence refused");
        regionArgs.push("--self", lease.id);
      }
      const region = await jsonCommand("fak", regionArgs, root);
      if (region.schema !== "fak.loop-region.v1" || region.admit !== true) {
        throw new Error("[dos-proof-guard] FAK lease admission refused");
      }

      // DOS owns WAL parsing, corruption handling, expiry and locking. Older DOS
      // versions reject strict=True, so a missing prerequisite blocks mutations.
      const snapshot = await jsonCommand("python", ["-c",
        "import json,sys; from dos import config,lane_lease; cfg=config.load_workspace_config(sys.argv[1],gather_env=False); print(json.dumps(lane_lease.live_leases(cfg, strict=True, expire_dead=True)))", root], root);
      if (!Array.isArray(snapshot) || snapshot.some((row) => !row || typeof row.lane !== "string" || !Array.isArray(row.tree))) {
        throw new Error("[dos-proof-guard] Invalid DOS lease snapshot");
      }
      const peers = snapshot.filter((row) => !(row.holder === owner && row.run_id === session));
      const decision = await jsonCommand("dos", ["arbitrate", "--workspace", root, "--lane", "opencode-native-write", "--kind", "keyword", "--output", "json", "--leases", JSON.stringify(peers), "--tree", ...trees], root);
      if (decision.outcome !== "acquire") throw new Error("[dos-proof-guard] DOS lease admission refused");
      // These are pre-execution observations, not an atomic filesystem fence.
      // Shell/MCP mutations and late lease changes need separate mediation.
    },
    "tool.execute.after": async (input, output) => {
      const tool = input?.tool || "";
      if (tool === "edit" || tool === "write") {
        fileModifiedCount++;
        const reminder = "\n\n[dos-proof-guard] Code modified. On-device proof required before completion:\n" +
          "1. Run tests: .\\test.ps1 ./internal/<pkg>/... -> CLAIM_TEST_GREEN\n" +
          "2. DOS commit audit: dos commit-audit HEAD -> diff-witnessed\n" +
          "3. Spawn cross-validator subagent to adversarial-audit the diff\n" +
          "4. Auto-ticket follow-ons/edge-cases by default (gh issue create / fak issue fanout)\n";

        if (output && typeof output === "object") {
          if (typeof output.content === "string") {
            output.content += reminder;
          } else if (Array.isArray(output.content)) {
            output.content.push({ type: "text", text: reminder });
          }
        }
        console.log(reminder);
      }
    },
    "chat.message": async (input, output) => {
      // Reset modification counter when new user message arrives
      if (input?.role === "user") {
        fileModifiedCount = 0;
      }
    }
  };
}
`
