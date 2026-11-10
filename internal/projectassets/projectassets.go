package projectassets

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	ManifestPath             = ".claude/project-assets.json"
	maxSkillDescriptionRunes = 1024 // Agent Skills spec limit for a frontmatter description.
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
	// Plugins declares the workspace-relative .opencode/plugins/** assets whose
	// integrity parity must observe. Each entry is verified against its canonical
	// asset; drift is a reported gap, never a silent one-way revert.
	Plugins []string `json:"plugins,omitempty"`
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
// PluginReceipt is the parity verdict for one manifest-declared .opencode/plugins/** asset.
type PluginReceipt struct {
	Path   string `json:"path"`
	Status string `json:"status"` // ok | drifted | missing | invalid | unsupported
	Reason string `json:"reason,omitempty"`
}
type Receipt struct {
	Schema              string                    `json:"schema"`
	Manifest            string                    `json:"manifest"`
	Harnesses           map[string]HarnessReceipt `json:"harnesses"`
	Plugins             []PluginReceipt           `json:"plugins,omitempty"`
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

// isGeneratedAdapter distinguishes generated one-hop mirrors from deliberate native adapters.
// .agents/skills holds generated mirrors - full-body copies of the canonical SKILL.md with
// provenance metadata injected in the frontmatter; the canonical file remains the single
// semantic source. Sync may refresh only the mirrors; overwriting a native adapter would
// erase harness-specific behavior.
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
// accepting an implied close lets sync read their descriptions without copying
// the malformed delimiter into a mirror.
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

// mirror generates the one-hop mirror of a canonical SKILL.md: a full-body copy
// whose frontmatter lines are preserved verbatim and whose metadata block carries
// the provenance pair (generated-by, canonical). .agents/skills holds these
// generated mirrors - full-body copies of the canonical SKILL.md with provenance
// metadata injected in the frontmatter; the canonical file remains the single
// semantic source. Regeneration is idempotent: stale provenance inside the
// metadata block is dropped before the fresh pair is injected.
func mirror(canonicalPath, name, rel string) ([]byte, error) {
	b, e := os.ReadFile(canonicalPath)
	if e != nil {
		return nil, e
	}
	lines := strings.Split(strings.ReplaceAll(string(b), "\r\n", "\n"), "\n")
	if len(lines) == 0 || lines[0] != "---" {
		return nil, fmt.Errorf("skill %s (%s): missing opening frontmatter delimiter", name, canonicalPath)
	}
	closeIdx := -1
	for i := 1; i < len(lines); i++ {
		if lines[i] == "---" {
			closeIdx = i
			break
		}
	}
	if closeIdx < 0 {
		return nil, fmt.Errorf("skill %s (%s): missing closing frontmatter delimiter", name, canonicalPath)
	}
	fm := lines[1:closeIdx]
	body := strings.Join(lines[closeIdx+1:], "\n")

	provenance := []string{"  generated-by: fak project-assets sync", "  canonical: " + rel}

	// Drop stale provenance inside the metadata block so regenerating a file that
	// already carries it (a former mirror) stays byte-identical.
	kept := make([]string, 0, len(fm))
	metaIdx := -1
	inMeta := false
	for _, l := range fm {
		trimmed := strings.TrimSpace(l)
		if trimmed == "metadata:" {
			if metaIdx < 0 {
				metaIdx = len(kept)
			}
			inMeta = true
			kept = append(kept, l)
			continue
		}
		if inMeta {
			switch {
			case trimmed == "":
				// Blank lines do not end the metadata block.
			case l[0] != ' ' && l[0] != '\t':
				inMeta = false
			default:
				if key := strings.TrimLeft(l, " \t"); strings.HasPrefix(key, "generated-by:") || strings.HasPrefix(key, "canonical:") {
					continue
				}
			}
		}
		kept = append(kept, l)
	}

	out := make([]string, 0, len(kept)+4)
	out = append(out, "---")
	for i, l := range kept {
		out = append(out, l)
		if i == metaIdx {
			out = append(out, provenance...)
		}
	}
	if metaIdx < 0 {
		out = append(out, "metadata:")
		out = append(out, provenance...)
	}
	out = append(out, "---")
	out = append(out, body)
	return []byte(strings.Join(out, "\n")), nil
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
		if runes := len([]rune(description)); runes > maxSkillDescriptionRunes {
			return r, fmt.Errorf("%s: frontmatter description has %d runes, above the Agent Skills limit of %d", p, runes, maxSkillDescriptionRunes)
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
			b, e := mirror(filepath.Join(root, filepath.FromSlash(p)), n, filepath.ToSlash(rel))
			if e != nil {
				return r, e
			}
			if e = os.WriteFile(abs, b, 0644); e != nil {
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
		want, mirrorErr := mirror(filepath.Join(root, filepath.FromSlash(source)), n, filepath.ToSlash(rel))
		if e != nil || mirrorErr != nil || (isGeneratedAdapter(b) && strings.ReplaceAll(string(b), "\r\n", "\n") != string(want)) {
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
	// Pi consumes the generated .agents/skills mirrors the same way codex/opencode do;
	// it differs only in that cmd/fak's launcher passes them explicitly via --skill.
	r.Harnesses["pi"] = HarnessReceipt{all, codexImports, ex, dup, stale}
	nativeImports := append(append([]string{}, canon...), prompts...)
	nativeImports = append(nativeImports, m.Memories.StartupCommand)
	sort.Strings(nativeImports)
	r.Harnesses["fak-native"] = HarnessReceipt{all, nativeImports, ex, dup, nil}
	_, cok := m.Harnesses["codex"]
	_, nok := m.Harnesses["fak-native"]
	_, ook := m.Harnesses["opencode"]
	// The pi harness is derived (not manifest-declared): every project that ships the
	// generated .agents/skills mirrors can be driven by `fak pi --skill`, so its
	// receipt must exist and carry no stale entries, exactly like codex/opencode.
	piStale := len(r.Harnesses["pi"].Stale) == 0
	pluginsOK := true
	for _, rel := range m.Plugins {
		pr := verifyDeclaredPlugin(root, rel)
		r.Plugins = append(r.Plugins, pr)
		if pr.Status != pluginStatusOK {
			pluginsOK = false
		}
	}
	r.ZeroUnexplainedGaps = len(dup) == 0 && len(stale) == 0 && cok && nok && ook && piStale && pluginsOK
	if write {
		opencodeDir := filepath.Join(root, ".opencode")
		pluginPath := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
		hasOpenCode := false
		if info, statErr := os.Stat(opencodeDir); statErr == nil && info.IsDir() {
			hasOpenCode = true
		} else if _, statErr := os.Stat(pluginPath); statErr == nil {
			hasOpenCode = true
		}
		if hasOpenCode {
			if err := SyncOpenCodePlugin(root); err != nil {
				return r, err
			}
		}
	}
	return r, nil
}

// EnsureSync checks parity with Build(root, false) and synchronizes mirrors with Build(root, true) if needed.
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
		if pluginErr := VerifyOpenCodePlugin(root); pluginErr != nil && !errors.Is(pluginErr, ErrOpenCodePluginDrifted) {
			// Drift alone never arms a sync: a present, structurally valid guard is
			// preserved and reported by parity, not reverted (fak-private#2960).
			pluginNeedsSync = true
		}
	}

	if err == nil && r.ZeroUnexplainedGaps && !pluginNeedsSync {
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

// Ensure checks parity with Build(root, false) and synchronizes mirrors with Build(root, true) if autoSync is true.
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
	if err := json.Unmarshal(stripUTF8BOM(b), &raw); err != nil {
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

// DOSMinVersion is the minimum dos-kernel version whose lane_lease API the guard
// probe is written against; DOS >= 0.22.0 removed the `strict` kwarg, so emitting
// it raises TypeError and blocks every mutation.
const DOSMinVersion = "0.22.0"

// ErrOpenCodePluginDrifted marks a plugin that is present and structurally valid
// but not byte-identical to its canonical asset. Drift is a reported condition
// (parity receipt, verify CLI); it never triggers an overwrite of the local copy.
var ErrOpenCodePluginDrifted = errors.New("opencode plugin present but diverges from canonical asset")

const (
	pluginStatusOK          = "ok"
	pluginStatusDrifted     = "drifted"
	pluginStatusMissing     = "missing"
	pluginStatusInvalid     = "invalid"
	pluginStatusUnsupported = "unsupported"
)

// verifyOpenCodePluginContent checks the structural admission contract of a
// dos-proof-guard.js body: every required hook/pattern is present and the DOS
// probe does not emit the removed strict kwarg. It deliberately does NOT compare
// bytes against DefaultOpenCodePlugin, so a locally evolved guard that still
// enforces the full admission chain verifies as valid and is preserved by sync.
func verifyOpenCodePluginContent(path, content string) error {
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
		{"live_leases(cfg, expire_dead=True)", "DOS live_leases probe compatible with DOS >= " + DOSMinVersion + " (no strict kwarg)"},
		{"arbitrate", "DOS lane lease arbitration check"},
		{"FAK_WRITE_GUARD_BUDGET_MS", "budgeted preflight wall-clock deadline"},
		{"FAK_WRITE_GUARD_STRICT", "strict fail-closed escape hatch"},
		{"fak.queueus.pending.v1", "QUEUEUS pending-ledger schema"},
		{"win32", "Windows platform binary extension check"},
	}
	for _, req := range requiredChecks {
		if !strings.Contains(content, req.pattern) {
			return fmt.Errorf("opencode plugin %s missing %s (%q)", path, req.desc, req.pattern)
		}
	}
	// DOS >= DOSMinVersion removed the strict kwarg from live_leases; a guard that
	// still emits strict= raises TypeError at probe time and blocks every mutation.
	if strings.Contains(content, "strict=") {
		return fmt.Errorf("opencode plugin %s emits strict= in the DOS live_leases probe; incompatible with DOS >= %s (remove the strict kwarg)", path, DOSMinVersion)
	}
	return nil
}

// verifyDeclaredPlugin checks one manifest-declared plugin against its canonical
// asset. It is the read-only parity surface for .opencode/plugins/**: drift is
// observable here, while the write path (SyncOpenCodePlugin) preserves any
// structurally valid local copy.
func verifyDeclaredPlugin(root, rel string) PluginReceipt {
	abs := filepath.Join(root, filepath.FromSlash(rel))
	switch filepath.Base(rel) {
	case "dos-proof-guard.js":
		b, err := os.ReadFile(abs)
		if err != nil {
			return PluginReceipt{Path: rel, Status: pluginStatusMissing, Reason: "declared plugin not present"}
		}
		if e := verifyOpenCodePluginContent(rel, string(b)); e != nil {
			return PluginReceipt{Path: rel, Status: pluginStatusInvalid, Reason: e.Error()}
		}
		if string(b) != DefaultOpenCodePlugin {
			return PluginReceipt{Path: rel, Status: pluginStatusDrifted, Reason: "differs from canonical DefaultOpenCodePlugin"}
		}
		return PluginReceipt{Path: rel, Status: pluginStatusOK}
	case "fak-grep.js":
		b, err := os.ReadFile(abs)
		if err != nil {
			return PluginReceipt{Path: rel, Status: pluginStatusMissing, Reason: "declared plugin not present"}
		}
		if string(b) != defaultOpenCodeGrep {
			return PluginReceipt{Path: rel, Status: pluginStatusDrifted, Reason: "differs from canonical fak-grep asset"}
		}
		return PluginReceipt{Path: rel, Status: pluginStatusOK}
	default:
		return PluginReceipt{Path: rel, Status: pluginStatusUnsupported, Reason: "no canonical asset for this plugin name"}
	}
}

// VerifyOpenCodePlugin asserts that .opencode/plugins/dos-proof-guard.js exists in root,
// is non-empty, enforces cross-harness lease admission before native OpenCode mutations,
// and is byte-identical to the canonical asset (drift is reported as
// ErrOpenCodePluginDrifted, mirroring verifyOpenCodeGrep's byte-equality contract).
func VerifyOpenCodePlugin(root string) error {
	path := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	b, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read opencode plugin %s: %w", path, err)
	}
	content := string(b)
	if err := verifyOpenCodePluginContent(OpenCodePluginPath, content); err != nil {
		return err
	}
	if content != DefaultOpenCodePlugin {
		return fmt.Errorf("opencode plugin %s: %w", OpenCodePluginPath, ErrOpenCodePluginDrifted)
	}
	return verifyOpenCodeGrep(root)
}

// SyncOpenCodePlugin installs the canonical dos-proof-guard.js plugin into
// .opencode/plugins/ under root, ensuring cross-harness lease admission is active
// for OpenCode. Anti-clobber contract (fak-private#2960): a present guard that
// still passes the structural admission check is hand-authored or locally
// evolved and is preserved byte-for-byte, exactly the way the SKILL.md adapter
// write treats a hand-authored SKILL.md; only a missing or structurally broken
// guard is (re)written from the canonical constant. Drift from canonical is a
// reported condition (VerifyOpenCodePlugin / parity), never a silent revert.
func SyncOpenCodePlugin(root string) error {
	path := filepath.Join(root, filepath.FromSlash(OpenCodePluginPath))
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return fmt.Errorf("create opencode plugin dir: %w", err)
	}
	if err := syncOpenCodeGrep(root); err != nil {
		return err
	}
	if existing, readErr := os.ReadFile(path); readErr == nil {
		if verifyOpenCodePluginContent(OpenCodePluginPath, string(existing)) == nil {
			return nil
		}
	}
	if err := os.WriteFile(path, []byte(DefaultOpenCodePlugin), 0644); err != nil {
		return fmt.Errorf("write opencode plugin: %w", err)
	}
	return nil
}

// EnsureOpenCodePlugin verifies that the OpenCode plugin asset exists and enforces cross-harness
// lease admission. When autoSync is true and the plugin is missing or structurally invalid, it
// synchronizes it first. A drifting-but-valid plugin is preserved and reported, not reverted.
func EnsureOpenCodePlugin(root string, autoSync bool) error {
	err := VerifyOpenCodePlugin(root)
	if err == nil {
		return nil
	}
	if errors.Is(err, ErrOpenCodePluginDrifted) {
		// Present, structurally valid, locally evolved: parity reports the drift;
		// the ordinary startup path stays green and the local copy is untouched.
		return nil
	}
	if !autoSync {
		return err
	}
	if syncErr := SyncOpenCodePlugin(root); syncErr != nil {
		return syncErr
	}
	return VerifyOpenCodePlugin(root)
}

// DefaultOpenCodePlugin is the canonical dos-proof-guard.js script enforcing
// cross-harness FAK reference fences and DOS lane lease admission before OpenCode native mutations.
// The admission chain runs under a shared preflight wall-clock budget
// (FAK_WRITE_GUARD_BUDGET_MS); availability failures degrade through the bounded
// QUEUEUS ledger (<root>/.fak/queueus/) instead of failing closed, and
// FAK_WRITE_GUARD_STRICT=1 restores the strict fail-closed throw.
const DefaultOpenCodePlugin = `import { execFile } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { mkdir, readFile, realpath, rename, writeFile } from "node:fs/promises";
import path from "node:path";
import { promisify } from "node:util";

/**
 * dos-proof-guard.js — opencode plugin for DOS on-device proof & cross-validation lifecycle.
 *
 * Shape: OpenCode 2.x default export { id, server, setup }. ` + "`" + `setup(api)` + "`" + ` registers
 * exactly two hooks (execute.before / execute.after) and returns a cleanup that
 * disposes both and clears in-memory session state; ` + "`" + `server({directory})` + "`" + ` keeps
 * the 1.x dotted-hook contract for direct callers.
 *
 * execute.before fences every write/edit/apply_patch/patch through lease
 * admission. Out-of-session-root targets are refused OUT_OF_LEASE unless they
 * lie in OS temp/scratch (admitted outright, fak-private#12166) or inside a
 * DECLARED sibling workspace root (FAK_PUBLIC_ROOT / FAK_ADMITTED_ROOTS), which
 * is re-based through the SAME admission chain against that root — never by
 * bypassing the fence. A declaration that contains the session root (an
 * ancestor) widens nothing and admits nothing.
 *
 * PREFLIGHT BUDGET: the admission chain (fak leaseref liveness -> fak leaseref
 * fence -> fak loop region -> python DOS snapshot -> dos arbitrate) runs under
 * one shared wall-clock deadline of FAK_WRITE_GUARD_BUDGET_MS ms (default 5000,
 * clamped to [250,120000]; invalid/absent env falls back to the default). Each
 * subprocess call gets the remaining budget as its timeout; when the budget is
 * spent BEFORE a call, the call is not spawned and reads as an availability
 * failure. Chain order is unchanged; each sibling group gets its own budget.
 *
 * AVAILABILITY vs VERDICT: a subprocess that cannot produce a parsed verdict
 * (spawn error, nonzero exit, timeout, stdout that fails JSON.parse) is an
 * AVAILABILITY failure. A PARSED verdict that refuses — region.admit !== true,
 * decision.outcome !== "acquire", fence.ok !== true, wrong-shape snapshots — is
 * a VERDICT and stays fail-closed exactly as before. Only availability degrades.
 *
 * DEGRADATION LADDER (availability path only; skipped entirely when
 * FAK_WRITE_GUARD_STRICT=1, the exact escape hatch that restores today's
 * fail-closed throw ` + "`" + `[dos-proof-guard] Lease admission unavailable or refused: ...` + "`" + `):
 *   1. live probe within budget succeeds -> admit, refresh the cache, reconcile;
 *   2. availability failure + fresh cache (age <= FAK_WRITE_GUARD_CACHE_TTL_MS,
 *      default 600000 via the same bounded-env clamp; cache owner+workspace
 *      match AND cache.trees cover ALL requested trees of the group) -> admit
 *      WITHOUT live consultation, DEGRADED_ADMISSION note;
 *   3. availability failure with no usable cache -> admit WITHOUT lease
 *      consultation, ADMISSION_UNVERIFIED note, pending record appended.
 * Policy refusals (OUT_OF_LEASE, COLLISION_RISK, verdict-plane refusals) always
 * run ahead of and dominate the ladder.
 *
 * QUEUEUS LEDGER — the guard's durable, bounded admission ledger under
 * <sessionRoot>/.fak/queueus/, best-effort: it NEVER blocks, throws into, or
 * meaningfully delays the admission decision (all IO is try/catch-swallowed
 * with at most one silent retry; corrupt files self-heal, ignored on read and
 * replaced on the next successful write; dir created recursively on first
 * write; every file written atomically via tmp + rename):
 *   admission-cache.json   schema fak.queueus.admission-cache.v1 —
 *                          {schema, at, owner, session, workspace, trees};
 *                          refreshed after EVERY fully-successful admission.
 *   admission-pending.json schema fak.queueus.pending.v1 —
 *                          {schema, records:[{at, owner, session, trees,
 *                          kind:"admission_unverified"}]}; appended on every
 *                          unverified admit; hard cap 256 records, drop oldest.
 *
 * NOTE BRIDGE: degraded/reconcile notes are held per session (bounded with the
 * same eviction discipline as session state) and PREPENDED to the next
 * after-event mutation note when that mutation SUCCEEDS; a failed mutation
 * drops the note (the failure dominates).
 *
 * RECONCILIATION (resume safely): a fully successful admission drains the
 * pending records for its workspace against the FRESH DOS peer snapshot (rows
 * whose holder/run_id differ from this owner/session); a record whose trees
 * intersect a peer row's tree array is collision_risk (overlapping trees +
 * holder named), else clean. Survivors are unadjudicable records only,
 * normally zero. When anything drained, a QUEUEUS_RECONCILE note names
 * drained=N collision_risk=M and, on collision, the overlapping trees plus
 * 'fak recover COLLISION_RISK'.
 *
 * execute.after annotates completed mutations: a claimed write whose target is
 * absent on disk carries a PHANTOM_WRITE note instead of the proof reminder.
 * It also observes successful capability-catalog reconnaissance (fak-dev
 * feature query / capabilities / index, fak-featureindex orient/refresh, or the
 * corresponding MCP invocations) in the same session+workspace; the first
 * direct mutation WITHOUT such a receipt gets one short DISCOVERY_ADVISORY,
 * labeled as a post-edit notice, at most once per session. The advisory never
 * blocks, throws, runs commands, or touches lease decisions; state is bounded
 * and cleared on disposal; child sessions (distinct sessionIDs) inherit nothing.
 */

const execute = promisify(execFile);
// patch is the 2.x mutation tool; apply_patch its 1.x name.
const mutations = new Set(["write", "edit", "apply_patch", "patch"]);
const SESSION_STATE_CAP = 64;

// Bounded env int: invalid/absent -> default, out of [min,max] -> clamped.
function boundedIntEnv(name, def, min, max) {
  const raw = process.env[name];
  if (raw === undefined || String(raw).trim() === "") return def;
  const n = Number(raw);
  if (!Number.isInteger(n)) return def;
  if (n < min) return min;
  if (n > max) return max;
  return n;
}
// The write-guard preflight budget and the cached-verdict TTL it degrades to.
const WRITE_GUARD_BUDGET_MS = boundedIntEnv("FAK_WRITE_GUARD_BUDGET_MS", 5000, 250, 120000);
const WRITE_GUARD_CACHE_TTL_MS = boundedIntEnv("FAK_WRITE_GUARD_CACHE_TTL_MS", 600000, 250, 3600000);
// QUEUEUS pending-ledger cap: drop oldest past this.
const QUEUEUS_PENDING_CAP = 256;

// AVAILABILITY failure marker: the lease plane could not produce a parsed
// verdict (spawn error, nonzero exit, timeout, unparseable stdout). A PARSED
// verdict that refuses is NOT this — it stays fail-closed as a plain Error.
class AdmissionUnavailableError extends Error {
  constructor(message) {
    super(message);
    this.name = "AdmissionUnavailableError";
  }
}

function hashString(str) {
  if (typeof str !== "string" || !str) return "";
  return createHash("sha256").update(str).digest("hex").slice(0, 16);
}

function extractHolder(errorOrOutput, peers) {
  if (!errorOrOutput) return (peers && peers[0]?.holder) || "peer";
  if (typeof errorOrOutput === "object") {
    if (errorOrOutput.conflict?.holder) return errorOrOutput.conflict.holder;
    if (errorOrOutput.holder) return errorOrOutput.holder;
  }
  const text = typeof errorOrOutput === "string" ? errorOrOutput : (errorOrOutput.stdout || errorOrOutput.stderr || errorOrOutput.message || "");
  const match = text.match(/"holder"\s*:\s*"([^"]+)"/);
  if (match) return match[1];
  if (peers && peers.length > 0 && peers[0]?.holder) return peers[0].holder;
  return "peer";
}

function formatCollisionRecovery(holder) {
  return "[RECOVERY]: Lane conflict detected with holder '" + (holder || "peer") + "'. DO NOT retry editing this path. Run 'fak recover COLLISION_RISK' or select a tree-disjoint task.";
}

// Out-of-tree scratch (OS temp dirs, the harness's own scratch) is not leased
// by the kernel, so a repo mutation tool must neither be blocked nor steered to
// shell for it: that steerage is what taught agents to route around the guard
// (fak-private#12166). Admit it outright; containment still governs every
// in-repo path. Scratch is matched ONLY under a temp/scratch root so a repo
// path that merely contains the word cannot escape admission.
function isAdmittedScratchPath(target, filename) {
  const normTarget = (target || "").replace(/\\/g, "/");
  const normName = (filename || "").replace(/\\/g, "/");
  const scratchRoot = /(?:^|\/)(?:appdata\/local\/temp|temp|tmp|scratch)\//i;
  const scratchLeaf = /(?:^|\/)(?:opencode|scratch)(?:\/|$)/i;
  return (scratchRoot.test(normTarget) || scratchRoot.test(normName)) &&
    (scratchLeaf.test(normTarget) || scratchLeaf.test(normName));
}

function extractOutputText(output) {
  if (!output) return "";
  if (typeof output === "string") return output;
  if (typeof output.error === "string") return output.error;
  if (output.error instanceof Error) return output.error.message;
  if (typeof output.output === "string") return output.output;
  if (typeof output.result === "string") return output.result;
  if (typeof output.text === "string") return output.text;
  if (typeof output.content === "string") return output.content;
  if (Array.isArray(output.content)) {
    return output.content.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n");
  }
  if (Array.isArray(output.result)) {
    return output.result.map((c) => (typeof c === "string" ? c : c?.text || "")).join("\n");
  }
  return "";
}

function isMutationFailure(output) {
  if (!output) return false;
  if (output.error || output.isError === true || output.metadata?.error) return true;
  const text = extractOutputText(output);
  if (!text) return false;
  if (/could not find oldstring/i.test(text)) return true;
  if (/found multiple matches/i.test(text)) return true;
  if (/oldstring not found/i.test(text)) return true;
  if (/^error:\s+/im.test(text)) return true;
  return false;
}

// Completed-call success: explicit failure status, result error markers, or a
// nonzero shell exit all read as not-ok. Absent metadata stays unknown-ok.
function callOk(ev) {
  if (ev?.status === "failed") return false;
  const r = ev?.result;
  if (r && typeof r === "object") {
    if (r.error || r.isError === true || r.metadata?.error) return false;
    const exit = r.metadata?.exit;
    if (exit !== undefined && exit !== null && exit !== 0) return false;
  }
  return true;
}

// timeoutMs is the per-call budget slice; omitted/<=0 keeps the historical
// 120000 default for any other caller. Every failure here is AVAILABILITY —
// the subprocess could not produce a parsed verdict — so it throws the marker
// class, keeping the strict-path message text intact.
async function jsonCommand(command, args, cwd, timeoutMs) {
  let stdout;
  const binary = process.platform === "win32" && !command.endsWith(".exe") ? ` + "`" + `${command}.exe` + "`" + ` : command;
  try {
    ({ stdout } = await execute(binary, args, {
      cwd, timeout: timeoutMs > 0 ? timeoutMs : 120000, maxBuffer: 4 * 1024 * 1024, windowsHide: true,
    }));
  } catch (error) {
    const rawOut = error.stdout || error.stderr || error.message || "";
    let recovery = "";
    if (rawOut.includes("COLLISION_RISK")) {
      const holder = extractHolder(error);
      recovery = "\n" + formatCollisionRecovery(holder);
    }
    throw new AdmissionUnavailableError("[dos-proof-guard] Lease admission unavailable or refused: " + command + ": " + rawOut + recovery);
  }
  try {
    return JSON.parse(stdout);
  } catch {
    throw new AdmissionUnavailableError("[dos-proof-guard] Malformed lease admission response from " + command);
  }
}

// QUEUEUS ledger IO: best-effort and bounded. EVERY failure (missing dir,
// unwritable, corrupt JSON) is swallowed after at most one silent retry, so the
// ledger never blocks, throws into, or meaningfully delays the admission
// decision. Writes are atomic (tmp file + rename); a corrupt file is ignored on
// read and replaced by the next successful write (self-heal).
async function queueusWrite(file, obj) {
  const payload = JSON.stringify(obj);
  const attempt = async () => {
    await mkdir(path.dirname(file), { recursive: true });
    const tmp = ` + "`" + `${file}.tmp-${process.pid}-${Date.now()}` + "`" + `;
    await writeFile(tmp, payload, "utf8");
    await rename(tmp, file);
  };
  try {
    await attempt();
  } catch {
    try { await attempt(); } catch {}
  }
}

async function queueusReadJson(file) {
  try {
    return JSON.parse(await readFile(file, "utf8"));
  } catch {
    return null; // absent or corrupt: the next successful write replaces it
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
  if (tool !== "apply_patch" && tool !== "patch") {
    const candidates = [args?.filePath, args?.file_path, args?.path, args?.target];
    return [...new Set(candidates
      .filter((p) => typeof p === "string" && p.trim() !== "")
      .map((p) => p.trim()))];
  }
  const patch = typeof args?.patchText === "string" ? args.patchText : (typeof args?.patch === "string" ? args.patch : "");
  if (!patch) return [];

  const paths = new Set();
  const addPath = (raw) => {
    if (!raw || typeof raw !== "string") return;
    let s = raw.trim();
    const tabIdx = s.indexOf("\t");
    if (tabIdx !== -1) s = s.slice(0, tabIdx).trim();
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
      s = s.slice(1, -1).trim();
    }
    if (s.startsWith("a/") || s.startsWith("b/")) {
      s = s.slice(2).trim();
    }
    if (s && s !== "/dev/null" && s !== "dev/null") {
      paths.add(s);
    }
  };

  for (const m of patch.matchAll(/^\*\*\* (?:Add File|Update File|Delete File|Move to): (.+)\r?$/gm)) {
    addPath(m[1]);
  }
  for (const m of patch.matchAll(/^diff --git\s+(?:"([^"]+)"|'([^']+)'|(\S+))\s+(?:"([^"]+)"|'([^']+)'|(\S+))\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
    addPath(m[4] || m[5] || m[6]);
  }
  for (const m of patch.matchAll(/^---\s+(?:"([^"]+)"|'([^']+)'|([^\t\r\n]+?))(?:\t.*)?\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
  }
  for (const m of patch.matchAll(/^\+\+\+\s+(?:"([^"]+)"|'([^']+)'|([^\t\r\n]+?))(?:\t.*)?\r?$/gm)) {
    addPath(m[1] || m[2] || m[3]);
  }
  return [...paths];
}

// Paths a patch deletes: absence on disk after a completed delete is expected,
// not a phantom.
function deletedPaths(args) {
  const patch = typeof args?.patchText === "string" ? args.patchText : (typeof args?.patch === "string" ? args.patch : "");
  if (!patch) return [];
  const paths = new Set();
  for (const m of patch.matchAll(/^\*\*\* Delete File: (.+)\r?$/gm)) {
    let s = m[1].trim();
    if ((s.startsWith('"') && s.endsWith('"')) || (s.startsWith("'") && s.endsWith("'"))) {
      s = s.slice(1, -1).trim();
    }
    if (s.startsWith("a/") || s.startsWith("b/")) {
      s = s.slice(2).trim();
    }
    if (s) paths.add(s);
  }
  return [...paths];
}

function normPath(p) {
  const s = String(p || "").replace(/\\/g, "/");
  return process.platform === "win32" ? s.toLowerCase() : s;
}

function isInside(target, container) {
  const t = normPath(target);
  const c = normPath(container);
  return t === c || t.startsWith(c + "/");
}

// Declared sibling workspace roots (FAK_PUBLIC_ROOT, FAK_ADMITTED_ROOTS list).
// A declaration that CONTAINS the session root (ancestor, or identical) must
// never widen the fence: it adds nothing beyond session containment itself.
function declaredAdmissionRoots(sessionRoot, env) {
  const raw = [];
  const pub = String(env.FAK_PUBLIC_ROOT || "").trim();
  if (pub) raw.push(pub);
  const list = String(env.FAK_ADMITTED_ROOTS || "");
  if (list) {
    for (const piece of list.split(path.delimiter)) {
      const p = piece.trim();
      if (p) raw.push(p);
    }
  }
  const out = [];
  for (const r of raw) {
    const abs = path.resolve(r);
    if (isInside(sessionRoot, abs)) continue;
    if (!out.includes(abs)) out.push(abs);
  }
  return out;
}

// The deepest declared root containing the target, or null.
function admissionRootFor(target, roots) {
  let best = null;
  for (const r of roots) {
    if (isInside(target, r) && (best === null || normPath(r).length > normPath(best).length)) {
      best = r;
    }
  }
  return best;
}

// Closed discovery contract: the capability-catalog CLI family (bare or wrapped
// in fak-run / go run) and the corresponding MCP tool names. Printed commands,
// schema-only search, and everything else do not qualify.
const DISCOVERY_SHELL_FAK_DEV = /\bfak-dev\s+(?:feature\s+query|capabilities|index)\b/i;
const DISCOVERY_SHELL_FEATUREINDEX = /\bfak-featureindex\s+(?:orient|refresh)\b/i;

function isDiscoveryShellCommand(args) {
  const cmd = typeof args?.command === "string" ? args.command : "";
  if (!cmd) return false;
  return DISCOVERY_SHELL_FAK_DEV.test(cmd) || DISCOVERY_SHELL_FEATUREINDEX.test(cmd);
}

function isDiscoveryToolName(tool) {
  const n = String(tool || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return /^(?:mcp_)?(?:fak|dos)_(?:feature_query|featureindex|capabilities|catalog_index|index)$/.test(n);
}

async function createGuard({ directory }) {
  let fileModifiedCount = 0;
  const failureHistory = new Map();
  // Capture host configuration once; tool arguments never supply ownership.
  const hostOwner = process.env.FAK_LEASE_OWNER;
  const hostSession = process.env.FAK_LEASE_SESSION;
  const hostLease = process.env.FAK_LEASE_ID;
  const root = await realpath(directory || process.cwd());
  // Bounded per-session discovery state: probed = a successful same-session
  // capability-catalog probe was observed; advised = the advisory fired (or was
  // suppressed by a probe). Keyed by sessionID, so child sessions inherit
  // nothing; cleared on disposal.
  const sessions = new Map();
  // Note bridge: before -> after. A degraded/reconcile note set by
  // admitWorkspace for a session is consumed (and cleared) by that session's
  // next afterEvent: prepended when the mutation succeeded, dropped when it
  // failed (the failure dominates). Same cap discipline as ` + "`" + `sessions` + "`" + `.
  const pendingNotes = new Map();
  // QUEUEUS ledger location: <sessionRoot>/.fak/queueus/, created on first write.
  const queueusDir = path.join(root, ".fak", "queueus");
  const queueusCacheFile = path.join(queueusDir, "admission-cache.json");
  const queueusPendingFile = path.join(queueusDir, "admission-pending.json");

  function setPendingNote(session, note) {
    if (!pendingNotes.has(session) && pendingNotes.size >= SESSION_STATE_CAP) {
      pendingNotes.delete(pendingNotes.keys().next().value);
    }
    pendingNotes.set(session, note);
  }

  function takePendingNote(session) {
    const note = pendingNotes.get(session);
    pendingNotes.delete(session);
    return note || "";
  }

  function normalizeKey(session, filename) {
    const full = path.resolve(root, filename);
    return session + ":" + full.toLowerCase().replace(/\\/g, "/");
  }

  function checkCircuitBreaker(session, filename) {
    const key = normalizeKey(session, filename);
    const entry = failureHistory.get(key);
    if (entry && entry.count >= 2) {
      entry.count++;
      throw new Error("[dos-proof-guard-circuit-breaker] Refusing 3rd consecutive failing edit on " + filename + ". Stop retrying. Inspect the file with 'read' or pivot to an alternate lane.");
    }
  }

  function recordFailure(session, filename, tool, args, reason) {
    const key = normalizeKey(session, filename);
    const existing = failureHistory.get(key);
    const count = (existing?.count || 0) + 1;
    const oldString = args?.oldString || args?.old_string || "";
    const oldStringHash = oldString ? hashString(oldString) : (existing?.oldStringHash || "");
    failureHistory.set(key, {
      tool: tool || existing?.tool || "unknown",
      filePath: filename,
      oldStringHash,
      failureReason: reason || existing?.failureReason || "",
      count,
      lastFailedAt: Date.now(),
    });
  }

  function resetFailure(session, filename) {
    const key = normalizeKey(session, filename);
    failureHistory.delete(key);
  }

  function sessionState(session) {
    let st = sessions.get(session);
    if (!st) {
      if (sessions.size >= SESSION_STATE_CAP) {
        sessions.delete(sessions.keys().next().value);
      }
      st = { probed: false, advised: false };
      sessions.set(session, st);
    }
    return st;
  }

  // A cached verdict is usable for degraded admission only while fresh, owned
  // by the same owner in the same workspace, and covering every requested tree.
  function cacheUsable(cache, workspace, owner, trees) {
    if (!cache || typeof cache !== "object" || cache.schema !== "fak.queueus.admission-cache.v1") return false;
    if (typeof cache.at !== "number" || !Number.isFinite(cache.at)) return false;
    if (Date.now() - cache.at > WRITE_GUARD_CACHE_TTL_MS) return false;
    if (cache.owner !== owner || normPath(cache.workspace) !== normPath(workspace)) return false;
    const cached = Array.isArray(cache.trees) ? cache.trees.map(normPath) : [];
    return trees.every((t) => cached.includes(normPath(t)));
  }

  // Reconciliation (resume safely): drain this workspace's pending records
  // against the FRESH peer snapshot. A record whose trees intersect a peer
  // row's tree array is collision_risk (overlapping trees + holder named),
  // else clean. Only unadjudicable records (no parseable tree list) survive;
  // normally zero. Never throws; sets the note bridge only when it is free.
  async function reconcileQueueus(session, owner, peers) {
    try {
      const doc = await queueusReadJson(queueusPendingFile);
      const records = (doc && Array.isArray(doc.records)) ? doc.records : [];
      if (records.length === 0) return;
      let drained = 0;
      let collisionRisk = 0;
      const overlapTrees = new Set();
      const peerHolders = new Set();
      const kept = [];
      for (const record of records) {
        const rt = record && Array.isArray(record.trees) ? record.trees.map(normPath) : null;
        if (rt === null) { kept.push(record); continue; } // unadjudicable: no tree list to intersect
        let hit = false;
        for (const row of peers) {
          const pt = Array.isArray(row?.tree) ? row.tree.map(normPath) : [];
          const shared = rt.filter((t) => pt.includes(t));
          if (shared.length > 0) {
            hit = true;
            for (const t of shared) overlapTrees.add(t);
            if (row?.holder) peerHolders.add(row.holder);
            break;
          }
        }
        drained++;
        if (hit) collisionRisk++;
      }
      const note = "[dos-proof-guard:QUEUEUS_RECONCILE] drained=" + drained + " collision_risk=" + collisionRisk +
        (collisionRisk > 0
          ? " — overlapping trees: " + [...overlapTrees].join(", ") + (peerHolders.size ? " (peer holder: " + [...peerHolders].join(", ") + ")" : "") + ". Run 'fak recover COLLISION_RISK' before editing those paths."
          : "");
      if (!pendingNotes.has(session)) setPendingNote(session, note);
      await queueusWrite(queueusPendingFile, { schema: "fak.queueus.pending.v1", records: kept });
    } catch {}
  }

  async function admitWorkspace(workspace, trees, session, owner) {
    // One shared wall-clock deadline for the whole chain: each call gets the
    // remaining budget as its timeout, and a spent budget skips the remaining
    // calls entirely instead of spawning doomed subprocesses.
    const deadline = Date.now() + WRITE_GUARD_BUDGET_MS;
    const consult = async (command, args) => {
      if (deadline - Date.now() <= 0) {
        throw new AdmissionUnavailableError("[dos-proof-guard] Lease admission unavailable or refused: " + command + ": budget exhausted before consultation");
      }
      return jsonCommand(command, args, workspace, Math.max(250, deadline - Date.now()));
    };
    let refs, own, regionArgs, lease, fence, region, snapshot, peers, decision;
    try {
      refs = await consult("fak", ["leaseref", "liveness", "--dir", workspace, "--session", session]);
      if (!Array.isArray(refs)) throw new Error("[dos-proof-guard] Invalid FAK lease snapshot");
      own = refs.filter((row) => row.holder === owner && row.session_id === session && (!hostLease || row.id === hostLease));
      if (hostLease && own.length !== 1) throw new Error("[dos-proof-guard] Host lease identity is not current");
      if (own.length > 1) throw new Error("[dos-proof-guard] Set host FAK_LEASE_ID to identify the active lease");
      regionArgs = ["loop", "region", "--dir", workspace, "--actor", owner, "--no-queue", "--json"];
      for (const tree of trees) regionArgs.push("--tree", tree);
      if (own.length === 1) {
        lease = own[0];
        if (!lease.id || !Number.isInteger(lease.generation) || lease.generation < 1) {
          throw new Error("[dos-proof-guard] Own FAK lease lacks a fencing generation");
        }
        fence = await consult("fak", ["leaseref", "fence", "--dir", workspace, "--id", lease.id, "--holder", owner, "--generation", String(lease.generation)]);
        if (fence.ok !== true) throw new Error("[dos-proof-guard] Own FAK lease fence refused");
        regionArgs.push("--self", lease.id);
      }
      region = await consult("fak", regionArgs);
      if (region.schema !== "fak.loop-region.v1" || region.admit !== true) {
        let recovery = "";
        if (region.reason === "COLLISION_RISK" || JSON.stringify(region).includes("COLLISION_RISK")) {
          const holder = extractHolder(region);
          recovery = "\n" + formatCollisionRecovery(holder);
        }
        throw new Error("[dos-proof-guard] FAK lease admission refused" + recovery);
      }

      // DOS owns WAL parsing, corruption handling, expiry and locking. DOS >= 0.22.0
      // removed the strict kwarg; emitting it raises TypeError and blocks mutations.
      // FAK_PYTHON overrides python resolution (the witness recording stubs use it).
      snapshot = await consult(process.env.FAK_PYTHON || "python", ["-c",
        "import json,sys; from dos import config,lane_lease; cfg=config.load_workspace_config(sys.argv[1],gather_env=False); print(json.dumps(lane_lease.live_leases(cfg, expire_dead=True)))", workspace]);
      if (!Array.isArray(snapshot) || snapshot.some((row) => !row || typeof row.lane !== "string" || !Array.isArray(row.tree))) {
        throw new Error("[dos-proof-guard] Invalid DOS lease snapshot");
      }
      peers = snapshot.filter((row) => !(row.holder === owner && row.run_id === session));
      decision = await consult("dos", ["arbitrate", "--workspace", workspace, "--lane", "opencode-native-write", "--kind", "keyword", "--output", "json", "--leases", JSON.stringify(peers), "--tree", ...trees]);
      if (decision.outcome !== "acquire") {
        let recovery = "";
        if (decision.reason === "COLLISION_RISK" || JSON.stringify(decision).includes("COLLISION_RISK")) {
          const holder = extractHolder(decision, peers);
          recovery = "\n" + formatCollisionRecovery(holder);
        }
        throw new Error("[dos-proof-guard] DOS lease admission refused" + recovery);
      }
    } catch (err) {
      // VERDICT refusals (and identity/snapshot-shape errors) propagate exactly
      // as before: fail-closed. Only AVAILABILITY degrades.
      if (!(err instanceof AdmissionUnavailableError)) throw err;
      // Strict escape hatch: restore today's fail-closed behavior verbatim.
      if (String(process.env.FAK_WRITE_GUARD_STRICT) === "1") throw err;
      // Ladder step 2: a fresh cached verdict admits without live consultation.
      const cache = await queueusReadJson(queueusCacheFile);
      if (cacheUsable(cache, workspace, owner, trees)) {
        setPendingNote(session, "[dos-proof-guard:DEGRADED_ADMISSION] Lease plane unreadable within " + WRITE_GUARD_BUDGET_MS + "ms; admitted on the cached verdict from " + Math.max(0, Math.round((Date.now() - cache.at) / 1000)) + "s ago (no live fence was taken). Re-probe when the plane recovers.");
        return;
      }
      // Ladder step 3: admit WITHOUT lease consultation, recorded pending.
      setPendingNote(session, "[dos-proof-guard:ADMISSION_UNVERIFIED] Lease plane unreadable within " + WRITE_GUARD_BUDGET_MS + "ms and no cached verdict; write admitted WITHOUT lease admission and recorded pending in the QUEUEUS ledger. Reconcile with 'fak leaseref liveness' when the plane recovers; if a peer collision surfaces afterward, run 'fak recover COLLISION_RISK'.");
      const doc = await queueusReadJson(queueusPendingFile);
      const records = (doc && Array.isArray(doc.records)) ? doc.records : [];
      records.push({ at: Date.now(), owner, session, trees: [...trees], kind: "admission_unverified" });
      while (records.length > QUEUEUS_PENDING_CAP) records.shift();
      await queueusWrite(queueusPendingFile, { schema: "fak.queueus.pending.v1", records });
      return;
    }
    // Fully successful admission: refresh the durable cache, then drain the
    // pending ledger against the FRESH peer snapshot (resume safely).
    await queueusWrite(queueusCacheFile, {
      schema: "fak.queueus.admission-cache.v1",
      at: Date.now(),
      owner, session, workspace,
      trees: [...trees],
    });
    await reconcileQueueus(session, owner, peers);
    // These are pre-execution observations, not an atomic filesystem fence.
    // Shell/MCP mutations and late lease changes need separate mediation.
  }

  /** execute.before: fence every mutation through lease admission. Throws OUT_OF_LEASE at the escape pre-check before any external command runs. */
  async function beforeEvent(ev) {
    const tool = ev?.tool;
    if (!mutations.has(tool)) return;
    const sessionID = ev?.sessionID;
    if (!sessionID || (!!hostOwner !== !!hostSession)) {
      throw new Error("[dos-proof-guard] A harness session and paired host lease owner/session are required");
    }
    const session = hostSession || sessionID;
    const owner = hostOwner || ` + "`" + `opencode:${session}` + "`" + `;
    const args = ev?.args;
    const filenames = mutationPaths(tool, args);
    if (!filenames.length || filenames.some((name) => typeof name !== "string" || !name.trim())) {
      throw new Error("[dos-proof-guard] Explicit mutation paths are required for lease admission");
    }

    for (const filename of filenames) {
      checkCircuitBreaker(session, filename);
    }

    try {
      const siblings = declaredAdmissionRoots(root, process.env);
      const groups = new Map(); // admission workspace root -> [tree, ...]
      let scratchOnly = true;
      for (const filename of filenames) {
        if (filename.includes("\0")) {
          throw new Error("[dos-proof-guard] Malformed mutation path: contains null byte");
        }
        const target = await canonicalPath(path.resolve(root, filename));
        const relative = path.relative(root, target);
        if (!relative || relative === ".." || relative.startsWith(` + "`" + `..${path.sep}` + "`" + `) || path.isAbsolute(relative)) {
          // Out-of-tree scratch is admitted outright: the kernel does not lease
          // OS temp, and the old "use shell" steerage taught agents to route
          // around the guard (fak-private#12166). In-repo containment is
          // unaffected; only temp/scratch subtrees short-circuit.
          if (isAdmittedScratchPath(target, filename)) {
            continue;
          }
          // A declared sibling root re-bases the target through the SAME
          // admission chain against that root — consultation, not bypass.
          const sibling = admissionRootFor(target, siblings);
          if (sibling) {
            scratchOnly = false;
            const srel = path.relative(sibling, target);
            const list = groups.get(sibling);
            if (list) list.push(srel.split(path.sep).join("/"));
            else groups.set(sibling, [srel.split(path.sep).join("/")]);
            continue;
          }
          throw new Error("[dos-proof-guard:OUT_OF_LEASE] Mutation path is outside this lease workspace: '" + filename + "'. The kernel leases in-repo trees only; this path is not covered by any lane. Take the lane that owns this path (fak leaseref acquire ... or dos arbitrate), or record a handoff - do NOT re-attempt the same write through a different tool (shell/edit/write all cross the same lease boundary). Out-of-tree SCRATCH belongs under a temp/scratch directory and is admitted automatically.");
        }
        scratchOnly = false;
        const list = groups.get(root);
        if (list) list.push(relative.split(path.sep).join("/"));
        else groups.set(root, [relative.split(path.sep).join("/")]);
      }

      if (scratchOnly) {
        // Nothing in-repo to admit; scratch writes need no lease consultation.
        return;
      }

      for (const [workspace, trees] of groups) {
        await admitWorkspace(workspace, trees, session, owner);
      }
    } catch (err) {
      for (const filename of filenames) {
        recordFailure(session, filename, tool, args, err.message);
      }
      throw err;
    }
  }

  /** execute.after: annotate completed mutations; observe discovery; advisory. Never throws. */
  async function afterEvent(ev) {
    try {
      const tool = ev?.tool || "";
      const result = ev?.result;
      const ok = callOk(ev);
      // Observer: a completed successful catalog probe in this session/workspace
      // earns the receipt that suppresses the advisory. Failed or empty-output
      // calls stay unknown and earn nothing.
      if (ok && (isDiscoveryShellCommand(ev?.args) || isDiscoveryToolName(tool))) {
        const text = extractOutputText(result);
        if (text.trim()) {
          sessionState(hostSession || ev?.sessionID || "default").probed = true;
        }
      }
      if (!mutations.has(tool)) return;

      const session = hostSession || ev?.sessionID || "default";
      const args = ev?.args;
      const files = mutationPaths(tool, args);

      // Note bridge: consume the degraded/reconcile note set by beforeEvent. A
      // failed mutation drops it (the failure dominates); a successful one
      // prepends it below.
      const pendingNote = takePendingNote(session);

      if (!ok || isMutationFailure(result)) {
        for (const f of files) {
          recordFailure(session, f, tool, args, extractOutputText(result) || "mutation_failed");
        }
        return;
      }

      for (const f of files) {
        resetFailure(session, f);
      }

      fileModifiedCount++;

      // Phantom write: the tool reported success but the target is absent on
      // disk. Deletes expect absence; everything else must exist.
      const deleted = new Set(deletedPaths(args));
      const phantoms = [];
      for (const f of files) {
        if (deleted.has(f)) continue;
        let target;
        try {
          target = await canonicalPath(path.resolve(root, f));
        } catch {
          continue;
        }
        try {
          if (!existsSync(target)) phantoms.push(f);
        } catch {}
      }
      let note = "";
      if (phantoms.length) {
        note = pendingNote + "\n\n[dos-proof-guard] PHANTOM_WRITE: " + phantoms.map((p) => "'" + p + "'").join(", ") +
          " reported written but absent on disk. Confirm the write actually landed before claiming completion.";
      } else {

        let reminder = "\n\n[dos-proof-guard] Code modified. On-device proof required before completion:\n" +
          "1. Run scoped tests for the touched packages: go test ./<touched-pkg>/... -count=1\n" +
          "2. Check the diff against its claim with the DOS truth syscall (dos verify)\n" +
          "3. Spawn cross-validator subagent to adversarial-audit the diff\n" +
          "4. Auto-ticket follow-ons/edge-cases by default (gh issue create / fak issue fanout)\n";

        const hasHardware = files.some((f) => typeof f === "string" && (f.includes("internal/compute") || f.includes("internal/amdgpu") || f.includes("vulkan") || f.includes("strix")));
        if (hasHardware) {
          reminder += "\n[dos-proof-guard] Halo-related compute/hardware code modified. Physical hardware evidence required:\n" +
            "- Run ` + "`" + `fak-dev amd-strix-probe` + "`" + ` early to discover the LAN AMD Strix Halo and choose a bounded device correctness witness.\n" +
            "- Hold an exclusive lease across the entire baseline/candidate performance comparison.\n";
        }

        // The discovery advisory precedes the standing proof reminder: it is
        // the one-time post-edit notice that this session skipped the catalog.
        // The pending note bridge (degraded/reconcile admission) precedes both.
        const advisory = maybeAdvisory(sessionState(session));
        note = pendingNote + (advisory || "") + reminder;
      }
      // Emit against the hook result under an injection-root receiver name so
      // the self-observation audit sees this text reach the model context.
      if (note && result && typeof result === "object") {
        if (typeof result.output === "string") {
          result.output += note;
        } else if (typeof result.content === "string") {
          result.content += note;
        } else if (Array.isArray(result.content)) {
          result.content.push({ type: "text", text: note });
        } else if (typeof result.result === "string") {
          result.result += note;
        } else if (Array.isArray(result.result)) {
          result.result.push({ type: "text", text: note });
        } else if (typeof result.text === "string") {
          result.text += note;
        }
      }
    } catch {}
  }

  /** Bounded state, cleared on disposal: child sessions inherit nothing. */
  function dispose() {
    sessions.clear();
    pendingNotes.clear();
    failureHistory.clear();
    fileModifiedCount = 0;
  }

  return {
    root,
    dispose,
    resetCounter: () => { fileModifiedCount = 0; },
    beforeEvent,
    afterEvent,
  };
}

// One short DISCOVERY_ADVISORY per session: the first landed mutation in a
// session with no successful same-session catalog probe yet. Labeled a
// post-edit notice because it arrives after the fact; it never blocks, throws,
// runs commands, or touches lease decisions.
function maybeAdvisory(st) {
  if (st.probed || st.advised) return "";
  st.advised = true;
  return "\n\n[dos-proof-guard:DISCOVERY_ADVISORY] Post-edit notice: this session made a direct code change without a prior capability-catalog probe. Query the catalog before the next edit - fak-dev feature query \"<axis>\", fak-featureindex orient \"<intent>\", or the corresponding MCP catalog tool - so work reuses existing capability instead of rediscovering it.";
}

/** OpenCode 1.x dotted-hook surface: the same guard behind a stable contract. */
async function dosProofGuardServer({ client, directory }) {
  const guard = await createGuard({ directory });
  return {
    "tool.execute.before": async (input, output) => {
      // Fail closed: OUT_OF_LEASE must reject this hook call, not leak as an
      // unhandled rejection after the caller has moved on.
      await guard.beforeEvent({
        tool: input?.tool,
        sessionID: input?.sessionID,
        args: output?.args ?? input?.args,
      });
    },
    "tool.execute.after": async (input, output) => {
      await guard.afterEvent({
        tool: input?.tool,
        sessionID: input?.sessionID,
        args: input?.args ?? output?.args,
        result: output,
        status: input?.status ?? output?.status,
      });
    },
    "chat.message": async (input) => {
      if (input?.role === "user") guard.resetCounter();
    },
  };
}

/** OpenCode 2.x entrypoint: exactly two hooks; cleanup disposes both and clears state. */
async function dosProofGuardSetup(api) {
  const guard = await createGuard({ directory: api?.location?.directory || process.cwd() });
  const registrations = [];
  registrations.push(await api.tool.hook("execute.before", async (event) => {
    // Fail closed on the fence: OUT_OF_LEASE must propagate to the caller.
    await guard.beforeEvent({
      tool: event?.tool,
      sessionID: event?.sessionID,
      args: event?.input ?? event?.args,
    });
  }));
  registrations.push(await api.tool.hook("execute.after", async (event) => {
    try {
      await guard.afterEvent({
        tool: event?.tool,
        sessionID: event?.sessionID,
        args: event?.input ?? event?.args,
        result: event?.result,
        status: event?.status,
      });
    } catch {}
  }));
  return async () => {
    for (const registration of registrations) await registration.dispose();
    guard.dispose();
  };
}

export default { id: "dos-proof-guard", server: dosProofGuardServer, setup: dosProofGuardSetup };`
