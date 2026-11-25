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
