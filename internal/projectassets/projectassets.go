package projectassets

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

const ManifestPath = ".claude/project-assets.json"

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
func adapter(name, rel string) string {
	return fmt.Sprintf("---\nname: %s\ndescription: Generated Codex adapter for the canonical fak project skill %s.\nmetadata:\n  generated-by: fak project-assets sync\n  canonical: %s\n---\n\n# Canonical project skill adapter\n\nLoad and follow [`%s`](%s). This generated discovery adapter contains no maintained workflow body.\n\n## Portability contract\n\n- The linked canonical `SKILL.md` is the single semantic workflow body for Claude, Codex, and fak-native loaders.\n- This adapter changes discovery only; it must not fork, summarize, or translate the workflow.\n- Harness-native invocation, permissions, hooks, model routing, and worker launch remain typed adapters outside the semantic body.\n", name, name, rel, rel, rel)
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
			e = os.WriteFile(abs, []byte(adapter(n, filepath.ToSlash(rel))), 0644)
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
		if e != nil || (isGeneratedAdapter(b) && string(b) != adapter(n, filepath.ToSlash(rel))) {
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
