package ctxknobs

import (
	"bufio"
	"github.com/anthony-chaudhary/fak/internal/sortkeys"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ReasonNewUserRequiredKnob is the closed-vocabulary refusal code the ratchet
// emits for a user-required context overlay that is not in the frozen baseline.
const ReasonNewUserRequiredKnob = "NEW_USER_REQUIRED_KNOB"

// Class is the two-valued classification of a context knob (doctrine L5).
type Class string

const (
	// OperatorDebug is a knob that survives as an operator/debug surface — fine.
	OperatorDebug Class = "operator-debug"
	// UserRequired is an overlay a user or agent must engage to manage context — a defect.
	UserRequired Class = "user-required"
)

// Kind is what sort of overlay a knob is.
type Kind string

const (
	KindFlag  Kind = "flag"  // a cmd/fak flag registration
	KindEnv   Kind = "env"   // a cmd/fak environment lookup
	KindSkill Kind = "skill" // a .claude/skills context-management skill
)

// Knob is one context-management overlay with file:line provenance.
type Knob struct {
	Kind     Kind   `json:"kind"`
	Name     string `json:"name"`
	Class    Class  `json:"class"`
	File     string `json:"file"` // repo-relative, forward slashes
	Line     int    `json:"line"`
	Evidence string `json:"evidence"`
}

// Key is the stable identity the ratchet baseline is keyed on (kind:name).
func (k Knob) Key() string { return string(k.Kind) + ":" + k.Name }

// Inventory is a full, sorted scan of the tree's context knobs.
type Inventory struct {
	Knobs         []Knob `json:"knobs"`
	UserRequired  int    `json:"user_required"`
	OperatorDebug int    `json:"operator_debug"`
}

// Scan walks root and returns the sorted context-knob inventory. It is
// deterministic: the same tree yields byte-identical output (the "run twice →
// identical" witness). A missing scan root (no cmd/fak or no .claude/skills) is
// not an error — that source simply contributes nothing.
func Scan(root string) (Inventory, error) {
	var knobs []Knob

	flagEnv, err := scanFlagsAndEnv(root)
	if err != nil {
		return Inventory{}, err
	}
	knobs = append(knobs, flagEnv...)

	skills, err := scanSkills(root)
	if err != nil {
		return Inventory{}, err
	}
	knobs = append(knobs, skills...)

	sort.Slice(knobs, func(i, j int) bool {
		a, b := knobs[i], knobs[j]
		return sortkeys.FileLine(a.File, a.Line, a.Key(), b.File, b.Line, b.Key())
	})

	inv := Inventory{Knobs: knobs}
	for _, k := range knobs {
		switch k.Class {
		case UserRequired:
			inv.UserRequired++
		case OperatorDebug:
			inv.OperatorDebug++
		}
	}
	return inv, nil
}

// RatchetOffenses returns the user-required knobs whose Key is NOT in the
// baseline set — the new manual overlays the ratchet refuses. Operator-debug
// knobs are never offenses (they are permitted surfaces). The result preserves
// the inventory's sorted order.
func RatchetOffenses(inv Inventory, baseline []string) []Knob {
	allowed := make(map[string]bool, len(baseline))
	for _, k := range baseline {
		allowed[k] = true
	}
	var off []Knob
	for _, k := range inv.Knobs {
		if k.Class == UserRequired && !allowed[k.Key()] {
			off = append(off, k)
		}
	}
	return off
}

// --- flag / env scanning (cmd/fak) ---

var (
	// The NAME is the first double-quoted string on a flag-registration line:
	// flag.String("name", ...) / fs.IntVar(&x, "name", ...) — the &x arg is not quoted.
	reFlagReg  = regexp.MustCompile(`\b(?:flag|fs)\.(?:String|Bool|Int|Int64|Uint|Uint64|Float64|Duration|Var|Func|TextVar)(?:Var)?\(`)
	reFirstStr = regexp.MustCompile(`"([^"]*)"`)
	reEnv      = regexp.MustCompile(`\bos\.(?:Getenv|LookupEnv)\("([^"]+)"\)`)
)

func scanFlagsAndEnv(root string) ([]Knob, error) {
	dir := filepath.Join(root, "cmd", "fak")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var knobs []Knob
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".go") || strings.HasSuffix(e.Name(), "_test.go") {
			continue
		}
		rel := "cmd/fak/" + e.Name()
		fileKnobs, err := scanGoFileForKnobs(filepath.Join(dir, e.Name()), rel)
		if err != nil {
			return nil, err
		}
		knobs = append(knobs, fileKnobs...)
	}
	return knobs, nil
}

func scanGoFileForKnobs(path, rel string) ([]Knob, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	var knobs []Knob
	seen := map[string]bool{} // dedup identical (kind,name) within a file
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	for sc.Scan() {
		line++
		text := sc.Text()
		if loc := reFlagReg.FindStringIndex(text); loc != nil {
			if m := reFirstStr.FindStringSubmatch(text[loc[1]-1:]); m != nil {
				name := m[1]
				if name != "" && isContextFlagName(name) && !seen["flag:"+name] {
					seen["flag:"+name] = true
					knobs = append(knobs, Knob{
						Kind: KindFlag, Name: name, Class: OperatorDebug,
						File: rel, Line: line,
						Evidence: "cmd/fak flag whose name touches context/cache/session budgets",
					})
				}
			}
		}
		for _, m := range reEnv.FindAllStringSubmatch(text, -1) {
			name := m[1]
			if isContextEnvName(name) && !seen["env:"+name] {
				seen["env:"+name] = true
				knobs = append(knobs, Knob{
					Kind: KindEnv, Name: name, Class: OperatorDebug,
					File: rel, Line: line,
					Evidence: "cmd/fak env lookup whose name touches context/cache/session budgets",
				})
			}
		}
	}
	return knobs, sc.Err()
}

// isContextFlagName decides whether a flag/base name is a context/cache/session
// BUDGET knob. It is deliberately narrow (name-only, not body text) so the
// inventory stays meaningful rather than sweeping in every cache/token flag.
func isContextFlagName(name string) bool {
	low := strings.ToLower(name)
	switch {
	case strings.Contains(low, "ctx"),
		strings.Contains(low, "context"),
		strings.Contains(low, "compact"),
		strings.Contains(low, "managed-cache"),
		strings.Contains(low, "managedcache"):
		return true
	case strings.Contains(low, "budget"):
		return strings.Contains(low, "context") || strings.Contains(low, "session") ||
			strings.Contains(low, "cache") || strings.Contains(low, "token") ||
			strings.Contains(low, "view") || strings.Contains(low, "ctx")
	}
	return false
}

func isContextEnvName(name string) bool { return isContextFlagName(name) }

// --- skill scanning (.claude/skills) ---

var (
	reSkillMgmtVerb = regexp.MustCompile(`(?i)\b(compact|prune|trim|shrink|tier|truncat|hygiene|manage|overlay|clear|/compact|/clear)\b`)
	reSkillCtxNoun  = regexp.MustCompile(`(?i)\b(context|memory|window|memory\.md|auto-memory|token budget)\b`)
	reSkillNameTok  = regexp.MustCompile(`(?i)(memory|context|compact|ctx)`)
)

func scanSkills(root string) ([]Knob, error) {
	dir := filepath.Join(root, ".claude", "skills")
	entries, err := os.ReadDir(dir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}
	var knobs []Knob
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		skillPath := filepath.Join(dir, e.Name(), "SKILL.md")
		name, desc, nameLine, err := readSkillFrontmatter(skillPath)
		if err != nil {
			if os.IsNotExist(err) {
				continue
			}
			return nil, err
		}
		if name == "" {
			name = e.Name()
		}
		if isUserRequiredSkill(name, desc) {
			knobs = append(knobs, Knob{
				Kind: KindSkill, Name: name, Class: UserRequired,
				File: ".claude/skills/" + e.Name() + "/SKILL.md", Line: nameLine,
				Evidence: "skill whose purpose is managing the context window or memory store",
			})
		}
	}
	return knobs, nil
}

// isUserRequiredSkill is true when a skill's REASON FOR EXISTING is context or
// memory management: its name carries a context/memory token AND its description
// pairs a management verb with a context noun. This isolates the manage-your-
// context overlays (memory-compact) from scorecards and other skills that merely
// mention context in passing.
func isUserRequiredSkill(name, desc string) bool {
	return reSkillNameTok.MatchString(name) &&
		reSkillMgmtVerb.MatchString(desc) &&
		reSkillCtxNoun.MatchString(desc)
}

// readSkillFrontmatter extracts name + description from the leading --- fenced
// YAML-ish block, plus the 1-based line of the name: field for provenance.
func readSkillFrontmatter(path string) (name, desc string, nameLine int, err error) {
	f, err := os.Open(path)
	if err != nil {
		return "", "", 0, err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	line := 0
	inFrontmatter := false
	for sc.Scan() {
		line++
		text := sc.Text()
		trimmed := strings.TrimSpace(text)
		if line == 1 {
			if trimmed == "---" {
				inFrontmatter = true
				continue
			}
			break // no frontmatter fence
		}
		if inFrontmatter && trimmed == "---" {
			break // end of frontmatter
		}
		if !inFrontmatter {
			continue
		}
		if v, ok := frontmatterValue(text, "name"); ok && name == "" {
			name = v
			nameLine = line
		}
		if v, ok := frontmatterValue(text, "description"); ok && desc == "" {
			desc = v
		}
	}
	if nameLine == 0 {
		nameLine = 1
	}
	return name, desc, nameLine, sc.Err()
}

// frontmatterValue parses a top-level `key: value` line, stripping optional
// surrounding quotes. It ignores indented (nested) keys.
func frontmatterValue(text, key string) (string, bool) {
	if len(text) > 0 && (text[0] == ' ' || text[0] == '\t') {
		return "", false // nested key, not a top-level field
	}
	prefix := key + ":"
	if !strings.HasPrefix(text, prefix) {
		return "", false
	}
	v := strings.TrimSpace(text[len(prefix):])
	v = strings.TrimSuffix(strings.TrimPrefix(v, `"`), `"`)
	v = strings.TrimSuffix(strings.TrimPrefix(v, `'`), `'`)
	return v, true
}
