package studymonitor

import (
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// maxSubmoduleNestingDepth bounds how deep .gitmodules files inside submodules
// are followed, so a cyclic or adversarial checkout cannot recurse forever.
const maxSubmoduleNestingDepth = 8

var (
	// ErrExcludedSubmoduleUndeclared refuses an exclusion naming a path that no
	// .gitmodules file in the checkout declares.
	ErrExcludedSubmoduleUndeclared = errors.New("excluded submodule is not declared in .gitmodules")
	// ErrExcludedSubmoduleReason refuses an exclusion without a recorded reason.
	ErrExcludedSubmoduleReason = errors.New("excluded submodule needs a reason")
	// ErrExcludedSubmodulePath refuses an exclusion path that is not relative to
	// the checkout root.
	ErrExcludedSubmodulePath = errors.New("excluded submodule path must be relative inside the checkout")
	// ErrExcludedSubmodulePopulated refuses excluding a submodule that is checked
	// out: its source is walkable, so excluding it would hide kernels the walk
	// can see.
	ErrExcludedSubmodulePopulated = errors.New("excluded submodule is checked out; walk it instead of excluding it")
)

// InventoryExcludedSubmodule records a declared submodule the study chose not to
// walk, with the reason that keeps it from blocking readiness.
type InventoryExcludedSubmodule struct {
	Path   string `json:"path"`
	Reason string `json:"reason"`
}

// readGitSubmodulePaths returns every submodule path declared by root/.gitmodules
// and, recursively, by the .gitmodules at each declared submodule root (paths
// prefixed by their parent submodule). Submodules are pinned first-class
// sources, so they are walked even under a directory name that is otherwise
// skipped (vendor/). Excluded paths are declared but never descended into.
func readGitSubmodulePaths(root string, excluded map[string]string) ([]string, error) {
	seen := map[string]bool{}
	var visit func(prefix string, depth int) error
	visit = func(prefix string, depth int) error {
		dir := root
		if prefix != "" {
			dir = filepath.Join(root, filepath.FromSlash(prefix))
			info, err := os.Lstat(dir)
			if err != nil || !info.IsDir() {
				return nil
			}
		}
		data, err := os.ReadFile(filepath.Join(dir, ".gitmodules"))
		if err != nil {
			if errors.Is(err, fs.ErrNotExist) {
				return nil
			}
			return err
		}
		for _, rel := range parseGitmodulesPaths(data) {
			full := rel
			if prefix != "" {
				full = prefix + "/" + rel
			}
			if seen[full] {
				continue
			}
			seen[full] = true
			if _, skip := excluded[full]; skip || depth+1 >= maxSubmoduleNestingDepth {
				continue
			}
			if err := visit(full, depth+1); err != nil {
				return err
			}
		}
		return nil
	}
	if err := visit("", 0); err != nil {
		return nil, err
	}
	return sortedMapKeys(seen), nil
}

// parseGitmodulesPaths reads the `path` keys of [submodule] sections using
// git-config syntax: case-insensitive keys, quoted values with \" \\ \t \n
// escapes, and # or ; comments outside quotes. Paths that are absolute, carry a
// drive letter, or escape the root are dropped.
func parseGitmodulesPaths(data []byte) []string {
	var out []string
	section := ""
	for _, raw := range strings.Split(string(data), "\n") {
		line := strings.TrimSpace(raw)
		if line == "" || line[0] == '#' || line[0] == ';' {
			continue
		}
		if line[0] == '[' {
			name, rest, ok := parseGitConfigSectionHeader(line)
			if !ok {
				section = ""
				continue
			}
			section = name
			line = strings.TrimSpace(rest)
			if line == "" || line[0] == '#' || line[0] == ';' {
				continue
			}
		}
		if section != "submodule" {
			continue
		}
		key, rawValue, hasValue := strings.Cut(line, "=")
		if !hasValue || !strings.EqualFold(strings.TrimSpace(key), "path") {
			continue
		}
		value, ok := parseGitConfigValue(rawValue)
		if !ok {
			continue
		}
		if cleaned, ok := cleanSubmodulePath(value); ok {
			out = append(out, cleaned)
		}
	}
	return out
}

// parseGitConfigSectionHeader returns the lower-cased section name of a
// `[section "subsection"]` or deprecated `[section.subsection]` header and the
// text after the closing bracket.
func parseGitConfigSectionHeader(line string) (string, string, bool) {
	inQuote := false
	for i := 1; i < len(line); i++ {
		switch c := line[i]; {
		case c == '\\' && inQuote:
			i++
		case c == '"':
			inQuote = !inQuote
		case c == ']' && !inQuote:
			name := strings.TrimSpace(line[1:i])
			if j := strings.IndexAny(name, " \t\""); j >= 0 {
				name = name[:j]
			}
			if j := strings.IndexByte(name, '.'); j >= 0 {
				name = name[:j]
			}
			return strings.ToLower(name), line[i+1:], true
		}
	}
	return "", "", false
}

func parseGitConfigValue(raw string) (string, bool) {
	var b strings.Builder
	inQuote, started := false, false
	keep := 0
scan:
	for i := 0; i < len(raw); i++ {
		c := raw[i]
		switch {
		case c == '\\':
			if i+1 >= len(raw) {
				return "", false
			}
			i++
			switch raw[i] {
			case '"', '\\':
				b.WriteByte(raw[i])
			case 't':
				b.WriteByte('\t')
			case 'n':
				b.WriteByte('\n')
			default:
				return "", false
			}
			started, keep = true, b.Len()
		case c == '"':
			inQuote = !inQuote
			started, keep = true, b.Len()
		case !inQuote && (c == '#' || c == ';'):
			break scan
		case !inQuote && (c == ' ' || c == '\t' || c == '\r'):
			if started {
				b.WriteByte(c)
			}
		default:
			b.WriteByte(c)
			started, keep = true, b.Len()
		}
	}
	if inQuote {
		return "", false
	}
	return b.String()[:keep], true
}

// cleanSubmodulePath normalizes a submodule path to a slash-separated path
// relative to the checkout root, refusing absolute, drive-letter, and escaping
// paths.
func cleanSubmodulePath(raw string) (string, bool) {
	p := strings.TrimSpace(strings.ReplaceAll(raw, "\\", "/"))
	if p == "" || strings.HasPrefix(p, "/") || hasDriveLetter(p) {
		return "", false
	}
	p = path.Clean(p)
	if p == "." || p == ".." || strings.HasPrefix(p, "../") {
		return "", false
	}
	return p, true
}

func hasDriveLetter(p string) bool {
	if len(p) < 2 || p[1] != ':' {
		return false
	}
	c := p[0] | 0x20
	return c >= 'a' && c <= 'z'
}

// submoduleCheckedOut reports a submodule directory holding anything besides
// its .git link. A missing path or a non-directory holds nothing walkable.
func submoduleCheckedOut(root, rel string) (bool, error) {
	dir := filepath.Join(root, filepath.FromSlash(rel))
	info, err := os.Lstat(dir)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return false, nil
		}
		return false, err
	}
	if !info.IsDir() {
		return false, nil
	}
	entries, err := os.ReadDir(dir)
	if err != nil {
		return false, err
	}
	for _, entry := range entries {
		if entry.Name() != ".git" {
			return true, nil
		}
	}
	return false, nil
}

// normalizeExcludedSubmodules cleans exclusion paths and refuses empty reasons.
func normalizeExcludedSubmodules(in map[string]string) (map[string]string, error) {
	out := make(map[string]string, len(in))
	raws := make([]string, 0, len(in))
	for raw := range in {
		raws = append(raws, raw)
	}
	sort.Strings(raws)
	for _, raw := range raws {
		cleaned, ok := cleanSubmodulePath(raw)
		if _, dup := out[cleaned]; !ok || dup {
			return nil, fmt.Errorf("%w: %q", ErrExcludedSubmodulePath, raw)
		}
		reason := strings.TrimSpace(in[raw])
		if reason == "" {
			return nil, fmt.Errorf("%w: %s", ErrExcludedSubmoduleReason, cleaned)
		}
		out[cleaned] = reason
	}
	return out, nil
}

// inventoryWalkPlan decides which directories the inventory walk enters. A
// skip-named directory (vendor/, build/) kept only because a submodule lies
// beneath it is pass-through: below it the walk enters only directories on a
// submodule path and counts only files inside a submodule.
type inventoryWalkPlan struct {
	walkedSubmodules []string
	excluded         map[string]string
	keepDirs         map[string]bool
	passThrough      map[string]bool
}

func newInventoryWalkPlan(declared []string, excluded map[string]string) inventoryWalkPlan {
	plan := inventoryWalkPlan{excluded: excluded, keepDirs: map[string]bool{}, passThrough: map[string]bool{}}
	for _, submodule := range declared {
		if _, skip := excluded[submodule]; skip || underAnyPath(submodule, excluded) {
			continue
		}
		plan.walkedSubmodules = append(plan.walkedSubmodules, submodule)
		parts := strings.Split(submodule, "/")
		for i := range parts {
			plan.keepDirs[strings.Join(parts[:i+1], "/")] = true
		}
	}
	return plan
}

func underAnyPath(rel string, prefixes map[string]string) bool {
	for prefix := range prefixes {
		if strings.HasPrefix(rel, prefix+"/") {
			return true
		}
	}
	return false
}

func (plan inventoryWalkPlan) submoduleRoot(rel string) bool {
	for _, submodule := range plan.walkedSubmodules {
		if rel == submodule {
			return true
		}
	}
	return false
}

func (plan inventoryWalkPlan) excludedDir(rel string) bool {
	_, excluded := plan.excluded[rel]
	return excluded
}

// enterDir reports whether the walk descends into directory rel (a refusal is
// a skipped directory) and records whether rel is pass-through. A submodule
// root is never pass-through, so inside it the normal skip rules apply again; a
// skip-named directory inside a submodule that leads to a nested submodule is
// pass-through in turn.
func (plan inventoryWalkPlan) enterDir(rel, name string) bool {
	parentPass := plan.passThrough[path.Dir(rel)]
	skipNamed := shouldSkipInventoryDir(name)
	if (parentPass || skipNamed) && !plan.keepDirs[rel] {
		return false
	}
	if (parentPass || skipNamed) && !plan.submoduleRoot(rel) {
		plan.passThrough[rel] = true
	}
	return true
}

// fileInPassThrough reports a file sitting directly in a pass-through
// directory: it is not inside a submodule and is not counted.
func (plan inventoryWalkPlan) fileInPassThrough(rel string) bool {
	return plan.passThrough[path.Dir(rel)]
}
