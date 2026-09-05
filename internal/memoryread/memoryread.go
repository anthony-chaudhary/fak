// Package memoryread renders the committed fleet memory mirror as a bounded digest.
package memoryread

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	StoreRel    = ".claude/memory"
	FakStoreRel = ".fak/memory"
	MemoryFile  = "MEMORY.md"
)

var (
	linkRE  = regexp.MustCompile(`\[([^\]]+)\]\(([^)#\s]+\.md)(?:#[^)]*)?\)`)
	nonFact = map[string]bool{"MEMORY.md": true, "MEMORY_archive.md": true, "README.md": true}
)

// ParseIndex extracts (title, filename) pairs for same-directory fact files linked by MEMORY.md.
func ParseIndex(indexText string) [][2]string {
	var out [][2]string
	seen := map[string]bool{}
	for _, m := range linkRE.FindAllStringSubmatch(indexText, -1) {
		title, fname := strings.TrimSpace(m[1]), m[2]
		if strings.ContainsAny(fname, `/\`) || nonFact[fname] || seen[fname] {
			continue
		}
		seen[fname] = true
		out = append(out, [2]string{title, fname})
	}
	return out
}

// StripFrontmatter removes a leading YAML frontmatter block.
func StripFrontmatter(text string) string {
	if !strings.HasPrefix(text, "---") {
		return text
	}
	end := strings.Index(text[3:], "\n---")
	if end == -1 {
		return text
	}
	end += 3
	nl := strings.Index(text[end+1:], "\n")
	if nl == -1 {
		return text
	}
	return strings.TrimLeft(text[end+1+nl+1:], "\n")
}

// DefaultStore resolves the committed memory mirror below root.
func DefaultStore(root string) string {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	return filepath.Join(root, filepath.FromSlash(StoreRel))
}

// DiscoverStore looks for default memory stores in the workspace root in priority order:
// 1. .fak/memory (with MEMORY.md)
// 2. .claude/memory (with MEMORY.md)
// 3. MEMORY.md (in workspace root)
// 4. .fak/memory (directory)
// 5. .claude/memory (directory)
// It returns the resolved path, or "" if no candidate exists.
func DiscoverStore(root string) string {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	fakPath := filepath.Join(root, filepath.FromSlash(FakStoreRel))
	fakMem := filepath.Join(fakPath, MemoryFile)
	if fi, err := os.Stat(fakMem); err == nil && !fi.IsDir() {
		return fakPath
	}

	claudePath := filepath.Join(root, filepath.FromSlash(StoreRel))
	claudeMem := filepath.Join(claudePath, MemoryFile)
	if fi, err := os.Stat(claudeMem); err == nil && !fi.IsDir() {
		return claudePath
	}

	rootMem := filepath.Join(root, MemoryFile)
	if fi, err := os.Stat(rootMem); err == nil && !fi.IsDir() {
		return rootMem
	}

	if fi, err := os.Stat(fakPath); err == nil && fi.IsDir() {
		return fakPath
	}

	if fi, err := os.Stat(claudePath); err == nil && fi.IsDir() {
		return claudePath
	}

	return ""
}

// ResolveStore resolves the effective memory store given an optional explicit path
// and workspace root. If explicit is non-empty, it is expanded and returned (or resolved
// against root if relative). If explicit is empty, DiscoverStore(root) is used.
func ResolveStore(root, explicit string) string {
	if strings.TrimSpace(root) == "" {
		root = "."
	}
	explicit = strings.TrimSpace(explicit)
	if explicit != "" {
		exp := expandTilde(explicit)
		if !filepath.IsAbs(exp) {
			exp = filepath.Join(root, exp)
		}
		return exp
	}
	return DiscoverStore(root)
}

func expandTilde(p string) string {
	if p == "~" || strings.HasPrefix(p, "~/") || strings.HasPrefix(p, `~\`) {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[1:])
		}
	}
	return p
}

// RenderDigest renders MEMORY.md plus linked fact bodies, bounded by maxBytes.
func RenderDigest(storeDir string, indexOnly bool, maxBytes int) string {
	dir := storeDir
	indexPath := filepath.Join(dir, MemoryFile)
	if fi, err := os.Stat(dir); err == nil && !fi.IsDir() {
		indexPath = dir
		dir = filepath.Dir(dir)
	}
	indexBytes, err := os.ReadFile(indexPath)
	if err != nil {
		return fmt.Sprintf("(no committed memory mirror at %s - fresh node or scrubbed clone; nothing to orient from)\n", filepath.ToSlash(storeDir))
	}
	indexText := string(indexBytes)
	parts := []string{
		"# Fleet memory (committed mirror: " + StoreRel + ") - read-only orientation",
		"",
		strings.TrimRight(indexText, "\n"),
	}
	if indexOnly {
		parts = append(parts, "")
		return strings.Join(parts, "\n") + "\n"
	}

	parts = append(parts, "", "---", "")
	budget := maxBytes
	emitted, unreadable := 0, 0
	var overflow []string // fact files dropped for BUDGET — named, never an anonymous count (#2430)
	for _, fact := range ParseIndex(indexText) {
		title, fname := fact[0], fact[1]
		bodyBytes, err := os.ReadFile(filepath.Join(dir, fname))
		if err != nil {
			unreadable++
			continue
		}
		body := strings.TrimRight(StripFrontmatter(string(bodyBytes)), "\n")
		block := fmt.Sprintf("## %s (%s)\n\n%s\n", title, fname, body)
		if budget-len(block) < 0 && emitted > 0 {
			overflow = append(overflow, fmt.Sprintf("%s (%s)", title, fname))
			continue
		}
		parts = append(parts, block)
		budget -= len(block)
		emitted++
	}
	// An over-budget index is a TYPED, NAMED event, not a silent tail-drop: name every
	// fact that fell past the line so the reader knows exactly what it did not get and
	// can page it in directly. The list is the compaction work-list (#2430). Emitted
	// only when something overflowed, so an in-budget index stays advisory-free.
	if len(overflow) > 0 {
		parts = append(parts, fmt.Sprintf("%s: %d fact file(s) past the %d-byte budget: %s - read them directly from %s/",
			OverflowReason, len(overflow), maxBytes, strings.Join(overflow, ", "), StoreRel))
	}
	if unreadable > 0 {
		parts = append(parts, fmt.Sprintf("...(%d fact file(s) unreadable and skipped)", unreadable))
	}
	return strings.TrimRight(strings.Join(parts, "\n"), "\n") + "\n"
}

// OverflowReason is the closed-vocabulary label RenderDigest stamps on an over-budget
// index advisory (#2430). It mirrors memq.OverflowReason by value; the two are kept as
// separate string literals ON PURPOSE because internal/memq imports internal/memoryread
// (notesbackend), so memoryread cannot import memq without a cycle. A test pins them
// equal so the wire label never drifts.
const OverflowReason = "MEMORY_INDEX_OVERFLOW"
