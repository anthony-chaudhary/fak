// Package memoryread renders the committed fleet memory mirror as a bounded digest.
package memoryread

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const StoreRel = ".claude/memory"

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

// RenderDigest renders MEMORY.md plus linked fact bodies, bounded by maxBytes.
func RenderDigest(storeDir string, indexOnly bool, maxBytes int) string {
	indexPath := filepath.Join(storeDir, "MEMORY.md")
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
	emitted, omitted := 0, 0
	for _, fact := range ParseIndex(indexText) {
		title, fname := fact[0], fact[1]
		bodyBytes, err := os.ReadFile(filepath.Join(storeDir, fname))
		if err != nil {
			omitted++
			continue
		}
		body := strings.TrimRight(StripFrontmatter(string(bodyBytes)), "\n")
		block := fmt.Sprintf("## %s (%s)\n\n%s\n", title, fname, body)
		if budget-len(block) < 0 && emitted > 0 {
			omitted++
			continue
		}
		parts = append(parts, block)
		budget -= len(block)
		emitted++
	}
	if omitted > 0 {
		parts = append(parts, fmt.Sprintf("...(%d more fact file(s) omitted - read directly from %s/ if needed)", omitted, StoreRel))
	}
	return strings.TrimRight(strings.Join(parts, "\n"), "\n") + "\n"
}
