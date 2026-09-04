package managedocs

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// PageLines is the deterministic source-page budget. Three 50-line rendered
// pages is deliberately approximate; the fixed source measure keeps CI stable.
const PageLines = 150

// DocumentSetMarker declares that an oversized landing page indexes bounded
// child pages rather than remaining a monolith.
const DocumentSetMarker = "<!-- fak:document-set -->"

// SizeFinding records a Markdown file exceeding the permitted source line budget.
type SizeFinding struct {
	Path  string
	Lines int
}

// AuditDocumentSets rejects oversized Markdown unless it is a compact document
// set index. Child pages remain subject to the same budget.
func AuditDocumentSets(repoRoot string, roots ...string) error {
	if len(roots) == 0 {
		roots = []string{"README.md", "docs", "examples", "CLAIMS.md", "STATUS.md", "ARCHITECTURE.md"}
	}
	var findings []SizeFinding
	for _, rel := range roots {
		path := filepath.Join(repoRoot, filepath.FromSlash(rel))
		info, err := os.Stat(path)
		if os.IsNotExist(err) {
			continue
		}
		if err != nil {
			return err
		}
		visit := func(path string, info os.FileInfo) error {
			if info.IsDir() || strings.ToLower(filepath.Ext(path)) != ".md" {
				return nil
			}
			f, e := os.Open(path)
			if e != nil {
				return e
			}
			defer f.Close()
			s := bufio.NewScanner(f)
			lines := 0
			marker := false
			for s.Scan() {
				lines++
				if strings.TrimSpace(s.Text()) == DocumentSetMarker {
					marker = true
				}
			}
			if e = s.Err(); e != nil {
				return e
			}
			if lines > PageLines && !marker {
				r, _ := filepath.Rel(repoRoot, path)
				findings = append(findings, SizeFinding{filepath.ToSlash(r), lines})
			}
			return nil
		}
		if info.IsDir() {
			err = filepath.Walk(path, func(p string, i os.FileInfo, e error) error {
				if e != nil {
					return e
				}
				if strings.Contains(filepath.ToSlash(p), "/completed/") {
					if i.IsDir() {
						return filepath.SkipDir
					}
					return nil
				}
				return visit(p, i)
			})
		} else {
			err = visit(path, info)
		}
		if err != nil {
			return err
		}
	}
	if len(findings) == 0 {
		return nil
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].Path < findings[j].Path })
	var b strings.Builder
	b.WriteString("oversized Markdown must be split into an indexed document set (>150 lines):")
	for _, f := range findings {
		fmt.Fprintf(&b, "\n%s: %d lines", f.Path, f.Lines)
	}
	return fmt.Errorf("%s", b.String())
}
