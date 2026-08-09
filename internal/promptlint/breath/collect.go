package breath

import (
	"fmt"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// InScope reports whether a repo-relative path is a page under contract: a `.md` file
// under one of the contract roots.
//
// Matching is on a path PREFIX boundary, not strings.HasPrefix, so a root of
// "docs/explainers" does not silently swallow "docs/explainers-archive/x.md".
func (c Contract) InScope(rel string) bool {
	rel = path.Clean(filepath.ToSlash(rel))
	if strings.ToLower(path.Ext(rel)) != ".md" {
		return false
	}
	for _, r := range c.Roots {
		r = path.Clean(filepath.ToSlash(r))
		if r == "." || r == "" {
			return true
		}
		if rel == r || strings.HasPrefix(rel, r+"/") {
			return true
		}
	}
	return false
}

// Collect loads the pages under contract from a caller-supplied tracked-path list.
//
// The path list is the caller's business — `fak breath` passes `git ls-files`, because the
// index is the one view of the tree two agents on the same commit agree about, so a peer's
// unstaged new page cannot turn this gate red in someone else's session. Keeping the git
// call at the composition root also keeps this leaf free of process spawning.
//
// A path in the list whose file cannot be read is a HARD ERROR naming the path, never a
// skipped entry: an unreadable page is not a compliant one, and skipping it would let a
// deleted-but-still-indexed page pass where a malformed one fails.
func Collect(root string, rels []string) ([]Doc, error) {
	seen := map[string]bool{}
	var out []Doc
	for _, rel := range rels {
		rel = path.Clean(filepath.ToSlash(strings.TrimSpace(rel)))
		if rel == "" || rel == "." || seen[rel] {
			continue
		}
		seen[rel] = true
		b, err := os.ReadFile(filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			return nil, fmt.Errorf("breath: %s is listed but could not be read (%w) — an unreadable "+
				"page is not a compliant one; check whether it was deleted without being staged, or "+
				"whether the list is being read from a different root than the files", rel, err)
		}
		out = append(out, Doc{Path: rel, Body: b})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Path < out[j].Path })
	return out, nil
}

// Filter keeps only the paths under contract, preserving order.
func (c Contract) Filter(rels []string) []string {
	var out []string
	for _, r := range rels {
		if c.InScope(r) {
			out = append(out, path.Clean(filepath.ToSlash(r)))
		}
	}
	return out
}
