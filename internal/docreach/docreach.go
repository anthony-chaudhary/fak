// Package docreach computes named document-reachability censuses from an
// immutable corpus. Repository I/O belongs at the caller so this package can
// prove resolver semantics without reading peer-dirty working trees.
//
// Invariant: docreach census resolution is fail-closed and deterministic across all executions.
// Guard: ambiguous document basenames or unresolvable relative targets are treated as broken links.
// Precondition: Blobs provided to Census are immutable, and file paths are repository-relative paths.
package docreach

import (
	"path"
	"regexp"
	"sort"
	"strings"
)

// Blob represents an immutable document path and its textual payload.
type Blob struct{ Path, Text string }

// Count captures reachability metrics for a specific named rule, including unreached targets.
type Count struct {
	Rule        string   `json:"rule"`
	Numerator   int      `json:"numerator"`
	Denominator int      `json:"denominator"`
	Unreached   []string `json:"unreached"`
}

// BrokenLink records an unresolvable or ambiguous document reference from source to target.
type BrokenLink struct{ Source, Target string }

// Report aggregates census counts across rules and all identified broken references.
type Report struct {
	Commit      string       `json:"commit"`
	Documents   int          `json:"documents"`
	Rules       []Count      `json:"rules"`
	BrokenLinks []BrokenLink `json:"broken_links"`
}

var linkRE = regexp.MustCompile(`\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)`)
var mentionRE = regexp.MustCompile(`(?:[A-Za-z0-9_.-]+/)+(?:[A-Za-z0-9_.-]+\.md)|[A-Za-z0-9_.-]+\.md`)

// Census applies R-LINK, R-MENTION, and their union. Every count carries the
// complete tracked-Markdown domain; broken link targets are kept separate.
func Census(commit string, blobs []Blob) Report {
	docs := map[string]bool{}
	text := map[string]string{}
	for _, b := range blobs {
		p := clean(b.Path)
		if strings.HasSuffix(strings.ToLower(p), ".md") {
			docs[p] = true
			text[p] = b.Text
		}
	}
	link, mention := map[string]bool{}, map[string]bool{}
	var broken []BrokenLink
	for src, body := range text {
		for _, m := range linkRE.FindAllStringSubmatch(body, -1) {
			target := resolve(src, m[1], docs)
			if target == "" {
				broken = append(broken, BrokenLink{src, m[1]})
			} else if target != src {
				link[target] = true
			}
		}
		for _, m := range mentionRE.FindAllString(body, -1) {
			if target := resolve(src, m, docs); target != "" && target != src {
				mention[target] = true
			}
		}
	}
	union := map[string]bool{}
	for p := range link {
		union[p] = true
	}
	for p := range mention {
		union[p] = true
	}
	sort.Slice(broken, func(i, j int) bool {
		if broken[i].Source == broken[j].Source {
			return broken[i].Target < broken[j].Target
		}
		return broken[i].Source < broken[j].Source
	})
	return Report{Commit: commit, Documents: len(docs), Rules: []Count{count("R-LINK", docs, link), count("R-MENTION", docs, mention), count("R-UNION", docs, union)}, BrokenLinks: broken}
}
func clean(p string) string {
	return strings.TrimPrefix(path.Clean(strings.ReplaceAll(p, "\\", "/")), "./")
}
func resolve(src, raw string, docs map[string]bool) string {
	raw = clean(strings.TrimSpace(raw))
	if raw == "" || strings.Contains(raw, "://") {
		return ""
	}
	candidates := []string{clean(path.Join(path.Dir(src), raw)), raw}
	// Ancestor re-anchoring, followed by a uniqueness-gated basename match.
	for d := path.Dir(src); d != "." && d != "/"; d = path.Dir(d) {
		candidates = append(candidates, clean(path.Join(path.Dir(d), raw)))
	}
	seen := map[string]bool{}
	for _, c := range candidates {
		if !seen[c] && docs[c] {
			return c
		}
		seen[c] = true
	}
	base := path.Base(raw)
	var hit string
	for p := range docs {
		if path.Base(p) == base {
			if hit != "" {
				return ""
			}
			hit = p
		}
	}
	return hit
}
func count(rule string, docs, hit map[string]bool) Count {
	u := []string{}
	for p := range docs {
		if !hit[p] {
			u = append(u, p)
		}
	}
	sort.Strings(u)
	return Count{rule, len(docs) - len(u), len(docs), u}
}
