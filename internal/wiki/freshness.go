package wiki

import (
	"bytes"
	"sort"
	"strings"
)

// This is L4 (#4281): a generated page pins the commit SHA it was built at and the
// set of source files it cites; it is stale the moment any cited file moves —
// freshness witnessed against git *now*, not on a schedule (DeepWiki's anti-pattern:
// snapshot pages regenerated on a timer, silently lagging main "by hours to days",
// `page.tsx:1709,1905@16f35a0`). Sibling of #4077 (the same generated_at drift idea,
// a different artifact). The kernel here is PURE — it owns no git, no clock: the CLI
// gathers "files changed since <sha>" and hands the set in, exactly the way
// devindex.ReferenceIndex takes a pre-gathered graph.

// PageMeta is the freshness contract a generated wiki page carries in its leading
// YAML frontmatter block:
//
//	---
//	generated_at_sha: 1a2b3c4
//	cited_files:
//	  - internal/gateway/gateway.go
//	  - internal/gateway/admit.go
//	---
//
// GeneratedAtSHA is the commit the page's prose was written against; CitedFiles is
// the exact source set it cites. HasFrontmatter distinguishes "no frontmatter at
// all" (an un-pinned page) from "frontmatter present but empty".
type PageMeta struct {
	GeneratedAtSHA string   `json:"generated_at_sha,omitempty"`
	CitedFiles     []string `json:"cited_files,omitempty"`
	HasFrontmatter bool     `json:"has_frontmatter"`
}

// StaleReason is the closed set of ways a page fails the freshness witness.
type StaleReason string

const (
	// ReasonNoSHA: the page carries no generated_at_sha, so its freshness cannot be
	// witnessed at all — a generated page that does not pin its build commit is
	// treated as stale by construction (it can never be proven current).
	ReasonNoSHA StaleReason = "no-generated-at-sha"
	// ReasonCitedCodeMoved: at least one cited file changed between the page's
	// generated_at_sha and now — the prose may describe code that no longer exists.
	ReasonCitedCodeMoved StaleReason = "cited-code-moved"
)

// StalePage is one freshness finding: the page, the SHA it pinned, why it is stale,
// and (for a code-moved failure) exactly which cited files drifted. It is the
// DriftStaleWikiPage analogue of a Dangler — enough to point a reader or a CI gate
// at the precise offending page.
type StalePage struct {
	Path    string      `json:"path"`
	SHA     string      `json:"generated_at_sha,omitempty"`
	Reason  StaleReason `json:"reason"`
	Touched []string    `json:"touched,omitempty"`
}

// DriftStaleWikiPage is the pure freshness kernel. Given a page's parsed meta and
// the set of repo-relative files that changed since the page's generated_at_sha
// (gathered by the caller via git), it reports whether the page is stale and, if so,
// why. It reads no files and runs no git: same (meta, changedSince) in → same
// verdict out.
//
//   - No generated_at_sha           → stale, ReasonNoSHA (cannot be witnessed).
//   - A cited file is in changedSince → stale, ReasonCitedCodeMoved + the touched set.
//   - Otherwise                     → fresh.
//
// A page with a SHA but no cited_files is vacuously fresh: it pins a build commit
// and cites nothing whose drift could invalidate it.
func DriftStaleWikiPage(path string, meta PageMeta, changedSince []string) (StalePage, bool) {
	if strings.TrimSpace(meta.GeneratedAtSHA) == "" {
		return StalePage{Path: path, Reason: ReasonNoSHA}, true
	}
	changed := make(map[string]bool, len(changedSince))
	for _, f := range changedSince {
		if f = normalizeRel(f); f != "" {
			changed[f] = true
		}
	}
	var touched []string
	for _, c := range meta.CitedFiles {
		if changed[normalizeRel(c)] {
			touched = append(touched, c)
		}
	}
	if len(touched) > 0 {
		sort.Strings(touched)
		return StalePage{
			Path:    path,
			SHA:     meta.GeneratedAtSHA,
			Reason:  ReasonCitedCodeMoved,
			Touched: touched,
		}, true
	}
	return StalePage{}, false
}

// ParseFrontmatter extracts the leading `---`-fenced YAML frontmatter block and
// returns the two keys the freshness contract needs. It is a deliberately tiny,
// dependency-free scanner (not a general YAML parser): it recognises `key: value`
// scalars and a `key:` followed by `  - item` list lines, which is all a generated
// page's frontmatter uses. A page with no leading `---` fence returns a zero
// PageMeta with HasFrontmatter=false.
func ParseFrontmatter(md []byte) PageMeta {
	lines := bytes.Split(md, []byte("\n"))
	// The fence must be the very first non-empty line.
	i := 0
	for i < len(lines) && len(bytes.TrimSpace(lines[i])) == 0 {
		i++
	}
	if i >= len(lines) || strings.TrimSpace(string(lines[i])) != "---" {
		return PageMeta{}
	}
	i++
	meta := PageMeta{HasFrontmatter: true}
	curKey := ""
	for ; i < len(lines); i++ {
		line := string(lines[i])
		if strings.TrimSpace(line) == "---" {
			break // end of frontmatter
		}
		// A `  - item` continuation of the current list key.
		if item, ok := listItem(line); ok {
			if curKey == "cited_files" && item != "" {
				meta.CitedFiles = append(meta.CitedFiles, item)
			}
			continue
		}
		key, val, ok := scalar(line)
		if !ok {
			continue
		}
		curKey = key
		switch key {
		case "generated_at_sha":
			meta.GeneratedAtSHA = strings.Trim(strings.TrimSpace(val), `"'`)
		case "cited_files":
			// Inline form `cited_files: [a, b]` or a following `- ` list.
			for _, f := range inlineList(val) {
				meta.CitedFiles = append(meta.CitedFiles, f)
			}
		}
	}
	return meta
}

// scalar splits a `key: value` frontmatter line. It returns ok=false for a list
// continuation, a comment, or a blank line.
func scalar(line string) (key, val string, ok bool) {
	t := strings.TrimSpace(line)
	if t == "" || strings.HasPrefix(t, "#") || strings.HasPrefix(t, "- ") {
		return "", "", false
	}
	colon := strings.Index(t, ":")
	if colon < 0 {
		return "", "", false
	}
	return strings.TrimSpace(t[:colon]), strings.TrimSpace(t[colon+1:]), true
}

// listItem recognises a `  - value` YAML list entry, returning the unquoted value.
func listItem(line string) (string, bool) {
	t := strings.TrimSpace(line)
	if !strings.HasPrefix(t, "- ") {
		return "", false
	}
	return strings.Trim(strings.TrimSpace(t[2:]), `"'`), true
}

// inlineList parses the `[a, b, c]` inline-array form of a frontmatter value. A
// non-bracketed value yields no items (the list is expected as following `- ` lines).
func inlineList(val string) []string {
	v := strings.TrimSpace(val)
	if !strings.HasPrefix(v, "[") || !strings.HasSuffix(v, "]") {
		return nil
	}
	inner := strings.TrimSpace(v[1 : len(v)-1])
	if inner == "" {
		return nil
	}
	var out []string
	for _, part := range strings.Split(inner, ",") {
		if p := strings.Trim(strings.TrimSpace(part), `"'`); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// normalizeRel canonicalises a repo-relative path for set comparison: forward
// slashes, no leading "./". It mirrors the path spelling git diff --name-only emits
// so a cited file and a changed file compare equal.
func normalizeRel(p string) string {
	p = strings.TrimSpace(strings.ReplaceAll(p, "\\", "/"))
	return strings.TrimPrefix(p, "./")
}
