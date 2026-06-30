package devindex

// work.go answers "what should I work on?" as a QUERY instead of a survey — the
// orient/plan-stage half of the self-index (epic #1287; the work-view #1291 that
// WORK-MAP names as its #1 drift: "ongoing work has no single live view"). It reads
// .github/issue-views.json — the curated, API-readable mirror of the GitHub saved
// views (which hydrate client-side and are NOT reachable through gh/REST/GraphQL) —
// so an agent landing cold gets the DEFAULT selection surface ("ready-leaves") and
// every named view's ready-to-run `gh issue list --search` query in one call, rather
// than re-reading the JSON or guessing the issue-search syntax. Like the rest of
// devindex it is a VIEW: it reads the file that owns the fact, never a cached copy,
// and never reaches the network itself.

import (
	"encoding/json"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// IssueView is one named issue-search view from .github/issue-views.json. Query is
// GitHub issue-search syntax fed verbatim to `gh issue list --search`; Note explains
// when to reach for it.
type IssueView struct {
	Slug  string `json:"slug"`
	Title string `json:"title"`
	Query string `json:"query"`
	Note  string `json:"note,omitempty"`
}

// IssueViews is the parsed selection surface: the Default view slug, the gh page
// Limit every query should be paired with (gh defaults to 30, so the file pins an
// explicit cap), and the named Views in file order (the default view sits first by
// convention).
type IssueViews struct {
	Default string      `json:"default"`
	Limit   int         `json:"limit"`
	Views   []IssueView `json:"views"`
}

// issueViewsPath is the in-repo selection surface, relative to the repo root.
var issueViewsPath = filepath.FromSlash(".github/issue-views.json")

// IssueViews reads .github/issue-views.json from the catalog root. A missing or
// unparseable file is returned as an error the caller surfaces — unlike the doc map
// (which degrades to empty), the selection surface has no taxonomy to invent, and an
// absent file is a real "this repo declares no views" answer, not a silent empty.
func (c *Catalog) IssueViews() (IssueViews, error) {
	var v IssueViews
	b, err := os.ReadFile(filepath.Join(c.Root, issueViewsPath))
	if err != nil {
		return v, err
	}
	if err := json.Unmarshal(b, &v); err != nil {
		return v, err
	}
	return v, nil
}

// PageLimit returns the gh page --limit to pair every query with: the file's declared
// Limit, or a safe default when it declares none (gh's own default of 30 silently
// truncates a real backlog).
func (v IssueViews) PageLimit() int {
	if v.Limit > 0 {
		return v.Limit
	}
	return 100
}

// DefaultView returns the view named by the Default slug, or false when the file
// declares no default or the slug names no view.
func (v IssueViews) DefaultView() (IssueView, bool) {
	for _, iv := range v.Views {
		if iv.Slug == v.Default {
			return iv, true
		}
	}
	return IssueView{}, false
}

// SearchViews returns the views matching the query — by slug (an exact slug match
// dominates), title, or note — best-first. An empty query returns every view in file
// order (the default view first by convention), mirroring `fak index leaf`/`verbs`.
func (v IssueViews) SearchViews(query string) []IssueView {
	toks := tokens(query)
	if len(toks) == 0 {
		out := make([]IssueView, len(v.Views))
		copy(out, v.Views)
		return out
	}
	type scored struct {
		iv IssueView
		s  int
	}
	var hits []scored
	for _, iv := range v.Views {
		slug, title, note := strings.ToLower(iv.Slug), strings.ToLower(iv.Title), strings.ToLower(iv.Note)
		score := 0
		for _, tk := range toks {
			if slug == tk {
				score += 10 // an exact slug match dominates a mere substring hit
			}
			if strings.Contains(slug, tk) {
				score += 3
			}
			if strings.Contains(title, tk) {
				score += 2
			}
			if strings.Contains(note, tk) {
				score++
			}
		}
		if score > 0 {
			hits = append(hits, scored{iv, score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].s != hits[j].s {
			return hits[i].s > hits[j].s
		}
		return hits[i].iv.Slug < hits[j].iv.Slug
	})
	out := make([]IssueView, len(hits))
	for i, h := range hits {
		out[i] = h.iv
	}
	return out
}
