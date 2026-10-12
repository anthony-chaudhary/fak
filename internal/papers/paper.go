// Package papers fetches research-paper metadata from arXiv and Hugging Face
// into a local JSONL index and searches it with BM25.
package papers

import (
	"regexp"
	"sort"
	"strings"
	"time"
)

const (
	SourceArxiv = "arxiv"
	SourceHF    = "hf"
)

type Paper struct {
	ID         string   `json:"id"`
	Source     string   `json:"source"`
	Title      string   `json:"title"`
	Abstract   string   `json:"abstract,omitempty"`
	Authors    []string `json:"authors,omitempty"`
	Published  string   `json:"published,omitempty"`
	URL        string   `json:"url,omitempty"`
	PDFURL     string   `json:"pdf_url,omitempty"`
	Categories []string `json:"categories,omitempty"`
	Upvotes    int      `json:"upvotes,omitempty"`
	FetchedAt  string   `json:"fetched_at,omitempty"`
}

var versionSuffix = regexp.MustCompile(`v\d+$`)

// BareArxivID strips URL/prefix/version decoration: "http://arxiv.org/abs/2610.13778v1" -> "2610.13778".
func BareArxivID(raw string) string {
	s := strings.TrimSpace(raw)
	for _, marker := range []string{"/abs/", "/pdf/", "/papers/"} {
		if i := strings.LastIndex(s, marker); i >= 0 {
			s = s[i+len(marker):]
		}
	}
	if len(s) >= 6 && strings.EqualFold(s[:6], "arxiv:") {
		s = s[6:]
	}
	s = strings.TrimSuffix(s, ".pdf")
	s = strings.Trim(s, "/ ")
	return versionSuffix.ReplaceAllString(s, "")
}

// CanonicalID is the index key. HF daily papers carry arXiv ids, so both
// sources collapse onto the same record.
func CanonicalID(raw string) string {
	bare := BareArxivID(raw)
	if bare == "" {
		return ""
	}
	return "arxiv:" + bare
}

func AbsURL(id string) string { return "https://arxiv.org/abs/" + BareArxivID(id) }
func PDFURL(id string) string { return "https://arxiv.org/pdf/" + BareArxivID(id) }
func HFURL(id string) string  { return "https://huggingface.co/papers/" + BareArxivID(id) }

func (p Paper) PublishedTime() time.Time {
	return parseTime(p.Published)
}

func parseTime(s string) time.Time {
	if s == "" {
		return time.Time{}
	}
	for _, layout := range []string{time.RFC3339Nano, time.RFC3339, "2006-01-02"} {
		if t, err := time.Parse(layout, s); err == nil {
			return t.UTC()
		}
	}
	return time.Time{}
}

func (p Paper) HasSource(src string) bool {
	for _, s := range strings.Split(p.Source, ",") {
		if s == src {
			return true
		}
	}
	return false
}

// Merge folds b into a (same ID): longest abstract, union of sources and
// categories, max upvotes, earliest publish date, latest fetch time.
func Merge(a, b Paper) Paper {
	out := a
	out.Source = unionCSV(a.Source, b.Source)
	if out.Title == "" || (b.HasSource(SourceArxiv) && !a.HasSource(SourceArxiv) && b.Title != "") {
		out.Title = b.Title
	}
	if len(b.Abstract) > len(out.Abstract) {
		out.Abstract = b.Abstract
	}
	if len(b.Authors) > len(out.Authors) {
		out.Authors = b.Authors
	}
	at, bt := a.PublishedTime(), b.PublishedTime()
	if at.IsZero() || (!bt.IsZero() && bt.Before(at)) {
		if b.Published != "" {
			out.Published = b.Published
		}
	}
	if b.HasSource(SourceArxiv) && b.URL != "" {
		out.URL = b.URL
	} else if out.URL == "" {
		out.URL = b.URL
	}
	if out.PDFURL == "" {
		out.PDFURL = b.PDFURL
	}
	out.Categories = unionSorted(a.Categories, b.Categories)
	if b.Upvotes > out.Upvotes {
		out.Upvotes = b.Upvotes
	}
	if parseTime(b.FetchedAt).After(parseTime(out.FetchedAt)) {
		out.FetchedAt = b.FetchedAt
	}
	return out
}

func unionCSV(a, b string) string {
	var parts []string
	for _, s := range []string{a, b} {
		for _, p := range strings.Split(s, ",") {
			if p != "" {
				parts = append(parts, p)
			}
		}
	}
	return strings.Join(unionSorted(parts, nil), ",")
}

func unionSorted(a, b []string) []string {
	seen := map[string]bool{}
	var out []string
	for _, list := range [][]string{a, b} {
		for _, s := range list {
			if s != "" && !seen[s] {
				seen[s] = true
				out = append(out, s)
			}
		}
	}
	sort.Strings(out)
	return out
}

func collapseSpace(s string) string {
	return strings.Join(strings.Fields(s), " ")
}
