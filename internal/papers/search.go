package papers

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"time"
	"unicode"
)

const (
	bm25K1      = 1.2
	bm25B       = 0.75
	titleRepeat = 2
)

var stopwords = map[string]bool{}

func init() {
	for _, w := range strings.Fields(`a an and are as at be by for from has have in into is it its of on or
		that the their this to via was we were with without our using based towards toward can not`) {
		stopwords[w] = true
	}
}

func Tokenize(s string) []string {
	fields := strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	out := fields[:0]
	for _, f := range fields {
		if !stopwords[f] {
			out = append(out, f)
		}
	}
	return out
}

type Filter struct {
	Since    time.Time
	Category string
	Source   string
}

func (f Filter) match(p Paper) bool {
	if !f.Since.IsZero() {
		t := p.PublishedTime()
		if t.IsZero() || t.Before(f.Since) {
			return false
		}
	}
	if f.Category != "" {
		ok := false
		for _, c := range p.Categories {
			if strings.EqualFold(c, f.Category) {
				ok = true
				break
			}
		}
		if !ok {
			return false
		}
	}
	if f.Source != "" && f.Source != "all" && !p.HasSource(f.Source) {
		return false
	}
	return true
}

type Hit struct {
	Paper Paper   `json:"paper"`
	Score float64 `json:"score"`
}

func docTokens(p Paper) []string {
	var toks []string
	title := Tokenize(p.Title)
	for range titleRepeat {
		toks = append(toks, title...)
	}
	toks = append(toks, Tokenize(p.Abstract)...)
	toks = append(toks, Tokenize(strings.Join(p.Authors, " "))...)
	return toks
}

// Search ranks filtered papers by BM25 (title counted twice), ties broken by Published desc, then upvotes.
func Search(ps []Paper, query string, f Filter, max int) []Hit {
	qtoks := Tokenize(query)
	if len(qtoks) == 0 {
		return nil
	}
	var docs [][]string
	var cands []Paper
	for _, p := range ps {
		if f.match(p) {
			cands = append(cands, p)
			docs = append(docs, docTokens(p))
		}
	}
	if len(cands) == 0 {
		return nil
	}
	df := map[string]int{}
	total := 0
	for _, d := range docs {
		total += len(d)
		seen := map[string]bool{}
		for _, t := range d {
			if !seen[t] {
				seen[t] = true
				df[t]++
			}
		}
	}
	n := float64(len(docs))
	avg := float64(total) / n
	if avg == 0 {
		avg = 1
	}
	uniq := unionSorted(qtoks, nil)
	var hits []Hit
	for i, d := range docs {
		tf := map[string]int{}
		for _, t := range d {
			tf[t]++
		}
		score := 0.0
		for _, q := range uniq {
			c := float64(tf[q])
			if c == 0 {
				continue
			}
			idf := math.Log(1 + (n-float64(df[q])+0.5)/(float64(df[q])+0.5))
			score += idf * c * (bm25K1 + 1) / (c + bm25K1*(1-bm25B+bm25B*float64(len(d))/avg))
		}
		if score > 0 {
			hits = append(hits, Hit{Paper: cands[i], Score: score})
		}
	}
	sort.SliceStable(hits, func(i, j int) bool {
		if hits[i].Score != hits[j].Score {
			return hits[i].Score > hits[j].Score
		}
		return newerFirst(hits[i].Paper, hits[j].Paper)
	})
	if max > 0 && len(hits) > max {
		hits = hits[:max]
	}
	return hits
}

func Latest(ps []Paper, f Filter, max int) []Paper {
	var out []Paper
	for _, p := range ps {
		if f.match(p) {
			out = append(out, p)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return newerFirst(out[i], out[j]) })
	if max > 0 && len(out) > max {
		out = out[:max]
	}
	return out
}

func newerFirst(a, b Paper) bool {
	at, bt := a.PublishedTime(), b.PublishedTime()
	if !at.Equal(bt) {
		return at.After(bt)
	}
	if a.Upvotes != b.Upvotes {
		return a.Upvotes > b.Upvotes
	}
	return a.ID < b.ID
}

// ParseSince accepts "30d", "2w", a Go duration ("48h"), YYYY-MM-DD, or RFC3339.
func ParseSince(s string, now time.Time) (time.Time, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return time.Time{}, nil
	}
	if t := parseTime(s); !t.IsZero() {
		return t, nil
	}
	if len(s) > 1 {
		unit := s[len(s)-1]
		if n, err := strconv.Atoi(s[:len(s)-1]); err == nil && n >= 0 {
			switch unit {
			case 'd':
				return now.AddDate(0, 0, -n), nil
			case 'w':
				return now.AddDate(0, 0, -7*n), nil
			}
		}
	}
	if d, err := time.ParseDuration(s); err == nil {
		return now.Add(-d), nil
	}
	return time.Time{}, fmt.Errorf("invalid --since %q (want 30d, 2w, 48h, or YYYY-MM-DD)", s)
}
