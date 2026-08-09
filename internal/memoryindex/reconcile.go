package memoryindex

import (
	"path/filepath"
	"sort"
	"strings"
)

func Reconcile(s Store, opt Options) Report {
	r := Report{Schema: Schema, Dir: s.Dir, Counts: zeroCounts(), Files: len(s.Files), Tiers: append([]string(nil), s.Tiers...), Rows: len(s.Rows)}
	allowed := map[string]bool{"project": true, "user": true, "feedback": true, "reference": true}
	for _, x := range opt.Types {
		allowed[x] = true
	}
	files := map[string]File{}
	slugs := map[string][]string{}
	for _, f := range s.Files {
		if IsIndexTier(f.Name) {
			continue
		}
		files[f.Name] = f
		slug := strings.TrimSuffix(f.Name, filepath.Ext(f.Name))
		if !f.Front.Present || !f.Front.Terminated || f.Front.Name == "" || f.Front.Description == "" || f.Front.Type == "" {
			r.add(KindFrontmatter, f.Name, f.Name, "")
		}
		if f.Front.Type != "" && len(allowed) > 0 && !allowed[f.Front.Type] {
			r.add(KindTypeVocabulary, f.Name, f.Name, f.Front.Type)
		}
		if f.Front.Name != "" {
			slugs[f.Front.Name] = append(slugs[f.Front.Name], f.Name)
			if f.Front.Name != slug {
				r.add(KindSlugMismatch, f.Name, f.Name, f.Front.Name+" != "+slug)
			}
		}
	}
	covered := map[string]bool{}
	present := map[string]bool{}
	for n := range files {
		present[n] = true
	}
	for _, n := range s.Present {
		present[n] = true
	}
	for _, row := range s.Rows {
		target := filepath.Base(row.Target)
		if IsIndexTier(target) {
			continue
		}
		if present[target] {
			if _, ok := files[target]; ok {
				covered[target] = true
			}
		} else {
			r.add(KindIndexLineNoFile, row.Target, row.Tier+":"+itoa(row.Line), "")
		}
	}
	for n, f := range files {
		if !covered[n] {
			r.add(KindMissingFromIndex, n, n, "")
		}
		for _, target := range f.Wikilinks {
			if _, ok := slugs[target]; !ok {
				r.add(KindUnresolvedLink, target, n, target)
			}
		}
	}
	for slug, names := range slugs {
		if len(names) > 1 {
			sort.Strings(names)
			r.add(KindDuplicateSlug, slug, strings.Join(names, ","), "")
		}
	}
	SortFindings(r.Findings)
	return r
}
func (r *Report) add(k, subject, where, detail string) {
	r.Findings = append(r.Findings, Finding{Kind: k, Subject: subject, Where: where, Detail: detail})
	r.Counts[k]++
}
func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	b := make([]byte, 0, 8)
	for n > 0 {
		b = append([]byte{byte('0' + n%10)}, b...)
		n /= 10
	}
	return string(b)
}
