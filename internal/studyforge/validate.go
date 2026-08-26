package studyforge

import (
	"errors"
	"fmt"
	"sort"
	"time"
)

// Validate verifies page continuity, classification, uniqueness, counts, status, and checksums.
func Validate(c Corpus) error {
	var es []error
	if c.Schema != CorpusSchema {
		es = append(es, fmt.Errorf("schema must be %q", CorpusSchema))
	}
	r := c.Receipt
	if r.Schema != ReceiptSchema {
		es = append(es, fmt.Errorf("receipt schema must be %q", ReceiptSchema))
	}
	if r.Repository == "" || r.Revision == "" {
		es = append(es, errors.New("repository and revision are required"))
	}
	if _, e := time.Parse(time.RFC3339Nano, r.Cutoff); e != nil {
		es = append(es, errors.New("cutoff must be RFC3339"))
	}
	if r.Status != StatusComplete && r.Status != StatusPartial && r.Status != StatusFailed {
		es = append(es, fmt.Errorf("invalid receipt status %q", r.Status))
	}
	seenSources := map[string]bool{}
	seenNodeIDs := map[string]bool{}
	complete := true
	var issuePulls, pullRows int
	for _, s := range r.Sources {
		if sourceRank(s.Name) >= len(SourceNames) {
			es = append(es, fmt.Errorf("unknown source %q", s.Name))
			continue
		}
		if seenSources[s.Name] {
			es = append(es, fmt.Errorf("duplicate source %q", s.Name))
		}
		seenSources[s.Name] = true
		if s.Status != StatusComplete {
			complete = false
		}
		for i, p := range s.Pages {
			if p.Number != i+1 {
				es = append(es, fmt.Errorf("%s missing page %d", s.Name, i+1))
			}
			if i < len(s.Pages)-1 && p.Next != s.Pages[i+1].URL {
				es = append(es, fmt.Errorf("%s page %d next cursor does not match page %d URL", s.Name, p.Number, p.Number+1))
			}
			if i == len(s.Pages)-1 && s.Status == StatusComplete && p.Next != "" {
				es = append(es, fmt.Errorf("%s complete with unfetched next page", s.Name))
			}
		}
		rs := recordsForSource(c.Records, s.Name)
		pageItems := 0
		for _, p := range s.Pages {
			pageItems += p.ItemCount
		}
		if pageItems != s.FetchedCount {
			es = append(es, fmt.Errorf("%s page item count mismatch", s.Name))
		}
		ids := map[int64]bool{}
		for _, x := range rs {
			if ids[x.ID] {
				es = append(es, fmt.Errorf("%s duplicate id %d", s.Name, x.ID))
			}
			ids[x.ID] = true
			if x.NodeID != "" {
				if seenNodeIDs[x.NodeID] {
					es = append(es, fmt.Errorf("duplicate node_id %s", x.NodeID))
				}
				seenNodeIDs[x.NodeID] = true
			}
			want := map[string]string{"issues": "issue", "pulls": "pull", "discussions": "discussion", "releases": "release", "labels": "label", "milestones": "milestone"}[s.Name]
			if x.Kind != want {
				es = append(es, fmt.Errorf("%s id %d has mixed or unclassified kind %q", s.Name, x.ID, x.Kind))
			}
			if x.CreatedAt != "" {
				if created, e := time.Parse(time.RFC3339, x.CreatedAt); e != nil || created.After(mustParseCutoff(r.Cutoff)) {
					es = append(es, fmt.Errorf("%s id %d violates cutoff", s.Name, x.ID))
				}
			}
		}
		if s.NormalizedCount != len(rs) || s.UniqueCount != len(ids) {
			es = append(es, fmt.Errorf("%s count mismatch", s.Name))
		}
		if s.FetchedCount != s.NormalizedCount+s.ClassifiedPullCount+s.CutoffExcludedCount {
			es = append(es, fmt.Errorf("%s fetched count mismatch", s.Name))
		}
		if s.Checksum != recordDigest(rs) {
			es = append(es, fmt.Errorf("%s checksum mismatch", s.Name))
		}
		if s.Name == "issues" {
			issuePulls = s.ClassifiedPullCount
		}
		if s.Name == "pulls" {
			pullRows = s.FetchedCount
		}
	}
	if issuePulls != pullRows {
		es = append(es, fmt.Errorf("issues pull partition count %d does not reconcile with pulls census %d", issuePulls, pullRows))
	}
	ordered := append([]Record(nil), c.Records...)
	sort.Slice(ordered, func(i, j int) bool {
		a, b := ordered[i], ordered[j]
		if sourceRank(a.Source) != sourceRank(b.Source) {
			return sourceRank(a.Source) < sourceRank(b.Source)
		}
		if a.ID != b.ID {
			return a.ID < b.ID
		}
		if a.Number != b.Number {
			return a.Number < b.Number
		}
		return a.Name < b.Name
	})
	for i := range ordered {
		if ordered[i].Source != c.Records[i].Source || ordered[i].ID != c.Records[i].ID {
			return fmt.Errorf("records are not in canonical order")
		}
	}
	if r.Status == StatusComplete {
		if !complete || len(seenSources) != len(SourceNames) {
			es = append(es, errors.New("partial receipt marked complete"))
		}
	}
	if r.IndexChecksum != recordDigest(c.Records) {
		es = append(es, errors.New("index checksum mismatch"))
	}
	return errors.Join(es...)
}
func mustParseCutoff(v string) time.Time { t, _ := time.Parse(time.RFC3339Nano, v); return t }
func validateResume(c Corpus, repo string, cutoff time.Time) error {
	if err := Validate(c); err != nil && c.Receipt.Status == StatusComplete {
		return fmt.Errorf("invalid resume: %w", err)
	}
	if c.Receipt.Repository != repo {
		return errors.New("resume repository mismatch")
	}
	t, e := time.Parse(time.RFC3339Nano, c.Receipt.Cutoff)
	if e != nil || !t.Equal(cutoff) {
		return errors.New("resume cutoff mismatch")
	}
	return nil
}
