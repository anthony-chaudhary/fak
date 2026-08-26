package studyforge

import (
	"encoding/hex"
	"errors"
	"fmt"
	"net/url"
	"sort"
	"strings"
	"time"
)

// Validate verifies that a corpus is terminally complete as well as internally sound.
func Validate(c Corpus) error { return validateCorpus(c, true) }

// validateCheckpoint accepts explicit partial/failed progress, but verifies every durable
// fact already claimed by the checkpoint. Resume must never build on a weak partial file.
func validateCheckpoint(c Corpus) error { return validateCorpus(c, false) }

func validateCorpus(c Corpus, requireComplete bool) error {
	var es []error
	if c.Schema != CorpusSchema {
		es = append(es, fmt.Errorf("schema must be %q", CorpusSchema))
	}
	r := c.Receipt
	if r.Schema != ReceiptSchema {
		es = append(es, fmt.Errorf("receipt schema must be %q", ReceiptSchema))
	}
	if r.Repository == "" {
		es = append(es, errors.New("repository is required"))
	}
	if r.Revision == "" && (r.Status != StatusFailed || len(r.Sources) != 0) {
		es = append(es, errors.New("revision is required once source capture starts"))
	}
	if _, e := time.Parse(time.RFC3339Nano, r.Cutoff); e != nil {
		es = append(es, errors.New("cutoff must be RFC3339"))
	}
	if r.APIBase != "" {
		base, e := url.Parse(r.APIBase)
		if e != nil || base.Scheme == "" || base.Host == "" {
			es = append(es, errors.New("api_base must be an absolute URL"))
		}
	}
	if r.StartedAt != "" {
		if _, e := time.Parse(time.RFC3339Nano, r.StartedAt); e != nil {
			es = append(es, errors.New("started_at must be RFC3339"))
		}
	}
	if r.CompletedAt != "" {
		if _, e := time.Parse(time.RFC3339Nano, r.CompletedAt); e != nil {
			es = append(es, errors.New("completed_at must be RFC3339"))
		}
	}
	if r.Status != StatusComplete && r.Status != StatusPartial && r.Status != StatusFailed {
		es = append(es, fmt.Errorf("invalid receipt status %q", r.Status))
	}
	for _, api := range r.API {
		if api.URL == "" || !validDigest(api.Checksum) {
			es = append(es, fmt.Errorf("%s API receipt has invalid URL or checksum", api.Purpose))
		}
	}

	seenSources := map[string]bool{}
	seenNodeIDs := map[string]bool{}
	complete := true
	var issuePulls, pullRows int
	var issuesComplete, pullsComplete bool
	for sourceIndex, s := range r.Sources {
		if sourceIndex >= len(SourceNames) || s.Name != SourceNames[sourceIndex] {
			es = append(es, fmt.Errorf("source order mismatch at position %d", sourceIndex+1))
		}
		if sourceRank(s.Name) >= len(SourceNames) {
			es = append(es, fmt.Errorf("unknown source %q", s.Name))
			continue
		}
		if seenSources[s.Name] {
			es = append(es, fmt.Errorf("duplicate source %q", s.Name))
		}
		seenSources[s.Name] = true
		if s.Endpoint == "" {
			es = append(es, fmt.Errorf("%s endpoint is required", s.Name))
		}
		if s.Status != StatusComplete && s.Status != StatusPartial && s.Status != StatusFailed {
			es = append(es, fmt.Errorf("%s has invalid status %q", s.Name, s.Status))
		}
		if s.Status != StatusComplete {
			complete = false
		}
		if s.Status == StatusComplete && s.Failure != "" {
			es = append(es, fmt.Errorf("%s complete with failure evidence", s.Name))
		}

		seenPageURLs := map[string]bool{}
		for i, p := range s.Pages {
			if p.Number != i+1 {
				es = append(es, fmt.Errorf("%s missing page %d", s.Name, i+1))
			}
			if p.URL == "" || seenPageURLs[p.URL] {
				es = append(es, fmt.Errorf("%s page %d has empty or repeated URL", s.Name, p.Number))
			}
			seenPageURLs[p.URL] = true
			if !validDigest(p.Checksum) {
				es = append(es, fmt.Errorf("%s page %d checksum is invalid", s.Name, p.Number))
			}
			if p.StatusCode < 200 || p.StatusCode >= 300 {
				es = append(es, fmt.Errorf("%s page %d status code is not successful", s.Name, p.Number))
			}
			if i < len(s.Pages)-1 && p.Next != s.Pages[i+1].URL {
				es = append(es, fmt.Errorf("%s page %d next cursor does not match page %d URL", s.Name, p.Number, p.Number+1))
			}
			if p.Next != "" && seenPageURLs[p.Next] {
				es = append(es, fmt.Errorf("%s page %d repeats an earlier next cursor", s.Name, p.Number))
			}
			if i == len(s.Pages)-1 && s.Status == StatusComplete && p.Next != "" {
				es = append(es, fmt.Errorf("%s complete with unfetched next page", s.Name))
			}
		}
		if s.Status == StatusComplete && len(s.Pages) == 0 {
			es = append(es, fmt.Errorf("%s complete without a terminal page", s.Name))
		}
		if s.PageChecksum != "" && s.PageChecksum != pageDigest(s.Pages) {
			es = append(es, fmt.Errorf("%s page chain checksum mismatch", s.Name))
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
			issuePulls, issuesComplete = s.ClassifiedPullCount, s.Status == StatusComplete
		}
		if s.Name == "pulls" {
			pullRows, pullsComplete = s.FetchedCount, s.Status == StatusComplete
		}
	}
	if issuesComplete && pullsComplete && issuePulls != pullRows {
		es = append(es, fmt.Errorf("issues pull partition count %d does not reconcile with pulls census %d", issuePulls, pullRows))
	}
	for _, record := range c.Records {
		if !seenSources[record.Source] {
			es = append(es, fmt.Errorf("record source %q has no receipt", record.Source))
		}
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
		if ordered[i].Source != c.Records[i].Source || ordered[i].ID != c.Records[i].ID || ordered[i].Number != c.Records[i].Number || ordered[i].Name != c.Records[i].Name {
			es = append(es, errors.New("records are not in canonical order"))
			break
		}
	}
	if r.Status == StatusComplete && (!complete || len(seenSources) != len(SourceNames)) {
		es = append(es, errors.New("partial receipt marked complete"))
	}
	if requireComplete && r.Status != StatusComplete {
		es = append(es, fmt.Errorf("receipt status must be complete, got %q", r.Status))
	}
	if r.IndexChecksum != recordDigest(c.Records) {
		es = append(es, errors.New("index checksum mismatch"))
	}
	return errors.Join(es...)
}

func validDigest(value string) bool {
	const prefix = "sha256:"
	if !strings.HasPrefix(value, prefix) || len(value) != len(prefix)+64 {
		return false
	}
	_, err := hex.DecodeString(strings.TrimPrefix(value, prefix))
	return err == nil
}

func mustParseCutoff(v string) time.Time { t, _ := time.Parse(time.RFC3339Nano, v); return t }

func validateResume(c Corpus, repo string, cutoff time.Time, baseURL string, expectedEndpoints map[string]string) error {
	if err := validateCheckpoint(c); err != nil {
		return fmt.Errorf("invalid resume: %w", err)
	}
	if c.Receipt.Repository != repo {
		return errors.New("resume repository mismatch")
	}
	t, e := time.Parse(time.RFC3339Nano, c.Receipt.Cutoff)
	if e != nil || !t.Equal(cutoff) {
		return errors.New("resume cutoff mismatch")
	}
	if c.Receipt.Revision == "" {
		return errors.New("resume revision is required")
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if c.Receipt.APIBase != "" && strings.TrimRight(c.Receipt.APIBase, "/") != baseURL {
		return errors.New("resume API base mismatch")
	}
	for _, source := range c.Receipt.Sources {
		if source.Endpoint != expectedEndpoints[source.Name] {
			return fmt.Errorf("resume %s endpoint does not match API base", source.Name)
		}
		for _, page := range source.Pages {
			if !sameAPIBase(page.URL, baseURL) || (page.Next != "" && !sameAPIBase(page.Next, baseURL)) {
				return fmt.Errorf("resume %s page chain leaves API base", source.Name)
			}
		}
	}
	for _, api := range c.Receipt.API {
		if !sameAPIBase(api.URL, baseURL) {
			return errors.New("resume API receipt base mismatch")
		}
	}
	return nil
}

func sameAPIBase(rawURL, baseURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	base, err := url.Parse(baseURL)
	if err != nil {
		return false
	}
	return u.Scheme == base.Scheme && u.Host == base.Host && (base.Path == "" || strings.HasPrefix(u.Path, strings.TrimRight(base.Path, "/")+"/"))
}
