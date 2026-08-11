// Package quantwatch provides a neutral, metadata-only watchlist for public
// quantization research and ecosystem releases. It ranks observations; it does
// not endorse methods or infer artifact, runtime, or hardware support.
package quantwatch

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"
)

const (
	SnapshotSchemaV1 = "fak.quantwatch.snapshot/v1"
	ResultSchemaV1   = "fak.quantwatch.result/v1"

	SourceArxiv            SourceKind = "arxiv"
	SourceEcosystemRelease SourceKind = "ecosystem_release"

	OutcomeRanked      Outcome = "ranked"
	OutcomeAbstain     Outcome = "abstain"
	OutcomeUnsupported Outcome = "unsupported"
	OutcomeRefused     Outcome = "refused"

	ReasonNone              Reason = "none"
	ReasonUnknownVersion    Reason = "unknown_version"
	ReasonSourceNotHandled  Reason = "source_not_handled"
	ReasonMalformedInput    Reason = "malformed_input"
	ReasonNoRecords         Reason = "no_records"
	ReasonSourceUnavailable Reason = "source_unavailable"
	ReasonBoundExceeded     Reason = "bound_exceeded"

	ClaimReported ClaimStatus = "reported"
	ClaimMeasured ClaimStatus = "measured"
	ClaimUnknown  ClaimStatus = "unknown"
)

type SourceKind string
type Outcome string
type Reason string
type ClaimStatus string

type Claim struct {
	Status ClaimStatus `json:"status"`
	Detail string      `json:"detail,omitempty"`
}

type Claims struct {
	Artifact          Claim `json:"artifact"`
	Recipe            Claim `json:"recipe"`
	RuntimeDelegation Claim `json:"runtime_delegation"`
	HardwareEnvelope  Claim `json:"hardware_envelope"`
}

type Record struct {
	ID          string    `json:"id"`
	Title       string    `json:"title"`
	URL         string    `json:"url"`
	PublishedAt time.Time `json:"published_at"`
	Adoption    int       `json:"adoption"`
	Tags        []string  `json:"tags,omitempty"`
	Claims      Claims    `json:"claims"`
}

type SnapshotSource struct {
	Kind    SourceKind `json:"kind"`
	Name    string     `json:"name"`
	Records []Record   `json:"records"`
}

type Snapshot struct {
	Schema    string           `json:"schema"`
	QueryTime time.Time        `json:"query_time"`
	Query     string           `json:"query,omitempty"`
	Sources   []SnapshotSource `json:"sources"`
}

type SourceReceipt struct {
	Kind      SourceKind `json:"kind,omitempty"`
	Name      string     `json:"name"`
	URL       string     `json:"url,omitempty"`
	QueryTime time.Time  `json:"query_time"`
	Records   int        `json:"records"`
	Abstained bool       `json:"abstained"`
	Reason    Reason     `json:"reason,omitempty"`
	Detail    string     `json:"detail,omitempty"`
}

type RankedItem struct {
	Record
	Score          float64 `json:"score"`
	RecencyScore   float64 `json:"recency_score"`
	AdoptionScore  float64 `json:"adoption_score"`
	RelevanceScore float64 `json:"relevance_score"`
	Source         string  `json:"source"`
}

type Result struct {
	Schema           string          `json:"schema"`
	Outcome          Outcome         `json:"outcome"`
	Reason           Reason          `json:"reason"`
	Query            string          `json:"query,omitempty"`
	QueryTime        time.Time       `json:"query_time"`
	Items            []RankedItem    `json:"items"`
	Sources          []SourceReceipt `json:"sources"`
	Deduplicated     int             `json:"deduplicated"`
	EvidenceBoundary string          `json:"evidence_boundary"`
}

func baseResult(query string, at time.Time) Result {
	return Result{Schema: ResultSchemaV1, Outcome: OutcomeAbstain, Reason: ReasonNoRecords, Query: query, QueryTime: at, Items: []RankedItem{}, Sources: []SourceReceipt{}, EvidenceBoundary: "metadata-only: reported is not measured; unknown is never promoted; ranking is not endorsement"}
}

// IngestSnapshot validates, deduplicates, and ranks a bounded offline snapshot.
func IngestSnapshot(raw []byte) Result {
	var header struct {
		Schema    string    `json:"schema"`
		QueryTime time.Time `json:"query_time"`
		Query     string    `json:"query"`
	}
	if err := json.Unmarshal(raw, &header); err != nil {
		got := baseResult("", time.Time{})
		got.Outcome, got.Reason = OutcomeRefused, ReasonMalformedInput
		got.Sources = []SourceReceipt{{Name: "snapshot", Abstained: true, Reason: ReasonMalformedInput, Detail: err.Error()}}
		return got
	}
	if header.Schema != SnapshotSchemaV1 {
		got := baseResult(header.Query, header.QueryTime)
		got.Reason = ReasonUnknownVersion
		got.Sources = []SourceReceipt{{Name: "snapshot", QueryTime: header.QueryTime, Abstained: true, Reason: ReasonUnknownVersion, Detail: fmt.Sprintf("schema %q is not %q", header.Schema, SnapshotSchemaV1)}}
		return got
	}
	var snapshot Snapshot
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&snapshot); err != nil {
		got := baseResult(header.Query, header.QueryTime)
		got.Outcome, got.Reason = OutcomeRefused, ReasonMalformedInput
		got.Sources = []SourceReceipt{{Name: "snapshot", QueryTime: header.QueryTime, Abstained: true, Reason: ReasonMalformedInput, Detail: err.Error()}}
		return got
	}
	return Rank(snapshot)
}

// Rank produces deterministic ranking: recency 50%, adoption 30%, query
// relevance 20%. Stable public IDs are the dedupe key; ties sort by ID.
func Rank(snapshot Snapshot) Result {
	got := baseResult(snapshot.Query, snapshot.QueryTime)
	seen := make(map[string]struct{})
	unsupported := false
	for _, source := range snapshot.Sources {
		receipt := SourceReceipt{Kind: source.Kind, Name: source.Name, QueryTime: snapshot.QueryTime, Records: len(source.Records)}
		if source.Kind != SourceArxiv && source.Kind != SourceEcosystemRelease {
			receipt.Abstained, receipt.Reason = true, ReasonSourceNotHandled
			unsupported = true
			got.Sources = append(got.Sources, receipt)
			continue
		}
		got.Sources = append(got.Sources, receipt)
		for _, record := range source.Records {
			if record.ID == "" || record.Title == "" || record.PublishedAt.IsZero() {
				continue
			}
			if _, ok := seen[record.ID]; ok {
				got.Deduplicated++
				continue
			}
			seen[record.ID] = struct{}{}
			normalizeClaims(&record.Claims)
			recency, adoption, relevance := scoreParts(snapshot.QueryTime, snapshot.Query, record)
			got.Items = append(got.Items, RankedItem{Record: record, Score: round6(.5*recency + .3*adoption + .2*relevance), RecencyScore: recency, AdoptionScore: adoption, RelevanceScore: relevance, Source: source.Name})
		}
	}
	sort.Slice(got.Items, func(i, j int) bool {
		if got.Items[i].Score != got.Items[j].Score {
			return got.Items[i].Score > got.Items[j].Score
		}
		if !got.Items[i].PublishedAt.Equal(got.Items[j].PublishedAt) {
			return got.Items[i].PublishedAt.After(got.Items[j].PublishedAt)
		}
		return got.Items[i].ID < got.Items[j].ID
	})
	if len(got.Items) > 0 {
		got.Outcome, got.Reason = OutcomeRanked, ReasonNone
	} else if unsupported {
		got.Outcome, got.Reason = OutcomeUnsupported, ReasonSourceNotHandled
	}
	return got
}

func normalizeClaims(c *Claims) {
	claims := []*Claim{&c.Artifact, &c.Recipe, &c.RuntimeDelegation, &c.HardwareEnvelope}
	for _, claim := range claims {
		if claim.Status != ClaimReported && claim.Status != ClaimMeasured {
			claim.Status = ClaimUnknown
		}
	}
	// Metadata ingestion can report a publisher's measured claim, but this leaf
	// never performed that measurement. Preserve honesty by refusing promotion.
	if c.HardwareEnvelope.Status == ClaimMeasured {
		c.HardwareEnvelope.Status = ClaimReported
		if c.HardwareEnvelope.Detail == "" {
			c.HardwareEnvelope.Detail = "publisher metadata reports a measurement; quantwatch did not reproduce it"
		}
	}
}

func scoreParts(now time.Time, query string, r Record) (float64, float64, float64) {
	age := now.Sub(r.PublishedAt).Hours() / 24
	if age < 0 {
		age = 0
	}
	recency := math.Max(0, 1-age/365)
	adoption := math.Log1p(float64(max(r.Adoption, 0))) / math.Log(1001)
	terms := strings.Fields(strings.ToLower(query))
	haystack := strings.ToLower(r.Title + " " + strings.Join(r.Tags, " "))
	matched := 0
	for _, term := range terms {
		if strings.Contains(haystack, term) {
			matched++
		}
	}
	relevance := 0.0
	if len(terms) == 0 {
		relevance = 1
	} else {
		relevance = float64(matched) / float64(len(terms))
	}
	return round6(recency), round6(math.Min(1, adoption)), round6(relevance)
}

func round6(v float64) float64 { return math.Round(v*1e6) / 1e6 }

// LiveRequest bounds public-source queries. GitHubRepositories must contain
// explicit owner/repo names; quantwatch does not discover or endorse projects.
type LiveRequest struct {
	Query              string
	QueryTime          time.Time
	MaxPerSource       int
	GitHubRepositories []string
	ArxivEndpoint      string
	GitHubAPI          string
}

// FetchLive records a receipt for every attempted source and returns a ranked
// result even when one source abstains. It never downloads model artifacts.
func FetchLive(ctx context.Context, client *http.Client, req LiveRequest) Result {
	if req.QueryTime.IsZero() {
		req.QueryTime = time.Now().UTC()
	}
	if req.MaxPerSource <= 0 {
		req.MaxPerSource = 10
	}
	if req.MaxPerSource > 100 {
		got := baseResult(req.Query, req.QueryTime)
		got.Outcome, got.Reason = OutcomeRefused, ReasonBoundExceeded
		got.Sources = []SourceReceipt{{Name: "live-request", QueryTime: req.QueryTime, Abstained: true, Reason: ReasonBoundExceeded, Detail: "max_per_source must be <= 100"}}
		return got
	}
	if client == nil {
		client = http.DefaultClient
	}
	if req.ArxivEndpoint == "" {
		req.ArxivEndpoint = "https://export.arxiv.org/api/query"
	}
	if req.GitHubAPI == "" {
		req.GitHubAPI = "https://api.github.com"
	}
	snapshot := Snapshot{Schema: SnapshotSchemaV1, QueryTime: req.QueryTime, Query: req.Query}
	receipts := []SourceReceipt{}

	records, endpoint, err := fetchArxiv(ctx, client, req.ArxivEndpoint, req.Query, req.MaxPerSource)
	ar := SourceReceipt{Kind: SourceArxiv, Name: "arxiv", URL: endpoint, QueryTime: req.QueryTime, Records: len(records)}
	if err != nil {
		ar.Abstained, ar.Reason, ar.Detail = true, ReasonSourceUnavailable, err.Error()
	}
	receipts = append(receipts, ar)
	if err == nil {
		snapshot.Sources = append(snapshot.Sources, SnapshotSource{Kind: SourceArxiv, Name: "arxiv", Records: records})
	}

	for _, repo := range req.GitHubRepositories {
		records, endpoint, err := fetchGitHubReleases(ctx, client, req.GitHubAPI, repo, req.MaxPerSource)
		receipt := SourceReceipt{Kind: SourceEcosystemRelease, Name: "github:" + repo, URL: endpoint, QueryTime: req.QueryTime, Records: len(records)}
		if err != nil {
			receipt.Abstained, receipt.Reason, receipt.Detail = true, ReasonSourceUnavailable, err.Error()
		}
		receipts = append(receipts, receipt)
		if err == nil {
			snapshot.Sources = append(snapshot.Sources, SnapshotSource{Kind: SourceEcosystemRelease, Name: "github:" + repo, Records: records})
		}
	}
	got := Rank(snapshot)
	got.Sources = receipts
	if len(got.Items) == 0 {
		got.Outcome, got.Reason = OutcomeAbstain, ReasonNoRecords
		for _, r := range receipts {
			if r.Abstained {
				got.Reason = ReasonSourceUnavailable
				break
			}
		}
	}
	return got
}

func fetchArxiv(ctx context.Context, client *http.Client, base, query string, limit int) ([]Record, string, error) {
	u, err := url.Parse(base)
	if err != nil {
		return nil, base, err
	}
	q := u.Query()
	q.Set("search_query", "all:"+query)
	q.Set("start", "0")
	q.Set("max_results", fmt.Sprint(limit))
	q.Set("sortBy", "submittedDate")
	q.Set("sortOrder", "descending")
	u.RawQuery = q.Encode()
	body, err := get(ctx, client, u.String(), "application/atom+xml")
	if err != nil {
		return nil, u.String(), err
	}
	// Atom fields need individual tags rather than chardata.
	var atom struct {
		Entries []struct {
			ID        string `xml:"id"`
			Title     string `xml:"title"`
			Published string `xml:"published"`
		} `xml:"entry"`
	}
	if err := xml.Unmarshal(body, &atom); err != nil {
		return nil, u.String(), err
	}
	out := make([]Record, 0, len(atom.Entries))
	for _, e := range atom.Entries {
		published, err := time.Parse(time.RFC3339, strings.TrimSpace(e.Published))
		if err != nil {
			continue
		}
		id := strings.TrimSpace(e.ID)
		paperID := strings.TrimPrefix(id, "http://arxiv.org/abs/")
		paperID = strings.TrimPrefix(paperID, "https://arxiv.org/abs/")
		out = append(out, Record{ID: "arxiv:" + paperID, Title: strings.Join(strings.Fields(e.Title), " "), URL: id, PublishedAt: published, Tags: []string{"quantization", "research"}, Claims: Claims{Recipe: Claim{Status: ClaimReported, Detail: "paper metadata reports a research recipe"}}})
	}
	return out, u.String(), nil
}

func fetchGitHubReleases(ctx context.Context, client *http.Client, base, repo string, limit int) ([]Record, string, error) {
	parts := strings.Split(repo, "/")
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return nil, base, errors.New("repository must be owner/name")
	}
	endpoint := strings.TrimRight(base, "/") + "/repos/" + url.PathEscape(parts[0]) + "/" + url.PathEscape(parts[1]) + "/releases?per_page=" + fmt.Sprint(limit)
	body, err := get(ctx, client, endpoint, "application/vnd.github+json")
	if err != nil {
		return nil, endpoint, err
	}
	var releases []struct {
		ID         int64  `json:"id"`
		Tag        string `json:"tag_name"`
		Name       string `json:"name"`
		URL        string `json:"html_url"`
		Published  string `json:"published_at"`
		Draft      bool   `json:"draft"`
		Prerelease bool   `json:"prerelease"`
	}
	if err := json.Unmarshal(body, &releases); err != nil {
		return nil, endpoint, err
	}
	out := make([]Record, 0, len(releases))
	for _, release := range releases {
		if release.Draft {
			continue
		}
		published, err := time.Parse(time.RFC3339, release.Published)
		if err != nil {
			continue
		}
		title := release.Name
		if title == "" {
			title = release.Tag
		}
		out = append(out, Record{ID: fmt.Sprintf("github:%s:%d", repo, release.ID), Title: title, URL: release.URL, PublishedAt: published, Tags: []string{"runtime", "release", "quantization"}, Claims: Claims{Artifact: Claim{Status: ClaimReported, Detail: "release metadata names an artifact; quantwatch did not download it"}, RuntimeDelegation: Claim{Status: ClaimReported, Detail: "execution would delegate to this external runtime"}}})
	}
	return out, endpoint, nil
}

func get(ctx context.Context, client *http.Client, endpoint, accept string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", accept)
	req.Header.Set("User-Agent", "fak-quantwatch/1")
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("GET %s: %s", endpoint, resp.Status)
	}
	return io.ReadAll(io.LimitReader(resp.Body, 4<<20))
}
