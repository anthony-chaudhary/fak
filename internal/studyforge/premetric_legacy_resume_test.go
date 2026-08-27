package studyforge

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestReadResumeAdmitsOnlyExactPreMetricLegacyCountOnlyCheckpoint(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	checkpoint := issue9337PreMetricLegacyCheckpoint(t, "https://api.example.test", cutoff)
	path := issue9337WriteRawCheckpoint(t, checkpoint)

	if _, err := Read(path); err == nil {
		t.Fatal("ordinary Read accepted the pre-metric resume-only checkpoint")
	}
	loaded, err := ReadResume(path)
	if err != nil {
		t.Fatalf("ReadResume rejected exact pre-metric checkpoint: %v", err)
	}
	if !reflect.DeepEqual(loaded, checkpoint) {
		t.Fatal("ReadResume mutated the pre-metric checkpoint")
	}
	delta := loaded.Receipt.NonAtomicDelta
	if delta == nil || delta.Policy.Metric != "" || delta.ObservedEndpointCardinalityDelta != nil ||
		delta.Verdict != NonAtomicDeltaVerdictCompatibleUnproven || delta.Accepted {
		t.Fatalf("loaded pre-metric delta = %+v", delta)
	}

	// The preserved live checkpoint was written before the explicit false
	// field existed, so omission is part of the exact historical shape.
	checkpoint.Receipt.NonAtomicDelta.IdentitySetsAvailable = nil
	b, err := json.MarshalIndent(checkpoint, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := ReadResume(path); err != nil {
		t.Fatalf("ReadResume rejected observed omitted identity availability: %v", err)
	}
}

func TestReadResumeRejectsPreMetricLegacyCountOnlyMutations(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	base := issue9337PreMetricLegacyCheckpoint(t, "https://api.example.test", cutoff)
	if err := validateResumeCheckpoint(base); err != nil {
		t.Fatalf("exact pre-metric checkpoint rejected: %v", err)
	}

	tests := []struct {
		name string
		edit func(*Corpus)
	}{
		{name: "wrong evidence mode", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.EvidenceMode = NonAtomicDeltaEvidenceModeExactIdentity
		}},
		{name: "missing reason", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.EvidenceReason = ""
		}},
		{name: "wrong reason", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.EvidenceReason = "legacy identities projected"
		}},
		{name: "identities marked available", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.IdentitySetsAvailable = boolPointer(true)
		}},
		{name: "missing identity basis", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.IdentityBasis = ""
		}},
		{name: "wrong identity basis", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.IdentityBasis = identityBasisLegacyProjection
		}},
		{name: "wrong mixed provenance", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.MixedEvidence = CrossEndpointEvidenceExactIdentities
		}},
		{name: "wrong dedicated provenance", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.DedicatedEvidence = CrossEndpointEvidenceExactCountOnly
		}},
		{name: "wrong relation provenance", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.RelationEvidence = CrossEndpointEvidenceExactIdentities
		}},
		{name: "mixed identity array present", edit: func(c *Corpus) {
			issues := issue9337SourceIndex(t, c, "issues")
			c.Receipt.Sources[issues].ClassifiedPullIdentities = []CrossEndpointIdentity{}
		}},
		{name: "mixed identity checksum present", edit: func(c *Corpus) {
			issues := issue9337SourceIndex(t, c, "issues")
			c.Receipt.Sources[issues].ClassifiedPullChecksum = identityDigest(nil)
		}},
		{name: "overlap array present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Overlap = []CrossEndpointIdentity{}
		}},
		{name: "only in mixed array present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.OnlyInMixed = []CrossEndpointIdentity{}
		}},
		{name: "only in dedicated array present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.OnlyInDedicated = []CrossEndpointIdentity{}
		}},
		{name: "overlap count present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.OverlapCount = intPointer(0)
		}},
		{name: "only in mixed count present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.OnlyInMixedCount = intPointer(0)
		}},
		{name: "only in dedicated count present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.OnlyInDedicatedCount = intPointer(0)
		}},
		{name: "mixed endpoint count mismatch", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.MixedCount--
		}},
		{name: "dedicated endpoint count mismatch", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.DedicatedCount--
		}},
		{name: "wrong lower symmetric bound", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.SymmetricDifferenceLowerBound++
		}},
		{name: "wrong upper symmetric bound", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.SymmetricDifferenceUpperBound--
		}},
		{name: "wrong policy type", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.Type = "endpoint_delta"
		}},
		{name: "wrong mixed policy bound", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.MaxOnlyInMixed--
		}},
		{name: "wrong dedicated policy bound", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.MaxOnlyInDedicated--
		}},
		{name: "wrong total policy bound", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.MaxTotal--
		}},
		{name: "current metric already present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.Metric = NonAtomicDeltaPolicyMetricEndpointCardinalityDelta
		}},
		{name: "mismatched exact metric", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Policy.Metric = NonAtomicDeltaPolicyMetricExactIdentitySymmetricDifference
		}},
		{name: "observed delta already present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta = intPointer(23)
		}},
		{name: "contradictory observed delta present", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.ObservedEndpointCardinalityDelta = intPointer(24)
		}},
		{name: "missing verdict", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Verdict = ""
		}},
		{name: "accepted verdict", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Verdict = NonAtomicDeltaVerdictAccepted
		}},
		{name: "accepted flag", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Accepted = true
		}},
		{name: "wrong delta type", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.Type = "atomic_delta"
		}},
		{name: "wrong mixed source", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.MixedSource = "pulls"
		}},
		{name: "wrong dedicated source", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.DedicatedSource = "issues"
		}},
		{name: "wrong mixed crawl", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.MixedCrawl.EndedAt = cutoff.Add(time.Second).Format(time.RFC3339Nano)
		}},
		{name: "wrong dedicated crawl", edit: func(c *Corpus) {
			c.Receipt.NonAtomicDelta.DedicatedCrawl.StartedAt = cutoff.Add(-time.Second).Format(time.RFC3339Nano)
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus := cloneCorpus(base)
			tt.edit(&corpus)
			if err := validateResumeCheckpoint(corpus); err == nil {
				t.Fatal("validateResumeCheckpoint accepted a mutated pre-metric checkpoint")
			}
		})
	}
}

func TestCaptureMigratesPreMetricLegacyCountOnlyAtNextCheckpointWithoutRefetch(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu    sync.Mutex
		calls = map[string]int{}
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path]++
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		case "/repos/acme/widget/issues", "/repos/acme/widget/pulls":
			t.Errorf("resume refetched completed endpoint: %s", r.URL.RequestURI())
			http.Error(w, "completed endpoint refetched", http.StatusConflict)
		default:
			t.Errorf("unexpected resume request: %s", r.URL.RequestURI())
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	checkpoint := issue9337PreMetricLegacyCheckpoint(t, server.URL, cutoff)
	rawPath := issue9337WriteRawCheckpoint(t, checkpoint)
	loaded, err := ReadResume(rawPath)
	if err != nil {
		t.Fatalf("ReadResume rejected exact pre-metric checkpoint: %v", err)
	}

	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.MaxRetries = 0
	collector.Now = func() time.Time { return cutoff }
	firstPath := filepath.Join(t.TempDir(), "first-migrated-checkpoint.json")
	var persisted []Corpus
	completed, err := collector.Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff, Resume: &loaded,
		Checkpoint: func(c Corpus) error {
			persisted = append(persisted, cloneCorpus(c))
			if len(persisted) == 1 {
				return Write(firstPath, c)
			}
			return nil
		},
	})
	if err != nil {
		t.Fatalf("Capture error = %v", err)
	}
	if len(persisted) == 0 {
		t.Fatal("resume did not persist a migrated checkpoint")
	}
	issue9337AssertMigratedLegacyDelta(t, persisted[0].Receipt.NonAtomicDelta)
	issue9337AssertMigratedLegacyDelta(t, completed.Receipt.NonAtomicDelta)
	if _, err := Read(firstPath); err != nil {
		t.Fatalf("ordinary Read rejected first migrated checkpoint: %v", err)
	}
	if err := Validate(completed); err != nil {
		t.Fatalf("completed migrated corpus did not validate: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()
	if calls["/repos/acme/widget/issues"] != 0 || calls["/repos/acme/widget/pulls"] != 0 {
		t.Fatalf("resume refetched completed endpoints: issues=%d pulls=%d", calls["/repos/acme/widget/issues"], calls["/repos/acme/widget/pulls"])
	}
	for _, source := range []string{"discussions", "releases", "labels", "milestones"} {
		if got := calls["/repos/acme/widget/"+source]; got != 1 {
			t.Fatalf("%s calls = %d, want 1", source, got)
		}
	}
}

func TestReadResumeLeavesExactIdentityModeUnchanged(t *testing.T) {
	exact, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "exact.json")
	if err := Write(path, exact); err != nil {
		t.Fatal(err)
	}
	loaded, err := ReadResume(path)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(loaded, exact) {
		t.Fatal("ReadResume changed an exact-identity checkpoint")
	}
}

func issue9337PreMetricLegacyCheckpoint(t *testing.T, baseURL string, cutoff time.Time) Corpus {
	t.Helper()
	const (
		mixedCount       = 35528
		dedicatedCount   = 35551
		normalizedIssues = 17606
	)
	baseURL = strings.TrimRight(baseURL, "/")
	repositoryURL := baseURL + "/repos/acme/widget"
	fetchedAt := cutoff.Format(time.RFC3339Nano)
	records := make([]Record, 0, normalizedIssues+dedicatedCount)
	for i := 1; i <= normalizedIssues; i++ {
		records = append(records, Record{Source: "issues", Kind: "issue", ID: int64(i), Number: i})
	}
	for i := 1; i <= dedicatedCount; i++ {
		records = append(records, Record{
			Source: "pulls", Kind: "pull", ID: int64(100000 + i), Number: i,
			NodeID: fmt.Sprintf("PR_%d", i),
		})
	}
	corpus := Corpus{
		Schema: CorpusSchema,
		Receipt: Receipt{
			Schema:     ReceiptSchema,
			Repository: "acme/widget",
			Revision:   "issue-9337-pre-metric",
			Cutoff:     fetchedAt,
			APIBase:    baseURL,
			StartedAt:  fetchedAt,
			Status:     StatusPartial,
			Sources: []SourceReceipt{
				{
					Name: "issues", Endpoint: repositoryURL + "/issues?state=all&sort=created&direction=asc&per_page=100&page=1",
					Status: StatusComplete, CrawlStartedAt: fetchedAt, CrawlEndedAt: fetchedAt,
					Pages: []PageReceipt{{
						Number: 1, URL: repositoryURL + "/issues?state=all&sort=created&direction=asc&per_page=100&page=1",
						ItemCount: normalizedIssues + mixedCount, Checksum: digest([]byte("issues page")),
						FetchedAt: fetchedAt, StatusCode: http.StatusOK,
					}},
					FetchedCount: normalizedIssues + mixedCount, ClassifiedPullCount: mixedCount,
				},
				{
					Name: "pulls", Endpoint: repositoryURL + "/pulls?state=all&sort=created&direction=asc&per_page=100&page=1",
					Status: StatusComplete, CrawlStartedAt: fetchedAt, CrawlEndedAt: fetchedAt,
					Pages: []PageReceipt{{
						Number: 1, URL: repositoryURL + "/pulls?state=all&sort=created&direction=asc&per_page=100&page=1",
						ItemCount: dedicatedCount, Checksum: digest([]byte("pulls page")),
						FetchedAt: fetchedAt, StatusCode: http.StatusOK,
					}},
					FetchedCount: dedicatedCount,
				},
			},
			NonAtomicDelta: &NonAtomicDeltaEvidence{
				Type:                          NonAtomicDeltaType,
				MixedSource:                   "issues",
				DedicatedSource:               "pulls",
				EvidenceMode:                  NonAtomicDeltaEvidenceModeLegacyCountOnly,
				EvidenceReason:                legacyCountOnlyReason,
				IdentitySetsAvailable:         boolPointer(false),
				IdentityBasis:                 identityBasisLegacyCountOnly,
				MixedEvidence:                 CrossEndpointEvidenceExactCountOnly,
				DedicatedEvidence:             CrossEndpointEvidenceExactIdentities,
				RelationEvidence:              CrossEndpointEvidenceUnavailable,
				MixedCrawl:                    CrawlWindow{StartedAt: fetchedAt, EndedAt: fetchedAt},
				DedicatedCrawl:                CrawlWindow{StartedAt: fetchedAt, EndedAt: fetchedAt},
				MixedCount:                    mixedCount,
				DedicatedCount:                dedicatedCount,
				SymmetricDifferenceLowerBound: 23,
				SymmetricDifferenceUpperBound: 71079,
				Policy: NonAtomicDeltaPolicy{
					Type:               NonAtomicDeltaPolicyType,
					MaxOnlyInMixed:     DefaultNonAtomicDeltaLimit,
					MaxOnlyInDedicated: DefaultNonAtomicDeltaLimit,
					MaxTotal:           DefaultNonAtomicDeltaLimit,
				},
				Verdict:  NonAtomicDeltaVerdictCompatibleUnproven,
				Accepted: false,
			},
			API: []APIReceipt{
				{Purpose: "repository", URL: repositoryURL, FetchedAt: fetchedAt, StatusCode: http.StatusOK, Checksum: digest([]byte("repository"))},
				{Purpose: "revision", URL: repositoryURL + "/commits/main", FetchedAt: fetchedAt, StatusCode: http.StatusOK, Checksum: digest([]byte("revision"))},
			},
		},
		Records: records,
	}
	sortCorpus(&corpus)
	refreshChecksums(&corpus)
	return corpus
}

func issue9337WriteRawCheckpoint(t *testing.T, corpus Corpus) string {
	t.Helper()
	data, err := json.MarshalIndent(corpus, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(t.TempDir(), "pre-metric.json")
	if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func issue9337SourceIndex(t *testing.T, corpus *Corpus, name string) int {
	t.Helper()
	for i := range corpus.Receipt.Sources {
		if corpus.Receipt.Sources[i].Name == name {
			return i
		}
	}
	t.Fatalf("source %q not found", name)
	return -1
}

func issue9337AssertMigratedLegacyDelta(t *testing.T, delta *NonAtomicDeltaEvidence) {
	t.Helper()
	if delta == nil ||
		delta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly ||
		delta.EvidenceReason != legacyCountOnlyReason ||
		delta.IdentityBasis != identityBasisLegacyCountOnly ||
		delta.MixedCount != 35528 ||
		delta.DedicatedCount != 35551 ||
		delta.SymmetricDifferenceLowerBound != 23 ||
		delta.SymmetricDifferenceUpperBound != 71079 ||
		delta.Policy.Metric != NonAtomicDeltaPolicyMetricEndpointCardinalityDelta ||
		pointedInt(delta.ObservedEndpointCardinalityDelta) != 23 ||
		delta.Verdict != NonAtomicDeltaVerdictAccepted ||
		!delta.Accepted {
		t.Fatalf("migrated legacy delta = %+v", delta)
	}
}
