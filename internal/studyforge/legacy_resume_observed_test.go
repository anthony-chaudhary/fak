package studyforge

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestLegacyObservedCheckpointResumesAfter532IssueAnd348PullPagesWithoutRefetch(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu    sync.Mutex
		calls = map[string]int{}
	)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.RequestURI()]++
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/acme/widget/pulls":
			if r.URL.Query().Get("page") != "349" {
				t.Errorf("unexpected pulls refetch: %s", r.URL.RequestURI())
				http.Error(w, "unexpected refetch", http.StatusConflict)
				return
			}
			fmt.Fprint(w, `[]`)
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			t.Errorf("unexpected request: %s", r.URL.RequestURI())
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	checkpoint := observedLegacyCheckpointForTest(server.URL, cutoff)
	if err := validateResumeCheckpoint(checkpoint); err != nil {
		t.Fatalf("observed legacy checkpoint rejected: %v", err)
	}
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.MaxRetries = 0
	collector.Now = func() time.Time { return cutoff }
	resumed, err := collector.Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff, Resume: &checkpoint,
	})
	if err == nil || !strings.Contains(err.Error(), "compatible but unproven") {
		t.Fatalf("Capture error = %v, want typed count-only partial", err)
	}
	if resumed.Receipt.Status != StatusPartial {
		t.Fatalf("resumed status = %q, want partial", resumed.Receipt.Status)
	}
	delta := resumed.Receipt.NonAtomicDelta
	if delta == nil || delta.EvidenceMode != NonAtomicDeltaEvidenceModeLegacyCountOnly || delta.MixedCount != 35528 || delta.DedicatedCount != 34800 || delta.SymmetricDifferenceLowerBound != 728 || delta.SymmetricDifferenceUpperBound != 70328 || delta.Verdict != NonAtomicDeltaVerdictCompatibleUnproven {
		t.Fatalf("count-only evidence = %+v", delta)
	}
	mu.Lock()
	defer mu.Unlock()
	if calls["/repos/acme/widget/pulls?page=349"] != 1 {
		t.Fatalf("pull page 349 calls = %d, want 1", calls["/repos/acme/widget/pulls?page=349"])
	}
	for uri := range calls {
		if strings.HasPrefix(uri, "/repos/acme/widget/issues") {
			t.Fatalf("resume refetched one of 532 completed issue pages: %s", uri)
		}
		if strings.HasPrefix(uri, "/repos/acme/widget/pulls") && uri != "/repos/acme/widget/pulls?page=349" {
			t.Fatalf("resume refetched one of 348 completed pull pages: %s", uri)
		}
	}
}

func observedLegacyCheckpointForTest(baseURL string, cutoff time.Time) Corpus {
	repositoryURL := baseURL + "/repos/acme/widget"
	issuesEndpoint := repositoryURL + "/issues?state=all&sort=created&direction=asc&per_page=100&page=1"
	pullsEndpoint := repositoryURL + "/pulls?state=all&sort=created&direction=asc&per_page=100&page=1"
	fetchedAt := cutoff.Format(time.RFC3339Nano)

	issuesPages := observedPagesForTest(issuesEndpoint, repositoryURL+"/issues?page=", 532, 53134, "", fetchedAt)
	pullsPages := observedPagesForTest(pullsEndpoint, repositoryURL+"/pulls?page=", 348, 34800, repositoryURL+"/pulls?page=349", fetchedAt)
	records := make([]Record, 0, 17606+34800)
	for i := 1; i <= 17606; i++ {
		records = append(records, Record{Source: "issues", Kind: "issue", ID: int64(i), Number: i})
	}
	for i := 1; i <= 34800; i++ {
		records = append(records, Record{Source: "pulls", Kind: "pull", ID: int64(100000 + i), Number: i, NodeID: fmt.Sprintf("PR_%d", i)})
	}
	corpus := Corpus{
		Schema: CorpusSchema,
		Receipt: Receipt{
			Schema:     ReceiptSchema,
			Repository: "acme/widget",
			Revision:   "observed-legacy-revision",
			Cutoff:     cutoff.Format(time.RFC3339Nano),
			APIBase:    baseURL,
			StartedAt:  fetchedAt,
			Status:     StatusPartial,
			Sources: []SourceReceipt{
				{
					Name: "issues", Endpoint: issuesEndpoint, Status: StatusComplete,
					CrawlStartedAt: fetchedAt, CrawlEndedAt: fetchedAt, Pages: issuesPages,
					FetchedCount: 53134, ClassifiedPullCount: 35528,
				},
				{
					Name: "pulls", Endpoint: pullsEndpoint, Status: StatusPartial,
					CrawlStartedAt: fetchedAt, Pages: pullsPages, FetchedCount: 34800,
				},
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

func observedPagesForTest(firstURL, pagePrefix string, pageCount, itemCount int, terminalNext, fetchedAt string) []PageReceipt {
	pages := make([]PageReceipt, pageCount)
	remaining := itemCount
	for i := range pages {
		url := firstURL
		if i > 0 {
			url = fmt.Sprintf("%s%d", pagePrefix, i+1)
		}
		count := min(100, remaining)
		remaining -= count
		next := terminalNext
		if i < len(pages)-1 {
			next = fmt.Sprintf("%s%d", pagePrefix, i+2)
		}
		pages[i] = PageReceipt{
			Number: i + 1, URL: url, ItemCount: count, Checksum: digest([]byte(fmt.Sprintf("page-%d", i+1))),
			Next: next, FetchedAt: fetchedAt, StatusCode: http.StatusOK,
		}
	}
	return pages
}
