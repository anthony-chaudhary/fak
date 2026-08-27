package studyforge

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestCaptureRecordsNonAtomicDeltaForConcurrentCreationAndDeletion(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(corpus); err != nil {
		t.Fatal(err)
	}
	delta := corpus.Receipt.NonAtomicDelta
	if delta == nil || !delta.Accepted || delta.Type != NonAtomicDeltaType || delta.IdentityBasis != identityBasisCaptured {
		t.Fatalf("non_atomic_delta = %+v", delta)
	}
	if delta.MixedCount != 2 || delta.DedicatedCount != 2 || delta.OverlapCount != 1 || delta.OnlyInMixedCount != 1 || delta.OnlyInDedicatedCount != 1 {
		t.Fatalf("non_atomic_delta counts = %+v", delta)
	}
	assertIdentityIDs(t, delta.Overlap, 12)
	assertIdentityIDs(t, delta.OnlyInMixed, 11)
	assertIdentityIDs(t, delta.OnlyInDedicated, 13)
	if delta.MixedCrawl.StartedAt == "" || delta.MixedCrawl.EndedAt == "" || delta.DedicatedCrawl.StartedAt == "" || delta.DedicatedCrawl.EndedAt == "" {
		t.Fatalf("missing crawl windows: %+v", delta)
	}
	if got := recordsForSource(corpus.Records, "issues"); len(got) != 0 {
		t.Fatalf("mixed endpoint PR-shaped rows leaked into issue-only records: %+v", got)
	}
	if got := recordsForSource(corpus.Records, "pulls"); len(got) != 2 || got[0].ID != 12 || got[1].ID != 13 {
		t.Fatalf("dedicated pulls are not authoritative normalized rows: %+v", got)
	}
}

func TestValidateRejectsMissingOrContradictoryNonAtomicDelta(t *testing.T) {
	base, err := captureReconciliationFixture(t, []int64{11, 12}, []int64{12, 13})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name string
		want string
		edit func(*Corpus)
	}{
		{name: "missing", want: "require non_atomic_delta", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta = nil }},
		{name: "contradictory set", want: "identity sets contradict", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.OnlyInDedicated[0].ID = 99 }},
		{name: "contradictory count", want: "counts contradict", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.OverlapCount++ }},
		{name: "contradictory verdict", want: "accepted verdict contradicts", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Accepted = false }},
		{name: "unbounded policy", want: "policy is missing or exceeds", edit: func(c *Corpus) { c.Receipt.NonAtomicDelta.Policy.MaxTotal = DefaultNonAtomicDeltaLimit + 1 }},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			corpus := cloneCorpus(base)
			tt.edit(&corpus)
			if err := Validate(corpus); err == nil || !strings.Contains(err.Error(), tt.want) {
				t.Fatalf("Validate error = %v, want %q", err, tt.want)
			}
		})
	}
}

func TestCaptureRecordsPolicyOverflowAsPartial(t *testing.T) {
	dedicated := make([]int64, DefaultNonAtomicDeltaLimit+1)
	for i := range dedicated {
		dedicated[i] = int64(i + 1)
	}
	corpus, err := captureReconciliationFixture(t, nil, dedicated)
	if err == nil || !strings.Contains(err.Error(), "exceeds policy") {
		t.Fatalf("Capture error = %v", err)
	}
	if corpus.Receipt.Status != StatusPartial || corpus.Receipt.NonAtomicDelta == nil || corpus.Receipt.NonAtomicDelta.Accepted {
		t.Fatalf("overflow receipt = %+v", corpus.Receipt)
	}
	if err := validateCheckpoint(corpus); err != nil {
		t.Fatalf("overflow partial must remain a resumable, internally consistent checkpoint: %v", err)
	}
	if err := Validate(corpus); err == nil || !strings.Contains(err.Error(), "policy overflow") {
		t.Fatalf("Validate error = %v, want policy overflow", err)
	}
}

func TestCaptureRejectsDuplicateMixedPullIdentities(t *testing.T) {
	corpus, err := captureReconciliationFixture(t, []int64{7, 7}, []int64{7})
	if err == nil || !strings.Contains(err.Error(), "duplicate issues id 7") {
		t.Fatalf("Capture error = %v", err)
	}
	issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
	if len(issues.Pages) != 0 || issues.ClassifiedPullCount != 0 || len(issues.ClassifiedPullIdentities) != 0 {
		t.Fatalf("duplicate identity page was accepted: %+v", issues)
	}
}

func TestCaptureResumesLegacyCheckpointWithoutRefetchingCompleteEndpoints(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var (
		mu              sync.Mutex
		calls           = map[string]int{}
		failDiscussions = true
		server          *httptest.Server
	)
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.Path]++
		fail := failDiscussions
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"legacy-revision"}`)
		case "/repos/acme/widget/issues":
			writeRows(w, []int64{21}, true)
		case "/repos/acme/widget/pulls":
			writeRows(w, []int64{21}, false)
		case "/repos/acme/widget/discussions":
			if fail {
				http.Error(w, "interrupted", http.StatusInternalServerError)
				return
			}
			fmt.Fprint(w, `[]`)
		case "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.MaxRetries = 0
	collector.Now = func() time.Time { return cutoff }
	partial, err := collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err == nil || partial.Receipt.Status != StatusPartial {
		t.Fatalf("initial Capture = %v status=%s", err, partial.Receipt.Status)
	}
	partial.Receipt.NonAtomicDelta = nil
	for i := range partial.Receipt.Sources {
		partial.Receipt.Sources[i].CrawlStartedAt = ""
		partial.Receipt.Sources[i].CrawlEndedAt = ""
		if partial.Receipt.Sources[i].Name == "issues" {
			partial.Receipt.Sources[i].ClassifiedPullIdentities = nil
			partial.Receipt.Sources[i].ClassifiedPullChecksum = ""
		}
	}
	if err := validateCheckpoint(partial); err != nil {
		t.Fatalf("legacy checkpoint rejected: %v", err)
	}
	mu.Lock()
	issuesBefore, pullsBefore := calls["/repos/acme/widget/issues"], calls["/repos/acme/widget/pulls"]
	failDiscussions = false
	mu.Unlock()
	completed, err := collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff, Resume: &partial})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(completed); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	issuesAfter, pullsAfter := calls["/repos/acme/widget/issues"], calls["/repos/acme/widget/pulls"]
	mu.Unlock()
	if issuesAfter != issuesBefore || pullsAfter != pullsBefore {
		t.Fatalf("resume refetched complete endpoints: issues %d->%d pulls %d->%d", issuesBefore, issuesAfter, pullsBefore, pullsAfter)
	}
	if completed.Receipt.NonAtomicDelta == nil || completed.Receipt.NonAtomicDelta.IdentityBasis != identityBasisLegacyProjection {
		t.Fatalf("legacy reconciliation evidence = %+v", completed.Receipt.NonAtomicDelta)
	}
}

func captureReconciliationFixture(t *testing.T, mixed, dedicated []int64) (Corpus, error) {
	t.Helper()
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"delta-revision"}`)
		case "/repos/acme/widget/issues":
			writeRows(w, mixed, true)
		case "/repos/acme/widget/pulls":
			writeRows(w, dedicated, false)
		case "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(server.Close)
	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.Now = func() time.Time { return cutoff }
	return collector.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
}

func writeRows(w http.ResponseWriter, ids []int64, mixed bool) {
	rows := make([]map[string]any, 0, len(ids))
	for _, id := range ids {
		row := map[string]any{
			"id": id, "node_id": fmt.Sprintf("PR_%d", id), "number": int(id),
			"title": fmt.Sprintf("pull %d", id), "created_at": "2026-08-25T00:00:00Z",
		}
		if mixed {
			row["pull_request"] = map[string]any{}
		}
		rows = append(rows, row)
	}
	_ = json.NewEncoder(w).Encode(rows)
}

func assertIdentityIDs(t *testing.T, got []CrossEndpointIdentity, want ...int64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("identity count = %d, want %d: %+v", len(got), len(want), got)
	}
	for i := range want {
		if got[i].ID != want[i] {
			t.Fatalf("identity[%d] = %d, want %d", i, got[i].ID, want[i])
		}
	}
}
