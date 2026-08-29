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

func TestCapturePaginatesClassifiesCutsOffAndIsDeterministic(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("X-GitHub-Request-Id", "fixture")
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits":
			fmt.Fprint(w, `[{"sha":"abc123"}]`)
		case "/repos/acme/widget/issues":
			if r.URL.Query().Get("page") == "1" {
				w.Header().Set("Link", `<`+server.URL+`/repos/acme/widget/issues?page=2>; rel="next"`)
				fmt.Fprint(w, `[{"id":2,"number":2,"title":"PR ref","created_at":"2026-08-25T00:00:00Z","pull_request":{}},{"id":1,"number":1,"title":"issue","labels":[{"name":"z"},{"name":"a"}],"created_at":"2026-08-25T00:00:00Z"}]`)
			} else {
				fmt.Fprint(w, `[{"id":3,"number":3,"title":"future","created_at":"2026-08-27T00:00:00Z"}]`)
			}
		case "/repos/acme/widget/pulls":
			fmt.Fprint(w, `[{"id":2,"number":2,"title":"pull","created_at":"2026-08-25T00:00:00Z","base":{"ref":"main","sha":"b"},"head":{"ref":"work","sha":"h"}}]`)
		case "/repos/acme/widget/discussions":
			fmt.Fprint(w, `[{"id":4,"number":4,"title":"talk","created_at":"2026-08-25T00:00:00Z","category":{"name":"Ideas"}}]`)
		case "/repos/acme/widget/releases":
			fmt.Fprint(w, `[{"id":5,"tag_name":"v1","created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/labels":
			fmt.Fprint(w, `[{"id":6,"name":"bug"}]`)
		case "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[{"id":7,"number":1,"title":"M1","created_at":"2026-08-25T00:00:00Z"}]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()
	c := NewCollector(server.Client())
	c.BaseURL = server.URL
	c.Now = func() time.Time { return cutoff }
	got, err := c.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err != nil {
		t.Fatal(err)
	}
	if err = Validate(got); err != nil {
		t.Fatal(err)
	}
	if got.Receipt.Status != StatusComplete || len(got.Records) != 6 { //boundarylint:ignore CHANGE_DETECTOR_TEST the collector fixture supplies six records and requires complete preservation
		t.Fatalf("status=%s records=%d", got.Receipt.Status, len(got.Records))
	}
	issues, _ := sourceByName(got.Receipt.Sources, "issues")
	if len(issues.Pages) != 2 || issues.ClassifiedPullCount != 1 || issues.CutoffExcludedCount != 1 {
		t.Fatalf("issues receipt: %+v", issues)
	}
	if strings.Join(got.Records[0].Labels, ",") != "a,z" {
		t.Fatalf("labels not normalized: %v", got.Records[0].Labels)
	}
	again, err := c.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err != nil {
		t.Fatal(err)
	}
	if got.Receipt.IndexChecksum != again.Receipt.IndexChecksum {
		t.Fatal("checksum changed")
	}
}

func TestCollectorDefaultsUseTimeoutBoundClients(t *testing.T) {
	fromConstructor := NewCollector(nil)
	if got := fromConstructor.Client.Timeout; got != defaultHTTPTimeout || got <= 0 {
		t.Fatalf("NewCollector default timeout = %s, want %s", got, defaultHTTPTimeout)
	}
	var zero Collector
	zero.defaults()
	if got := zero.Client.Timeout; got != defaultHTTPTimeout || got <= 0 {
		t.Fatalf("zero Collector default timeout = %s, want %s", got, defaultHTTPTimeout)
	}
}

func TestCapturePinsDefaultBranchRevisionAtInclusiveCutoff(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	const historicalRevision = "revision-at-cutoff"
	var liveHeadCalls int
	var revisionRequestURI string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			liveHeadCalls++
			fmt.Fprint(w, `{"sha":"post-cutoff-live-head"}`)
		case "/repos/acme/widget/commits":
			revisionRequestURI = r.URL.RequestURI()
			w.Header().Set("X-GitHub-Request-Id", "cutoff-revision-request")
			fmt.Fprint(w, `[{"sha":"`+historicalRevision+`","commit":{"committer":{"date":"2026-08-26T12:00:00Z"}}}]`)
		default:
			fmt.Fprint(w, `[]`)
		}
	}))
	defer server.Close()

	collector := NewCollector(server.Client())
	collector.BaseURL = server.URL
	collector.Now = func() time.Time { return cutoff.Add(time.Hour) }
	corpus, err := collector.Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: cutoff,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := corpus.Receipt.Revision; got != historicalRevision {
		t.Fatalf("revision = %q, want historical cutoff revision %q", got, historicalRevision)
	}
	if liveHeadCalls != 0 {
		t.Fatalf("capture resolved post-cutoff live head %d time(s)", liveHeadCalls)
	}
	const wantRequestURI = "/repos/acme/widget/commits?per_page=1&sha=main&until=2026-08-26T12%3A00%3A00Z"
	if revisionRequestURI != wantRequestURI {
		t.Fatalf("revision request = %q, want %q", revisionRequestURI, wantRequestURI)
	}
	if len(corpus.Receipt.API) != 2 {
		t.Fatalf("API receipts = %+v", corpus.Receipt.API)
	}
	receipt := corpus.Receipt.API[1]
	if receipt.Purpose != "revision" || receipt.URL != server.URL+wantRequestURI || receipt.RequestID != "cutoff-revision-request" {
		t.Fatalf("revision API evidence = %+v", receipt)
	}
}

func TestCaptureRetriesAndResumesFirstMissingPage(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	var mu sync.Mutex
	calls := map[string]int{}
	failPage2 := true
	var server *httptest.Server
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		calls[r.URL.RequestURI()]++
		mu.Unlock()
		switch r.URL.Path {
		case "/repos/o/r":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/o/r/commits":
			fmt.Fprint(w, `[{"sha":"rev"}]`)
		case "/repos/o/r/issues":
			if r.URL.Query().Get("page") == "1" {
				w.Header().Set("Link", `<`+server.URL+`/repos/o/r/issues?page=2>; rel="next"`)
				fmt.Fprint(w, `[{"id":1,"created_at":"2026-08-25T00:00:00Z"}]`)
			} else if failPage2 {
				http.Error(w, "temporary", 500)
			} else {
				fmt.Fprint(w, `[{"id":2,"created_at":"2026-08-25T00:00:00Z"}]`)
			}
		default:
			fmt.Fprint(w, `[]`)
		}
	}))
	defer server.Close()
	c := NewCollector(server.Client())
	c.BaseURL = server.URL
	c.MaxRetries = 1
	c.RetryWait = func(context.Context, time.Duration) error { return nil }
	c.Now = func() time.Time { return cutoff }
	partial, err := c.Capture(context.Background(), CaptureRequest{Owner: "o", Repository: "r", Cutoff: cutoff})
	if err == nil || partial.Receipt.Status != StatusPartial {
		t.Fatalf("expected partial: %v %+v", err, partial.Receipt)
	}
	failPage2 = false
	done, err := c.Capture(context.Background(), CaptureRequest{Owner: "o", Repository: "r", Resume: &partial})
	if err != nil {
		t.Fatal(err)
	}
	if err = Validate(done); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	page1 := calls["/repos/o/r/issues?state=all&sort=created&direction=asc&per_page=100&page=1"]
	mu.Unlock()
	if page1 != 1 {
		t.Fatalf("resume refetched page 1: calls=%d", page1)
	}
}

func TestCaptureAcceptsDiscussionsDisabledAsTerminalEmpty(t *testing.T) {
	cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits":
			fmt.Fprint(w, `[{"sha":"abc123"}]`)
		case "/repos/acme/widget/discussions":
			w.Header().Set("X-GitHub-Request-Id", "disabled-fixture")
			w.WriteHeader(http.StatusGone)
			fmt.Fprint(w, discussionsDisabledPayload)
		default:
			fmt.Fprint(w, `[]`)
		}
	}))
	defer server.Close()

	c := NewCollector(server.Client())
	c.BaseURL = server.URL
	c.Now = func() time.Time { return cutoff }
	got, err := c.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(got); err != nil {
		t.Fatal(err)
	}
	if got.Receipt.Status != StatusComplete {
		t.Fatalf("receipt status = %q", got.Receipt.Status)
	}
	discussions, ok := sourceByName(got.Receipt.Sources, "discussions")
	if !ok {
		t.Fatal("missing discussions receipt")
	}
	if discussions.Status != StatusComplete || discussions.Failure != "" || discussions.FetchedCount != 0 || discussions.NormalizedCount != 0 {
		t.Fatalf("discussions receipt = %+v", discussions)
	}
	if len(discussions.Pages) != 1 {
		t.Fatalf("discussion pages = %d", len(discussions.Pages))
	}
	page := discussions.Pages[0]
	if page.StatusCode != http.StatusGone || page.ItemCount != 0 || page.Checksum != digest([]byte(discussionsDisabledPayload)) || page.RequestID != "disabled-fixture" {
		t.Fatalf("disabled discussion page receipt = %+v", page)
	}

	tampered := cloneCorpus(got)
	tamperedDiscussions, _ := sourceByName(tampered.Receipt.Sources, "discussions")
	tamperedDiscussions.Pages[0].Checksum = digest([]byte(`{"message":"Gone"}`))
	upsertSource(&tampered.Receipt.Sources, tamperedDiscussions)
	refreshChecksums(&tampered)
	if Validate(tampered) == nil {
		t.Fatal("accepted a 410 receipt without the documented disabled-discussions evidence")
	}
}

func TestCaptureRejectsOtherDiscussionsGoneResponses(t *testing.T) {
	cases := map[string]string{
		"unrelated":     `{"message":"Gone","documentation_url":"https://docs.github.com/rest","status":"410"}`,
		"wrong message": `{"message":"Discussions are disabled","documentation_url":"https://docs.github.com/rest/repos/discussions#list-discussions-for-repository","status":"410"}`,
		"extra field":   `{"message":"Discussions are disabled for this repo","documentation_url":"https://docs.github.com/rest/repos/discussions#list-discussions-for-repository","status":"410","extra":true}`,
	}
	for name, body := range cases {
		t.Run(name, func(t *testing.T) {
			cutoff := time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC)
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				switch r.URL.Path {
				case "/repos/acme/widget":
					fmt.Fprint(w, `{"default_branch":"main"}`)
				case "/repos/acme/widget/commits":
					fmt.Fprint(w, `[{"sha":"abc123"}]`)
				case "/repos/acme/widget/discussions":
					w.WriteHeader(http.StatusGone)
					fmt.Fprint(w, body)
				default:
					fmt.Fprint(w, `[]`)
				}
			}))
			defer server.Close()

			c := NewCollector(server.Client())
			c.BaseURL = server.URL
			c.Now = func() time.Time { return cutoff }
			got, err := c.Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: cutoff})
			if err == nil || !strings.Contains(err.Error(), "discussions: GET") || !strings.Contains(err.Error(), "HTTP 410") {
				t.Fatalf("error = %v", err)
			}
			discussions, ok := sourceByName(got.Receipt.Sources, "discussions")
			if !ok || discussions.Status != StatusFailed || !strings.Contains(discussions.Failure, "HTTP 410") {
				t.Fatalf("discussions receipt = %+v", discussions)
			}
			if got.Receipt.Status == StatusComplete {
				t.Fatalf("receipt status = %q", got.Receipt.Status)
			}
		})
	}
}

func TestValidateRejectsCorruptReceipts(t *testing.T) {
	base := Corpus{Schema: CorpusSchema, Receipt: Receipt{Schema: ReceiptSchema, Repository: "o/r", Revision: "x", Cutoff: "2026-08-26T00:00:00Z", Status: StatusComplete}, Records: []Record{{Source: "issues", Kind: "issue", ID: 1}}}
	for _, n := range SourceNames {
		base.Receipt.Sources = append(base.Receipt.Sources, SourceReceipt{Name: n, Status: StatusComplete, Pages: []PageReceipt{{Number: 1}}, FetchedCount: len(recordsForSource(base.Records, n)), Checksum: recordDigest(recordsForSource(base.Records, n))})
	}
	refreshChecksums(&base)
	cases := map[string]func(*Corpus){"duplicate id": func(c *Corpus) { c.Records = append(c.Records, c.Records[0]); refreshChecksums(c) }, "mixed row": func(c *Corpus) { c.Records[0].Kind = "pull"; refreshChecksums(c) }, "count mismatch": func(c *Corpus) { c.Receipt.Sources[0].FetchedCount++ }, "partial marked complete": func(c *Corpus) { c.Receipt.Sources[0].Status = StatusPartial }, "missing page": func(c *Corpus) { c.Receipt.Sources[0].Pages[0].Number = 2 }, "checksum": func(c *Corpus) { c.Receipt.IndexChecksum = "sha256:tampered" }}
	for name, mutate := range cases {
		t.Run(name, func(t *testing.T) {
			c := cloneCorpus(base)
			mutate(&c)
			if Validate(c) == nil {
				t.Fatal("expected validation error")
			}
		})
	}
}
