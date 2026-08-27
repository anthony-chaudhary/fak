package studyforge

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

var errFixtureInterrupted = errors.New("fixture interrupted after durable checkpoint")

type checkpointFixture struct {
	server         *httptest.Server
	cutoff         time.Time
	mu             sync.Mutex
	calls          map[string]int
	liveRevision   string
	duplicatePage2 bool
	repeatPage1    bool
}

func newCheckpointFixture(t *testing.T) *checkpointFixture {
	t.Helper()
	fixture := &checkpointFixture{
		cutoff:       time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC),
		calls:        map[string]int{},
		liveRevision: "checkpoint-revision-a",
	}
	fixture.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fixture.mu.Lock()
		fixture.calls[r.URL.RequestURI()]++
		liveRevision := fixture.liveRevision
		duplicatePage2 := fixture.duplicatePage2
		repeatPage1 := fixture.repeatPage1
		fixture.mu.Unlock()
		w.Header().Set("X-GitHub-Request-Id", "checkpoint-fixture")
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprintf(w, `{"sha":%q}`, liveRevision)
		case "/repos/acme/widget/issues":
			page := r.URL.Query().Get("page")
			switch page {
			case "", "1":
				next := fixture.server.URL + "/repos/acme/widget/issues?page=2"
				if repeatPage1 {
					next = fixture.server.URL + r.URL.RequestURI()
				}
				w.Header().Set("Link", `<`+next+`>; rel="next"`)
				fmt.Fprint(w, `[{"id":1,"number":1,"title":"one","created_at":"2026-08-25T00:00:00Z"}]`)
			case "2":
				w.Header().Set("Link", `<`+fixture.server.URL+`/repos/acme/widget/issues?page=3>; rel="next"`)
				id := 2
				if duplicatePage2 {
					id = 1
				}
				fmt.Fprintf(w, `[{"id":%d,"number":2,"title":"two","created_at":"2026-08-25T00:00:00Z"}]`, id)
			case "3":
				fmt.Fprint(w, `[{"id":3,"number":3,"title":"three","created_at":"2026-08-25T00:00:00Z"}]`)
			default:
				http.NotFound(w, r)
			}
		case "/repos/acme/widget/pulls":
			fmt.Fprint(w, `[]`)
		case "/repos/acme/widget/discussions":
			fmt.Fprint(w, `[{"id":10,"number":10,"title":"talk","created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/releases":
			fmt.Fprint(w, `[{"id":11,"tag_name":"v1","created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/labels":
			fmt.Fprint(w, `[{"id":12,"name":"bug"}]`)
		case "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[{"id":13,"number":1,"title":"M1","created_at":"2026-08-25T00:00:00Z"}]`)
		default:
			http.NotFound(w, r)
		}
	}))
	t.Cleanup(fixture.server.Close)
	return fixture
}

func (f *checkpointFixture) collector() *Collector {
	collector := NewCollector(f.server.Client())
	collector.BaseURL = f.server.URL
	collector.Now = func() time.Time { return f.cutoff }
	return collector
}

func (f *checkpointFixture) callCount(uri string) int {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.calls[uri]
}

func (f *checkpointFixture) totalCalls() int {
	f.mu.Lock()
	defer f.mu.Unlock()
	total := 0
	for _, count := range f.calls {
		total += count
	}
	return total
}

func (f *checkpointFixture) setDuplicatePage2(value bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.duplicatePage2 = value
}

func (f *checkpointFixture) setRepeatPage1(value bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.repeatPage1 = value
}

func (f *checkpointFixture) setLiveRevision(revision string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.liveRevision = revision
}

func TestCaptureCheckpointsInsideSourceAndResumesByteEquivalently(t *testing.T) {
	fixture := newCheckpointFixture(t)
	dir := t.TempDir()
	resumedPath := filepath.Join(dir, "resumed.json")
	uninterruptedPath := filepath.Join(dir, "uninterrupted.json")

	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error {
			if err := Write(resumedPath, corpus); err != nil {
				return err
			}
			issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
			if len(issues.Pages) == 2 {
				return errFixtureInterrupted
			}
			return nil
		},
	})
	if !errors.Is(err, errFixtureInterrupted) {
		t.Fatalf("capture error = %v, want interruption", err)
	}
	partial, err := Read(resumedPath)
	if err != nil {
		t.Fatal(err)
	}
	issues, _ := sourceByName(partial.Receipt.Sources, "issues")
	if partial.Receipt.Status != StatusPartial || len(issues.Pages) != 2 || issues.Pages[1].Next == "" {
		t.Fatalf("durable checkpoint = %+v", partial.Receipt)
	}
	if err := Validate(partial); err == nil || !strings.Contains(err.Error(), "receipt status must be complete") {
		t.Fatalf("Validate(partial) error = %v", err)
	}

	page1URI := "/repos/acme/widget/issues?state=all&sort=created&direction=asc&per_page=100&page=1"
	page2URI := "/repos/acme/widget/issues?page=2"
	page1Before, page2Before := fixture.callCount(page1URI), fixture.callCount(page2URI)
	completed, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff, Resume: &partial,
		Checkpoint: func(corpus Corpus) error { return Write(resumedPath, corpus) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(completed); err != nil {
		t.Fatal(err)
	}
	if fixture.callCount(page1URI) != page1Before || fixture.callCount(page2URI) != page2Before {
		t.Fatalf("resume refetched accepted pages: page1 %d->%d page2 %d->%d", page1Before, fixture.callCount(page1URI), page2Before, fixture.callCount(page2URI))
	}

	uninterrupted, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error { return Write(uninterruptedPath, corpus) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(uninterrupted); err != nil {
		t.Fatal(err)
	}
	resumedBytes, err := os.ReadFile(resumedPath)
	if err != nil {
		t.Fatal(err)
	}
	uninterruptedBytes, err := os.ReadFile(uninterruptedPath)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(resumedBytes, uninterruptedBytes) {
		t.Fatalf("resumed output differs from uninterrupted output\nresumed:\n%s\nuninterrupted:\n%s", resumedBytes, uninterruptedBytes)
	}
}

func TestCaptureResumeRetainsCheckpointRevisionAndAPIProvenance(t *testing.T) {
	fixture := newCheckpointFixture(t)
	path := filepath.Join(t.TempDir(), "corpus.json")

	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error {
			if err := Write(path, corpus); err != nil {
				return err
			}
			return errFixtureInterrupted
		},
	})
	if !errors.Is(err, errFixtureInterrupted) {
		t.Fatalf("capture error = %v, want interruption", err)
	}
	partial, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	originalAPI := append([]APIReceipt(nil), partial.Receipt.API...)
	metadataURI := "/repos/acme/widget"
	revisionURI := "/repos/acme/widget/commits/main"
	metadataBefore := fixture.callCount(metadataURI)
	revisionBefore := fixture.callCount(revisionURI)
	if metadataBefore != 1 || revisionBefore != 1 {
		t.Fatalf("new capture revision resolution calls: metadata=%d revision=%d, want 1 each", metadataBefore, revisionBefore)
	}
	fixture.setLiveRevision("live-revision-b")

	completed, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff, Resume: &partial,
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := completed.Receipt.Revision, "checkpoint-revision-a"; got != want {
		t.Fatalf("resumed revision = %q, want checkpoint revision %q", got, want)
	}
	if !equalAPIReceipts(completed.Receipt.API, originalAPI) {
		t.Fatalf("resume changed API provenance:\n got: %+v\nwant: %+v", completed.Receipt.API, originalAPI)
	}
	if got := fixture.callCount(metadataURI); got != metadataBefore {
		t.Fatalf("resume called repository metadata endpoint: %d->%d", metadataBefore, got)
	}
	if got := fixture.callCount(revisionURI); got != revisionBefore {
		t.Fatalf("resume called revision endpoint: %d->%d", revisionBefore, got)
	}
}

func equalAPIReceipts(a, b []APIReceipt) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestCaptureBoundsCheckpointWriteAmplification(t *testing.T) {
	fixture := newCheckpointFixture(t)
	writes := 0
	completed, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff, CheckpointEvery: 3,
		Checkpoint: func(corpus Corpus) error {
			writes++
			return validateCheckpoint(corpus)
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(completed); err != nil {
		t.Fatal(err)
	}
	// Eight accepted pages produce two interval writes and one forced terminal write.
	if writes != 3 {
		t.Fatalf("checkpoint writes = %d, want 3", writes)
	}
}

func TestCaptureAtomicRenameFailurePreservesLastCheckpoint(t *testing.T) {
	fixture := newCheckpointFixture(t)
	path := filepath.Join(t.TempDir(), "corpus.json")
	renameErr := errors.New("simulated crash before atomic rename")

	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error {
			issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
			if len(issues.Pages) == 2 {
				return writeCorpus(path, corpus, func(string, string) error { return renameErr })
			}
			return Write(path, corpus)
		},
	})
	if !errors.Is(err, renameErr) {
		t.Fatalf("capture error = %v, want rename failure", err)
	}
	partial, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	issues, _ := sourceByName(partial.Receipt.Sources, "issues")
	if len(issues.Pages) != 1 || len(recordsForSource(partial.Records, "issues")) != 1 {
		t.Fatalf("atomic target advanced past last rename: pages=%d records=%d", len(issues.Pages), len(recordsForSource(partial.Records, "issues")))
	}

	page2URI := "/repos/acme/widget/issues?page=2"
	before := fixture.callCount(page2URI)
	completed, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff, Resume: &partial,
		Checkpoint: func(corpus Corpus) error { return Write(path, corpus) },
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(completed); err != nil {
		t.Fatal(err)
	}
	if fixture.callCount(page2URI) != before+1 {
		t.Fatalf("resume should redo exactly the unrenamed page: calls %d->%d", before, fixture.callCount(page2URI))
	}
}

func TestReadRejectsCorruptedCheckpointPageChain(t *testing.T) {
	fixture := newCheckpointFixture(t)
	path := filepath.Join(t.TempDir(), "corpus.json")
	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error {
			if err := Write(path, corpus); err != nil {
				return err
			}
			return errFixtureInterrupted
		},
	})
	if !errors.Is(err, errFixtureInterrupted) {
		t.Fatal(err)
	}
	partial, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	partial.Receipt.Sources[0].Pages[0].Checksum = "sha256:" + strings.Repeat("f", 64)
	data, err := json.MarshalIndent(partial, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(data, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Read(path); err == nil || !strings.Contains(err.Error(), "page chain checksum mismatch") {
		t.Fatalf("Read error = %v, want page chain checksum mismatch", err)
	}
}

func TestCaptureRejectsResumeIdentityAndSourceOrderMismatchBeforeRequests(t *testing.T) {
	fixture := newCheckpointFixture(t)
	base, err := fixture.collector().Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name      string
		mutate    func(*Corpus)
		owner     string
		repo      string
		cutoff    time.Time
		baseURL   string
		wantError string
	}{
		{name: "corpus schema", mutate: func(c *Corpus) { c.Schema = "wrong" }, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "schema"},
		{name: "receipt schema", mutate: func(c *Corpus) { c.Receipt.Schema = "wrong" }, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "receipt schema"},
		{name: "repository", mutate: func(*Corpus) {}, owner: "acme", repo: "other", cutoff: fixture.cutoff, wantError: "repository mismatch"},
		{name: "cutoff", mutate: func(*Corpus) {}, owner: "acme", repo: "widget", cutoff: fixture.cutoff.Add(time.Second), wantError: "cutoff mismatch"},
		{name: "API base", mutate: func(*Corpus) {}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, baseURL: fixture.server.URL + "/v2", wantError: "API base mismatch"},
		{name: "missing repository receipt", mutate: func(c *Corpus) {
			c.Receipt.API = c.Receipt.API[1:]
		}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "repository API receipt is required"},
		{name: "missing revision receipt", mutate: func(c *Corpus) {
			c.Receipt.API = c.Receipt.API[:1]
		}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "revision API receipt is required"},
		{name: "contradictory repository receipt", mutate: func(c *Corpus) {
			c.Receipt.API[0].URL = fixture.server.URL + "/repos/acme/other"
		}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "repository API receipt contradicts checkpoint repository"},
		{name: "contradictory revision receipt", mutate: func(c *Corpus) {
			c.Receipt.API[1].URL = fixture.server.URL + "/repos/acme/other/commits/main"
		}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "revision API receipt contradicts checkpoint repository"},
		{name: "source order", mutate: func(c *Corpus) {
			c.Receipt.Sources[0], c.Receipt.Sources[1] = c.Receipt.Sources[1], c.Receipt.Sources[0]
		}, owner: "acme", repo: "widget", cutoff: fixture.cutoff, wantError: "source order mismatch"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			requestsBefore := fixture.totalCalls()
			resume := cloneCorpus(base)
			tt.mutate(&resume)
			collector := fixture.collector()
			if tt.baseURL != "" {
				collector.BaseURL = tt.baseURL
			}
			_, err := collector.Capture(context.Background(), CaptureRequest{Owner: tt.owner, Repository: tt.repo, Cutoff: tt.cutoff, Resume: &resume})
			if err == nil || !strings.Contains(err.Error(), tt.wantError) {
				t.Fatalf("Capture error = %v, want %q", err, tt.wantError)
			}
			if got := fixture.totalCalls(); got != requestsBefore {
				t.Fatalf("resume mismatch made network request: %d->%d", requestsBefore, got)
			}
		})
	}
}

func TestCaptureRejectsRepeatedNextCursor(t *testing.T) {
	fixture := newCheckpointFixture(t)
	fixture.setRepeatPage1(true)
	corpus, err := fixture.collector().Capture(context.Background(), CaptureRequest{Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff})
	if err == nil || !strings.Contains(err.Error(), "repeated current next cursor") {
		t.Fatalf("Capture error = %v", err)
	}
	issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
	if len(issues.Pages) != 0 || len(recordsForSource(corpus.Records, "issues")) != 0 {
		t.Fatalf("repeated cursor page was accepted: pages=%d records=%d", len(issues.Pages), len(recordsForSource(corpus.Records, "issues")))
	}
}

func TestCaptureRejectsDuplicateRecordAcrossResumeBoundary(t *testing.T) {
	fixture := newCheckpointFixture(t)
	path := filepath.Join(t.TempDir(), "corpus.json")
	_, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff,
		Checkpoint: func(corpus Corpus) error {
			if err := Write(path, corpus); err != nil {
				return err
			}
			return errFixtureInterrupted
		},
	})
	if !errors.Is(err, errFixtureInterrupted) {
		t.Fatal(err)
	}
	partial, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	fixture.setDuplicatePage2(true)
	corpus, err := fixture.collector().Capture(context.Background(), CaptureRequest{
		Owner: "acme", Repository: "widget", Cutoff: fixture.cutoff, Resume: &partial,
		Checkpoint: func(corpus Corpus) error { return Write(path, corpus) },
	})
	if err == nil || !strings.Contains(err.Error(), "duplicate issues id 1 across page boundary") {
		t.Fatalf("Capture error = %v", err)
	}
	issues, _ := sourceByName(corpus.Receipt.Sources, "issues")
	issueRecords := recordsForSource(corpus.Records, "issues")
	if len(issues.Pages) != 1 || len(issueRecords) != 1 || issueRecords[0].ID != 1 {
		t.Fatalf("duplicate crossed checkpoint boundary: pages=%d records=%+v", len(issues.Pages), issueRecords)
	}
	stored, err := Read(path)
	if err != nil {
		t.Fatal(err)
	}
	if got := recordsForSource(stored.Records, "issues"); len(got) != 1 || got[0].ID != 1 {
		t.Fatalf("stored checkpoint contains duplicate: %+v", got)
	}
}
