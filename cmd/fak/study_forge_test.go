package main

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/studyforge"
)

func TestStudyForgeCaptureCLIResumesWithinSourceCheckpoint(t *testing.T) {
	var (
		mu        sync.Mutex
		page1Hits int
		failPage2 = true
		server    *httptest.Server
	)
	server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/repos/acme/widget":
			fmt.Fprint(w, `{"default_branch":"main"}`)
		case "/repos/acme/widget/commits/main":
			fmt.Fprint(w, `{"sha":"cli-revision"}`)
		case "/repos/acme/widget/issues":
			if r.URL.Query().Get("page") == "2" {
				mu.Lock()
				fail := failPage2
				mu.Unlock()
				if fail {
					http.Error(w, "stop after checkpoint", http.StatusInternalServerError)
					return
				}
				fmt.Fprint(w, `[{"id":2,"created_at":"2026-08-25T00:00:00Z"}]`)
				return
			}
			mu.Lock()
			page1Hits++
			mu.Unlock()
			w.Header().Set("Link", `<`+server.URL+`/repos/acme/widget/issues?page=2>; rel="next"`)
			fmt.Fprint(w, `[{"id":1,"created_at":"2026-08-25T00:00:00Z"}]`)
		case "/repos/acme/widget/pulls", "/repos/acme/widget/discussions", "/repos/acme/widget/releases", "/repos/acme/widget/labels", "/repos/acme/widget/milestones":
			fmt.Fprint(w, `[]`)
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	out := filepath.Join(t.TempDir(), "corpus.json")
	args := []string{
		"capture", "--repository", "acme/widget", "--cutoff", "2026-08-26T12:00:00Z",
		"--out", out, "--base-url", server.URL, "--retries", "0",
	}
	var stdout, stderr bytes.Buffer
	if code := runStudyForge(&stdout, &stderr, args); code != 1 {
		t.Fatalf("first capture exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	partial, err := studyforge.Read(out)
	if err != nil {
		t.Fatal(err)
	}
	if partial.Receipt.Status != studyforge.StatusPartial || len(partial.Receipt.Sources[0].Pages) != 1 {
		t.Fatalf("partial checkpoint = %+v", partial.Receipt)
	}

	mu.Lock()
	failPage2 = false
	mu.Unlock()
	stdout.Reset()
	stderr.Reset()
	if code := runStudyForge(&stdout, &stderr, append(args, "--resume")); code != 0 {
		t.Fatalf("resume exit=%d stdout=%q stderr=%q", code, stdout.String(), stderr.String())
	}
	mu.Lock()
	gotPage1Hits := page1Hits
	mu.Unlock()
	if gotPage1Hits != 1 {
		t.Fatalf("resume refetched page 1: hits=%d", gotPage1Hits)
	}
	completed, err := studyforge.Read(out)
	if err != nil {
		t.Fatal(err)
	}
	if err := studyforge.Validate(completed); err != nil {
		t.Fatal(err)
	}
}
