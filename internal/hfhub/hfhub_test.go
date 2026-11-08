package hfhub

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

type unsolicitedPartialTransport func(*http.Request) (*http.Response, error)

func (f unsolicitedPartialTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return f(r)
}

// fak-test:runtime fast est=1ms lane=default
func TestDownloadRejectsUnsolicitedPartial(t *testing.T) {
	for _, existing := range []bool{false, true} {
		t.Run(fmt.Sprintf("existing=%v", existing), func(t *testing.T) {
			dst := filepath.Join(t.TempDir(), "object")
			part := dst + partSuffix
			if existing {
				if err := os.WriteFile(dst, []byte("existing complete object"), 0o600); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(part, []byte("retained partial"), 0o600); err != nil {
					t.Fatal(err)
				}
			}
			calls := 0
			client := &Client{HTTP: &http.Client{Transport: unsolicitedPartialTransport(func(r *http.Request) (*http.Response, error) {
				calls++
				if got := r.Header.Get("Range"); got != "" {
					t.Fatalf("full-object request sent Range %q", got)
				}
				return &http.Response{StatusCode: http.StatusPartialContent, Header: http.Header{"Content-Range": {"bytes 7-12/13"}}, Body: io.NopCloser(strings.NewReader("suffix"))}, nil
			})}}
			want := "" // No trusted digest exists to catch a truncated object later.
			if err := client.attemptTransfer(context.Background(), "https://fixture.invalid/object", part, dst, &want, nil); err == nil {
				t.Fatal("unsolicited partial response was accepted")
			}
			if calls != 1 {
				t.Fatalf("attempt made %d requests", calls)
			}
			for path, content := range map[string]string{dst: "existing complete object", part: "retained partial"} {
				got, err := os.ReadFile(path)
				if existing {
					if err != nil || string(got) != content {
						t.Fatalf("refusal changed existing file %s: %q, %v", path, got, err)
					}
				} else if !errors.Is(err, os.ErrNotExist) {
					t.Fatalf("refusal created file %s: %v", path, err)
				}
			}
		})
	}
}

func TestParseURI(t *testing.T) {
	cases := []struct {
		uri                  string
		repo, revision, file string
		wantErr              bool
	}{
		{"hf://mradermacher/Qwen2.5-1.5B-GGUF/model.Q8_0.gguf", "mradermacher/Qwen2.5-1.5B-GGUF", "main", "model.Q8_0.gguf", false},
		{"hf://meta-llama/Llama-3.1-8B@main/orig/consolidated.00.pth", "meta-llama/Llama-3.1-8B", "main", "orig/consolidated.00.pth", false},
		{"hf://Qwen/Qwen2.5-1.5B-Instruct@v1.0/config.json", "Qwen/Qwen2.5-1.5B-Instruct", "v1.0", "config.json", false},
		{"hf://owner/repo/a/b/c.gguf", "owner/repo", "main", "a/b/c.gguf", false},
		{"hf://owner/repo", "owner/repo", "main", "", false},
		{"hf://owner/repo@v1", "owner/repo", "v1", "", false},
		{"https://huggingface.co/x/y/z", "", "", "", true}, // not hf://
		{"hf://owner//file", "", "", "", true},             // empty repo
		{"hf://owner/repo@/file", "", "", "", true},        // empty revision
		{"hf://owner/repo/", "", "", "", true},             // trailing slash, no file
	}
	for _, tc := range cases {
		got, err := ParseURI(tc.uri)
		if tc.wantErr {
			if err == nil {
				t.Errorf("ParseURI(%q): want error, got %+v", tc.uri, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseURI(%q): unexpected error: %v", tc.uri, err)
			continue
		}
		if got.Repo != tc.repo || got.Revision != tc.revision || got.File != tc.file {
			t.Errorf("ParseURI(%q) = %+v, want {%s %s %s}", tc.uri, got, tc.repo, tc.revision, tc.file)
		}
	}
}

func TestResolveURL(t *testing.T) {
	r := Ref{Repo: "owner/repo", Revision: "main", File: "a/b.gguf"}
	want := "https://huggingface.co/owner/repo/resolve/main/a/b.gguf"
	if got := r.ResolveURL(DefaultBaseURL); got != want {
		t.Errorf("ResolveURL = %q, want %q", got, want)
	}
	// trailing slash on base must not double up
	if got := r.ResolveURL("https://example.test/"); got != "https://example.test/owner/repo/resolve/main/a/b.gguf" {
		t.Errorf("ResolveURL trailing-slash = %q", got)
	}
}

// hubStub serves the resolve endpoint for one file, stamping the LFS sha256 on
// X-Linked-Etag, and counts GETs so a cache hit is observable.
type hubStub struct {
	body     []byte
	etag     string // value stamped on X-Linked-Etag (verbatim)
	gets     atomic.Int32
	heads    atomic.Int32
	lastAuth atomic.Value // string
}

func (h *hubStub) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		h.lastAuth.Store(r.Header.Get("Authorization"))
		if h.etag != "" {
			w.Header().Set("X-Linked-Etag", h.etag)
		}
		if r.Method == http.MethodHead {
			h.heads.Add(1)
			w.WriteHeader(http.StatusOK)
			return
		}
		h.gets.Add(1)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(h.body)
	}
}

func sha256hex(b []byte) string {
	s := sha256.Sum256(b)
	return hex.EncodeToString(s[:])
}

func TestDownloadVerifiesAndCaches(t *testing.T) {
	body := []byte("GGUF\x00 pretend weights")
	stub := &hubStub{body: body, etag: `"` + sha256hex(body) + `"`}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	ref := Ref{Repo: "owner/repo", Revision: "main", File: "model.gguf"}

	path1, err := c.Download(context.Background(), ref, nil)
	if err != nil {
		t.Fatalf("Download: %v", err)
	}
	got, err := os.ReadFile(path1)
	if err != nil {
		t.Fatalf("read cached file: %v", err)
	}
	if string(got) != string(body) {
		t.Fatalf("cached content = %q, want %q", got, body)
	}
	if n := stub.gets.Load(); n != 1 {
		t.Fatalf("first download: got %d GETs, want 1", n)
	}

	// Second call must be a pure cache hit — no further GET.
	path2, err := c.Download(context.Background(), ref, nil)
	if err != nil {
		t.Fatalf("second Download: %v", err)
	}
	if path2 != path1 {
		t.Fatalf("cache path changed: %q != %q", path2, path1)
	}
	if n := stub.gets.Load(); n != 1 {
		t.Fatalf("cache hit refetched: got %d GETs, want 1", n)
	}
}

func TestDownloadRefusesCorruptedVerifiedCache(t *testing.T) {
	body := []byte("verified weights")
	stub := &hubStub{body: body, etag: `"` + sha256hex(body) + `"`}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	ref := Ref{Repo: "owner/repo", Revision: "pinned", File: "model.safetensors"}
	path, err := c.Download(context.Background(), ref, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte("corrupted cache"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, err = c.Download(context.Background(), ref, nil)
	if err == nil || !strings.Contains(err.Error(), "cached sha256 mismatch") {
		t.Fatalf("error = %v, want cached sha256 mismatch", err)
	}
}

func TestDownloadSHA256Mismatch(t *testing.T) {
	body := []byte("real bytes")
	wrong := sha256hex([]byte("different bytes"))
	stub := &hubStub{body: body, etag: `"` + wrong + `"`}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	ref := Ref{Repo: "owner/repo", Revision: "main", File: "model.gguf"}

	_, err := c.Download(context.Background(), ref, nil)
	if err == nil {
		t.Fatal("Download: want sha256 mismatch error, got nil")
	}
	if !strings.Contains(err.Error(), "sha256 mismatch") {
		t.Fatalf("error = %v, want sha256 mismatch", err)
	}
	// A failed verification must leave no cached (partial) file behind.
	if _, statErr := os.Stat(c.CachePath(ref)); !os.IsNotExist(statErr) {
		t.Fatalf("cache file present after mismatch: %v", statErr)
	}
}

func TestDownloadSendsToken(t *testing.T) {
	body := []byte("gated weights")
	stub := &hubStub{body: body, etag: `"` + sha256hex(body) + `"`}
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), Token: "testtok"}
	ref := Ref{Repo: "meta-llama/gated", Revision: "main", File: "model.gguf"}
	if _, err := c.Download(context.Background(), ref, nil); err != nil {
		t.Fatalf("Download: %v", err)
	}
	if auth, _ := stub.lastAuth.Load().(string); auth != "Bearer testtok" {
		t.Fatalf("Authorization = %q, want %q", auth, "Bearer testtok")
	}
}

func TestNewClientResolvesDotEnvToken(t *testing.T) {
	// A .env in the working directory supplies the HF token when nothing is
	// exported — the acceptance box "Respects .env HF token" (issue #294).
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("HF_TOKEN=from-dotenv\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	t.Setenv("HF_TOKEN", "") // exported-but-blank must not shadow .env
	t.Setenv("HUGGING_FACE_HUB_TOKEN", "")

	if got := NewClient().Token; got != "from-dotenv" {
		t.Fatalf("NewClient().Token = %q, want %q (from .env)", got, "from-dotenv")
	}
}

func TestNewClientEnvWinsOverDotEnv(t *testing.T) {
	// A real exported HF_TOKEN always wins over a .env value — os-env is the
	// highest-priority source, so .env can never shadow an operator's export.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, ".env"), []byte("HF_TOKEN=from-dotenv\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Chdir(dir)
	t.Setenv("HF_TOKEN", "from-export")

	if got := NewClient().Token; got != "from-export" {
		t.Fatalf("NewClient().Token = %q, want %q (export wins)", got, "from-export")
	}
}

func TestFetchURIRepoOnlyResolvesUnambiguousGGUF(t *testing.T) {
	body := []byte("GGUF\x00 selected by repo-only URI")
	sum := sha256hex(body)
	var infoGets atomic.Int32
	var downloaded atomic.Value
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/api/models/owner/repo/revision/main":
			infoGets.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"README.md"},{"rfilename":"model.Q8_0.gguf"}]}`))
		case "/owner/repo/resolve/main/model.Q8_0.gguf":
			w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
			if r.Method == http.MethodHead {
				return
			}
			downloaded.Store(r.URL.Path)
			_, _ = w.Write(body)
		default:
			http.NotFound(w, r)
		}
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	path, err := c.FetchURI(context.Background(), "hf://owner/repo", nil)
	if err != nil {
		t.Fatalf("FetchURI repo-only: %v", err)
	}
	got, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read cached file: %v", err)
	}
	if string(got) != string(body) {
		t.Fatalf("cached content = %q, want %q", got, body)
	}
	if n := infoGets.Load(); n != 1 {
		t.Fatalf("repo info GETs = %d, want 1", n)
	}
	if p, _ := downloaded.Load().(string); p != "/owner/repo/resolve/main/model.Q8_0.gguf" {
		t.Fatalf("download path = %q", p)
	}
}

func TestResolveRepoFileRefusesAmbiguousGGUF(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/repo/revision/main" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"a.gguf"},{"rfilename":"b.gguf"}]}`))
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	_, err := c.ResolveRepoFile(context.Background(), Ref{Repo: "owner/repo", Revision: "main"}, nil)
	if err == nil {
		t.Fatal("ResolveRepoFile: want ambiguous artifact error, got nil")
	}
	if !strings.Contains(err.Error(), "ambiguous GGUF artifacts") {
		t.Fatalf("error = %v, want ambiguous GGUF artifacts", err)
	}
}

func TestResolveRepoFileFallsBackToSingleSafetensors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/repo/revision/main" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"config.json"},{"rfilename":"model.safetensors"}]}`))
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	got, err := c.ResolveRepoFile(context.Background(), Ref{Repo: "owner/repo", Revision: "main"}, nil)
	if err != nil {
		t.Fatalf("ResolveRepoFile: %v", err)
	}
	if got.File != "model.safetensors" {
		t.Fatalf("resolved file = %q, want model.safetensors", got.File)
	}
}

func TestResolveRepoFileResolvesShardedSafetensors(t *testing.T) {
	// A sharded repo (a model.safetensors.index.json plus model-0000N shards, no
	// single model.safetensors) resolves to the index — the entry point FetchURI
	// expands to the full shard set (issue #294's hf://meta-llama/Llama-3.1-8B).
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/models/owner/repo/revision/main" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"config.json"},` +
			`{"rfilename":"model.safetensors.index.json"},` +
			`{"rfilename":"model-00001-of-00002.safetensors"},` +
			`{"rfilename":"model-00002-of-00002.safetensors"}]}`))
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	got, err := c.ResolveRepoFile(context.Background(), Ref{Repo: "owner/repo", Revision: "main"}, nil)
	if err != nil {
		t.Fatalf("ResolveRepoFile sharded: %v", err)
	}
	if got.File != "model.safetensors.index.json" {
		t.Fatalf("resolved file = %q, want model.safetensors.index.json", got.File)
	}
}

func TestFetchURIShardedSafetensorsFansOut(t *testing.T) {
	// meta-llama/Llama-3.1-8B-shaped repo: a bare hf://owner/repo must resolve to
	// the sharded index, then download the index plus every shard its weight_map
	// names into ONE cache directory (the unit model.LoadSafetensorsDir consumes).
	// Every file is verified against its X-Linked-Etag sha256 (issue #294).
	shard1 := []byte("shard-one-pretend-weights")
	shard2 := []byte("shard-two-pretend-weights")
	indexJSON := []byte(`{"metadata":{"total_size":50},"weight_map":{` +
		`"model.layers.0.weight":"model-00001-of-00002.safetensors",` +
		`"model.layers.1.weight":"model-00001-of-00002.safetensors",` +
		`"model.layers.2.weight":"model-00002-of-00002.safetensors"}}`)
	files := map[string][]byte{
		"config.json":                      []byte(`{"model_type":"test"}`),
		"tokenizer.json":                   []byte(`{"model":{"type":"BPE"}}`),
		"tokenizer_config.json":            []byte(`{"chat_template":"test"}`),
		"model.safetensors.index.json":     indexJSON,
		"model-00001-of-00002.safetensors": shard1,
		"model-00002-of-00002.safetensors": shard2,
	}
	var infoGets atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/owner/repo/revision/main" {
			infoGets.Add(1)
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"siblings":[{"rfilename":"config.json"},` +
				`{"rfilename":"model.safetensors.index.json"},` +
				`{"rfilename":"model-00001-of-00002.safetensors"},` +
				`{"rfilename":"model-00002-of-00002.safetensors"}]}`))
			return
		}
		name, ok := strings.CutPrefix(r.URL.Path, "/owner/repo/resolve/main/")
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, ok := files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("X-Linked-Etag", `"`+sha256hex(body)+`"`)
		if r.Method == http.MethodHead {
			return
		}
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	dir, err := c.FetchURI(context.Background(), "hf://owner/repo", nil)
	if err != nil {
		t.Fatalf("FetchURI sharded: %v", err)
	}
	if fi, err := os.Stat(dir); err != nil || !fi.IsDir() {
		t.Fatalf("FetchURI returned %q, want a directory (stat err=%v)", dir, err)
	}
	for name, want := range files {
		got, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("read %s from cache dir: %v", name, err)
		}
		if string(got) != string(want) {
			t.Fatalf("cached %s = %q, want %q", name, got, want)
		}
	}
	if n := infoGets.Load(); n != 1 {
		t.Fatalf("repo info GETs = %d, want 1", n)
	}
}

func TestDownloadUnverifiedWhenNoEtag(t *testing.T) {
	body := []byte("no oid stamped")
	stub := &hubStub{body: body, etag: ""} // Hub stamps nothing
	srv := httptest.NewServer(stub.handler())
	defer srv.Close()

	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	ref := Ref{Repo: "owner/repo", Revision: "main", File: "small.json"}
	p, err := c.Download(context.Background(), ref, nil)
	if err != nil {
		t.Fatalf("Download (unverified): %v", err)
	}
	got, _ := os.ReadFile(p)
	if string(got) != string(body) {
		t.Fatalf("content = %q, want %q", got, body)
	}
}

func TestIsURI(t *testing.T) {
	if !IsURI("hf://a/b/c") {
		t.Error("IsURI hf:// = false")
	}
	if IsURI("/local/path.gguf") {
		t.Error("IsURI local = true")
	}
}

// TestDownloadResumableRangeRetryAndGitSha1 pins the TensorFold-adapted
// transfer contract: Range resume of a retained .part prefix fed into the
// same digest, a zero restart when the server answers 200 instead of 206,
// bounded retries with doubling backoff, exactly one digest-mismatch restart,
// git-blob-sha1 fallback verification, and publication only after the
// verified rename.
func TestDownloadResumableRangeRetryAndGitSha1(t *testing.T) {
	const size = 8192
	body := make([]byte, size)
	for i := range body {
		body[i] = byte((i * 7) % 251)
	}
	sum := sha256hex(body)
	ref := Ref{Repo: "owner/repo", Revision: "main", File: "model.gguf"}

	seedPart := func(t *testing.T, c *Client, prefix []byte) {
		t.Helper()
		dst := c.CachePath(ref)
		if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(dst+".part", prefix, 0o644); err != nil { // an interrupted run left a prefix
			t.Fatal(err)
		}
	}

	t.Run("resumes_range_prefix_into_same_digest", func(t *testing.T) {
		var mu sync.Mutex
		var ranges []string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
				return
			}
			mu.Lock()
			ranges = append(ranges, r.Header.Get("Range"))
			mu.Unlock()
			w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
			start := 0
			if rng := r.Header.Get("Range"); strings.HasPrefix(rng, "bytes=") {
				start, _ = strconv.Atoi(strings.TrimSuffix(strings.TrimPrefix(rng, "bytes="), "-"))
				w.WriteHeader(http.StatusPartialContent)
				_, _ = w.Write(body[start:])
				return
			}
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond}
		seedPart(t, c, body[:1000])

		path, err := c.Download(context.Background(), ref, nil)
		if err != nil {
			t.Fatalf("Download: %v", err)
		}
		dst := c.CachePath(ref)
		if path != dst {
			t.Fatalf("path = %q, want %q", path, dst)
		}
		got, err := os.ReadFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, body) {
			t.Fatalf("content = %d bytes, want the full %d-byte object", len(got), size)
		}
		mu.Lock()
		defer mu.Unlock()
		if len(ranges) != 1 || ranges[0] != "bytes=1000-" {
			t.Fatalf("GETs saw Range %v, want exactly [bytes=1000-]", ranges)
		}
		if _, err := os.Stat(dst + ".part"); !os.IsNotExist(err) {
			t.Fatalf("partial left behind after publication: %v", err)
		}
	})

	t.Run("ignored_range_restarts_from_zero", func(t *testing.T) {
		var gets atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
				return
			}
			gets.Add(1)
			w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
			_, _ = w.Write(body) // the server ignores Range: full object, status 200
		}))
		defer srv.Close()

		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond}
		seedPart(t, c, body[:1000])

		if _, err := c.Download(context.Background(), ref, nil); err != nil {
			t.Fatalf("Download: %v", err)
		}
		got, _ := os.ReadFile(c.CachePath(ref))
		if !bytes.Equal(got, body) {
			t.Fatalf("content = %d bytes, want the full %d-byte object (the stale prefix must be discarded)", len(got), size)
		}
		if n := gets.Load(); n != 1 {
			t.Fatalf("GETs = %d, want 1", n)
		}
	})

	t.Run("transient_failures_retry_with_doubling_backoff", func(t *testing.T) {
		var gets atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
				return
			}
			if gets.Add(1) <= 2 {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		var delays []time.Duration
		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond,
			backoffProbe: func(d time.Duration) { delays = append(delays, d) }}
		if _, err := c.Download(context.Background(), ref, nil); err != nil {
			t.Fatalf("Download: %v", err)
		}
		if n := gets.Load(); n != 3 {
			t.Fatalf("GETs = %d, want 3 (two 503s, then success)", n)
		}
		want := []time.Duration{time.Millisecond, 2 * time.Millisecond}
		if len(delays) != len(want) {
			t.Fatalf("backoff delays = %v, want %v", delays, want)
		}
		for i := range want {
			if delays[i] != want[i] {
				t.Fatalf("backoff delay[%d] = %v, want %v (doubling)", i, delays[i], want[i])
			}
		}
	})

	t.Run("retry_exhaustion_is_bounded_at_five_attempts", func(t *testing.T) {
		var gets atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				return
			}
			gets.Add(1)
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer srv.Close()

		var delays []time.Duration
		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond,
			backoffProbe: func(d time.Duration) { delays = append(delays, d) }}
		_, err := c.Download(context.Background(), ref, nil)
		if err == nil {
			t.Fatal("Download: want failure after retry exhaustion, got nil")
		}
		if errors.Is(err, ErrDigestMismatch) {
			t.Fatalf("error %v is a digest mismatch, want a transport/status failure", err)
		}
		if n := gets.Load(); n != 5 {
			t.Fatalf("GETs = %d, want exactly 5 attempts", n)
		}
		want := []time.Duration{time.Millisecond, 2 * time.Millisecond, 4 * time.Millisecond, 8 * time.Millisecond}
		if len(delays) != len(want) {
			t.Fatalf("backoff delays = %v, want %v", delays, want)
		}
		for i := range want {
			if delays[i] != want[i] {
				t.Fatalf("backoff delay[%d] = %v, want %v (doubling)", i, delays[i], want[i])
			}
		}
	})

	t.Run("digest_mismatch_deletes_partial_and_restarts_once", func(t *testing.T) {
		wrong := make([]byte, size)
		copy(wrong, body)
		wrong[0] ^= 0xFF
		var gets atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				w.Header().Set("X-Linked-Etag", `"`+sum+`"`) // the hub claims the real digest...
				return
			}
			gets.Add(1)
			w.Header().Set("X-Linked-Etag", `"`+sum+`"`)
			_, _ = w.Write(wrong) // ...but serves corrupt bytes
		}))
		defer srv.Close()

		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond}
		dst := c.CachePath(ref)
		_, err := c.Download(context.Background(), ref, nil)
		if !errors.Is(err, ErrDigestMismatch) {
			t.Fatalf("error = %v, want ErrDigestMismatch", err)
		}
		if n := gets.Load(); n != 2 {
			t.Fatalf("GETs = %d, want 2 (one mismatch restart, then fail)", n)
		}
		if _, statErr := os.Stat(dst); !os.IsNotExist(statErr) {
			t.Fatalf("destination published despite mismatch: %v", statErr)
		}
		if _, statErr := os.Stat(dst + ".part"); !os.IsNotExist(statErr) {
			t.Fatalf("corrupt partial not deleted: %v", statErr)
		}
	})

	t.Run("git_blob_sha1_verifies_without_sha256", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				return // no LFS oid stamped anywhere
			}
			w.Header().Set("ETag", `"`+gitBlobSHA1Fixture(body)+`"`)
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond}
		dst, err := c.Download(context.Background(), ref, nil)
		if err != nil {
			t.Fatalf("Download (blob sha1): %v", err)
		}
		got, _ := os.ReadFile(dst)
		if !bytes.Equal(got, body) {
			t.Fatalf("content = %d bytes, want %d", len(got), size)
		}
	})

	t.Run("git_blob_sha1_mismatch_fails_after_one_restart", func(t *testing.T) {
		var gets atomic.Int32
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodHead {
				return
			}
			gets.Add(1)
			w.Header().Set("ETag", "\"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"") // a wrong blob id
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir(), retryBackoff: time.Millisecond}
		dst := c.CachePath(ref)
		_, err := c.Download(context.Background(), ref, nil)
		if !errors.Is(err, ErrDigestMismatch) {
			t.Fatalf("error = %v, want ErrDigestMismatch", err)
		}
		if n := gets.Load(); n != 2 {
			t.Fatalf("GETs = %d, want 2", n)
		}
		if _, statErr := os.Stat(dst); !os.IsNotExist(statErr) {
			t.Fatalf("destination published despite blob sha1 mismatch: %v", statErr)
		}
	})
}

// gitBlobSHA1Fixture computes the git blob identifier of b: sha1 over the
// "blob <size>\x00" header plus the raw bytes.
func gitBlobSHA1Fixture(b []byte) string {
	h := sha1.New()
	fmt.Fprintf(h, "blob %d\x00", len(b))
	h.Write(b)
	return hex.EncodeToString(h.Sum(nil))
}
