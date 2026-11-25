package hfhub

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// splitHub is a fake Hub serving a fixed file set with per-file LFS sha256
// etags, a siblings listing, and a record of which files were GET-downloaded.
type splitHub struct {
	files    map[string][]byte
	listing  []string
	mu       sync.Mutex
	gets     map[string]int
	infoGets int
}

func (h *splitHub) server(t *testing.T) *httptest.Server {
	t.Helper()
	h.gets = map[string]int{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/models/owner/repo/revision/main" {
			h.mu.Lock()
			h.infoGets++
			h.mu.Unlock()
			type sib struct {
				RFilename string `json:"rfilename"`
			}
			var info struct {
				Siblings []sib `json:"siblings"`
			}
			for _, f := range h.listing {
				info.Siblings = append(info.Siblings, sib{f})
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(info)
			return
		}
		name, ok := strings.CutPrefix(r.URL.Path, "/owner/repo/resolve/main/")
		if !ok {
			http.NotFound(w, r)
			return
		}
		body, ok := h.files[name]
		if !ok {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("X-Linked-Etag", `"`+sha256hex(body)+`"`)
		if r.Method == http.MethodHead {
			return
		}
		h.mu.Lock()
		h.gets[name]++
		h.mu.Unlock()
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv
}

func smallSplitFiles(dir string, n int) map[string][]byte {
	files := map[string][]byte{}
	for k := 1; k <= n; k++ {
		s := splitGGUF{Dir: dir, Prefix: "Inkling-2-Small-UD-Q2_K_XL", Ext: ".gguf", Shard: k, Total: n}
		files[s.shardFile(k)] = []byte("GGUF shard " + s.shardFile(k))
	}
	return files
}

func keys(m map[string][]byte) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func assertSplitFetched(t *testing.T, got string, files map[string][]byte, first string) {
	t.Helper()
	if filepath.Base(got) != filepath.Base(first) {
		t.Fatalf("FetchURI returned %q, want shard 1 %q", got, first)
	}
	dir := filepath.Dir(got)
	for name, want := range files {
		b, err := os.ReadFile(filepath.Join(dir, filepath.Base(name)))
		if err != nil {
			t.Fatalf("shard %s not beside shard 1: %v", name, err)
		}
		if string(b) != string(want) {
			t.Fatalf("shard %s = %q, want %q", name, b, want)
		}
	}
}

func TestFetchURIGGUFSplitFansOutToAllShards(t *testing.T) {
	files := smallSplitFiles("UD-Q2_K_XL", 5)
	hub := &splitHub{files: files, listing: keys(files)}
	srv := hub.server(t)
	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}

	// Naming a later shard still fetches the whole set and returns shard 1.
	third := "UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00003-of-00005.gguf"
	got, err := c.FetchURI(context.Background(), "hf://owner/repo/"+third, nil)
	if err != nil {
		t.Fatalf("FetchURI split: %v", err)
	}
	assertSplitFetched(t, got, files, "Inkling-2-Small-UD-Q2_K_XL-00001-of-00005.gguf")
	for name := range files {
		if hub.gets[name] != 1 {
			t.Fatalf("GETs for %s = %d, want 1", name, hub.gets[name])
		}
	}
	if hub.infoGets != 0 {
		t.Fatalf("explicit file URI hit repo info %d times, want 0", hub.infoGets)
	}

	// A re-run is a pure cache hit.
	if _, err := c.FetchURI(context.Background(), "hf://owner/repo/"+third, nil); err != nil {
		t.Fatalf("FetchURI split re-run: %v", err)
	}
	for name := range files {
		if hub.gets[name] != 1 {
			t.Fatalf("re-run GETs for %s = %d, want 1 (cache hit)", name, hub.gets[name])
		}
	}
}

func TestFetchURIGGUFSplitBareRepoSingleSetResolves(t *testing.T) {
	files := smallSplitFiles("UD-Q2_K_XL", 3)
	listing := append(keys(files), "README.md", "config.json")
	hub := &splitHub{files: files, listing: listing}
	srv := hub.server(t)
	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}

	got, err := c.FetchURI(context.Background(), "hf://owner/repo", nil)
	if err != nil {
		t.Fatalf("FetchURI bare repo with one split set: %v", err)
	}
	assertSplitFetched(t, got, files, "Inkling-2-Small-UD-Q2_K_XL-00001-of-00003.gguf")
}

func TestFetchURIGGUFSplitMultipleSetsStayAmbiguous(t *testing.T) {
	files := smallSplitFiles("UD-Q2_K_XL", 2)
	for k, v := range smallSplitFiles("Q4_K_M", 3) {
		files[k] = v
	}
	hub := &splitHub{files: files, listing: keys(files)}
	srv := hub.server(t)
	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}

	_, err := c.FetchURI(context.Background(), "hf://owner/repo", nil)
	if err == nil || !strings.Contains(err.Error(), "ambiguous GGUF artifacts") {
		t.Fatalf("FetchURI multi-set = %v, want ambiguous GGUF artifacts", err)
	}
	if len(hub.gets) != 0 {
		t.Fatalf("ambiguous repo downloaded %v, want nothing", hub.gets)
	}

	// A split set beside a single-file GGUF is ambiguous too.
	single := smallSplitFiles("UD-Q2_K_XL", 2)
	single["model.Q8_0.gguf"] = []byte("GGUF single")
	hub2 := &splitHub{files: single, listing: keys(single)}
	srv2 := hub2.server(t)
	c2 := &Client{BaseURL: srv2.URL, HTTP: srv2.Client(), CacheDir: t.TempDir()}
	if _, err := c2.FetchURI(context.Background(), "hf://owner/repo", nil); err == nil ||
		!strings.Contains(err.Error(), "ambiguous GGUF artifacts") {
		t.Fatalf("FetchURI split+single = %v, want ambiguous GGUF artifacts", err)
	}
}

func TestFetchURIGGUFSplitMissingShardFails(t *testing.T) {
	files := smallSplitFiles("UD-Q2_K_XL", 4)
	missing := "UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00003-of-00004.gguf"
	delete(files, missing)

	// Explicit URI: the Hub 404s the absent shard during the fan-out.
	hub := &splitHub{files: files, listing: keys(files)}
	srv := hub.server(t)
	c := &Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	_, err := c.FetchURI(context.Background(), "hf://owner/repo/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00004.gguf", nil)
	if err == nil {
		t.Fatal("FetchURI with a missing shard: want error, got nil")
	}
	for _, want := range []string{"shard 3 of 4", missing, "404"} {
		if !strings.Contains(err.Error(), want) {
			t.Fatalf("error = %v, want it to mention %q", err, want)
		}
	}

	// Bare repo: the listing hole is refused before any download.
	_, err = c.FetchURI(context.Background(), "hf://owner/repo", nil)
	if err == nil || !strings.Contains(err.Error(), "incomplete") || !strings.Contains(err.Error(), "shard 3 of 4") {
		t.Fatalf("FetchURI bare repo with a hole = %v, want incomplete shard 3 of 4", err)
	}
}

func TestFetchURIGGUFSplitParseRejectsNonSplitNames(t *testing.T) {
	for _, name := range []string{
		"model.Q8_0.gguf",
		"model-0001-of-0002.gguf",
		"model-00000-of-00002.gguf",
		"model-00003-of-00002.gguf",
		"model-00001-of-00002.safetensors",
	} {
		if _, ok := parseSplitGGUF(name); ok {
			t.Fatalf("parseSplitGGUF(%q) ok, want rejected", name)
		}
	}
	s, ok := parseSplitGGUF("Q4/m-00002-of-00007.GGUF")
	if !ok || s.Shard != 2 || s.Total != 7 || s.shardFile(1) != "Q4/m-00001-of-00007.GGUF" {
		t.Fatalf("parseSplitGGUF = %+v ok=%v", s, ok)
	}
}
