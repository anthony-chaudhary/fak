package modelreg

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

func TestMacDefaultAliasIsPinnedCatalogTarget(t *testing.T) {
	uri, ok := Catalog[MacDefaultAlias]
	if !ok {
		t.Fatalf("%s is not in the catalog", MacDefaultAlias)
	}
	ref, err := hfhub.ParseURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	if len(ref.Revision) != 40 || ref.File == "" {
		t.Fatalf("%s target %q is not pinned to a commit and file", MacDefaultAlias, uri)
	}
	if DefaultAlias == MacDefaultAlias {
		t.Fatal("the Apple Silicon default must not replace the cross-platform default")
	}
}

func TestCompanionsAreMetalOnlyAndPinned(t *testing.T) {
	uri := Catalog[MacDefaultAlias]
	got := Companions(uri, "darwin", "arm64")
	if len(got) != 1 || got[0].Dir != "Inkling-2-Mini-DFlash2" {
		t.Fatalf("Companions on darwin/arm64 = %+v", got)
	}
	c := got[0]
	if c.Repo != "incoai/Inkling-2-Mini-DFlash2" || len(c.Revision) != 40 || strings.Trim(c.Revision, "0123456789abcdef") != "" {
		t.Fatalf("companion is not pinned to a commit: %+v", c)
	}
	if strings.Join(c.Files, ",") != "model.safetensors,config.json" {
		t.Fatalf("companion files = %v", c.Files)
	}
	for _, platform := range [][2]string{{"linux", "amd64"}, {"darwin", "amd64"}, {"windows", "arm64"}} {
		if extra := Companions(uri, platform[0], platform[1]); len(extra) != 0 {
			t.Fatalf("Companions on %s/%s = %+v, want none", platform[0], platform[1], extra)
		}
	}
	if extra := Companions("hf://someone/else/model.gguf", "darwin", "arm64"); len(extra) != 0 {
		t.Fatalf("an unrelated target has companions: %+v", extra)
	}
}

func TestFetchCompanionsPlacesDirectoryBesideArtifact(t *testing.T) {
	served := map[string]string{
		"/owner/draft/resolve/rev1/config.json":       `{"architectures":["DFlash2DraftModel"]}`,
		"/owner/draft/resolve/rev1/model.safetensors": "weights",
	}
	requests := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, ok := served[r.URL.Path]
		if !ok {
			http.NotFound(w, r)
			return
		}
		if r.Method == http.MethodGet {
			requests++
		}
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	root := t.TempDir()
	artifact := filepath.Join(root, "models", "target.gguf")
	if err := os.MkdirAll(filepath.Dir(artifact), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(artifact, []byte("gguf"), 0o644); err != nil {
		t.Fatal(err)
	}
	client := &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: filepath.Join(root, "hub")}
	list := []Companion{{Dir: "Draft", Repo: "owner/draft", Revision: "rev1", Files: []string{"model.safetensors", "config.json"}}}

	if err := fetchCompanions(context.Background(), client, list, artifact, nil); err != nil {
		t.Fatal(err)
	}
	for file, want := range map[string]string{"config.json": `{"architectures":["DFlash2DraftModel"]}`, "model.safetensors": "weights"} {
		got, err := os.ReadFile(filepath.Join(filepath.Dir(artifact), "Draft", file))
		if err != nil || string(got) != want {
			t.Fatalf("companion %s = %q, %v", file, got, err)
		}
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(artifact), "Draft.partial")); !os.IsNotExist(err) {
		t.Fatalf("staging directory left behind: %v", err)
	}

	before := requests
	if err := fetchCompanions(context.Background(), client, list, artifact, nil); err != nil {
		t.Fatal(err)
	}
	if requests != before {
		t.Fatalf("an existing companion directory was fetched again (%d new requests)", requests-before)
	}
}

func TestFetchCompanionsLeavesNoDirectoryOnFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/config.json") {
			_, _ = w.Write([]byte("{}"))
			return
		}
		http.NotFound(w, r)
	}))
	defer srv.Close()

	root := t.TempDir()
	artifact := filepath.Join(root, "target.gguf")
	client := &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: filepath.Join(root, "hub")}
	list := []Companion{{Dir: "Draft", Repo: "owner/draft", Revision: "rev1", Files: []string{"config.json", "model.safetensors"}}}
	if err := fetchCompanions(context.Background(), client, list, artifact, nil); err == nil {
		t.Fatal("a missing companion file was accepted")
	}
	if _, err := os.Stat(filepath.Join(root, "Draft")); !os.IsNotExist(err) {
		t.Fatalf("a partial companion directory was published: %v", err)
	}
}

func TestLocalMacDefaultArtifactMatchesNameAndExactSize(t *testing.T) {
	dir := withCacheRoot(t)
	const size = 64
	old := macDefaultArtifactBytes
	macDefaultArtifactBytes = size
	t.Cleanup(func() { macDefaultArtifactBytes = old })
	uri := Catalog[MacDefaultAlias]

	if p, ok := LocalMacDefaultArtifact(uri); ok {
		t.Fatalf("found %q with no local copy", p)
	}
	nested := filepath.Join(dir, "inkling2")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		t.Fatal(err)
	}
	local := filepath.Join(nested, "Inkling-2-Mini-UD-Q3_K_XL.gguf")
	if err := os.WriteFile(local, make([]byte, size-1), 0o644); err != nil {
		t.Fatal(err)
	}
	if p, ok := LocalMacDefaultArtifact(uri); ok {
		t.Fatalf("accepted %q with the wrong size", p)
	}
	if err := os.WriteFile(local, make([]byte, size), 0o644); err != nil {
		t.Fatal(err)
	}
	if p, ok := LocalMacDefaultArtifact(uri); !ok || p != local {
		t.Fatalf("LocalMacDefaultArtifact = %q, %v; want %q", p, ok, local)
	}
	if p, ok := LocalMacDefaultArtifact("hf://someone/else/Inkling-2-Mini-UD-Q3_K_XL.gguf"); ok {
		t.Fatalf("matched %q for an overridden target", p)
	}
}
