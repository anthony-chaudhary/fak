package ociartifact

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

func mixed(t *testing.T) (Artifact, map[string][]byte) {
	t.Helper()
	payloads := map[string][]byte{"skills/review.json": []byte(`{"name":"review"}`), "mcp/fak.server.json": []byte(`{"name":"io.example/fak","version":"1.2.3","packages":[{"registryType":"oci","identifier":"example/fak:1.2.3"}],"remotes":[{"type":"sse","url":"https://example.invalid/mcp"}],"repository":{"url":"https://example.invalid/repo"},"status":"active"}`), "future/opaque.bin": {0, 1, 2, 255, 10}}
	skill := Digest(payloads["skills/review.json"])
	cfg := Config{Schema: "fak.oci.collection/v1", Name: "mixed", Version: "1.0.0", Objects: []Object{{Name: "review", Kind: "skill", MediaType: SkillMediaType, Path: "skills/review.json"}, {Name: "fak", Kind: "mcp-service", MediaType: MCPServerMediaType, Path: "mcp/fak.server.json", Dependencies: []string{skill}}, {Name: "opaque", Kind: "future-object", MediaType: "application/vnd.example.future.v9+binary", Path: "future/opaque.bin"}}}
	a, err := Build(cfg, payloads, map[string]string{"org.opencontainers.image.description": "golden mixed collection", "example.unknown": "preserve-me"})
	if err != nil {
		t.Fatal(err)
	}
	return a, payloads
}

func TestGoldenLayoutRegistryProxyAndReferrers(t *testing.T) {
	a, payloads := mixed(t)
	dir := t.TempDir()
	if err := ExportLayout(dir, a); err != nil {
		t.Fatal(err)
	}
	fromLayout, lr, err := ImportLayout(dir)
	if err != nil {
		t.Fatal(err)
	}
	if lr.Activated {
		t.Fatal("layout import activated content")
	}
	if !bytes.Equal(fromLayout.Blobs[Digest(payloads["future/opaque.bin"])], payloads["future/opaque.bin"]) {
		t.Fatal("unknown layer bytes changed")
	}
	reg := newTestRegistry()
	srv := httptest.NewServer(reg)
	defer srv.Close()
	c := Client{Base: srv.URL, Repository: "acme/fak"}
	if err = c.Push("mixed:mutable", fromLayout); err != nil {
		t.Fatal(err)
	}
	resolved, err := c.Resolve("mixed:mutable")
	if err != nil {
		t.Fatal(err)
	}
	if resolved.Digest != a.Manifest.Digest {
		t.Fatalf("resolved %s", resolved.Digest)
	}
	if _, _, err = c.Pull("mixed:mutable"); Code(err) != "DIGEST_REQUIRED" {
		t.Fatalf("tag-only pull: %v", err)
	}
	pulled, rr, err := c.Pull(resolved.Digest)
	if err != nil {
		t.Fatal(err)
	}
	if rr.Activated || rr.ManifestDigest != a.Manifest.Digest || len(rr.LayerDigests) != 3 {
		t.Fatalf("bad inert receipt %#v", rr)
	}
	proxy := t.TempDir()
	if err = ExportLayout(proxy, pulled); err != nil {
		t.Fatal(err)
	}
	again, _, err := ImportLayout(proxy)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(a.RawManifest, again.RawManifest) {
		t.Fatal("proxy/re-export changed manifest bytes or descriptors")
	}
	if fmt.Sprint(a.Parsed.Annotations) != fmt.Sprint(again.Parsed.Annotations) {
		t.Fatal("unknown annotations changed")
	}
	ref := makeReferrer(t, a.Manifest, "application/vnd.example.signature+json", []byte(`{"signature":"detached"}`))
	if err = c.Push("signature", ref); err != nil {
		t.Fatal(err)
	}
	refs, err := c.Referrers(a.Manifest.Digest)
	if err != nil || len(refs) != 1 || refs[0].Digest != ref.Manifest.Digest {
		t.Fatalf("standard referrers: %#v %v", refs, err)
	}
	reg.standard = false
	refs, err = c.Referrers(a.Manifest.Digest)
	if err != nil || len(refs) != 1 {
		t.Fatalf("fallback referrers: %#v %v", refs, err)
	}
	if atomic.LoadInt32(&activationCalls) != 0 {
		t.Fatal("transport caused activation side effect")
	}
}

var activationCalls int32

func TestTypedFailClosedCorpus(t *testing.T) {
	a, _ := mixed(t)
	cases := []struct {
		name, want string
		mutate     func(*Manifest, map[string][]byte)
	}{
		{"digest", "DIGEST_MISMATCH", func(m *Manifest, b map[string][]byte) { b[m.Layers[0].Digest] = []byte("tampered-but-same") }},
		{"size", "SIZE_MISMATCH", func(m *Manifest, b map[string][]byte) { m.Layers[0].Size++ }},
		{"media-type", "UNSUPPORTED_MEDIA_TYPE", func(m *Manifest, b map[string][]byte) { m.Layers[0].MediaType = "application/octet-stream" }},
		{"path-escape", "PATH_ESCAPE", func(m *Manifest, b map[string][]byte) {
			rewriteConfig(t, m, b, func(c *Config) { c.Objects[0].Path = "../escape" })
		}},
		{"missing-layer", "MISSING_LAYER", func(m *Manifest, b map[string][]byte) { delete(b, m.Layers[0].Digest) }},
		{"dependency-cycle", "DEPENDENCY_CYCLE", func(m *Manifest, b map[string][]byte) {
			rewriteConfig(t, m, b, func(c *Config) {
				c.Objects[0].Dependencies = []string{c.Objects[1].Digest}
				c.Objects[1].Dependencies = []string{c.Objects[0].Digest}
			})
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var m Manifest
			_ = json.Unmarshal(a.RawManifest, &m)
			bl := copyBlobs(a.Blobs)
			tc.mutate(&m, bl)
			raw, _ := json.Marshal(m)
			_, _, err := Inspect(raw, bl)
			if Code(err) != tc.want {
				t.Fatalf("got %s %v want %s", Code(err), err, tc.want)
			}
		})
	}
}
func rewriteConfig(t *testing.T, m *Manifest, b map[string][]byte, f func(*Config)) {
	t.Helper()
	var c Config
	if err := json.Unmarshal(b[m.Config.Digest], &c); err != nil {
		t.Fatal(err)
	}
	f(&c)
	cb, _ := json.Marshal(c)
	delete(b, m.Config.Digest)
	m.Config = descriptor(ConfigMediaType, cb)
	b[m.Config.Digest] = cb
}

func TestMCPServerJSONBridgeIsNarrowAndLossless(t *testing.T) {
	raw := []byte(`{"$schema":"https://example/schema","name":"io.example/server","version":"2.0.0","packages":[{"registryType":"npm","identifier":"x"}],"remotes":[{"type":"sse","url":"https://example.invalid"}],"repository":{"url":"https://example.invalid/repo","source":"git"},"status":"deprecated"}`)
	s, err := ImportServerJSON("mcp-service", raw)
	if err != nil {
		t.Fatal(err)
	}
	out, err := ExportServerJSON("mcp-service", s)
	if err != nil {
		t.Fatal(err)
	}
	var x, y map[string]any
	_ = json.Unmarshal(raw, &x)
	_ = json.Unmarshal(out, &y)
	if fmt.Sprint(x) != fmt.Sprint(y) {
		t.Fatalf("metadata changed\n%s", out)
	}
	if _, err = ImportServerJSON("collection", raw); Code(err) != "BRIDGE_SCOPE" {
		t.Fatalf("server.json became collection manifest: %v", err)
	}
}

func TestMutableTagMustResolveBeforePull(t *testing.T) {
	a, _ := mixed(t)
	reg := newTestRegistry()
	ts := httptest.NewServer(reg)
	defer ts.Close()
	c := Client{Base: ts.URL, Repository: "acme/fak"}
	if err := c.Push("latest", a); err != nil {
		t.Fatal(err)
	}
	if _, _, err := c.Pull("latest"); Code(err) != "DIGEST_REQUIRED" {
		t.Fatalf("tag-only pull was not refused: %v", err)
	}
	resolved, err := c.Resolve("latest")
	if err != nil || resolved.Digest != a.Manifest.Digest || resolved.MediaType != ManifestMediaType {
		t.Fatalf("resolve latest: %#v %v", resolved, err)
	}
	_, receipt, err := c.Pull(resolved.Digest)
	if err != nil || receipt.ManifestDigest != resolved.Digest || receipt.Activated {
		t.Fatalf("digest pull: %#v %v", receipt, err)
	}
}

func TestReferrerKindsAreDetachedAndDiscoverable(t *testing.T) {
	a, _ := mixed(t)
	reg := newTestRegistry()
	ts := httptest.NewServer(reg)
	defer ts.Close()
	c := Client{Base: ts.URL, Repository: "acme/fak"}
	if err := c.Push("collection", a); err != nil {
		t.Fatal(err)
	}
	kinds := []string{SignatureMediaType, AttestationMediaType, SBOMMediaType, StatementMediaType + ";kind=replacement", StatementMediaType + ";kind=revocation"}
	for i, kind := range kinds {
		ref := makeReferrer(t, a.Manifest, kind, []byte(fmt.Sprintf(`{"kind":%q}`, kind)))
		if err := c.Push(fmt.Sprintf("ref-%d", i), ref); err != nil {
			t.Fatal(err)
		}
	}
	refs, err := c.Referrers(a.Manifest.Digest)
	if err != nil || len(refs) != len(kinds) {
		t.Fatalf("standard referrers: %d %v", len(refs), err)
	}
	reg.standard = false
	refs, err = c.Referrers(a.Manifest.Digest)
	if err != nil || len(refs) != len(kinds) {
		t.Fatalf("fallback referrers: %d %v", len(refs), err)
	}
}

func TestHandAuthoredGoldenFixture(t *testing.T) {
	root := filepath.Join("..", "..", "docs", "portability", "oci", "fixtures", "mixed-layout")
	a, r, err := ImportLayout(root)
	if err != nil {
		t.Fatal(err)
	}
	if r.Activated || a.Parsed.ArtifactType != ConfigMediaType || len(a.Parsed.Layers) != 3 {
		t.Fatalf("bad fixture %#v %#v", a.Parsed, r)
	}
}

func makeReferrer(t *testing.T, subject Descriptor, typ string, payload []byte) Artifact {
	t.Helper()
	pd := descriptor(typ, payload)
	pd.Annotations = map[string]string{annotationTitle: "statement.json", annotationKind: "referrer"}
	cfg := []byte(fmt.Sprintf(`{"schema":"fak.oci.collection/v1","name":"referrer","version":"1","objects":[{"name":"statement","kind":"referrer","mediaType":%q,"digest":%q,"path":"statement.json"}]}`, typ, pd.Digest))
	cd := descriptor(ConfigMediaType, cfg)
	m := Manifest{SchemaVersion: 2, MediaType: ManifestMediaType, ArtifactType: typ, Config: cd, Layers: []Descriptor{pd}, Subject: &subject}
	raw, _ := json.Marshal(m)
	return Artifact{Manifest: descriptor(ManifestMediaType, raw), RawManifest: raw, Parsed: m, Blobs: map[string][]byte{cd.Digest: cfg, pd.Digest: payload}}
}

type testRegistry struct {
	mu        sync.Mutex
	blobs     map[string][]byte
	manifests map[string][]byte
	tags      map[string]string
	refs      map[string][]Descriptor
	standard  bool
}

func newTestRegistry() *testRegistry {
	return &testRegistry{blobs: map[string][]byte{}, manifests: map[string][]byte{}, tags: map[string]string{}, refs: map[string][]Descriptor{}, standard: true}
}
func (r *testRegistry) ServeHTTP(w http.ResponseWriter, q *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()
	p := q.URL.Path
	repoPrefix := "/v2/acme/fak"
	if !strings.HasPrefix(p, repoPrefix) {
		http.NotFound(w, q)
		return
	}
	tail := strings.TrimPrefix(p, repoPrefix)
	if strings.HasPrefix(tail, "/blobs/uploads/") && q.Method == "POST" {
		b, _ := os.ReadFile("")
		_ = b
		body, _ := ioReadAll(q)
		d := q.URL.Query().Get("digest")
		if d == "" || Digest(body) != d {
			http.Error(w, "bad digest", 400)
			return
		}
		r.blobs[d] = body
		w.WriteHeader(201)
		return
	}
	if strings.HasPrefix(tail, "/blobs/") && q.Method == "GET" {
		d := strings.TrimPrefix(tail, "/blobs/")
		b, ok := r.blobs[d]
		if !ok {
			http.NotFound(w, q)
			return
		}
		w.Write(b)
		return
	}
	if strings.HasPrefix(tail, "/manifests/") {
		ref := strings.TrimPrefix(tail, "/manifests/")
		if q.Method == "PUT" {
			b, _ := ioReadAll(q)
			d := Digest(b)
			r.manifests[d] = b
			r.tags[ref] = d
			var m Manifest
			_ = json.Unmarshal(b, &m)
			if m.Subject != nil {
				desc := descriptor(ManifestMediaType, b)
				desc.ArtifactType = m.ArtifactType
				r.refs[m.Subject.Digest] = append(r.refs[m.Subject.Digest], desc)
				idx, _ := json.Marshal(struct {
					SchemaVersion int          `json:"schemaVersion"`
					MediaType     string       `json:"mediaType"`
					Manifests     []Descriptor `json:"manifests"`
				}{2, IndexMediaType, r.refs[m.Subject.Digest]})
				r.manifests[strings.Replace(m.Subject.Digest, ":", "-", 1)] = idx
			}
			w.Header().Set("Docker-Content-Digest", d)
			w.WriteHeader(201)
			return
		}
		d := ref
		if x, ok := r.tags[ref]; ok {
			d = x
		}
		b, ok := r.manifests[d]
		if !ok {
			http.NotFound(w, q)
			return
		}
		w.Header().Set("Docker-Content-Digest", d)
		w.Header().Set("Content-Type", ManifestMediaType)
		if q.Method == "HEAD" {
			return
		}
		w.Write(b)
		return
	}
	if strings.HasPrefix(tail, "/referrers/") {
		if !r.standard {
			http.NotFound(w, q)
			return
		}
		d := strings.TrimPrefix(tail, "/referrers/")
		b, _ := json.Marshal(struct {
			SchemaVersion int          `json:"schemaVersion"`
			MediaType     string       `json:"mediaType"`
			Manifests     []Descriptor `json:"manifests"`
		}{2, IndexMediaType, r.refs[d]})
		w.Header().Set("Content-Type", IndexMediaType)
		w.Write(b)
		return
	}
	http.NotFound(w, q)
}
func ioReadAll(r *http.Request) ([]byte, error) {
	var b bytes.Buffer
	_, e := b.ReadFrom(r.Body)
	return b.Bytes(), e
}

func TestVerifyArtifact(t *testing.T) {
	dir := t.TempDir()
	validPath := filepath.Join(dir, "artifact.bin")
	if err := os.WriteFile(validPath, []byte("payload"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Run("empty path returns error", func(t *testing.T) {
		err := VerifyArtifact("", "test-key")
		if err == nil {
			t.Fatal("expected error for empty path, got nil")
		}
		if Code(err) != "EMPTY_PATH" {
			t.Fatalf("expected EMPTY_PATH, got %v", err)
		}
	})

	t.Run("non-existent path returns error", func(t *testing.T) {
		nonExistent := filepath.Join(dir, "does-not-exist")
		err := VerifyArtifact(nonExistent, "test-key")
		if err == nil {
			t.Fatal("expected error for non-existent path, got nil")
		}
		if Code(err) != "NOT_FOUND" {
			t.Fatalf("expected NOT_FOUND, got %v", err)
		}
	})

	t.Run("empty verifyKey returns error", func(t *testing.T) {
		err := VerifyArtifact(validPath, "")
		if err == nil {
			t.Fatal("expected error for empty verifyKey, got nil")
		}
		if Code(err) != "EMPTY_KEY" {
			t.Fatalf("expected EMPTY_KEY, got %v", err)
		}
	})

	t.Run("valid path and verifyKey returns nil", func(t *testing.T) {
		err := VerifyArtifact(validPath, "test-key")
		if err != nil {
			t.Fatalf("expected nil error, got %v", err)
		}
	})
}

