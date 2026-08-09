package modelsrc

import (
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpenResolvesFileAndHTTPSThroughOneEntryPoint(t *testing.T) {
	payload := []byte("model-weights")
	path := filepath.Join(t.TempDir(), "tiny model.gguf")
	if err := os.WriteFile(path, payload, 0o600); err != nil {
		t.Fatal(err)
	}
	fileURL := (&url.URL{Scheme: "file", Path: "/" + filepath.ToSlash(path)}).String()
	assertObject(t, Open, fileURL, payload)

	server := httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/tiny.gguf" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write(payload)
	}))
	defer server.Close()

	registry := New(WithHTTPClient(server.Client()))
	assertObject(t, registry.Open, server.URL+"/tiny.gguf", payload)
}

func TestRegistryRejectsUnknownAndAllowsRegistration(t *testing.T) {
	registry := New()
	if _, _, err := registry.Open("s3://bucket/model"); err == nil || !strings.Contains(err.Error(), "unsupported scheme") {
		t.Fatalf("unknown scheme error = %v", err)
	}
	registry.Register("mem", func(string) (io.ReaderAt, int64, error) {
		return strings.NewReader("ok"), 2, nil
	})
	assertObject(t, registry.Open, "mem://model", []byte("ok"))
}

func assertObject(t *testing.T, open func(string) (io.ReaderAt, int64, error), ref string, want []byte) {
	t.Helper()
	reader, size, err := open(ref)
	if err != nil {
		t.Fatalf("Open(%q): %v", ref, err)
	}
	if closer, ok := reader.(io.Closer); ok {
		defer closer.Close()
	}
	if size != int64(len(want)) {
		t.Fatalf("size = %d, want %d", size, len(want))
	}
	got := make([]byte, size)
	if _, err := reader.ReadAt(got, 0); err != nil && err != io.EOF {
		t.Fatalf("ReadAt: %v", err)
	}
	if string(got) != string(want) {
		t.Fatalf("content = %q, want %q", got, want)
	}
}
