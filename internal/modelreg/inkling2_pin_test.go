package modelreg

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/hfhub"
)

const inklingMiniSHA256 = "e2d03e8c92ac21cff0e3309b95f40dfd27d1825184f0d2bc09db249797273639"

func TestInklingMiniPinsGGUFSHA256(t *testing.T) {
	uri := Catalog["inkling2-mini"]
	if uri != inklingMiniURI {
		t.Fatalf("inkling2-mini = %q, want the pinned URI %q", uri, inklingMiniURI)
	}
	pin, ok := PinnedArtifact(uri)
	if !ok {
		t.Fatal("inkling2-mini has no sha256 pin")
	}
	if pin.SHA256 != inklingMiniSHA256 {
		t.Fatalf("pinned sha256 = %s, want %s", pin.SHA256, inklingMiniSHA256)
	}
	if pin.Bytes != 13631779296 {
		t.Fatalf("pinned bytes = %d, want 13631779296", pin.Bytes)
	}
	if got, _ := Resolve("inkling"); got != uri {
		t.Fatalf("Resolve(inkling) = %q, want the same pinned URI", got)
	}
}

func TestFetchPinnedRejectsMismatchAndEvictsCache(t *testing.T) {
	body := []byte("not the inkling gguf")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Write(body)
	}))
	defer srv.Close()
	client := &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	const uri = "hf://owner/repo@0123456789abcdef0123456789abcdef01234567/model.gguf"
	ref, err := hfhub.ParseURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{SHA256: inklingMiniSHA256, Bytes: int64(len(body))}

	_, err = fetchPinned(context.Background(), client, uri, pin, nil)
	if err == nil || !strings.Contains(err.Error(), "does not match pin") {
		t.Fatalf("fetchPinned error = %v, want a pin mismatch", err)
	}
	if _, statErr := os.Stat(client.CachePath(ref)); !os.IsNotExist(statErr) {
		t.Fatalf("mismatched file still cached at %s (stat err %v)", client.CachePath(ref), statErr)
	}
}

func TestFetchPinnedReverifiesCacheHit(t *testing.T) {
	const uri = "hf://owner/repo@0123456789abcdef0123456789abcdef01234567/model.gguf"
	// No X-Linked-Etag, so hfhub serves a cache hit with no digest check of its own.
	// The HEAD probe for that etag is allowed; any body download is not.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			t.Errorf("unexpected network body fetch for a cache hit: %s %s", r.Method, r.URL)
		}
	}))
	defer srv.Close()
	client := &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	ref, err := hfhub.ParseURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	corrupt := []byte("corrupted cache bytes")
	if err := os.MkdirAll(filepath.Dir(client.CachePath(ref)), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(client.CachePath(ref), corrupt, 0o644); err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{SHA256: inklingMiniSHA256, Bytes: int64(len(corrupt))}

	if _, err := fetchPinned(context.Background(), client, uri, pin, nil); err == nil {
		t.Fatal("corrupt cache hit passed the pin check")
	}
	if _, statErr := os.Stat(client.CachePath(ref)); !os.IsNotExist(statErr) {
		t.Fatal("corrupt cache file was not evicted")
	}
}

func TestVerifyArtifactAcceptsMatchingDigest(t *testing.T) {
	body := []byte("pinned bytes")
	sum := sha256.Sum256(body)
	path := t.TempDir() + "/artifact.gguf"
	if err := os.WriteFile(path, body, 0o644); err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(body))}
	if err := verifyArtifact(path, pin); err != nil {
		t.Fatalf("matching digest rejected: %v", err)
	}
}

func TestFetchPinnedStampSkipsRehashOfVerifiedFile(t *testing.T) {
	good := []byte("pinned gguf bytes")
	sum := sha256.Sum256(good)
	pin := ArtifactPin{SHA256: hex.EncodeToString(sum[:]), Bytes: int64(len(good))}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodHead {
			t.Errorf("unexpected network body fetch: %s %s", r.Method, r.URL)
		}
	}))
	defer srv.Close()
	client := &hfhub.Client{BaseURL: srv.URL, HTTP: srv.Client(), CacheDir: t.TempDir()}
	const uri = "hf://owner/repo@0123456789abcdef0123456789abcdef01234567/model.gguf"
	ref, err := hfhub.ParseURI(uri)
	if err != nil {
		t.Fatal(err)
	}
	path := client.CachePath(ref)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, good, 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	ctx := context.Background()

	if _, err := fetchPinned(ctx, client, uri, pin, nil); err != nil {
		t.Fatalf("first verified load: %v", err)
	}
	if _, err := os.Stat(stampPath(path)); err != nil {
		t.Fatalf("verified load wrote no stamp: %v", err)
	}

	// Same size and mtime, different bytes: only a stamp-trusting load accepts it.
	if err := os.WriteFile(path, []byte(strings.Repeat("x", len(good))), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, fi.ModTime(), fi.ModTime()); err != nil {
		t.Fatal(err)
	}
	if _, err := fetchPinned(ctx, client, uri, pin, nil); err != nil {
		t.Fatalf("stamped file was re-hashed: %v", err)
	}

	// Without the stamp the same bytes are hashed, fail closed, and are evicted.
	if err := os.Remove(stampPath(path)); err != nil {
		t.Fatal(err)
	}
	if _, err := fetchPinned(ctx, client, uri, pin, nil); err == nil || !strings.Contains(err.Error(), "does not match pin") {
		t.Fatalf("unstamped corrupt file error = %v, want a pin mismatch", err)
	}
	if _, statErr := os.Stat(path); !os.IsNotExist(statErr) {
		t.Fatalf("corrupt file not evicted (stat err %v)", statErr)
	}
}

func TestStampBindsToPinDigest(t *testing.T) {
	path := filepath.Join(t.TempDir(), "artifact.gguf")
	if err := os.WriteFile(path, []byte("abc"), 0o644); err != nil {
		t.Fatal(err)
	}
	fi, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	pin := ArtifactPin{SHA256: strings.Repeat("a", 64), Bytes: 3}
	if err := writeStamp(path, fi, pin); err != nil {
		t.Fatal(err)
	}
	if !stampMatches(path, fi, pin) {
		t.Fatal("stamp does not match the pin that wrote it")
	}
	other := ArtifactPin{SHA256: strings.Repeat("b", 64), Bytes: 3}
	if stampMatches(path, fi, other) {
		t.Fatal("stamp accepted under a different pinned digest")
	}
}
