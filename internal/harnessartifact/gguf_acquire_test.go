package harnessartifact

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestDefaultGGUFHTTPClientHasLongBoundedTimeout(t *testing.T) {
	if got := defaultGGUFHTTPClient().Timeout; got != defaultGGUFDownloadTimeout || got < time.Hour {
		t.Fatalf("default GGUF client timeout = %s, want long bounded timeout %s", got, defaultGGUFDownloadTimeout)
	}
}

func TestGGUFAcquisitionPlanApplyAndOfflineReuse(t *testing.T) {
	fixture, err := os.ReadFile("testdata/tiny.gguf")
	if err != nil {
		t.Fatal(err)
	}
	sum := sha256.Sum256(fixture)
	digest := hex.EncodeToString(sum[:])
	requests := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { requests++; _, _ = w.Write(fixture) }))
	cache := filepath.Join(t.TempDir(), "cache")
	req := GGUFRequest{Source: server.URL + "/tiny.gguf", License: "test-fixture-only", Bytes: int64(len(fixture)), SHA256: digest, CacheDir: cache}
	plan, err := PlanGGUFAcquisition(req)
	if err != nil {
		t.Fatal(err)
	}
	if requests != 0 {
		t.Fatalf("plan made %d network calls", requests)
	}
	if _, err := os.Stat(cache); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("plan mutated cache: %v", err)
	}
	receipt, err := ApplyGGUFAcquisition(context.Background(), plan, GGUFApplyOptions{Client: server.Client()})
	if err != nil {
		t.Fatal(err)
	}
	if receipt.CacheHit || requests != 1 {
		t.Fatalf("first receipt=%+v requests=%d", receipt, requests)
	}
	promoted, err := os.ReadFile(receipt.Path)
	if err != nil {
		t.Fatal(err)
	}
	got := sha256.Sum256(promoted)
	if got != sum {
		t.Fatalf("independent digest = %x, want %x", got, sum)
	}
	server.Close()
	offline, err := ApplyGGUFAcquisition(context.Background(), plan, GGUFApplyOptions{Offline: true})
	if err != nil {
		t.Fatal(err)
	}
	if !offline.CacheHit || requests != 1 {
		t.Fatalf("offline receipt=%+v requests=%d", offline, requests)
	}
}

func TestGGUFAcquisitionRefusesMismatchWithoutPromotion(t *testing.T) {
	fixture := []byte("GGUF-bad")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(fixture) }))
	defer server.Close()
	cache := filepath.Join(t.TempDir(), "cache")
	digest := hex.EncodeToString(make([]byte, sha256.Size))
	plan, err := PlanGGUFAcquisition(GGUFRequest{Source: server.URL, License: "test", Bytes: int64(len(fixture)), SHA256: digest, CacheDir: cache})
	if err != nil {
		t.Fatal(err)
	}
	_, err = ApplyGGUFAcquisition(context.Background(), plan, GGUFApplyOptions{Client: server.Client()})
	if !errors.Is(err, ErrGGUFDigestMismatch) {
		t.Fatalf("error=%v", err)
	}
	if _, err := os.Stat(plan.Destination); !errors.Is(err, os.ErrNotExist) {
		t.Fatalf("mismatch promoted: %v", err)
	}
	entries, err := os.ReadDir(cache)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Fatalf("temporary residue: %v", entries)
	}
	_, err = ApplyGGUFAcquisition(context.Background(), plan, GGUFApplyOptions{Offline: true})
	if !errors.Is(err, ErrGGUFOfflineMiss) {
		t.Fatalf("offline error=%v", err)
	}
}
