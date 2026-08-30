package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestSelfUpdateManifestAuthenticatedSelectionAndCache(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	oldKey, oldNow := selfUpdateManifestPublicKey, selfUpdateManifestNow
	selfUpdateManifestPublicKey = pub
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	selfUpdateManifestNow = func() time.Time { return now }
	t.Cleanup(func() { selfUpdateManifestPublicKey, selfUpdateManifestNow = oldKey, oldNow })

	cache := filepath.Join(t.TempDir(), "manifest.json")
	q := selfUpdateManifestRequest{ManifestID: "m1", CachePath: cache, Channel: "stable", Cohort: "c1", Platform: runtime.GOOS, Architecture: runtime.GOARCH, InstalledIdentity: "installed-r1"}
	payload := validManifestPayload(now)
	envelope := signedManifestEnvelope(t, priv, payload)
	var requests atomic.Int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		w.Header().Set("ETag", `"m1"`)
		_ = json.NewEncoder(w).Encode(envelope)
	}))
	defer server.Close()
	q.URL = server.URL

	s, err := selfUpdateManifestSelect(context.Background(), q)
	if err != nil || s.Disposition != "update" || s.TargetRevision != "target-r2" {
		t.Fatalf("authenticated 200 = %+v, %v", s, err)
	}
	cached, err := loadSelfUpdateManifestState(cache)
	if err != nil || cached.ETag != `"m1"` {
		t.Fatalf("cache = %+v, %v", cached, err)
	}
	if requests.Load() != 1 {
		t.Fatalf("requests = %d", requests.Load())
	}
}

func TestSelfUpdateManifest304RequiresValidCache(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	valid := signedManifestEnvelope(t, priv, validManifestPayload(now))

	for _, tc := range []struct {
		name    string
		mutate  func(*selfUpdateManifestPayload)
		wantErr bool
	}{
		{"valid", func(*selfUpdateManifestPayload) {}, false},
		{"expired", func(p *selfUpdateManifestPayload) { p.ExpiresAt = now.Add(-time.Second).Format(time.RFC3339) }, true},
		{"identity-mismatch", func(p *selfUpdateManifestPayload) { p.InstalledIdentity = "other" }, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			cache := filepath.Join(t.TempDir(), "m.json")
			env := valid
			if tc.name != "valid" {
				p := validManifestPayload(now)
				tc.mutate(&p)
				env = signedManifestEnvelope(t, priv, p)
			}
			if err := saveSelfUpdateManifestState(cache, selfUpdateManifestState{ETag: `"x"`, Envelope: env}); err != nil {
				t.Fatal(err)
			}
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				if r.Header.Get("If-None-Match") != `"x"` {
					t.Errorf("If-None-Match missing")
				}
				w.WriteHeader(http.StatusNotModified)
			}))
			defer srv.Close()
			_, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, cache))
			if (err != nil) != tc.wantErr {
				t.Fatalf("error=%v wantErr=%v", err, tc.wantErr)
			}
		})
	}

	missing := filepath.Join(t.TempDir(), "missing.json")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { w.WriteHeader(http.StatusNotModified) }))
	defer srv.Close()
	if _, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, missing)); err == nil {
		t.Fatal("304 without cache accepted")
	}
}

func TestSelfUpdateManifestRejectsManifestIDMismatch(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	p := validManifestPayload(selfUpdateManifestNow().UTC())
	srv := serveManifest(t, http.StatusOK, signedManifestEnvelope(t, priv, p), nil)
	defer srv.Close()
	q := manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json"))
	q.ManifestID = "different-manifest"
	if _, err := selfUpdateManifestSelect(context.Background(), q); err == nil || !strings.Contains(err.Error(), "identity mismatch") {
		t.Fatalf("manifest ID mismatch err = %v", err)
	}
}
func TestSelfUpdateManifestHoldDispositionsAndForgedSignature(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	for _, disposition := range []string{"no-update", "cohort-hold"} {
		t.Run(disposition, func(t *testing.T) {
			p := validManifestPayload(now)
			p.Disposition = disposition
			env := signedManifestEnvelope(t, priv, p)
			srv := serveManifest(t, http.StatusOK, env, nil)
			defer srv.Close()
			s, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json")))
			if err != nil || s.Disposition != disposition {
				t.Fatalf("selection=%+v err=%v", s, err)
			}
		})
	}
	p := validManifestPayload(now)
	env := signedManifestEnvelope(t, priv, p)
	env.Signature = base64.StdEncoding.EncodeToString(make([]byte, ed25519.SignatureSize))
	srv := serveManifest(t, http.StatusOK, env, nil)
	defer srv.Close()
	if _, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json"))); err == nil {
		t.Fatal("forged signature accepted")
	}
}

func TestSelfUpdateManifestRetryAfterBackoffAndForce(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	cache := filepath.Join(t.TempDir(), "m.json")
	if err := saveSelfUpdateManifestState(cache, selfUpdateManifestState{Envelope: signedManifestEnvelope(t, priv, validManifestPayload(now))}); err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := calls.Add(1)
		if n == 1 {
			w.Header().Set("Retry-After", "999999")
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		_ = json.NewEncoder(w).Encode(signedManifestEnvelope(t, priv, validManifestPayload(now)))
	}))
	defer srv.Close()
	q := manifestRequest(srv.URL, cache)
	s, err := selfUpdateManifestSelect(context.Background(), q)
	if err != nil || s.Disposition != "backoff" || s.RetryAt.Sub(now) != 24*time.Hour {
		t.Fatalf("retry=%v err=%v", s.RetryAt, err)
	}
	if _, err = selfUpdateManifestSelect(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("cached backoff contacted server: %d", calls.Load())
	}
	q.Force = true
	if _, err = selfUpdateManifestSelect(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 2 {
		t.Fatalf("force did not refresh: %d", calls.Load())
	}
}

func TestSelfUpdateManifestSignedRetryCachesBackoff(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	p := validManifestPayload(now)
	p.RetryAt = now.Add(15 * time.Minute).Format(time.RFC3339)
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		_ = json.NewEncoder(w).Encode(signedManifestEnvelope(t, priv, p))
	}))
	defer srv.Close()
	q := manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json"))
	s, err := selfUpdateManifestSelect(context.Background(), q)
	if err != nil || s.Disposition != "backoff" {
		t.Fatalf("signed retry selection = %+v, %v", s, err)
	}
	if _, err := selfUpdateManifestSelect(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 1 {
		t.Fatalf("signed cached backoff requests = %d", calls.Load())
	}
}
func TestSelfUpdateManifestOfflineZeroRequest(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	cache := filepath.Join(t.TempDir(), "m.json")
	if err := saveSelfUpdateManifestState(cache, selfUpdateManifestState{Envelope: signedManifestEnvelope(t, priv, validManifestPayload(selfUpdateManifestNow().UTC()))}); err != nil {
		t.Fatal(err)
	}
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) { calls.Add(1) }))
	defer srv.Close()
	q := manifestRequest(srv.URL, cache)
	q.Offline = true
	if _, err := selfUpdateManifestSelect(context.Background(), q); err != nil {
		t.Fatal(err)
	}
	if calls.Load() != 0 {
		t.Fatalf("offline requests=%d", calls.Load())
	}
	q.CachePath = filepath.Join(t.TempDir(), "missing")
	if _, err := selfUpdateManifestSelect(context.Background(), q); err == nil {
		t.Fatal("offline missing cache accepted")
	}
}

func validManifestPayload(now time.Time) selfUpdateManifestPayload {
	return selfUpdateManifestPayload{Schema: selfUpdateManifestSchema, ManifestID: "m1", Channel: "stable", Cohort: "c1", Platform: runtime.GOOS, Architecture: runtime.GOARCH, InstalledIdentity: "installed-r1", ExpiresAt: now.Add(time.Hour).Format(time.RFC3339), Disposition: "update", TargetVersion: "v2", TargetRevision: "target-r2"}
}
func signedManifestEnvelope(t *testing.T, priv ed25519.PrivateKey, p selfUpdateManifestPayload) selfUpdateManifestEnvelope {
	t.Helper()
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	return selfUpdateManifestEnvelope{Payload: b, Signature: base64.StdEncoding.EncodeToString(ed25519.Sign(priv, b))}
}
func manifestRequest(url, cache string) selfUpdateManifestRequest {
	return selfUpdateManifestRequest{URL: url, ManifestID: "m1", CachePath: cache, Channel: "stable", Cohort: "c1", Platform: runtime.GOOS, Architecture: runtime.GOARCH, InstalledIdentity: "installed-r1"}
}
func withManifestTrust(t *testing.T, pub ed25519.PublicKey) {
	t.Helper()
	oldKey, oldNow := selfUpdateManifestPublicKey, selfUpdateManifestNow
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	selfUpdateManifestPublicKey = pub
	selfUpdateManifestNow = func() time.Time { return now }
	t.Cleanup(func() { selfUpdateManifestPublicKey, selfUpdateManifestNow = oldKey, oldNow })
}
func serveManifest(t *testing.T, status int, env selfUpdateManifestEnvelope, headers map[string]string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for k, v := range headers {
			w.Header().Set(k, v)
		}
		w.WriteHeader(status)
		if status == http.StatusOK {
			_ = json.NewEncoder(w).Encode(env)
		}
	}))
}

func TestSelfUpdateManifestHoldReturnsBeforeFetchAndPreservesInstalledIdentity(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	for _, disposition := range []string{"no-update", "cohort-hold"} {
		t.Run(disposition, func(t *testing.T) {
			p := validManifestPayload(selfUpdateManifestNow().UTC())
			p.Disposition = disposition
			srv := serveManifest(t, http.StatusOK, signedManifestEnvelope(t, priv, p), nil)
			defer srv.Close()
			installedIdentity := "installed-r1"
			fetches := 0
			proceed, got, err := selfUpdateManifestBeforeFetch(context.Background(), manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json")))
			if err != nil || proceed || got != disposition {
				t.Fatalf("gate = proceed %v disposition %q err %v", proceed, got, err)
			}
			if proceed {
				fetches++
				installedIdentity = "target-r2"
			}
			if fetches != 0 || installedIdentity != "installed-r1" {
				t.Fatalf("hold changed update state: fetches=%d identity=%q", fetches, installedIdentity)
			}
		})
	}
}
func TestSelfUpdateManifestRetryAfterSecondsParser(t *testing.T) {
	now := time.Now().UTC()
	got, err := boundedSelfUpdateRetryAfter(strconv.Itoa(60), now)
	if err != nil || got.Sub(now) != time.Minute {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
