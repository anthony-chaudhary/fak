package main

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
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
	if err != nil || s.Disposition != "update" || s.TargetRevision != strings.Repeat("a", 40) {
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
				if tc.name == "valid" && r.Header.Get("If-None-Match") != `"x"` {
					t.Errorf("If-None-Match missing for authenticated cache")
				}
				if tc.name != "valid" && r.Header.Get("If-None-Match") != "" {
					t.Errorf("invalid cache was used conditionally")
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

func TestSelfUpdateManifestAuthenticatesFullArtifactTarget(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	artifact := []byte("release artifact")
	artifactServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(artifact)
	}))
	defer artifactServer.Close()
	p := validManifestPayload(now)
	p.TargetVersion = "2.0.0"
	p.TargetRevision = strings.Repeat("a", 40)
	p.Targets = []selfUpdateArtifactTarget{validArtifactTarget(artifactServer.URL, artifact, p)}
	p.Targets[0].Deltas = []selfUpdateArtifactDelta{{
		URL: artifactServer.URL, Format: selfUpdateDeltaFormat, SourceSHA256: strings.Repeat("b", 64),
		SHA256: digestManifestFixture(artifact), Size: int64(len(artifact)),
	}}
	srv := serveManifest(t, http.StatusOK, signedManifestEnvelope(t, priv, p), nil)
	defer srv.Close()
	s, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json")))
	if err != nil || s.Artifact == nil || s.MetadataGeneration != p.MetadataGeneration || len(s.Artifact.Deltas) != 1 ||
		s.Artifact.Deltas[0].SourceSHA256 != strings.Repeat("b", 64) {
		t.Fatalf("artifact selection = %+v, %v", s, err)
	}
	path, err := downloadSelfUpdateArtifact(context.Background(), *s.Artifact, t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	if got, _ := os.ReadFile(path); string(got) != string(artifact) {
		t.Fatalf("downloaded artifact = %q", got)
	}
}

func TestSelfUpdateManifestRejectsTamperedDeltaMetadata(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	p := validManifestPayload(selfUpdateManifestNow().UTC())
	body := []byte("release artifact")
	p.Targets = []selfUpdateArtifactTarget{validArtifactTarget("https://updates.example/full", body, p)}
	p.Targets[0].Deltas = []selfUpdateArtifactDelta{{
		URL: "https://updates.example/delta", Format: selfUpdateDeltaFormat,
		SourceSHA256: strings.Repeat("b", 64), SHA256: strings.Repeat("c", 64), Size: 7,
	}}
	env := signedManifestEnvelope(t, priv, p)
	var payload selfUpdateManifestPayload
	if err := json.Unmarshal(env.Payload, &payload); err != nil {
		t.Fatal(err)
	}
	payload.Targets[0].Deltas[0].Size++
	env.Payload, _ = json.Marshal(payload)
	srv := serveManifest(t, http.StatusOK, env, nil)
	defer srv.Close()
	if _, err := selfUpdateManifestSelect(context.Background(), manifestRequest(srv.URL, filepath.Join(t.TempDir(), "m.json"))); err == nil ||
		!strings.Contains(err.Error(), "signature verification failed") {
		t.Fatalf("tampered signed delta accepted: %v", err)
	}
}

func TestSelfUpdateManifestRejectsRollbackFreezeAndMixMatch(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	cache := filepath.Join(t.TempDir(), "m.json")
	base := validManifestPayload(now)
	base.MetadataGeneration = 9
	base.TargetVersion = "2.0.0"
	base.TargetRevision = strings.Repeat("a", 40)
	body := []byte("artifact")
	base.Targets = []selfUpdateArtifactTarget{validArtifactTarget("https://updates.example/fak", body, base)}
	if err := saveSelfUpdateManifestState(cache, selfUpdateManifestState{Envelope: signedManifestEnvelope(t, priv, base)}); err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name string
		edit func(*selfUpdateManifestPayload)
	}{
		{"generation rollback", func(p *selfUpdateManifestPayload) { p.MetadataGeneration = 8 }},
		{"version rollback", func(p *selfUpdateManifestPayload) {
			p.MetadataGeneration = 10
			p.TargetVersion = "1.9.9"
			p.Targets[0].AppVersion = p.TargetVersion
		}},
		{"freeze", func(p *selfUpdateManifestPayload) { p.Targets[0].URL = "https://updates.example/changed" }},
		{"mix-match version", func(p *selfUpdateManifestPayload) {
			p.MetadataGeneration = 10
			p.Targets[0].AppVersion = "9.9.9"
		}},
		{"mix-match revision", func(p *selfUpdateManifestPayload) {
			p.MetadataGeneration = 10
			p.Targets[0].SourceRevision = strings.Repeat("b", 40)
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			p := base
			p.Targets = append([]selfUpdateArtifactTarget(nil), base.Targets...)
			tc.edit(&p)
			srv := serveManifest(t, http.StatusOK, signedManifestEnvelope(t, priv, p), nil)
			defer srv.Close()
			q := manifestRequest(srv.URL, cache)
			q.Force = true
			q.InstalledVersion = "2.0.0"
			q.InstalledGeneration = 9
			if _, err := selfUpdateManifestSelect(context.Background(), q); err == nil {
				t.Fatal("rollback/freeze/mix-match manifest accepted")
			}
		})
	}
}

func TestSelfUpdateManifestCarriesGenerationAcrossInstalledIdentityChange(t *testing.T) {
	pub, priv, _ := ed25519.GenerateKey(rand.Reader)
	withManifestTrust(t, pub)
	now := selfUpdateManifestNow().UTC()
	cache := filepath.Join(t.TempDir(), "m.json")
	prior := validManifestPayload(now)
	prior.MetadataGeneration = 9
	prior.InstalledIdentity = "old-installed-revision"
	if err := saveSelfUpdateManifestState(cache, selfUpdateManifestState{ETag: `"old"`, Envelope: signedManifestEnvelope(t, priv, prior)}); err != nil {
		t.Fatal(err)
	}
	next := prior
	next.MetadataGeneration = 10
	next.InstalledIdentity = "new-installed-revision"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("If-None-Match") != "" {
			t.Error("identity-mismatched cache was used for a conditional request")
		}
		_ = json.NewEncoder(w).Encode(signedManifestEnvelope(t, priv, next))
	}))
	defer srv.Close()
	q := manifestRequest(srv.URL, cache)
	q.InstalledIdentity = next.InstalledIdentity
	q.InstalledGeneration = 9
	s, err := selfUpdateManifestSelect(context.Background(), q)
	if err != nil || s.MetadataGeneration != 10 {
		t.Fatalf("identity rollover selection = %+v, %v", s, err)
	}
}

func TestDownloadSelfUpdateArtifactRejectsCorruption(t *testing.T) {
	body := []byte("corrupt artifact")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) { _, _ = w.Write(body) }))
	defer srv.Close()
	target := selfUpdateArtifactTarget{URL: srv.URL, SHA256: strings.Repeat("0", 64), Size: int64(len(body))}
	if _, err := downloadSelfUpdateArtifact(context.Background(), target, t.TempDir()); err == nil {
		t.Fatal("corrupt artifact accepted")
	}
	target.SHA256 = digestManifestFixture(body)
	target.Size--
	if _, err := downloadSelfUpdateArtifact(context.Background(), target, t.TempDir()); err == nil {
		t.Fatal("wrong artifact size accepted")
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
	return selfUpdateManifestPayload{Schema: selfUpdateManifestSchema, ManifestID: "m1", Channel: "stable", Cohort: "c1", Platform: runtime.GOOS, Architecture: runtime.GOARCH, InstalledIdentity: "installed-r1", MetadataGeneration: 2, ExpiresAt: now.Add(time.Hour).Format(time.RFC3339), Disposition: "update", TargetVersion: "v2", TargetRevision: strings.Repeat("a", 40)}
}

func validArtifactTarget(url string, body []byte, p selfUpdateManifestPayload) selfUpdateArtifactTarget {
	return selfUpdateArtifactTarget{
		URL: url, Platform: p.Platform, Architecture: p.Architecture,
		SHA256: digestManifestFixture(body), Size: int64(len(body)),
		AppVersion: p.TargetVersion, SourceRevision: p.TargetRevision,
	}
}

func digestManifestFixture(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
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
