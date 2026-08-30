package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/selfupdate"
)

const selfUpdateManifestSchema = "fak.self-update-manifest/v2"

// This key is part of the client trust root, not manifest-controlled configuration.
// Release builds replace it only through a reviewed source change.
var selfUpdateManifestPublicKey = mustSelfUpdateManifestKey("11qYAYdk9Jf73B6NfP3qF0PW0mCz9v7sS5Yp2wT0G0A=")

var (
	selfUpdateManifestHTTPClient = http.DefaultClient
	selfUpdateManifestNow        = time.Now
)

type selfUpdateManifestPayload struct {
	Schema             string                     `json:"schema"`
	ManifestID         string                     `json:"manifest_id"`
	Channel            string                     `json:"channel"`
	Cohort             string                     `json:"cohort"`
	Platform           string                     `json:"platform"`
	Architecture       string                     `json:"architecture"`
	InstalledIdentity  string                     `json:"installed_identity"`
	MetadataGeneration uint64                     `json:"metadata_generation"`
	ExpiresAt          string                     `json:"expires_at"`
	Disposition        string                     `json:"disposition"`
	TargetVersion      string                     `json:"target_version"`
	TargetRevision     string                     `json:"target_revision"`
	RetryAt            string                     `json:"retry_at,omitempty"`
	Targets            []selfUpdateArtifactTarget `json:"targets,omitempty"`
}

type selfUpdateArtifactTarget struct {
	URL            string `json:"url"`
	Platform       string `json:"platform"`
	Architecture   string `json:"architecture"`
	SHA256         string `json:"sha256"`
	Size           int64  `json:"size"`
	AppVersion     string `json:"app_version"`
	SourceRevision string `json:"source_revision"`
}

type selfUpdateManifestEnvelope struct {
	Payload   json.RawMessage `json:"payload"`
	Signature string          `json:"signature"`
}

type selfUpdateManifestState struct {
	ETag         string                     `json:"etag,omitempty"`
	Envelope     selfUpdateManifestEnvelope `json:"envelope"`
	BackoffUntil string                     `json:"backoff_until,omitempty"`
}

type selfUpdateManifestRequest struct {
	URL, ManifestID, CachePath, Channel, Cohort, Platform, Architecture, InstalledIdentity string
	InstalledVersion                                                                       string
	InstalledGeneration                                                                    uint64
	Offline, Force                                                                         bool
}

type selfUpdateManifestSelection struct {
	Disposition, TargetVersion, TargetRevision string
	MetadataGeneration                         uint64
	Artifact                                   *selfUpdateArtifactTarget
	RetryAt                                    time.Time
}

func selfUpdateManifestBeforeFetch(ctx context.Context, q selfUpdateManifestRequest) (bool, string, error) {
	selection, err := selfUpdateManifestSelect(ctx, q)
	if err != nil {
		return false, "", err
	}
	return selection.Disposition == "update", selection.Disposition, nil
}
func selfUpdateManifestSelect(ctx context.Context, q selfUpdateManifestRequest) (selfUpdateManifestSelection, error) {
	if strings.TrimSpace(q.URL) == "" {
		return selfUpdateManifestSelection{Disposition: "update"}, nil
	}
	now := selfUpdateManifestNow().UTC()
	cache, cacheErr := loadSelfUpdateManifestState(q.CachePath)
	authenticateStoredManifest := func() (selfUpdateManifestSelection, error) {
		if cacheErr != nil {
			return selfUpdateManifestSelection{}, cacheErr
		}
		return authenticateSelfUpdateManifest(cache.Envelope, q, now)
	}
	if q.Offline {
		return authenticateStoredManifest()
	}
	if !q.Force && cacheErr == nil && strings.TrimSpace(cache.BackoffUntil) != "" {
		until, err := time.Parse(time.RFC3339, cache.BackoffUntil)
		if err == nil && now.Before(until) {
			s, err := authenticateStoredManifest()
			if err == nil {
				s.Disposition = "backoff"
				s.RetryAt = until
				return s, nil
			}
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, q.URL, nil)
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	cacheAuthenticated := false
	if cacheErr == nil {
		_, cacheAuthErr := authenticateStoredManifest()
		cacheAuthenticated = cacheAuthErr == nil
	}
	if cacheAuthenticated && cache.ETag != "" && !q.Force {
		req.Header.Set("If-None-Match", cache.ETag)
	}
	resp, err := selfUpdateManifestHTTPClient.Do(req)
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified {
		if !cacheAuthenticated {
			return selfUpdateManifestSelection{}, errors.New("manifest 304 without authenticated identity-matched cache")
		}
		return authenticateStoredManifest()
	}
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable {
		until, err := boundedSelfUpdateRetryAfter(resp.Header.Get("Retry-After"), now)
		if err != nil {
			return selfUpdateManifestSelection{}, err
		}
		if !cacheAuthenticated {
			return selfUpdateManifestSelection{}, errors.New("manifest retry response without authenticated cache")
		}
		s, err := authenticateStoredManifest()
		if err != nil {
			return selfUpdateManifestSelection{}, fmt.Errorf("manifest retry cache: %w", err)
		}
		cache.BackoffUntil = until.Format(time.RFC3339)
		if err := saveSelfUpdateManifestState(q.CachePath, cache); err != nil {
			return selfUpdateManifestSelection{}, err
		}
		s.Disposition = "backoff"
		s.RetryAt = until
		return s, nil
	}
	if resp.StatusCode != http.StatusOK {
		return selfUpdateManifestSelection{}, fmt.Errorf("manifest HTTP status %s", resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	var env selfUpdateManifestEnvelope
	if err := json.Unmarshal(body, &env); err != nil {
		return selfUpdateManifestSelection{}, fmt.Errorf("manifest envelope: %w", err)
	}
	s, err := authenticateSelfUpdateManifest(env, q, now)
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	if cacheErr == nil {
		previous, previousErr := authenticatePriorSelfUpdateManifest(cache.Envelope, q)
		if previousErr != nil {
			var priorPayload selfUpdateManifestPayload
			if json.Unmarshal(cache.Envelope.Payload, &priorPayload) == nil && priorPayload.Schema == selfUpdateManifestSchema {
				return selfUpdateManifestSelection{}, fmt.Errorf("manifest prior cache: %w", previousErr)
			}
		} else {
			if s.MetadataGeneration < previous.MetadataGeneration {
				return selfUpdateManifestSelection{}, errors.New("manifest metadata generation rollback")
			}
			if s.MetadataGeneration == previous.MetadataGeneration && !bytes.Equal(env.Payload, cache.Envelope.Payload) {
				return selfUpdateManifestSelection{}, errors.New("manifest freeze detected: same metadata generation changed payload")
			}
		}
	}
	stored := selfUpdateManifestState{ETag: resp.Header.Get("ETag"), Envelope: env}
	if !s.RetryAt.IsZero() && now.Before(s.RetryAt) {
		stored.BackoffUntil = s.RetryAt.Format(time.RFC3339)
	}
	if err := saveSelfUpdateManifestState(q.CachePath, stored); err != nil {
		return selfUpdateManifestSelection{}, err
	}
	return s, nil
}

func authenticatePriorSelfUpdateManifest(env selfUpdateManifestEnvelope, q selfUpdateManifestRequest) (selfUpdateManifestSelection, error) {
	var prior selfUpdateManifestPayload
	if err := json.Unmarshal(env.Payload, &prior); err != nil {
		return selfUpdateManifestSelection{}, err
	}
	q.InstalledIdentity = prior.InstalledIdentity
	q.InstalledVersion = ""
	q.InstalledGeneration = 0
	return authenticateSelfUpdateManifest(env, q, time.Time{})
}

func authenticateSelfUpdateManifest(env selfUpdateManifestEnvelope, q selfUpdateManifestRequest, now time.Time) (selfUpdateManifestSelection, error) {
	sig, err := base64.StdEncoding.DecodeString(env.Signature)
	if err != nil || !ed25519.Verify(selfUpdateManifestPublicKey, env.Payload, sig) {
		return selfUpdateManifestSelection{}, errors.New("manifest signature verification failed")
	}
	var p selfUpdateManifestPayload
	if err := json.Unmarshal(env.Payload, &p); err != nil {
		return selfUpdateManifestSelection{}, fmt.Errorf("manifest payload: %w", err)
	}
	canonical, err := json.Marshal(p)
	if err != nil || !bytes.Equal(canonical, env.Payload) {
		return selfUpdateManifestSelection{}, errors.New("manifest payload is not canonical JSON")
	}
	if p.Schema != selfUpdateManifestSchema || p.ManifestID != q.ManifestID || strings.TrimSpace(q.ManifestID) == "" || p.Channel != q.Channel || p.Cohort != q.Cohort || p.Platform != q.Platform || p.Architecture != q.Architecture || p.InstalledIdentity != q.InstalledIdentity {
		return selfUpdateManifestSelection{}, errors.New("manifest identity mismatch")
	}
	if p.MetadataGeneration == 0 || p.MetadataGeneration < q.InstalledGeneration {
		return selfUpdateManifestSelection{}, errors.New("manifest metadata generation rollback")
	}
	expires, err := time.Parse(time.RFC3339, p.ExpiresAt)
	if err != nil || (!now.IsZero() && !now.Before(expires)) {
		return selfUpdateManifestSelection{}, errors.New("manifest expired")
	}
	if p.Disposition != "update" && p.Disposition != "no-update" && p.Disposition != "cohort-hold" {
		return selfUpdateManifestSelection{}, errors.New("manifest disposition invalid")
	}
	if p.Disposition == "update" {
		if !isFullGitCommit(p.TargetRevision) {
			return selfUpdateManifestSelection{}, errors.New("manifest target revision is not a full Git object ID")
		}
		if _, err := selfupdate.CompareReleaseVersions(p.TargetVersion, p.TargetVersion); err != nil {
			return selfUpdateManifestSelection{}, fmt.Errorf("manifest target app version: %w", err)
		}
	}
	if strings.TrimSpace(q.InstalledVersion) != "" && strings.TrimSpace(p.TargetVersion) != "" {
		cmp, err := selfupdate.CompareReleaseVersions(p.TargetVersion, q.InstalledVersion)
		if err != nil {
			return selfUpdateManifestSelection{}, fmt.Errorf("manifest version ordering: %w", err)
		}
		if cmp < 0 {
			return selfUpdateManifestSelection{}, errors.New("manifest app version rollback")
		}
	}
	var artifact *selfUpdateArtifactTarget
	for i := range p.Targets {
		target := p.Targets[i]
		if target.Platform != q.Platform || target.Architecture != q.Architecture {
			continue
		}
		if artifact != nil {
			return selfUpdateManifestSelection{}, errors.New("manifest has multiple usable artifact targets")
		}
		if err := validateSelfUpdateArtifactTarget(target, p); err != nil {
			return selfUpdateManifestSelection{}, err
		}
		target.SHA256 = strings.ToLower(target.SHA256)
		artifact = &target
	}
	var retry time.Time
	if p.RetryAt != "" {
		retry, err = time.Parse(time.RFC3339, p.RetryAt)
		if err != nil {
			return selfUpdateManifestSelection{}, errors.New("manifest retry_at invalid")
		}
	}
	selection := selfUpdateManifestSelection{
		Disposition: p.Disposition, TargetVersion: p.TargetVersion, TargetRevision: p.TargetRevision,
		MetadataGeneration: p.MetadataGeneration, Artifact: artifact, RetryAt: retry,
	}
	if !retry.IsZero() && now.Before(retry) {
		selection.Disposition = "backoff"
	}
	return selection, nil
}

func validateSelfUpdateArtifactTarget(target selfUpdateArtifactTarget, payload selfUpdateManifestPayload) error {
	if target.Platform != payload.Platform || target.Architecture != payload.Architecture ||
		target.AppVersion != payload.TargetVersion || !strings.EqualFold(target.SourceRevision, payload.TargetRevision) {
		return errors.New("manifest artifact identity mismatch")
	}
	if !isFullGitCommit(target.SourceRevision) || !validSelfUpdateSHA256(target.SHA256) || target.Size < 1 {
		return errors.New("manifest artifact identity invalid")
	}
	u, err := url.Parse(target.URL)
	if err != nil || u.Host == "" || (u.Scheme != "https" && !(u.Scheme == "http" && (u.Hostname() == "127.0.0.1" || u.Hostname() == "localhost"))) {
		return errors.New("manifest artifact URL must be HTTPS")
	}
	if _, err := selfupdate.CompareReleaseVersions(target.AppVersion, target.AppVersion); err != nil {
		return fmt.Errorf("manifest artifact app version: %w", err)
	}
	return nil
}

func validSelfUpdateSHA256(value string) bool {
	value = strings.TrimSpace(value)
	if len(value) != sha256.Size*2 {
		return false
	}
	_, err := hex.DecodeString(value)
	return err == nil
}

func downloadSelfUpdateArtifact(ctx context.Context, target selfUpdateArtifactTarget, dir string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, target.URL, nil)
	if err != nil {
		return "", err
	}
	resp, err := selfUpdateManifestHTTPClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("artifact HTTP status %s", resp.Status)
	}
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return "", err
	}
	out, err := os.CreateTemp(dir, "fak-self-update-artifact-*")
	if err != nil {
		return "", err
	}
	path := out.Name()
	ok := false
	defer func() {
		_ = out.Close()
		if !ok {
			_ = os.Remove(path)
		}
	}()
	hash := sha256.New()
	n, err := io.Copy(io.MultiWriter(out, hash), io.LimitReader(resp.Body, target.Size+1))
	if err != nil {
		return "", err
	}
	if n != target.Size {
		return "", fmt.Errorf("artifact size mismatch: got %d want %d", n, target.Size)
	}
	if digest := hex.EncodeToString(hash.Sum(nil)); !strings.EqualFold(digest, target.SHA256) {
		return "", errors.New("artifact SHA-256 mismatch")
	}
	if err := out.Sync(); err != nil {
		return "", err
	}
	if err := out.Chmod(0o755); err != nil {
		return "", err
	}
	if err := out.Close(); err != nil {
		return "", err
	}
	ok = true
	return path, nil
}

func selfUpdateArtifactBindingDigest(target selfUpdateArtifactTarget, generation uint64) string {
	body, _ := json.Marshal(struct {
		Generation uint64                   `json:"metadata_generation"`
		Target     selfUpdateArtifactTarget `json:"target"`
	}{Generation: generation, Target: target})
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func boundedSelfUpdateRetryAfter(v string, now time.Time) (time.Time, error) {
	const max = 24 * time.Hour
	var until time.Time
	if seconds, err := strconv.Atoi(strings.TrimSpace(v)); err == nil && seconds >= 0 {
		until = now.Add(time.Duration(seconds) * time.Second)
	} else if parsed, err := http.ParseTime(v); err == nil {
		until = parsed
	} else {
		return time.Time{}, errors.New("manifest retry response has invalid Retry-After")
	}
	if until.Before(now) {
		until = now
	}
	if until.After(now.Add(max)) {
		until = now.Add(max)
	}
	return until, nil
}

func loadSelfUpdateManifestState(path string) (selfUpdateManifestState, error) {
	var c selfUpdateManifestState
	if strings.TrimSpace(path) == "" {
		return c, errors.New("manifest cache path is empty")
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return c, err
	}
	if err = json.Unmarshal(b, &c); err != nil {
		return c, err
	}
	return c, nil
}
func saveSelfUpdateManifestState(path string, c selfUpdateManifestState) error {
	if strings.TrimSpace(path) == "" {
		return errors.New("manifest cache path is empty")
	}
	b, err := json.Marshal(c)
	if err != nil {
		return err
	}
	if err = os.MkdirAll(filepath.Dir(path), 0700); err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err = os.WriteFile(tmp, b, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}
func defaultSelfUpdateManifestStatePath() string {
	d, err := os.UserCacheDir()
	if err != nil {
		return ""
	}
	return filepath.Join(d, "fak", "self-update-manifest.json")
}
func mustSelfUpdateManifestKey(s string) ed25519.PublicKey {
	b, err := base64.StdEncoding.DecodeString(s)
	if err != nil || len(b) != ed25519.PublicKeySize {
		panic("invalid self-update manifest public key")
	}
	return ed25519.PublicKey(b)
}
