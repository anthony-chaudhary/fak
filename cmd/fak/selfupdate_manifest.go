package main

import (
	"bytes"
	"context"
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const selfUpdateManifestSchema = "fak.self-update-manifest/v1"

// This key is part of the client trust root, not manifest-controlled configuration.
// Release builds replace it only through a reviewed source change.
var selfUpdateManifestPublicKey = mustSelfUpdateManifestKey("11qYAYdk9Jf73B6NfP3qF0PW0mCz9v7sS5Yp2wT0G0A=")

var (
	selfUpdateManifestHTTPClient = http.DefaultClient
	selfUpdateManifestNow        = time.Now
)

type selfUpdateManifestPayload struct {
	Schema            string `json:"schema"`
	ManifestID        string `json:"manifest_id"`
	Channel           string `json:"channel"`
	Cohort            string `json:"cohort"`
	Platform          string `json:"platform"`
	Architecture      string `json:"architecture"`
	InstalledIdentity string `json:"installed_identity"`
	ExpiresAt         string `json:"expires_at"`
	Disposition       string `json:"disposition"`
	TargetVersion     string `json:"target_version"`
	TargetRevision    string `json:"target_revision"`
	RetryAt           string `json:"retry_at,omitempty"`
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
	Offline, Force                                                                         bool
}

type selfUpdateManifestSelection struct {
	Disposition, TargetVersion, TargetRevision string
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
			if err != nil {
				return selfUpdateManifestSelection{}, fmt.Errorf("manifest backoff cache: %w", err)
			}
			s.Disposition = "backoff"
			s.RetryAt = until
			return s, nil
		}
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, q.URL, nil)
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	if cacheErr == nil && cache.ETag != "" && !q.Force {
		req.Header.Set("If-None-Match", cache.ETag)
	}
	resp, err := selfUpdateManifestHTTPClient.Do(req)
	if err != nil {
		return selfUpdateManifestSelection{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotModified {
		return authenticateStoredManifest()
	}
	if resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode == http.StatusServiceUnavailable {
		until, err := boundedSelfUpdateRetryAfter(resp.Header.Get("Retry-After"), now)
		if err != nil {
			return selfUpdateManifestSelection{}, err
		}
		if cacheErr != nil {
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
	stored := selfUpdateManifestState{ETag: resp.Header.Get("ETag"), Envelope: env}
	if !s.RetryAt.IsZero() && now.Before(s.RetryAt) {
		stored.BackoffUntil = s.RetryAt.Format(time.RFC3339)
	}
	if err := saveSelfUpdateManifestState(q.CachePath, stored); err != nil {
		return selfUpdateManifestSelection{}, err
	}
	return s, nil
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
	expires, err := time.Parse(time.RFC3339, p.ExpiresAt)
	if err != nil || !now.Before(expires) {
		return selfUpdateManifestSelection{}, errors.New("manifest expired")
	}
	if p.Disposition != "update" && p.Disposition != "no-update" && p.Disposition != "cohort-hold" {
		return selfUpdateManifestSelection{}, errors.New("manifest disposition invalid")
	}
	var retry time.Time
	if p.RetryAt != "" {
		retry, err = time.Parse(time.RFC3339, p.RetryAt)
		if err != nil {
			return selfUpdateManifestSelection{}, errors.New("manifest retry_at invalid")
		}
	}
	selection := selfUpdateManifestSelection{Disposition: p.Disposition, TargetVersion: p.TargetVersion, TargetRevision: p.TargetRevision, RetryAt: retry}
	if !retry.IsZero() && now.Before(retry) {
		selection.Disposition = "backoff"
	}
	return selection, nil
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
