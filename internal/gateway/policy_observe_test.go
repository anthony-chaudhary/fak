package gateway

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestPolicyObserveRouteAttestsInstalledFloor is the core #3960 acceptance: GET
// /v1/fak/policy answers 200 with the source + effective digest of the floor that
// governs the process right now, and does it WITHOUT touching the reloader.
func TestPolicyObserveRouteAttestsInstalledFloor(t *testing.T) {
	srv := newTestServer(t)
	observes, reloads := 0, 0
	srv.observePolicy = func(context.Context) (PolicyObservation, error) {
		observes++
		return PolicyObservation{
			Source:          "floor.json",
			EffectiveDigest: "policy-sha256:abc123",
			Summary:         "posture: fail_closed",
		}, nil
	}
	srv.reloadPolicy = func(context.Context) (PolicyReloadResponse, error) {
		reloads++
		return PolicyReloadResponse{Reloaded: true}, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/policy")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("policy observe status = %d, want 200", r.StatusCode)
	}
	var resp PolicyObservation
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if observes != 1 || resp.Source != "floor.json" || resp.EffectiveDigest != "policy-sha256:abc123" || resp.Summary != "posture: fail_closed" {
		t.Fatalf("observes=%d response=%+v, want one observation carrying source+digest", observes, resp)
	}
	// The whole point of the read-only route: looking must not re-read (and possibly
	// CHANGE) the floor the way a POST reload does.
	if reloads != 0 {
		t.Fatalf("reloads=%d, want 0 — observing the floor must have no reload side effect", reloads)
	}
}

// TestPolicyObserveRouteDisabledWithoutCallback pins the "404 when not configured"
// half of the acceptance: an unconfigured deployment must say so, never answer a
// blank/zero floor that reads like a real attestation.
func TestPolicyObserveRouteDisabledWithoutCallback(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/policy")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled policy observe status = %d, want 404", r.StatusCode)
	}
}

// TestPolicyObserveRouteIsGetOnly keeps the read-only route read-only: the mutating
// verb lives at /v1/fak/policy/reload, and a POST here is a 405, not a silent reload.
func TestPolicyObserveRouteIsGetOnly(t *testing.T) {
	srv := newTestServer(t)
	srv.observePolicy = func(context.Context) (PolicyObservation, error) {
		return PolicyObservation{Source: "floor.json"}, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/policy", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusMethodNotAllowed {
		t.Fatalf("POST policy observe status = %d, want 405", r.StatusCode)
	}
}

// TestPolicyObserveRouteReportsObserverFailure: a floor that cannot be attested is a
// 400, never a 200 carrying an empty digest — an empty digest would read as "no floor".
func TestPolicyObserveRouteReportsObserverFailure(t *testing.T) {
	srv := newTestServer(t)
	srv.observePolicy = func(context.Context) (PolicyObservation, error) {
		return PolicyObservation{}, errors.New("policy not loaded")
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Get(ts.URL + "/v1/fak/policy")
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("failed policy observe status = %d, want 400", r.StatusCode)
	}
}

// TestPolicyObserveReloadCountIsGatewayOwned proves the counter is the GATEWAY's, not
// the host func's: a loader that reports a bogus count cannot spoof it, and the count
// tracks successful reload swaps so an operator can tell a hot-swapped floor from one
// that has stood since launch.
func TestPolicyObserveReloadCountIsGatewayOwned(t *testing.T) {
	srv := newTestServer(t)
	srv.observePolicy = func(context.Context) (PolicyObservation, error) {
		// A lying host func: it claims 99 swaps. The gateway must overwrite this.
		return PolicyObservation{Source: "floor.json", EffectiveDigest: "policy-sha256:abc123", ReloadCount: 99}, nil
	}
	srv.reloadPolicy = func(context.Context) (PolicyReloadResponse, error) {
		return PolicyReloadResponse{Reloaded: true, Source: "floor.json", EffectiveDigest: "policy-sha256:abc123"}, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	observe := func() PolicyObservation {
		t.Helper()
		r, err := http.Get(ts.URL + "/v1/fak/policy")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		var resp PolicyObservation
		if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		return resp
	}

	// A never-reloaded floor reports 0 — emitted, not omitted.
	if got := observe(); got.ReloadCount != 0 {
		t.Fatalf("initial reload_count = %d, want 0 (gateway-owned, host's 99 ignored)", got.ReloadCount)
	}

	for i := 0; i < 2; i++ {
		r, err := http.Post(ts.URL+"/v1/fak/policy/reload", "application/json", strings.NewReader(`{}`))
		if err != nil {
			t.Fatal(err)
		}
		r.Body.Close()
		if r.StatusCode != http.StatusOK {
			t.Fatalf("reload %d status = %d, want 200", i, r.StatusCode)
		}
	}

	got := observe()
	if got.ReloadCount != 2 {
		t.Fatalf("reload_count after 2 reloads = %d, want 2", got.ReloadCount)
	}
	// Identical reloads leave the digest untouched — the count moves, the floor does
	// not. That equality is what lets an operator prove a reload was a no-op.
	if got.EffectiveDigest != "policy-sha256:abc123" {
		t.Fatalf("effective_digest = %q, want it stable across identical reloads", got.EffectiveDigest)
	}
}

// TestPolicyObserveTracksDigestChange: when the effective floor moves (a file edit or
// an overlay change re-applied on reload), the attested digest moves with it. This is
// the "digest changes when the file or overlay changes" half of the acceptance.
func TestPolicyObserveTracksDigestChange(t *testing.T) {
	srv := newTestServer(t)
	digest := "policy-sha256:before"
	srv.observePolicy = func(context.Context) (PolicyObservation, error) {
		return PolicyObservation{Source: "floor.json", EffectiveDigest: digest}, nil
	}
	srv.reloadPolicy = func(context.Context) (PolicyReloadResponse, error) {
		digest = "policy-sha256:after" // the overlay is re-applied: the effective floor moved
		return PolicyReloadResponse{Reloaded: true, Source: "floor.json", EffectiveDigest: digest}, nil
	}
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	observeDigest := func() string {
		t.Helper()
		r, err := http.Get(ts.URL + "/v1/fak/policy")
		if err != nil {
			t.Fatal(err)
		}
		defer r.Body.Close()
		var resp PolicyObservation
		if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
			t.Fatalf("decode response: %v", err)
		}
		return resp.EffectiveDigest
	}

	if got := observeDigest(); got != "policy-sha256:before" {
		t.Fatalf("pre-reload digest = %q, want policy-sha256:before", got)
	}

	r, err := http.Post(ts.URL+"/v1/fak/policy/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	var reloadResp PolicyReloadResponse
	if err := json.NewDecoder(r.Body).Decode(&reloadResp); err != nil {
		t.Fatalf("decode reload response: %v", err)
	}
	r.Body.Close()

	got := observeDigest()
	if got != "policy-sha256:after" {
		t.Fatalf("post-reload digest = %q, want policy-sha256:after", got)
	}
	// POST and GET answer the same question the same way — the field and the
	// computation are shared, so the two are directly comparable.
	if reloadResp.EffectiveDigest != got {
		t.Fatalf("reload digest %q != observed digest %q — POST and GET must be comparable", reloadResp.EffectiveDigest, got)
	}
}
