package main

// serve_route_reload_test.go — the #4003 acceptance witnesses for the SERVE half of
// model-routing hot reload.
//
// internal/gateway already proved the endpoint against a hand-installed watcher
// (route_reload_test.go). What was unproven — and what #4003 actually is — is that the
// shipped `fak serve` ever installs one: before this, every SetRouteWatcher caller in the
// tree was a test, so POST /v1/fak/route/reload answered 404 on every real deployment and
// Reload()'s gate bypass had no production trigger. So these tests drive the real serve
// arming function against a real gateway.Server over real HTTP, and the primary one
// constructs the exact edit the background poller provably cannot see (size AND
// mtime-nanos preserved) — the operator situation the issue is about.

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// writeServeRouteManifest writes a tiny valid route manifest routing tool "x" to model.
// Callers pass equal-length model names when they need the on-disk BYTE COUNT to stay
// fixed across an edit (the size half of the poller's change gate).
func writeServeRouteManifest(t *testing.T, path, model string) {
	t.Helper()
	m := modelroute.Manifest{
		Version: modelroute.Version,
		Default: modelroute.Plan{Members: []modelroute.Member{{Model: "default", Role: "primary"}}},
		Rules: []modelroute.Rule{{
			Name:  "route-x",
			Match: modelroute.Match{Aspect: modelroute.AspectToolCall, Tool: "x"},
			Plan:  modelroute.Plan{Members: []modelroute.Member{{Model: model}}},
		}},
	}
	if err := os.WriteFile(path, m.JSON(), 0o600); err != nil {
		t.Fatalf("write manifest: %v", err)
	}
}

// newServeWithRouteManifest builds a real gateway.Server with a route manifest installed,
// exactly as `fak serve --route-manifest` does, and returns it with the manifest path.
func newServeWithRouteManifest(t *testing.T, model string) (*gateway.Server, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "route.json")
	writeServeRouteManifest(t, path, model)
	loaded, err := modelroute.LoadManifest(path)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless", RouteManifest: &loaded})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	t.Cleanup(func() { srv.Close() })
	return srv, path
}

// liveRoutedModel reads the model the LIVE routing policy binds tool "x" to. This reads the
// same atomic modelroute.Live the gateway classifies through on the hot path, so it is the
// real "did the swap land", not a counter that could move without the policy changing.
func liveRoutedModel(t *testing.T, srv *gateway.Server) string {
	t.Helper()
	m := srv.RouteLive().Manifest()
	if len(m.Rules) != 1 || len(m.Rules[0].Plan.Members) != 1 {
		t.Fatalf("live manifest has unexpected shape: %+v", m)
	}
	return m.Rules[0].Plan.Members[0].Model
}

// postRouteReload posts the forced reload and returns the status and decoded body.
func postRouteReload(t *testing.T, base string) (int, gateway.RouteReloadResponse) {
	t.Helper()
	r, err := http.Post(base+"/v1/fak/route/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	var resp gateway.RouteReloadResponse
	if r.StatusCode == http.StatusOK {
		if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
			t.Fatalf("decode reload response: %v", err)
		}
	}
	return r.StatusCode, resp
}

// TestServeRouteReloadForcesSwapOnMtimeInvisibleEdit is the PRIMARY #4003 acceptance: an
// edit that preserves BOTH halves of the poller's change gate (size + mtime-nanos) is
// permanently invisible to the background poll loop, and before this wiring an operator who
// made one had to restart the server. It asserts the blindness first (so the test cannot
// pass by accident on a poll that happened to fire), then proves the POST forces the
// content-compare reload and the live policy actually swapped.
func TestServeRouteReloadForcesSwapOnMtimeInvisibleEdit(t *testing.T) {
	srv, path := newServeWithRouteManifest(t, "alpha")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if w := armRouteHotReload(ctx, srv, path); w == nil {
		t.Fatal("armRouteHotReload returned nil with a route manifest installed")
	}

	before, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	// "gamma" is the same byte length as "alpha", so the manifest's serialized size is
	// unchanged; restoring mtime completes the gate-invisible edit a deploy tool that
	// preserves timestamps (or a coarse-granularity mount) produces in the field.
	writeServeRouteManifest(t, path, "gamma")
	if err := os.Chtimes(path, before.ModTime(), before.ModTime()); err != nil {
		t.Fatalf("restore mtime: %v", err)
	}
	after, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	if after.Size() != before.Size() || !after.ModTime().Equal(before.ModTime()) {
		t.Fatalf("edit did not preserve the change gate (size %d->%d, mtime %v->%v); "+
			"this filesystem cannot express the mtime-invisible edit the test is about",
			before.Size(), after.Size(), before.ModTime(), after.ModTime())
	}
	// The poll loop is running against this file and is structurally blind to the edit:
	// size+mtime are byte-identical, so poll() early-returns unchanged forever.
	if got := liveRoutedModel(t, srv); got != "alpha" {
		t.Fatalf("live model = %q before the forced reload, want %q (the poller should be blind to this edit)", got, "alpha")
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	status, resp := postRouteReload(t, ts.URL)
	if status != http.StatusOK {
		t.Fatalf("forced reload status = %d, want 200", status)
	}
	if !resp.Reloaded || resp.Source != path || resp.Reloads != 1 {
		t.Fatalf("forced reload response = %+v, want one reload of %q", resp, path)
	}
	if got := liveRoutedModel(t, srv); got != "gamma" {
		t.Fatalf("live model = %q after the forced reload, want %q — the swap did not reach the hot path", got, "gamma")
	}
}

// TestServeRouteReloadRejectsMalformedManifestKeepingLastGood covers the second acceptance
// bullet at the SERVE seam: the forced trigger inherits the poll loop's fail-loud contract
// because both drive the same watcher — a malformed edit is a 400, never a silent success,
// and the last-good policy stays live.
func TestServeRouteReloadRejectsMalformedManifestKeepingLastGood(t *testing.T) {
	srv, path := newServeWithRouteManifest(t, "alpha")

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if w := armRouteHotReload(ctx, srv, path); w == nil {
		t.Fatal("armRouteHotReload returned nil with a route manifest installed")
	}
	if err := os.WriteFile(path, []byte("this is not a valid manifest"), 0o600); err != nil {
		t.Fatalf("corrupt manifest: %v", err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	status, _ := postRouteReload(t, ts.URL)
	if status != http.StatusBadRequest {
		t.Fatalf("forced reload of a malformed manifest = %d, want 400", status)
	}
	if got := liveRoutedModel(t, srv); got != "alpha" {
		t.Fatalf("live model = %q after a REJECTED reload, want the last-good %q", got, "alpha")
	}
}

// TestServeWithoutRouteManifestLeavesRouteReloadDisabled covers the third acceptance
// bullet: with no --route-manifest there is no watcher to arm, so the route stays 404
// rather than pretending to reload a policy that does not exist.
func TestServeWithoutRouteManifestLeavesRouteReloadDisabled(t *testing.T) {
	srv, err := gateway.New(gateway.Config{ExposeProfile: "headless"})
	if err != nil {
		t.Fatalf("gateway.New: %v", err)
	}
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if w := armRouteHotReload(ctx, srv, filepath.Join(t.TempDir(), "absent.json")); w != nil {
		t.Fatal("armRouteHotReload armed a watcher with no route manifest installed")
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	if status, _ := postRouteReload(t, ts.URL); status != http.StatusNotFound {
		t.Fatalf("route reload with no manifest = %d, want 404", status)
	}
}
