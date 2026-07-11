package gateway

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// writeRouteManifest writes a tiny valid route manifest that routes tool "x" to
// model, with a fail-closed default. Distinct models across writes let a reload
// prove WHICH manifest became live.
func writeRouteManifest(t *testing.T, path, model string) {
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

// newRouteReloadWatcher builds a real modelroute.Watcher over a fresh temp manifest
// installed as the live policy, so a test can drive a genuine hot-reload through the
// gateway's /v1/fak/route/reload seam. It returns the watcher and the manifest path.
func newRouteReloadWatcher(t *testing.T, model string) (*modelroute.Watcher, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "route.json")
	writeRouteManifest(t, path, model)
	loaded, err := modelroute.LoadManifest(path)
	if err != nil {
		t.Fatalf("load manifest: %v", err)
	}
	live := modelroute.NewLive(&loaded)
	return modelroute.NewWatcher(path, live, 0, nil), path
}

func TestRouteReloadRouteInvokesConfiguredWatcher(t *testing.T) {
	srv := newTestServer(t)
	watcher, path := newRouteReloadWatcher(t, "alpha")
	srv.SetRouteWatcher(watcher)
	// Edit the on-disk manifest so the forced reload actually swaps (Reloaded=true).
	writeRouteManifest(t, path, "beta")

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/route/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusOK {
		t.Fatalf("reload status = %d, want 200", r.StatusCode)
	}
	var resp RouteReloadResponse
	if err := json.NewDecoder(r.Body).Decode(&resp); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if !resp.Reloaded || resp.Source != path || resp.Reloads != 1 {
		t.Fatalf("response=%+v, want one reload of %q", resp, path)
	}
}

func TestRouteReloadRouteDisabledWithoutWatcher(t *testing.T) {
	srv := newTestServer(t)
	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/route/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusNotFound {
		t.Fatalf("disabled reload status = %d, want 404", r.StatusCode)
	}
}

func TestRouteReloadRouteReportsRejectedManifest(t *testing.T) {
	srv := newTestServer(t)
	watcher, path := newRouteReloadWatcher(t, "alpha")
	srv.SetRouteWatcher(watcher)
	// A malformed edit must be REJECTED (last-good kept) and surface as a 400, not a
	// silent success.
	if err := os.WriteFile(path, []byte("this is not a valid manifest"), 0o600); err != nil {
		t.Fatalf("corrupt manifest: %v", err)
	}

	ts := httptest.NewServer(srv.Handler())
	defer ts.Close()

	r, err := http.Post(ts.URL+"/v1/fak/route/reload", "application/json", strings.NewReader(`{}`))
	if err != nil {
		t.Fatal(err)
	}
	defer r.Body.Close()
	if r.StatusCode != http.StatusBadRequest {
		t.Fatalf("rejected reload status = %d, want 400", r.StatusCode)
	}
}
