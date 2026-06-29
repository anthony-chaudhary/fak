package grafanapost

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// TestCreateSnapshotForDashboardHappyPath stubs Grafana with httptest (no network):
// the client fetches the dashboard model, POSTs it to /api/snapshots, and returns the
// url Grafana minted — never a URL the client built.
func TestCreateSnapshotForDashboardHappyPath(t *testing.T) {
	var gotAuth, gotDashboardModel string
	var gotExpires int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		switch r.URL.Path {
		case "/api/dashboards/uid/abc":
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"dashboard":{"uid":"abc","title":"Gateway Obs"},"meta":{}}`))
		case "/api/snapshots":
			var body struct {
				Dashboard json.RawMessage `json:"dashboard"`
				Expires   int             `json:"expires"`
			}
			_ = json.NewDecoder(r.Body).Decode(&body)
			gotDashboardModel = string(body.Dashboard)
			gotExpires = body.Expires
			w.Header().Set("Content-Type", "application/json")
			// The server mints an OPAQUE snapshot key; the url is not derivable from the uid.
			_, _ = w.Write([]byte(`{"key":"K9","deleteKey":"D9","url":"` + srvURL(r) + `/dashboard/snapshot/K9","deleteUrl":"x/api/snapshots-delete/D9"}`))
		default:
			http.Error(w, "unexpected path "+r.URL.Path, http.StatusNotFound)
		}
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "tok-123")
	res, err := c.CreateSnapshotForDashboard(context.Background(), "abc", 604800)
	if err != nil {
		t.Fatalf("CreateSnapshotForDashboard: %v", err)
	}
	if gotAuth != "Bearer tok-123" {
		t.Fatalf("auth header = %q, want Bearer tok-123", gotAuth)
	}
	if !strings.Contains(gotDashboardModel, `"uid":"abc"`) {
		t.Fatalf("snapshot request did not carry the fetched dashboard model: %q", gotDashboardModel)
	}
	if gotExpires != 604800 {
		t.Fatalf("expires forwarded = %d, want 604800", gotExpires)
	}
	if res.Key != "K9" || res.DeleteKey != "D9" {
		t.Fatalf("result keys = %+v", res)
	}
	if !strings.Contains(res.URL, "/dashboard/snapshot/K9") {
		t.Fatalf("result url = %q, want the server-minted snapshot url", res.URL)
	}

	// The whole point: SnapshotPost publishes the REAL url, never a fabricated one.
	post := SnapshotPost(Snapshot{Title: "p99 spike", URL: res.URL, Dashboard: "Gateway Obs"})
	if !strings.Contains(post.Text(), res.URL) {
		t.Fatalf("post should carry the real snapshot url:\n%s", post.Text())
	}
}

// srvURL reconstructs the test server origin from the request, so the stubbed response
// url matches the live host without Date.now()/randomness.
func srvURL(r *http.Request) string {
	return "http://" + r.Host
}

// TestCreateSnapshotRefusesNoURL is the no-fabricated-URL guarantee: when Grafana's
// response carries no url, the client errors rather than returning an empty-URL result
// that a caller might post.
func TestCreateSnapshotRefusesNoURL(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"key":"K9","deleteKey":"D9"}`)) // no url field
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "tok")
	_, err := c.CreateSnapshot(context.Background(), json.RawMessage(`{"uid":"abc"}`), 0)
	if err == nil {
		t.Fatal("CreateSnapshot must refuse a response with no url")
	}
	if !strings.Contains(err.Error(), "no url") {
		t.Fatalf("error should name the missing url, got: %v", err)
	}
}

// TestCreateSnapshotOmitsExpiresWhenZero confirms a zero TTL leaves Grafana's default
// (the "expires" field is absent) rather than sending expires=0 (immediate expiry).
func TestCreateSnapshotOmitsExpiresWhenZero(t *testing.T) {
	var hadExpires bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var raw map[string]any
		_ = json.NewDecoder(r.Body).Decode(&raw)
		_, hadExpires = raw["expires"]
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"key":"K","url":"` + srvURL(r) + `/dashboard/snapshot/K"}`))
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "tok")
	if _, err := c.CreateSnapshot(context.Background(), json.RawMessage(`{"uid":"abc"}`), 0); err != nil {
		t.Fatalf("CreateSnapshot: %v", err)
	}
	if hadExpires {
		t.Fatal("expires=0 must omit the field, not send an immediate expiry")
	}
}

// TestDoMapsNon2xxToError confirms an upstream non-2xx (e.g. 401 bad token, 404 unknown
// uid) becomes an error that surfaces Grafana's own status and body.
func TestDoMapsNon2xxToError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"message":"invalid API key"}`, http.StatusUnauthorized)
	}))
	defer srv.Close()

	c := NewClient(srv.URL, "bad")
	_, err := c.FetchDashboardModel(context.Background(), "abc")
	if err == nil {
		t.Fatal("a 401 must become an error")
	}
	if !strings.Contains(err.Error(), "401") || !strings.Contains(err.Error(), "invalid API key") {
		t.Fatalf("error should surface the upstream status+reason, got: %v", err)
	}
}

// TestDoRequiresToken confirms the client refuses to call Grafana unauthenticated — a
// missing token is an error at the request boundary, never a silent anonymous call.
func TestDoRequiresToken(t *testing.T) {
	c := NewClient("http://localhost:3000", "")
	if _, err := c.FetchDashboardModel(context.Background(), "abc"); err == nil {
		t.Fatal("an empty token must be refused")
	}
}

// TestResolveAPITokenIsSeparateFromSlack confirms the Grafana API token resolves from
// its own env and does NOT fall back to the Slack scoreboard token (a Slack bot token is
// not a Grafana credential).
func TestResolveAPITokenIsSeparateFromSlack(t *testing.T) {
	clearEnv(t)
	t.Setenv("FAK_SCOREBOARD_TOKEN", "slack-bot-tok")
	if got := ResolveAPIToken(); got != "" {
		t.Fatalf("ResolveAPIToken must not inherit the Slack token, got %q", got)
	}
	t.Setenv("FAK_GRAFANA_API_TOKEN", "grafana-api-tok")
	if got := ResolveAPIToken(); got != "grafana-api-tok" {
		t.Fatalf("ResolveAPIToken = %q, want grafana-api-tok", got)
	}
}
