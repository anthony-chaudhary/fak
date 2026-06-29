package grafanapost

// ingest.go is the INBOUND Grafana half named as the follow-on in this package's
// doc comment: instead of posting a snapshot URL the operator exported by hand, it
// calls a live Grafana's snapshots API to create/export a snapshot for a dashboard,
// then hands back the URL Grafana actually returned. The outbound SnapshotPost fold
// then publishes that real URL.
//
// The no-fabricated-URL discipline is structural, not a convention a caller must
// remember: CreateSnapshot returns ONLY the fields Grafana's response carried, and
// errors out when the response has no url. There is no code path that builds a
// snapshot URL from a uid + base — a snapshot URL is opaque (keyed by a server-side
// snapshot key), so the only honest source is the API response itself.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

// apiTokenEnvs is the dedicated Grafana API token key. It is SEPARATE from the Slack
// tokenEnvs/ResolveToken chain: the Grafana API token authenticates against the
// Grafana instance, whereas the Slack token posts the card. There is no fallback —
// an inbound pull with no Grafana token is an error, not a silent skip, so a missing
// credential never quietly degrades to "posted nothing".
var apiTokenEnvs = []string{"FAK_GRAFANA_API_TOKEN"}

// ResolveAPIToken returns the Grafana API token from FAK_GRAFANA_API_TOKEN env, then a
// FAK_GRAFANA_API_TOKEN= line in .env.slack.local. Returns "" if neither is set;
// callers treat "" as "no inbound pull configured" and refuse the --create path rather
// than calling Grafana unauthenticated. It deliberately does NOT fall back to the Slack
// ResolveToken — a Slack bot token is not a Grafana credential.
func ResolveAPIToken() string {
	for _, e := range apiTokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_GRAFANA_API_TOKEN")
}

// Client talks to a live Grafana instance's HTTP API with a bearer API token. BaseURL
// is the Grafana root (e.g. https://grafana.example or http://localhost:3000); Token is
// a Grafana service-account / API token. HTTP is the http.Client used (nil => a client
// with a sane timeout), injected so a test can stub the transport with no network.
type Client struct {
	BaseURL string
	Token   string
	HTTP    *http.Client
}

// NewClient builds a Client with the given base URL and token and a default 30s-timeout
// http.Client. base is trimmed of a trailing slash so endpoint joins are clean.
func NewClient(base, token string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(strings.TrimSpace(base), "/"),
		Token:   strings.TrimSpace(token),
		HTTP:    &http.Client{Timeout: 30 * time.Second},
	}
}

// httpClient returns the configured client or a default-timeout one.
func (c *Client) httpClient() *http.Client {
	if c.HTTP != nil {
		return c.HTTP
	}
	return &http.Client{Timeout: 30 * time.Second}
}

// SnapshotResult is the subset of Grafana's POST /api/snapshots response this package
// needs. The field names match the documented Grafana response verbatim (key, deleteKey,
// url, deleteUrl) so a future caller can also revoke a snapshot via DeleteKey. URL is
// the public snapshot link Grafana minted — the ONLY value SnapshotPost should publish.
type SnapshotResult struct {
	Key       string `json:"key"`
	DeleteKey string `json:"deleteKey"`
	URL       string `json:"url"`
	DeleteURL string `json:"deleteUrl"`
}

// FetchDashboardModel pulls the full dashboard model for a uid via the stable
// GET /api/dashboards/uid/:uid endpoint, which returns {"dashboard": {...}, "meta": ...};
// it returns the inner dashboard object as raw JSON, ready to embed in a snapshot
// create request. An empty/missing dashboard object is an error rather than an empty
// snapshot.
func (c *Client) FetchDashboardModel(ctx context.Context, uid string) (json.RawMessage, error) {
	uid = strings.TrimSpace(uid)
	if uid == "" {
		return nil, fmt.Errorf("grafana: dashboard uid is required")
	}
	if c.BaseURL == "" {
		return nil, fmt.Errorf("grafana: base URL is required")
	}
	endpoint := c.BaseURL + "/api/dashboards/uid/" + uid
	body, err := c.do(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, err
	}
	var wrap struct {
		Dashboard json.RawMessage `json:"dashboard"`
	}
	if err := json.Unmarshal(body, &wrap); err != nil {
		return nil, fmt.Errorf("grafana: decode dashboard %s: %w", uid, err)
	}
	if len(bytes.TrimSpace(wrap.Dashboard)) == 0 || string(bytes.TrimSpace(wrap.Dashboard)) == "null" {
		return nil, fmt.Errorf("grafana: dashboard %s has no model in the response", uid)
	}
	return wrap.Dashboard, nil
}

// CreateSnapshot creates a snapshot from a dashboard model via POST /api/snapshots and
// returns the result Grafana minted. expiresSeconds, when > 0, sets the snapshot TTL
// (Grafana's "expires" field, in seconds); 0 leaves Grafana's default (no expiry).
//
// It enforces the no-fabricated-URL rule at the boundary: if Grafana's response carries
// no url, it returns an error rather than a SnapshotResult with an empty URL — the
// caller can then post nothing instead of an empty link.
func (c *Client) CreateSnapshot(ctx context.Context, dashboard json.RawMessage, expiresSeconds int) (SnapshotResult, error) {
	var zero SnapshotResult
	if len(bytes.TrimSpace(dashboard)) == 0 {
		return zero, fmt.Errorf("grafana: dashboard model is empty")
	}
	if c.BaseURL == "" {
		return zero, fmt.Errorf("grafana: base URL is required")
	}
	reqBody := map[string]any{"dashboard": dashboard}
	if expiresSeconds > 0 {
		reqBody["expires"] = expiresSeconds
	}
	payload, err := json.Marshal(reqBody)
	if err != nil {
		return zero, fmt.Errorf("grafana: encode snapshot request: %w", err)
	}
	body, err := c.do(ctx, http.MethodPost, c.BaseURL+"/api/snapshots", payload)
	if err != nil {
		return zero, err
	}
	var res SnapshotResult
	if err := json.Unmarshal(body, &res); err != nil {
		return zero, fmt.Errorf("grafana: decode snapshot response: %w", err)
	}
	if strings.TrimSpace(res.URL) == "" {
		return zero, fmt.Errorf("grafana: snapshot response carried no url (refusing to post a fabricated link)")
	}
	return res, nil
}

// CreateSnapshotForDashboard is the one-call inbound path: fetch the dashboard model for
// uid, then create a snapshot of it, returning the real snapshot Grafana minted. It is
// the function the `--create` CLI mode calls.
func (c *Client) CreateSnapshotForDashboard(ctx context.Context, uid string, expiresSeconds int) (SnapshotResult, error) {
	model, err := c.FetchDashboardModel(ctx, uid)
	if err != nil {
		return SnapshotResult{}, err
	}
	return c.CreateSnapshot(ctx, model, expiresSeconds)
}

// do issues one authenticated JSON request and returns the response body, mapping a
// non-2xx status to an error that includes the (truncated) body so an operator sees
// Grafana's own reason (e.g. 401 invalid token, 404 unknown uid).
func (c *Client) do(ctx context.Context, method, url string, body []byte) ([]byte, error) {
	if strings.TrimSpace(c.Token) == "" {
		return nil, fmt.Errorf("grafana: API token is required (set FAK_GRAFANA_API_TOKEN or pass --grafana-token)")
	}
	var rdr io.Reader
	if body != nil {
		rdr = bytes.NewReader(body)
	}
	req, err := http.NewRequestWithContext(ctx, method, url, rdr)
	if err != nil {
		return nil, fmt.Errorf("grafana: build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+c.Token)
	req.Header.Set("Accept", "application/json")
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	resp, err := c.httpClient().Do(req)
	if err != nil {
		return nil, fmt.Errorf("grafana: %s %s: %w", method, url, err)
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return nil, fmt.Errorf("grafana: read response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("grafana: %s %s returned %d: %s", method, url, resp.StatusCode, truncate(string(respBody), 300))
	}
	return respBody, nil
}

// truncate clips s to n runes with an ellipsis, so an error never dumps a multi-KB
// Grafana error page into the log.
func truncate(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
