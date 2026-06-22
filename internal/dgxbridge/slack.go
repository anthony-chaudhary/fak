// Package dgxbridge is a pure-Go (stdlib-only) client for the Slack control-bridge
// that fronts the lab DGX. The DGX is reachable only through a slack-helpers
// "control session" thread in #dgx-control: a message posted to the thread is typed
// as stdin into a persistent remote shell, and the shell's output is mirrored back.
//
// The live bridge runs in PTY mode, so its rolling stdout mirror wedges on Slack's
// msg_too_long limit and live stdout is unreliable. The reliable read path is the
// bridge's !dump verb, which uploads the full raw transcript JSONL as a *file*
// (files API, no length limit). This package drives commands through that path and
// reads results back as files, never by scraping the live mirror. See rpc.go.
//
// No third-party deps: net/http + encoding/json only.
package dgxbridge

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"
)

// DefaultChannel is a PLACEHOLDER for the #dgx-control channel id. The real id is
// operator infra kept out of source (per PUBLIC-SCRUB-POLICY) — pass it with the
// -channel flag (or set FAK_SLACK_* env + your own default).
const DefaultChannel = "C00000000000"

const slackAPI = "https://slack.com/api/"

// tokenEnvs is the resolution order, matching the Python tooling.
var tokenEnvs = []string{"FAK_SLACK_BOT_TOKEN", "SLACK_BOT_TOKEN"}

// Client is a minimal Slack Web API client scoped to the bridge's needs.
type Client struct {
	token   string
	http    *http.Client
	apiBase string // override for tests
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient injects a custom *http.Client (used in tests to avoid the network).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

// WithAPIBase overrides the Slack API base URL (used in tests).
func WithAPIBase(base string) Option { return func(c *Client) { c.apiBase = base } }

// NewClient builds a Client. If token is empty it is resolved from the environment
// (FAK_SLACK_BOT_TOKEN, then SLACK_BOT_TOKEN) and finally from a .env.slack.local
// file walked up from the working directory.
func NewClient(token string, opts ...Option) (*Client, error) {
	if token == "" {
		token = ResolveToken()
	}
	if token == "" {
		return nil, fmt.Errorf("no Slack token: set %s, or add SLACK_BOT_TOKEN=... to .env.slack.local",
			strings.Join(tokenEnvs, "/"))
	}
	c := &Client{token: token, http: &http.Client{Timeout: 40 * time.Second}, apiBase: slackAPI}
	for _, o := range opts {
		o(c)
	}
	return c, nil
}

// ResolveToken applies the documented resolution order and returns "" if none found.
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return tokenFromEnvFile()
}

// tokenFromEnvFile walks up from the cwd looking for .env.slack.local with a
// SLACK_BOT_TOKEN= line. This is the durable, gitignored persistence location.
func tokenFromEnvFile() string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 6; i++ {
		p := filepath.Join(dir, ".env.slack.local")
		if b, err := os.ReadFile(p); err == nil {
			for _, ln := range strings.Split(string(b), "\n") {
				ln = strings.TrimSpace(ln)
				if v, ok := strings.CutPrefix(ln, "SLACK_BOT_TOKEN="); ok {
					return strings.TrimSpace(v)
				}
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return ""
}

// apiError carries Slack's structured "ok=false" failures.
type apiError struct {
	Method string
	Err    string
}

func (e *apiError) Error() string { return fmt.Sprintf("slack %s: %s", e.Method, e.Err) }

// callJSON does a POST with a JSON body and unmarshals the response into out.
// It retries on Slack rate-limiting (HTTP 429 / "ratelimited") with backoff —
// the Python tooling had none, this is a cheap robustness win.
func (c *Client) callJSON(ctx context.Context, method string, body any, out any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	return c.do(ctx, http.MethodPost, c.apiBase+method, "application/json; charset=utf-8", raw, method, out)
}

// callGet does a GET with query params and unmarshals into out.
func (c *Client) callGet(ctx context.Context, method string, params url.Values, out any) error {
	u := c.apiBase + method + "?" + params.Encode()
	return c.do(ctx, http.MethodGet, u, "", nil, method, out)
}

func (c *Client) do(ctx context.Context, httpMethod, u, contentType string, body []byte, slackMethod string, out any) error {
	var lastErr error
	for attempt := 0; attempt < 4; attempt++ {
		var rdr io.Reader
		if body != nil {
			rdr = bytes.NewReader(body)
		}
		req, err := http.NewRequestWithContext(ctx, httpMethod, u, rdr)
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+c.token)
		if contentType != "" {
			req.Header.Set("Content-Type", contentType)
		}
		resp, err := c.http.Do(req)
		if err != nil {
			lastErr = err
			time.Sleep(backoff(attempt))
			continue
		}
		data, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == http.StatusTooManyRequests {
			lastErr = fmt.Errorf("slack %s: ratelimited", slackMethod)
			time.Sleep(retryAfter(resp, attempt))
			continue
		}
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("slack %s: decode: %w (body=%.200s)", slackMethod, err, string(data))
		}
		// Slack puts errors in the body even on HTTP 200.
		if ae := okError(slackMethod, data); ae != nil {
			if ae.Err == "ratelimited" {
				lastErr = ae
				time.Sleep(backoff(attempt))
				continue
			}
			return ae
		}
		return nil
	}
	return lastErr
}

// okError extracts {ok:false, error:...} from a raw Slack response.
func okError(method string, data []byte) *apiError {
	var probe struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	if err := json.Unmarshal(data, &probe); err != nil {
		return nil // out already failed to decode if malformed; callers see decode error
	}
	if !probe.OK {
		return &apiError{Method: method, Err: probe.Error}
	}
	return nil
}

func backoff(attempt int) time.Duration {
	return time.Duration(1<<attempt) * time.Second
}

func retryAfter(resp *http.Response, attempt int) time.Duration {
	if v := resp.Header.Get("Retry-After"); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return time.Duration(n) * time.Second
		}
	}
	return backoff(attempt)
}

// ---- Typed responses ----

type postMessageResp struct {
	OK    bool   `json:"ok"`
	TS    string `json:"ts"`
	Error string `json:"error"`
}

// Message is one thread reply.
type Message struct {
	TS    string `json:"ts"`
	Text  string `json:"text"`
	User  string `json:"user"`
	BotID string `json:"bot_id"`
	Files []File `json:"files"`
}

// File is one channel file (e.g. an uploaded transcript.jsonl).
type File struct {
	ID          string `json:"id"`
	Name        string `json:"name"`
	Created     int64  `json:"created"`
	Size        int64  `json:"size"`
	URLPrivate  string `json:"url_private"`
	URLDownload string `json:"url_private_download"`
}

type repliesResp struct {
	OK       bool      `json:"ok"`
	Error    string    `json:"error"`
	Messages []Message `json:"messages"`
}

type filesListResp struct {
	OK    bool   `json:"ok"`
	Error string `json:"error"`
	Files []File `json:"files"`
}

// ---- Bridge primitives ----

// Post sends text to a thread (or the channel root if threadTS==""). Returns the
// posted message ts as a float-seconds string.
func (c *Client) Post(ctx context.Context, channel, threadTS, text string) (string, error) {
	body := map[string]any{"channel": channel, "text": text}
	if threadTS != "" {
		body["thread_ts"] = threadTS
	}
	var r postMessageResp
	if err := c.callJSON(ctx, "chat.postMessage", body, &r); err != nil {
		return "", err
	}
	return r.TS, nil
}

// Replies returns thread messages with ts >= oldest (oldest may be "" for all).
func (c *Client) Replies(ctx context.Context, channel, threadTS, oldest string, limit int) ([]Message, error) {
	p := url.Values{}
	p.Set("channel", channel)
	p.Set("ts", threadTS)
	if oldest != "" {
		p.Set("oldest", oldest)
	}
	p.Set("limit", strconv.Itoa(limit))
	var r repliesResp
	if err := c.callGet(ctx, "conversations.replies", p, &r); err != nil {
		return nil, err
	}
	return r.Messages, nil
}

// History returns recent channel messages (used to discover a control thread).
func (c *Client) History(ctx context.Context, channel string, limit int) ([]Message, error) {
	p := url.Values{}
	p.Set("channel", channel)
	p.Set("limit", strconv.Itoa(limit))
	var r repliesResp
	if err := c.callGet(ctx, "conversations.history", p, &r); err != nil {
		return nil, err
	}
	return r.Messages, nil
}

// HistoryPage is one page of channel history with a pagination cursor.
type HistoryPage struct {
	Messages   []Message
	NextCursor string
}

type historyResp struct {
	OK               bool      `json:"ok"`
	Error            string    `json:"error"`
	Messages         []Message `json:"messages"`
	ResponseMetadata struct {
		NextCursor string `json:"next_cursor"`
	} `json:"response_metadata"`
}

// HistoryPaged returns one page of channel history, with optional time bounds and a
// cursor for the next page. oldest/latest are float-seconds strings ("" to omit).
func (c *Client) HistoryPaged(ctx context.Context, channel, oldest, latest, cursor string, limit int) (HistoryPage, error) {
	p := url.Values{}
	p.Set("channel", channel)
	p.Set("limit", strconv.Itoa(limit))
	if oldest != "" {
		p.Set("oldest", oldest)
	}
	if latest != "" {
		p.Set("latest", latest)
	}
	if cursor != "" {
		p.Set("cursor", cursor)
	}
	var r historyResp
	if err := c.callGet(ctx, "conversations.history", p, &r); err != nil {
		return HistoryPage{}, err
	}
	return HistoryPage{Messages: r.Messages, NextCursor: r.ResponseMetadata.NextCursor}, nil
}

// DeleteMessage deletes a message by ts. Requires chat:write and (for messages the
// bot did not author) admin/appropriate scope; the API returns cant_delete_message
// otherwise, which the caller can treat as skippable.
func (c *Client) DeleteMessage(ctx context.Context, channel, ts string) error {
	var r struct {
		OK    bool   `json:"ok"`
		Error string `json:"error"`
	}
	return c.callJSON(ctx, "chat.delete", map[string]any{"channel": channel, "ts": ts}, &r)
}

// ListFiles returns recent channel files, newest first.
func (c *Client) ListFiles(ctx context.Context, channel string, count int) ([]File, error) {
	p := url.Values{}
	p.Set("channel", channel)
	p.Set("count", strconv.Itoa(count))
	var r filesListResp
	if err := c.callGet(ctx, "files.list", p, &r); err != nil {
		return nil, err
	}
	return r.Files, nil
}

// Download fetches a file's bytes by its url_private_download (Bearer auth).
func (c *Client) Download(ctx context.Context, downloadURL string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, downloadURL, nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	cl := &http.Client{Timeout: 90 * time.Second}
	resp, err := cl.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("download %s: status %d", downloadURL, resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// ControlSession is a discovered live bridge thread.
type ControlSession struct {
	ThreadTS string
	Host     string
	Banner   string
}

// FindControlSession returns the newest "control session started" thread, optionally
// filtered to a host (matched as `host` in the banner). Returns nil if none found.
// Note: "newest banner" is not the same as "live" — a session's shell can die while
// its banner remains in history. Use Bridge.Alive to confirm driveability.
func (c *Client) FindControlSession(ctx context.Context, channel, host string) (*ControlSession, error) {
	sessions, err := c.FindControlSessions(ctx, channel, host)
	if err != nil || len(sessions) == 0 {
		return nil, err
	}
	return &sessions[0], nil
}

// FindControlSessions returns all "control session started" threads (optionally
// host-filtered), newest first. Lets callers probe each for liveness.
func (c *Client) FindControlSessions(ctx context.Context, channel, host string) ([]ControlSession, error) {
	msgs, err := c.History(ctx, channel, 600)
	if err != nil {
		return nil, err
	}
	resolvedHost := host
	if resolvedHost == "" {
		resolvedHost = parseBannerHost("")
	}
	var out []ControlSession
	for _, m := range msgs {
		if !strings.Contains(m.Text, "control session started") {
			continue
		}
		if host != "" && !strings.Contains(m.Text, "`"+host+"`") {
			continue
		}
		out = append(out, ControlSession{ThreadTS: m.TS, Host: parseBannerHost(m.Text), Banner: m.Text})
	}
	sort.Slice(out, func(i, j int) bool {
		a, _ := strconv.ParseFloat(out[i].ThreadTS, 64)
		b, _ := strconv.ParseFloat(out[j].ThreadTS, 64)
		return a > b
	})
	return out, nil
}

// parseBannerHost extracts the host from a "...started on `host` for..." banner.
func parseBannerHost(banner string) string {
	const marker = "started on `"
	i := strings.Index(banner, marker)
	if i < 0 {
		return ""
	}
	rest := banner[i+len(marker):]
	if j := strings.IndexByte(rest, '`'); j >= 0 {
		return rest[:j]
	}
	return ""
}
