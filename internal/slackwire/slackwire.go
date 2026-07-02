// Package slackwire is the ONE Slack Web API transport for fak: chat.postMessage,
// chat.update, conversations.history, and auth.test in a single tested client with
// 429/Retry-After handling and a typed error.
//
// Before this package fak spoke to Slack through two parallel hand-rolled HTTP
// clients — internal/scoreboard.Client (post + auth, no thread_ts, no chat.update,
// no 429 handling) and internal/chatrelay.HTTPSlack (history + threaded post, its
// own decode, its own error surfacing). A transport fix had to land twice, and
// message *editing* existed only on the chatrelay side, so no feeder could update a
// card in place. Both now delegate here (#2261, epic #2259) — only transport moved:
// token/channel RESOLUTION stays in internal/slackenv, render logic stays in the
// per-surface *post packages, and post gating stays in scoreboard.
//
// Boundary: this is the PUBLIC side of the GPU-server/Slack boundary
// (docs/gpu-server-private-boundary.md) — a generic Web API client with no lab
// identifiers, no shell, no control protocol; the token arrives at runtime from the
// caller. Pure stdlib (net/http + encoding/json); tier-1 foundation, off the hot path.
//
// Rate limits (verified 2026-07-02, sources in
// docs/notes/SLACK-CONTROL-FOUNDATION-2026-07-02.md): chat.postMessage is
// special-tier (~1 msg/s per channel, burst-tolerant); a 429 carries Retry-After in
// seconds, scoped per-method-per-workspace. The client honors Retry-After with a
// BOUNDED in-call retry — pacing, spooling, and give-up-then-dead-letter policy
// belong to the outbox layer above, not the wire.
package slackwire

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"time"
)

// apiBaseDefault is the Slack Web API root every method name is appended to.
const apiBaseDefault = "https://slack.com/api/"

const (
	// retries429Default bounds how many times one call re-sends after a 429 before the
	// rate-limit error is returned to the caller. Two retries rides out a burst
	// collision; a sustained limit is the caller's (outbox's) problem, not the wire's.
	retries429Default = 2
	// retryAfterFallback is the wait when a 429 arrives without a parseable
	// Retry-After header (the header is documented but this client never trusts it to
	// be present).
	retryAfterFallback = 1 * time.Second
	// retryAfterCap bounds a single Retry-After wait so a hostile/broken header can
	// never park a caller for minutes inside what looks like one HTTP call.
	retryAfterCap = 30 * time.Second
	// bodyLimit bounds how much of a response is read; Slack envelopes are small and
	// an unbounded read of a misrouted endpoint must not balloon memory.
	bodyLimit = 1 << 20
	// httpTimeoutDefault matches the timeout both predecessor clients used.
	httpTimeoutDefault = 40 * time.Second
)

// Client is the Slack Web API transport. The zero value is not usable; construct
// with New. Safe for concurrent use (it holds no per-call state).
type Client struct {
	token   string
	apiBase string
	httpc   *http.Client
	retries int
	// sleep is the 429 wait, injectable so tests witness the waits instead of
	// serving them. It must honor ctx cancellation.
	sleep func(ctx context.Context, d time.Duration) error
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient injects a custom *http.Client (tests, proxies).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.httpc = h } }

// WithAPIBase overrides the Slack API base URL (tests, proxies). The value should
// end with "/" — method names are appended verbatim.
func WithAPIBase(base string) Option { return func(c *Client) { c.apiBase = base } }

// WithRetryBudget sets how many times one call re-sends after a 429 (0 = fail on
// the first 429). Negative values are treated as 0.
func WithRetryBudget(n int) Option {
	return func(c *Client) {
		if n < 0 {
			n = 0
		}
		c.retries = n
	}
}

// New builds a Client for a bot token. An empty token is allowed — the transport
// does not resolve or validate credentials (that is internal/slackenv's and the
// caller's job); Slack answers an unauthenticated call with ok:false invalid_auth,
// which surfaces as the typed *APIError.
func New(token string, opts ...Option) *Client {
	c := &Client{
		token:   token,
		apiBase: apiBaseDefault,
		httpc:   &http.Client{Timeout: httpTimeoutDefault},
		retries: retries429Default,
		sleep:   ctxSleep,
	}
	for _, o := range opts {
		o(c)
	}
	return c
}

// ctxSleep waits d or until ctx is cancelled, whichever comes first.
func ctxSleep(ctx context.Context, d time.Duration) error {
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// APIError is a Slack Web API failure: an ok:false envelope, a 429 that outlived
// the retry budget, or a non-JSON response. Error() leads with "<method>: <code>"
// so existing callers' Contains(<error token>) checks keep working.
type APIError struct {
	Method string // Slack method, e.g. "chat.postMessage"
	Status int    // HTTP status (200 for an ok:false body)
	Code   string // Slack's error token, e.g. "invalid_auth", "ratelimited"; "" when the body was not a Slack envelope
	Body   string // truncated response body, kept for the non-envelope case
}

func (e *APIError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s (http %d)", e.Method, e.Code, e.Status)
	}
	return fmt.Sprintf("%s: http %d (body=%.200s)", e.Method, e.Status, e.Body)
}

// Message is one Slack channel message — the subset of conversations.history the
// inbound surfaces (chatrelay, chatops) decide on.
type Message struct {
	Type     string `json:"type"`      // "message" for a real post
	Subtype  string `json:"subtype"`   // non-empty for edits/joins/bot posts
	TS       string `json:"ts"`        // "1719600000.000100" — the message id + thread anchor
	ThreadTS string `json:"thread_ts"` // parent thread ts when the message is a threaded reply
	User     string `json:"user"`      // posting user id ("" for a bot post)
	BotID    string `json:"bot_id"`    // non-empty when posted by a bot
	Text     string `json:"text"`      // message body
}

// AuthInfo is the identity a token resolves to — the auth.test subset a diagnostic
// reports, enough to answer "does this token work, and as whom?".
type AuthInfo struct {
	URL    string `json:"url"`     // workspace URL
	Team   string `json:"team"`    // workspace name
	User   string `json:"user"`    // authenticating user/bot handle
	TeamID string `json:"team_id"` // T... team id
	UserID string `json:"user_id"` // U... user id
	BotID  string `json:"bot_id"`  // B... bot id (set for bot tokens)
}

// envelope is the ok/error pair every Slack Web API response carries (ok:false
// arrives with HTTP 200 — the status code alone never decides success).
type envelope struct {
	OK    bool   `json:"ok"`
	Error string `json:"error"`
}

// PostMessage calls chat.postMessage and returns the posted message ts. blocks,
// when non-empty, attaches a Block Kit payload (text stays the notification
// fallback). threadTS, when non-empty, posts the message as a reply in that thread.
func (c *Client) PostMessage(ctx context.Context, channel, text string, blocks []any, threadTS string) (string, error) {
	body := map[string]any{"channel": channel, "text": text}
	if len(blocks) > 0 {
		body["blocks"] = blocks
	}
	if threadTS != "" {
		body["thread_ts"] = threadTS
	}
	var r struct {
		envelope
		TS string `json:"ts"`
	}
	if err := c.callJSON(ctx, "chat.postMessage", body, &r); err != nil {
		return "", err
	}
	return r.TS, nil
}

// UpdateMessage calls chat.update, replacing message ts in channel with text (and
// blocks when non-empty) — the edit-in-place primitive live run-cards build on.
func (c *Client) UpdateMessage(ctx context.Context, channel, ts, text string, blocks []any) error {
	body := map[string]any{"channel": channel, "ts": ts, "text": text}
	if len(blocks) > 0 {
		body["blocks"] = blocks
	}
	var r struct{ envelope }
	return c.callJSON(ctx, "chat.update", body, &r)
}

// History calls conversations.history: messages with ts after oldestTS (""
// means from the beginning), capped at limit (<=0 lets Slack default). Callers
// re-filter by ts themselves — the inclusive/exclusive `oldest` nuance stays theirs.
func (c *Client) History(ctx context.Context, channel, oldestTS string, limit int) ([]Message, error) {
	q := url.Values{}
	q.Set("channel", channel)
	if oldestTS != "" {
		q.Set("oldest", oldestTS)
	}
	if limit > 0 {
		q.Set("limit", strconv.Itoa(limit))
	}
	var r struct {
		envelope
		Messages []Message `json:"messages"`
	}
	if err := c.callGet(ctx, "conversations.history", q, &r); err != nil {
		return nil, err
	}
	return r.Messages, nil
}

// AuthTest calls auth.test — the "does this token actually work, and as whom?"
// probe behind `fak slack check --auth`.
func (c *Client) AuthTest(ctx context.Context) (*AuthInfo, error) {
	var r struct {
		envelope
		AuthInfo
	}
	if err := c.callGet(ctx, "auth.test", nil, &r); err != nil {
		return nil, err
	}
	info := r.AuthInfo
	return &info, nil
}

// okReporter lets call decode method-specific responses while reading the shared
// ok/error envelope they all embed.
type okReporter interface {
	ok() bool
	errCode() string
}

func (e envelope) ok() bool        { return e.OK }
func (e envelope) errCode() string { return e.Error }

// callJSON POSTs a JSON body to method; callGet GETs method with query q. Both
// decode into out (which must embed envelope) and apply the 429 retry.
func (c *Client) callJSON(ctx context.Context, method string, body map[string]any, out okReporter) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	return c.call(ctx, method, func() (*http.Request, error) {
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiBase+method, bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/json; charset=utf-8")
		return req, nil
	}, out)
}

func (c *Client) callGet(ctx context.Context, method string, q url.Values, out okReporter) error {
	target := c.apiBase + method
	if len(q) > 0 {
		target += "?" + q.Encode()
	}
	return c.call(ctx, method, func() (*http.Request, error) {
		return http.NewRequestWithContext(ctx, http.MethodGet, target, nil)
	}, out)
}

// call runs one Web API exchange: build request (fresh per attempt — a retried
// body reader must not be half-drained), send, honor 429/Retry-After within the
// bounded budget, decode the envelope, surface ok:false as *APIError.
func (c *Client) call(ctx context.Context, method string, build func() (*http.Request, error), out okReporter) error {
	for attempt := 0; ; attempt++ {
		req, err := build()
		if err != nil {
			return err
		}
		req.Header.Set("Authorization", "Bearer "+c.token)
		resp, err := c.httpc.Do(req)
		if err != nil {
			return err
		}
		data, _ := io.ReadAll(io.LimitReader(resp.Body, bodyLimit))
		resp.Body.Close()

		if resp.StatusCode == http.StatusTooManyRequests {
			if attempt >= c.retries {
				return &APIError{Method: method, Status: resp.StatusCode, Code: "ratelimited", Body: string(data)}
			}
			if err := c.sleep(ctx, retryAfterWait(resp.Header.Get("Retry-After"))); err != nil {
				return err
			}
			continue
		}
		if err := json.Unmarshal(data, out); err != nil {
			return fmt.Errorf("%s: decode: %w (status=%d body=%.200s)", method, err, resp.StatusCode, string(data))
		}
		if !out.ok() {
			return &APIError{Method: method, Status: resp.StatusCode, Code: out.errCode(), Body: string(data)}
		}
		return nil
	}
}

// retryAfterWait maps a Retry-After header (integer seconds per the Slack docs) to
// a bounded wait: fallback when absent/garbled, capped so one call never parks long.
func retryAfterWait(header string) time.Duration {
	secs, err := strconv.Atoi(header)
	if err != nil || secs < 0 {
		return retryAfterFallback
	}
	d := time.Duration(secs) * time.Second
	if d > retryAfterCap {
		return retryAfterCap
	}
	if d == 0 {
		return retryAfterFallback
	}
	return d
}
