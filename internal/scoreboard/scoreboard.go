// Package scoreboard posts fak status — scorecard results, scores, run events —
// to a Slack "scoreboard" channel. It is the OUTBOUND notification half of fak's
// Slack surface: local agents and CI/CD call it to publish a number the moment it
// changes, so a human watching #scoreboard sees code-debt drop or a gate go green
// without reading a log.
//
// This is deliberately NOT the lab control bridge. There is no remote shell, no
// !send, no transcript readback — just chat.postMessage with a formatted block.
// It also targets a SEPARATE Slack workspace from the lab/DGX bridge: that bridge
// drives the GPU boxes over a private control-hub, this one is a public status
// feed. The two never share a token. The lab bot is SLACK_BOT_TOKEN; this one is
// FAK_SCOREBOARD_TOKEN, so wiring a scoreboard never disturbs the lab plumbing.
//
// Resolution order (token, channel) matches the bridge's .env.slack.local idiom so
// an operator configures both workspaces in one gitignored file:
//
//	FAK_SCOREBOARD_TOKEN   then a FAK_SCOREBOARD_TOKEN=   line in .env.slack.local
//	FAK_SCOREBOARD_CHANNEL then a FAK_SCOREBOARD_CHANNEL= line in .env.slack.local
//
// No third-party deps: net/http + encoding/json only (same constraint as the bridge).
package scoreboard

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

const slackAPI = "https://slack.com/api/"

// tokenEnvs / channelEnvs are the resolution order. FAK_SCOREBOARD_* are the
// dedicated keys for THIS workspace; they intentionally do NOT fall through to the
// lab SLACK_BOT_TOKEN — posting fak status to the lab control channel would be a
// cross-workspace mistake, so an unset scoreboard token is an error, not a silent
// reuse of the bridge token.
var (
	tokenEnvs   = []string{"FAK_SCOREBOARD_TOKEN"}
	channelEnvs = []string{"FAK_SCOREBOARD_CHANNEL"}
)

// Client is a minimal Slack Web API client scoped to posting scoreboard updates.
type Client struct {
	token   string
	http    *http.Client
	apiBase string // override for tests
	lastMu  sync.RWMutex
	lastPost map[string]Update // keyed by title
}

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient injects a custom *http.Client (used in tests to avoid the network).
func WithHTTPClient(h *http.Client) Option { return func(c *Client) { c.http = h } }

// WithAPIBase overrides the Slack API base URL (used in tests).
func WithAPIBase(base string) Option { return func(c *Client) { c.apiBase = base } }

// NewClient builds a Client. If token is empty it is resolved from the environment
// (FAK_SCOREBOARD_TOKEN) and finally from a .env.slack.local file walked up from the
// working directory.
func NewClient(token string, opts ...Option) (*Client, error) {
	if token == "" {
		token = ResolveToken()
	}
	if token == "" {
		return nil, fmt.Errorf("no scoreboard token: set %s, or add FAK_SCOREBOARD_TOKEN=... to .env.slack.local",
			strings.Join(tokenEnvs, "/"))
	}
	c := &Client{
		token:   token,
		http:    &http.Client{Timeout: 40 * time.Second},
		apiBase: slackAPI,
		lastPost: make(map[string]Update),
	}
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
	return envFileValue("FAK_SCOREBOARD_TOKEN")
}

// ResolveChannel returns the scoreboard channel id from FAK_SCOREBOARD_CHANNEL, then a
// FAK_SCOREBOARD_CHANNEL= line in .env.slack.local. Returns "" if none found so a caller
// can require an explicit --channel.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_SCOREBOARD_CHANNEL")
}

// envFileValue walks up from the cwd looking for .env.slack.local and returns the value
// of the first `KEY=...` line for the given key (an optional `export ` prefix is tolerated).
// This mirrors the bridge's resolver so one gitignored file configures both workspaces.
func envFileValue(key string) string {
	dir, err := os.Getwd()
	if err != nil {
		return ""
	}
	for i := 0; i < 6; i++ {
		p := filepath.Join(dir, ".env.slack.local")
		if b, err := os.ReadFile(p); err == nil {
			for _, ln := range strings.Split(string(b), "\n") {
				ln = strings.TrimSpace(ln)
				ln = strings.TrimPrefix(ln, "export ")
				ln = strings.TrimSpace(ln)
				if v, ok := strings.CutPrefix(ln, key+"="); ok {
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

// shouldPost returns true if the update differs from the last posted update
// for the same title. Posts are gated so #scoreboard stays signal, not heartbeat.
func (c *Client) shouldPost(up Update) bool {
	c.lastMu.RLock()
	defer c.lastMu.RUnlock()

	last, ok := c.lastPost[up.Title]
	if !ok {
		return true // first post for this title
	}
	return up.Grade != last.Grade ||
		up.Score != last.Score ||
		up.Debt != last.Debt ||
		up.Verdict != last.Verdict
}

// recordLast saves the update as the last posted for its title.
func (c *Client) recordLast(up Update) {
	c.lastMu.Lock()
	defer c.lastMu.Unlock()
	c.lastPost[up.Title] = up
}

// postMessageResp carries the chat.postMessage outcome (Slack returns ok=false in
// the body even on HTTP 200).
type postMessageResp struct {
	OK      bool   `json:"ok"`
	TS      string `json:"ts"`
	Channel string `json:"channel"`
	Error   string `json:"error"`
}

// Post sends text to a channel and returns the posted message ts. blocks, when
// non-empty, attaches a Block Kit payload (used for the formatted scorecard card);
// text is the notification fallback Slack shows in the sidebar/badge.
// Posts are gated to avoid heartbeat noise: only posts when the key fields
// (grade, score, debt, verdict) change from the last post for the same title.
func (c *Client) Post(ctx context.Context, channel, text string, blocks []any) (string, error) {
	return c.PostWithUpdate(ctx, channel, Update{}, text, blocks)
}

// PostWithUpdate sends an update with explicit Update state for gating.
// It is the low-level entry point used by the scoreboard CLI; Post is a
// convenience wrapper for callers without an Update struct.
func (c *Client) PostWithUpdate(ctx context.Context, channel string, up Update, text string, blocks []any) (string, error) {
	if up.Title != "" && !c.shouldPost(up) {
		return "", nil // skip: no change from last post for this title
	}

	body := map[string]any{"channel": channel, "text": text}
	if len(blocks) > 0 {
		body["blocks"] = blocks
	}
	raw, err := json.Marshal(body)
	if err != nil {
		return "", err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiBase+"chat.postMessage", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	req.Header.Set("Content-Type", "application/json; charset=utf-8")
	resp, err := c.http.Do(req)
	if err != nil {
		return "", err
	}
	data, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	var r postMessageResp
	if err := json.Unmarshal(data, &r); err != nil {
		return "", fmt.Errorf("chat.postMessage: decode: %w (body=%.200s)", err, string(data))
	}
	if !r.OK {
		return "", fmt.Errorf("chat.postMessage: %s", r.Error)
	}
	if up.Title != "" {
		c.recordLast(up)
	}
	return r.TS, nil
}
