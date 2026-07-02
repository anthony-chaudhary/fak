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
	"context"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/slackenv"
	"github.com/anthony-chaudhary/fak/internal/slackwire"
)

const slackAPI = "https://slack.com/api/"

// tokenEnvs / channelEnvs are the resolution order. FAK_SCOREBOARD_* are the
// dedicated keys for THIS workspace; they intentionally do NOT fall through to the
// lab SLACK_BOT_TOKEN — posting fak status to the lab control channel would be a
// cross-workspace mistake, so an unset scoreboard token is an error, not a silent
// reuse of the bridge token.
//
// The scoreboard workspace has TWO post targets that share the one bot token:
// #scoreboard (FAK_SCOREBOARD_CHANNEL) carries scores and scorecard numbers; #product
// (FAK_PRODUCT_CHANNEL, see ResolveProductChannel) carries product direction, persona
// findings, and product-status snapshots. Same workspace, same token, different channel
// — so the channel resolvers are split but the token resolver is shared.
var (
	tokenEnvs          = []string{"FAK_SCOREBOARD_TOKEN"}
	channelEnvs        = []string{"FAK_SCOREBOARD_CHANNEL"}
	productChannelEnvs = []string{"FAK_PRODUCT_CHANNEL"}
)

// Client posts scoreboard updates to Slack. The wire protocol (post/auth, 429
// handling, typed errors) lives in internal/slackwire — the ONE Slack transport;
// this type keeps what is scoreboard-specific: token resolution and the
// change-gating that keeps #scoreboard signal instead of heartbeat.
type Client struct {
	token    string
	http     *http.Client // optional injected client, passed through to the wire
	apiBase  string       // override for tests, passed through to the wire
	wire     *slackwire.Client
	lastMu   sync.RWMutex
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
		token:    token,
		apiBase:  slackAPI,
		lastPost: make(map[string]Update),
	}
	for _, o := range opts {
		o(c)
	}
	wopts := []slackwire.Option{slackwire.WithAPIBase(c.apiBase)}
	if c.http != nil {
		wopts = append(wopts, slackwire.WithHTTPClient(c.http))
	}
	c.wire = slackwire.New(token, wopts...)
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

// ResolveProductChannel returns the #product channel id from FAK_PRODUCT_CHANNEL, then a
// FAK_PRODUCT_CHANNEL= line in .env.slack.local. It mirrors ResolveChannel but targets the
// product-direction channel rather than #scoreboard; returns "" if none found so a caller
// can require an explicit --channel. The two never share a default: a product post must not
// silently fall back to #scoreboard, so `fak product post` requires this key (or --channel)
// and does not call ResolveChannel.
func ResolveProductChannel() string {
	for _, e := range productChannelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_PRODUCT_CHANNEL")
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by delegating
// to internal/slackenv — the single shared, tested resolver now used by every Slack
// surface (the byte-identical per-package walk-up that used to live here is gone).
func envFileValue(key string) string {
	return slackenv.FileValue(key)
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

	ts, err := c.wire.PostMessage(ctx, channel, text, blocks, "")
	if err != nil {
		return "", err
	}
	if up.Title != "" {
		c.recordLast(up)
	}
	return ts, nil
}

// AuthInfo is the identity a bot token resolves to — the subset of auth.test a
// diagnostic reports, enough to answer "does this token work, and as whom?".
type AuthInfo struct {
	URL    string // workspace URL, e.g. https://acme.slack.com/
	Team   string // workspace name
	User   string // the authenticating user/bot handle
	TeamID string // T... team id
	UserID string // U... user id
	BotID  string // B... bot id (set when the token is a bot token)
}

// AuthTest calls auth.test to verify the token is valid and report the identity it
// resolves to. It is the "does this token actually work" probe behind `fak slack check
// --auth`: a wrong, revoked, or workspace-mismatched bot token is the most common Slack
// failure, and it surfaces here as a concrete error (e.g. "invalid_auth") instead of a
// downstream chat.postMessage rejection with no context.
func (c *Client) AuthTest(ctx context.Context) (*AuthInfo, error) {
	info, err := c.wire.AuthTest(ctx)
	if err != nil {
		return nil, err
	}
	return &AuthInfo{
		URL:    info.URL,
		Team:   info.Team,
		User:   info.User,
		TeamID: info.TeamID,
		UserID: info.UserID,
		BotID:  info.BotID,
	}, nil
}
