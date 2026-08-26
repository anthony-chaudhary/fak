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
	return slackenv.Lookup(tokenEnvs[0]).Value
}

// ResolveChannel returns the scoreboard channel id from FAK_SCOREBOARD_CHANNEL, then a
// FAK_SCOREBOARD_CHANNEL= line in .env.slack.local. Returns "" if none found so a caller
// can require an explicit --channel.
func ResolveChannel() string {
	return slackenv.Lookup(channelEnvs[0]).Value
}

// ResolveProductChannel returns the #product channel id from FAK_PRODUCT_CHANNEL, then a
// FAK_PRODUCT_CHANNEL= line in .env.slack.local, then the CI/CD reporting sink
// (ResolveCICDReportChannel) — product is one of the reporting feeders folded onto that
// single channel. It never inherits FAK_SCOREBOARD_CHANNEL: a product post must not
// silently fall back to the scoreboard CLI's #scoreboard default, so this dedicated key
// (or --channel), then the family sink, is used — never ResolveChannel's key.
func ResolveProductChannel() string {
	for _, e := range productChannelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := slackenv.FileValue("FAK_PRODUCT_CHANNEL"); v != "" {
		return v
	}
	return ResolveCICDReportChannel()
}

// Feeder is one member of the CI/CD reporting family — a Slack status surface that
// publishes a number or card on a cadence and folds onto the shared CICDReportChannel
// sink. The family was declared exactly once before this — as PROSE in the
// CICDReportChannel doc comment — with no machine-readable roster, so the scoreboard
// notifier, the super-loop walker (#4863), and the liveness read (#4864) each risked
// re-hardcoding the list and silently drifting: a feeder wired into one but not another
// goes invisible to the operator, the exact silent-gap class this epic (#4862) kills.
// [ReportingFamily] promotes it to one exported ordered roster all three consume.
type Feeder struct {
	// Name is the canonical feeder name (the surface's slug — matches its leaf and its
	// #channel), rendered in the operator's timeline. The roster order is the family's.
	Name string
	// ChannelEnv is this surface's dedicated FAK_<SURFACE>_CHANNEL override key. It
	// resolves first; on a miss the surface folds onto [ResolveCICDReportChannel] — the
	// shared sink — so the family moves together while any one surface can peel off.
	ChannelEnv string
	// PostCommand is the fak verb that generates and posts this surface's card — the
	// walk's `Enter` hint for refreshing it. The three feeders whose payload is computed
	// by their cadence workflow (capacity, backlog, releases) post through the generic
	// `fak scoreboard post`; the rest have a dedicated verb.
	PostCommand string
}

// reportingFamily is the canonical ordered roster — the ONE place the CI/CD reporting
// family is enumerated. Order is the family's declaration order (scoreboard first).
// Add or remove a feeder HERE and every consumer (notifier, walker, liveness read)
// follows; there is deliberately no second hardcoded list to keep in sync.
var reportingFamily = []Feeder{
	{Name: "scoreboard", ChannelEnv: "FAK_SCOREBOARD_CHANNEL", PostCommand: "fak scoreboard post"},
	{Name: "blockers", ChannelEnv: "FAK_BLOCKERS_CHANNEL", PostCommand: "fak blockers feed"},
	{Name: "bench", ChannelEnv: "FAK_BENCH_CHANNEL", PostCommand: "fak bench post"},
	{Name: "cachevalue", ChannelEnv: "FAK_CACHEVALUE_CHANNEL", PostCommand: "fak cachevalue feed"},
	{Name: "capacity", ChannelEnv: "FAK_CAPACITY_CHANNEL", PostCommand: "fak scoreboard post"},
	{Name: "node-usage", ChannelEnv: "FAK_NODE_USAGE_CHANNEL", PostCommand: "fak nodeusage post"},
	{Name: "backlog", ChannelEnv: "FAK_BACKLOG_CHANNEL", PostCommand: "fak scoreboard post"},
	{Name: "dojo", ChannelEnv: "FAK_DOJO_CHANNEL", PostCommand: "fak dojo post"},
	{Name: "product", ChannelEnv: "FAK_PRODUCT_CHANNEL", PostCommand: "fak product post"},
	{Name: "releases", ChannelEnv: "FAK_RELEASES_CHANNEL", PostCommand: "fak scoreboard post"},
	{Name: "steering", ChannelEnv: "FAK_STEERING_CHANNEL", PostCommand: "fak steering report"},
}

// ReportingFamily returns a copy of the canonical ordered reporting-family roster — the
// single source of truth for "which surfaces make up the CI/CD reporting family". The
// notifier, the super-loop walker (#4863), and the liveness read (#4864) all enumerate
// the family through this one accessor instead of re-hardcoding it, so a feeder can
// never live in one consumer and be invisible to another. Returning a copy keeps the
// roster immutable to callers.
func ReportingFamily() []Feeder {
	out := make([]Feeder, len(reportingFamily))
	copy(out, reportingFamily)
	return out
}

// CICDReportChannel is the single Slack sink every CI/CD reporting feeder publishes to
// by default — the fleet's "put all CI/CD reporting in one channel" decision made
// concrete. The reporting family — enumerated ONCE as [ReportingFamily], the canonical
// roster; do NOT re-list its members here or the lists drift (#4865) — folds status /
// debt / issue / blocker / release reporting here, so an operator watches one timeline
// instead of a dozen near-silent
// rooms. It is a PUBLIC, non-secret channel id in the scoreboard Slack workspace (team
// T0BDEJF1HGB): the id grants nothing without the bot token, exactly like every other
// public ChannelDefault. A surface still overrides it with its own dedicated
// FAK_<SURFACE>_CHANNEL, and the whole family at once with CICDReportChannelEnv.
const CICDReportChannel = "C0BGQ411TCJ"

// CICDReportChannelEnv is the family-wide override key: set it (env or a
// FAK_CICD_REPORT_CHANNEL= line in .env.slack.local) to repoint EVERY CI/CD reporting
// feeder's default at once, without touching each surface's own FAK_<SURFACE>_CHANNEL.
// A per-surface key still wins over it — this is the shared FALLBACK, not a hard sink.
const CICDReportChannelEnv = "FAK_CICD_REPORT_CHANNEL"

// ResolveCICDReportChannel returns the CI/CD reporting sink: FAK_CICD_REPORT_CHANNEL
// (env then .env.slack.local) if the operator repointed the family, else the built-in
// CICDReportChannel default. It is the shared fallback every reporting surface's
// ResolveChannel lands on after its own dedicated FAK_<SURFACE>_CHANNEL key misses — so
// the sink id lives in exactly one place and the whole family moves together.
func ResolveCICDReportChannel() string {
	if v := strings.TrimSpace(os.Getenv(CICDReportChannelEnv)); v != "" {
		return v
	}
	if v := slackenv.FileValue(CICDReportChannelEnv); v != "" {
		return v
	}
	return CICDReportChannel
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
