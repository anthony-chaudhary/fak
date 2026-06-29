// Package grafanapost posts fak #grafana-channel cards — exported Grafana
// snapshots and long-lived dashboard / debug links — to a Slack channel. It is
// the OUTBOUND Grafana half of fak's Slack surface, the twin of internal/benchpost
// and internal/dojopost: a local agent or CI folds a source (a snapshot URL the
// operator exported from Grafana, or the committed link registry) into a channel
// card, so a human watching #grafana sees the snapshot or the dashboard list
// without leaving Slack.
//
// What it is NOT: there is no inbound listener and no Grafana API client here. A
// snapshot post is a POST of a URL the operator already exported; a links rollup
// folds a committed registry. Pulling snapshots from a live Grafana via its API is
// a named follow-on, not this spine. Transport is reused verbatim from
// internal/scoreboard (a plain chat.postMessage client, no third-party deps); only
// token/channel resolution and the Grafana folds are new here.
//
// Resolution order mirrors the scoreboard/bench/dojo .env.slack.local idiom so one
// gitignored file configures every workspace:
//
//	FAK_GRAFANA_TOKEN    then a FAK_GRAFANA_TOKEN=   line in .env.slack.local,
//	                     then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                     so one bot token serves every channel in the shared
//	                     workspace. It NEVER falls back to the lab SLACK_BOT_TOKEN (a
//	                     cross-workspace mistake scoreboard.ResolveToken refuses).
//	FAK_GRAFANA_CHANNEL  then a FAK_GRAFANA_CHANNEL= line in .env.slack.local, then
//	                     the built-in ChannelDefault. The channel id is a PUBLIC,
//	                     non-secret value (the same posture #dojo and #blockers keep):
//	                     a channel id, not a credential, so the surface lands with
//	                     zero config. It deliberately does NOT inherit the generic
//	                     FAK_SCOREBOARD_CHANNEL (that is the scoreboard CLI's own
//	                     #scoreboard default, so reusing it would misroute #grafana).
package grafanapost

import (
	"encoding/json"
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is the #grafana channel in the scoreboard Slack workspace (team
// T0BDEJF1HGB). It is a PUBLIC channel id (not a secret): the @agent bot is a member
// and posts here with FAK_SCOREBOARD_TOKEN. Override with --channel or
// FAK_GRAFANA_CHANNEL — NOT FAK_SCOREBOARD_CHANNEL, which is the scoreboard CLI's own
// default (#scoreboard). This mirrors dojopost.ChannelDefault's posture: the channel
// id is public, only the token is secret.
const ChannelDefault = "C0BDV5M3XM3"

// DefaultBaseURL is the Grafana base used to build a dashboard link when a Link has
// no absolute url of its own. http://localhost:3000 matches the shipped tools/grafana
// stack (admin / fleet); point a registry at a tailnet/public Grafana by setting
// base_url, or override per-post with --base-url.
const DefaultBaseURL = "http://localhost:3000"

// tokenEnvs is the dedicated grafana token key; the resolver adds a scoreboard
// fallback (below). channelEnvs is the dedicated grafana channel key; the channel
// resolver adds the public ChannelDefault, never the generic FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_GRAFANA_TOKEN"}
	channelEnvs = []string{"FAK_GRAFANA_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_GRAFANA_TOKEN env, then a
// FAK_GRAFANA_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard
// token (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate: the #grafana channel lives in the same Slack workspace
// as #scoreboard, so one bot token serves both. It falls back only to the scoreboard
// token, NEVER to the lab SLACK_BOT_TOKEN (scoreboard.ResolveToken already refuses
// that cross-workspace fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_GRAFANA_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the #grafana channel id from FAK_GRAFANA_CHANNEL, then a
// FAK_GRAFANA_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute
// the surface whenever an operator sources .env.slack.local. The surface owns its own
// default, so it lands with zero config.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_GRAFANA_CHANNEL"); v != "" {
		return v
	}
	return ChannelDefault
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by
// delegating to internal/slackenv — the single shared, tested resolver every Slack
// surface uses.
func envFileValue(key string) string {
	return slackenv.FileValue(key)
}

// Categories an operator may tag a Link with. They drive the rollup grouping and the
// public-demo / debug split the channel cares about; an unrecognized value is carried
// through verbatim (UNCATEGORIZED in the rollup) rather than dropped.
const (
	CategoryPublicDemo = "public-demo" // a long-lived dashboard kept up for a public demo
	CategoryDebug      = "debug"       // a debug / triage dashboard (often stack-local)
	CategoryRollup     = "rollup"      // a saved rollup / overview view
)

// Link is one registered Grafana dashboard or debug link in the committed registry.
// Either UID (resolved against the registry base_url into a /d/<uid> route) or an
// absolute URL identifies it; a Link with neither still renders its title but carries
// no link (the rollup flags it rather than fabricating a URL).
type Link struct {
	Title       string `json:"title"`
	UID         string `json:"uid,omitempty"`
	URL         string `json:"url,omitempty"`
	Category    string `json:"category"`              // public-demo | debug | rollup
	Lifetime    string `json:"lifetime,omitempty"`    // long-lived | stack-local | ephemeral
	Description string `json:"description,omitempty"` // one-line: what the dashboard shows
	Source      string `json:"source,omitempty"`      // provenance: the dashboard JSON it came from
}

// ResolveURL returns the link's absolute URL: its own url if set, else
// base/d/<uid> (the Grafana dashboard route) when both a uid and a base are
// available. Returns "" when neither path yields a URL — callers render "(no url)"
// rather than invent one.
func (l Link) ResolveURL(base string) string {
	if u := strings.TrimSpace(l.URL); u != "" {
		return u
	}
	uid := strings.TrimSpace(l.UID)
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if uid != "" && base != "" {
		return base + "/d/" + uid
	}
	return ""
}

// Registry is the committed link corpus (docs/grafana/links.json), schema
// fak-grafana-links/1: a base_url plus the long-lived dashboard / debug links the
// rollup folds.
type Registry struct {
	Schema  string `json:"schema"`
	BaseURL string `json:"base_url,omitempty"`
	Links   []Link `json:"links"`
}

// Base returns the effective Grafana base URL for the registry: its own base_url, or
// DefaultBaseURL when unset.
func (r *Registry) Base() string {
	if r != nil {
		if b := strings.TrimSpace(r.BaseURL); b != "" {
			return b
		}
	}
	return DefaultBaseURL
}

// Find returns the registry link whose UID matches (case-insensitive), and whether it
// was found.
func (r *Registry) Find(uid string) (Link, bool) {
	want := strings.ToLower(strings.TrimSpace(uid))
	if r == nil || want == "" {
		return Link{}, false
	}
	for _, l := range r.Links {
		if strings.ToLower(strings.TrimSpace(l.UID)) == want {
			return l, true
		}
	}
	return Link{}, false
}

// LoadRegistry reads links.json and returns its parsed contents.
func LoadRegistry(path string) (*Registry, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var r Registry
	if err := json.Unmarshal(b, &r); err != nil {
		return nil, err
	}
	return &r, nil
}

// Snapshot is one exported Grafana snapshot to post: the operator exported a share
// link / snapshot URL from Grafana and posts it with this context. URL is the only
// required field for a useful card; the rest annotate it.
type Snapshot struct {
	Title     string // headline, e.g. "p99 latency spike during the 14:00 deploy"
	URL       string // the exported snapshot / share URL (required for a live post)
	Dashboard string // source dashboard name, e.g. "FAK Gateway Observability"
	TimeRange string // the captured window, e.g. "last 6h" or "14:00–14:30 UTC"
	Expires   string // when the snapshot link expires (Grafana snapshots can be TTL'd)
	Note      string // optional free-form note under the headline
}
