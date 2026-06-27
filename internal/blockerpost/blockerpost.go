// Package blockerpost posts BLOCKERS — the things that stop forward progress —
// to a single Slack "blockers" channel, so the fleet has one central place where an
// ongoing impediment is recorded and a human-needed one is surfaced. It is the
// OUTBOUND blocker half of fak's Slack surface, the twin of internal/benchpost,
// internal/dispatchpost, and internal/scoreboard: any agent, CI job, or background
// loop that hits a wall folds it into one channel post.
//
// The load-bearing distinction this package owns is SEVERITY-as-surfacing. The other
// feeders post a number; this one posts a blocker at one of three render states, and
// the state decides how LOUD the post is:
//
//	status    an ongoing, tracked blocker the channel should RECORD but not page
//	          anyone about — a muted line that scrolls by (no broadcast mention).
//	          "GPU-gated, waiting on DGX hours", "peer merge in flight".
//	operator  a blocker that needs a HUMAN to act — SURFACED: a broadcast mention
//	          (<!here> by default, or a named owner), a red glyph, an OPERATOR-NEEDED
//	          banner, and a "do this next" affordance. "FAK_SCOREBOARD_TOKEN missing",
//	          "DA33 host unreachable — needs a manual restart".
//	clear     an all-clear heartbeat (no open blockers) — a quiet green card so the
//	          daily cadence shows the pipe is alive without paging anyone.
//
// Only `operator` is surfaced. {status, clear} are the background tiers. That two-tier
// (background vs surfaced) split is the whole reason this channel is distinct from the
// status feeders — see Blocker.Text / Blocker.Blocks for the mechanics (the mention
// rides in BOTH the notification fallback and the lead section, which is what makes
// Slack actually page on an operator blocker).
//
// What it is NOT: there is no inbound listener and no remote shell. fak does not take
// an order from a Slack message here — it POSTS a blocker it already detected locally.
// The lab control bridge (SLACK_BOT_TOKEN) is a separate concern in a separate
// workspace; this never shares that token.
//
// Transport is reused verbatim from internal/scoreboard (a plain chat.postMessage
// client, no third-party deps). Only token/channel resolution and the blocker fold are
// new here.
//
// Resolution order (token, channel) mirrors the scoreboard/bench/dispatch
// .env.slack.local idiom so one gitignored file configures every workspace:
//
//	FAK_BLOCKERS_TOKEN    then a FAK_BLOCKERS_TOKEN=   line in .env.slack.local,
//	                      then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                      so one bot token serves both channels in a shared workspace.
//	FAK_BLOCKERS_CHANNEL  then a FAK_BLOCKERS_CHANNEL= line in .env.slack.local,
//	                      then the built-in #blockers default (ChannelDefault).
//
// Unlike dispatchpost (whose channel id is a gitignored value), the blockers channel
// id is a PUBLIC, non-secret default — exactly like steeringChannelDefault — so a local
// agent posts a blocker with zero config. Redirect it with FAK_BLOCKERS_CHANNEL or
// --channel; it never inherits FAK_SCOREBOARD_CHANNEL (that is #scoreboard's default,
// and a blocker must never misroute there).
package blockerpost

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
)

// ChannelDefault is #blockers in the scoreboard Slack workspace (team T0BDEJF1HGB). It
// is a PUBLIC channel id (not a secret): the @agent bot is a member and posts here with
// FAK_SCOREBOARD_TOKEN. Override with --channel or FAK_BLOCKERS_CHANNEL.
const ChannelDefault = "C0BDHRJJPTP"

// Severity is the render state of a blocker — it decides how loud the post is.
type Severity string

const (
	// SeverityStatus is an ongoing, tracked blocker: recorded, but no broadcast mention.
	SeverityStatus Severity = "status"
	// SeverityOperator is a blocker that needs a human: surfaced with a broadcast mention.
	SeverityOperator Severity = "operator"
	// SeverityClear is an all-clear heartbeat: a quiet green card, no mention.
	SeverityClear Severity = "clear"
)

// ParseSeverity maps a flag string to a Severity, defaulting to SeverityStatus for the
// empty string and reporting an error for an unknown value so a typo never silently
// downgrades an operator page to a muted status line.
func ParseSeverity(s string) (Severity, bool) {
	switch Severity(strings.ToLower(strings.TrimSpace(s))) {
	case "", SeverityStatus:
		return SeverityStatus, true
	case SeverityOperator:
		return SeverityOperator, true
	case SeverityClear:
		return SeverityClear, true
	default:
		return "", false
	}
}

// Blocker is one blockers-channel post, decoupled from what produced it. A manual
// `fak blockers post` builds one directly; the CI feeder's FoldIssues builds one from
// the open blocker backlog — so the renderer (Text/Blocks) has a single input shape,
// the same pattern as benchpost.Post and dojopost.Post.
type Blocker struct {
	Severity  Severity // status | operator | clear (default status)
	Title     string   // short headline, e.g. "DA33 CPU host unreachable"
	Detail    string   // one-line what is blocked / why
	Lines     []string // optional body, one per sub-item (the feeder's per-issue rows)
	Owner     string   // operator mention target ("<@U123>" / "<!here>"); defaults to <!here>
	Action    string   // optional "do this next" label, e.g. "restart the DA33 serve"
	ActionURL string   // optional link target (a runbook / issue / docs URL)
	Ref       string   // optional stable key shown in context, e.g. "#921" or a hostname
	Source    string   // who posted: "ci" | "agent" | <hostname> (optional)
}

// tokenEnvs / channelEnvs are the dedicated blockers keys. The token resolver adds a
// scoreboard fallback (below); the channel resolver falls back to the public default,
// never to FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_BLOCKERS_TOKEN"}
	channelEnvs = []string{"FAK_BLOCKERS_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_BLOCKERS_TOKEN env, then a
// FAK_BLOCKERS_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard token
// (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate and matches benchpost/dispatchpost: the blockers channel
// commonly lives in the same Slack workspace as #scoreboard, so one bot token serves
// both. It falls back ONLY to the scoreboard token, never to the lab SLACK_BOT_TOKEN
// (scoreboard.ResolveToken already refuses that fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_BLOCKERS_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the blockers channel id from FAK_BLOCKERS_CHANNEL, then a
// FAK_BLOCKERS_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute a
// blocker to #scoreboard whenever an operator has sourced .env.slack.local. Blockers
// owns its own default, so the surface lands in #blockers with zero config.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_BLOCKERS_CHANNEL"); v != "" {
		return v
	}
	return ChannelDefault
}

// envFileValue walks up from the cwd looking for .env.slack.local and returns the value
// of the first `KEY=...` line for the given key (an optional `export ` prefix is
// tolerated). Mirrors the scoreboard/bench/dispatch resolver so one gitignored file
// configures every workspace.
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
