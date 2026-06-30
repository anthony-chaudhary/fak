// Package nodeusagepost posts COMPUTE-NODE-USAGE status — the latest fleet/node
// readiness, the active worker count, and inbound load — to a Slack "node-usage"
// channel. It is the OUTBOUND status half of fak's node-usage surface, the twin of
// internal/scoreboard / internal/benchpost / internal/dispatchpost: a local agent or a
// CI cadence calls it to publish the current node-usage picture so a human watching
// #node-usage sees which compute nodes are live and how loaded the fleet is without
// reading a roster.
//
// The headline signal is the FLEET SNAPSHOT — the exact JSON `fak lab status --json`
// emits (a fleet.Snapshot). FromSnapshot folds that into one card, so "what is the
// latest state of compute node usage?" is one command. The ad-hoc --kpi path (in the
// CLI) carries the other node-usage signals — the active worker count and the
// open-issue inbound-load proxy — the same way `fak scoreboard post --kpi` does.
//
// This is deliberately NOT the lab control bridge. There is no remote shell, no
// !send, no transcript readback — just chat.postMessage with a formatted block. It
// targets the SAME Slack workspace as #scoreboard/#capacity/#product (team
// T0BDEJF1HGB), so the bot token is shared; only the channel differs. A node-usage
// post must NEVER fall back to #scoreboard — node-usage status in the number feed
// would be a cross-channel mistake — so ResolveChannel returns "" when unset and the
// caller requires an explicit --channel.
//
// Resolution order (token, channel) mirrors the scoreboard/bench/dispatch/bridge
// .env.slack.local idiom so one gitignored file configures every workspace:
//
//	FAK_NODE_USAGE_TOKEN    then a FAK_NODE_USAGE_TOKEN=   line in .env.slack.local,
//	                        then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                        so one bot token serves both channels in a shared workspace.
//	FAK_NODE_USAGE_CHANNEL  then a FAK_NODE_USAGE_CHANNEL= line in .env.slack.local.
//
// The channel id is NEVER hard-coded in tracked source (a real id is a gitignored
// value, per the scrubbing convention) — ResolveChannel returns "" when unset so the
// caller errors rather than posting to a wrong default.
//
// Transport is reused verbatim from internal/scoreboard (a plain chat.postMessage
// client, no third-party deps). Only token/channel resolution and the snapshot fold
// are new here.
package nodeusagepost

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is #node-usage in the scoreboard Slack workspace — the public built-in
// the node-usage feeder (.github/workflows/node-usage-feed.yml) already documents as its
// default. Mirrors dojopost/blockerpost/steeringChannelDefault's posture: the channel id
// is public (it grants nothing without the bot token), only the token is secret. Wiring it
// here stops the surface resolving to NO channel and silently dry-running (was INCOMPLETE
// in `fak slack health`, #1428).
const ChannelDefault = "C0BEFFPCSAU"

// tokenEnvs / channelEnvs are the dedicated node-usage keys. The token resolver adds a
// scoreboard fallback (below); the channel resolver falls through to ChannelDefault — a
// node-usage post goes to the node-usage channel, never silently to #scoreboard.
var (
	tokenEnvs   = []string{"FAK_NODE_USAGE_TOKEN"}
	channelEnvs = []string{"FAK_NODE_USAGE_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_NODE_USAGE_TOKEN env, then a
// FAK_NODE_USAGE_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard
// token (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate and matches benchpost/dispatchpost: the node-usage
// channel lives in the same Slack workspace as #scoreboard, so one bot token serves
// both. It falls back ONLY to the scoreboard token, never to the lab SLACK_BOT_TOKEN
// (scoreboard.ResolveToken already refuses that fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_NODE_USAGE_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the node-usage channel id from FAK_NODE_USAGE_CHANNEL, then a
// FAK_NODE_USAGE_CHANNEL= line in .env.slack.local. Returns "" if none found so a
// caller can require an explicit --channel (the real id is never a tracked default)
// rather than silently posting to #scoreboard.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_NODE_USAGE_CHANNEL"); v != "" {
		return v
	}
	return ChannelDefault
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by delegating
// to internal/slackenv — the single shared, tested resolver now used by every Slack
// surface (the byte-identical per-package walk-up that used to live here is gone).
func envFileValue(key string) string {
	return slackenv.FileValue(key)
}
