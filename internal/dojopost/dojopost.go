// Package dojopost posts fak DOJO calibration rollups — the latest
// prediction-vs-reality run and the across-tick calibration trend — to a Slack
// dojo channel. It is the OUTBOUND dojo half of fak's Slack surface, the twin of
// internal/benchpost and internal/scoreboard: a local agent or CI folds the dojo
// report (or its durable history ledger) into a channel post the moment a tick
// lands, so a human watching the dojo channel sees "are our predictors getting
// better calibrated over time" without reading docs/dojo/history.jsonl.
//
// What it is NOT: there is no inbound listener and no scoring here. The scoring,
// fold, and ledger live in the pure internal/dojo package; this package only turns
// a folded dojo.Report (or a parsed ledger) into a Slack message and resolves the
// dojo token/channel. Transport is reused verbatim from internal/scoreboard (a
// plain chat.postMessage client, no third-party deps).
//
// Resolution order mirrors the scoreboard/bench/steering .env.slack.local idiom so
// one gitignored file configures every workspace:
//
//	FAK_DOJO_TOKEN    then a FAK_DOJO_TOKEN=   line in .env.slack.local,
//	                  then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN) so
//	                  one bot token serves every channel in the shared workspace. It
//	                  NEVER falls back to the lab SLACK_BOT_TOKEN (a cross-workspace
//	                  mistake scoreboard.ResolveToken already refuses).
//	FAK_DOJO_CHANNEL  then a FAK_DOJO_CHANNEL= line in .env.slack.local, then the
//	                  built-in dojo channel default. Unlike the bench channel, the
//	                  dojo channel id is a PUBLIC, non-secret value (a channel id, not
//	                  a credential — the same posture #steering-guard keeps), so the
//	                  surface lands with zero config; redirect it only via --channel or
//	                  FAK_DOJO_CHANNEL. It deliberately does NOT inherit the generic
//	                  FAK_SCOREBOARD_CHANNEL (that is the scoreboard CLI's #scoreboard
//	                  default, so reusing it would misroute the dojo surface).
package dojopost

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is the dojo channel in the scoreboard Slack workspace (team
// T0BDEJF1HGB). It is a PUBLIC channel id (not a secret): the @agent bot is a
// member and posts here with FAK_SCOREBOARD_TOKEN. Override with --channel or
// FAK_DOJO_CHANNEL — NOT FAK_SCOREBOARD_CHANNEL, which is the scoreboard CLI's own
// default (#scoreboard). This mirrors steeringChannelDefault's posture: the channel
// id is public, only the token is secret.
const ChannelDefault = "C0BDP2V51L1"

// tokenEnvs is the dedicated dojo token key; the resolver adds a scoreboard fallback
// (below). channelEnvs is the dedicated dojo channel key; the channel resolver adds
// the public ChannelDefault, never the generic FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_DOJO_TOKEN"}
	channelEnvs = []string{"FAK_DOJO_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_DOJO_TOKEN env, then a
// FAK_DOJO_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard token
// (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate: the dojo channel lives in the same Slack workspace as
// #scoreboard, so one bot token serves both. It falls back only to the scoreboard
// token, NEVER to the lab SLACK_BOT_TOKEN (scoreboard.ResolveToken already refuses
// that fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_DOJO_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the dojo channel id from FAK_DOJO_CHANNEL, then a
// FAK_DOJO_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute
// the dojo surface whenever an operator sources .env.slack.local. The dojo surface
// owns its own default, so it lands with zero config.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_DOJO_CHANNEL"); v != "" {
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
