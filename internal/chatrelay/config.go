package chatrelay

import (
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// Token / channel resolution mirrors internal/blockerpost: one gitignored .env.slack.local
// (or the matching env var) configures the relay, and NOTHING — no token, no channel id —
// is ever baked into source, so the public tree stays clear of the secret-needle scan.
//
//	FAK_CHATRELAY_TOKEN    then a FAK_CHATRELAY_TOKEN=   line in .env.slack.local, then a
//	                       FALLBACK to the shared scoreboard bot token (scoreboard.ResolveToken),
//	                       so one workspace bot serves the relay and the status feeders alike.
//	FAK_CHATRELAY_CHANNEL  then a FAK_CHATRELAY_CHANNEL= line in .env.slack.local. NO fallback —
//	                       a chat relay must target a DELIBERATE channel and must never silently
//	                       inherit #scoreboard, so an unset channel forces an explicit --channel.

var (
	tokenEnvs   = []string{"FAK_CHATRELAY_TOKEN"}
	channelEnvs = []string{"FAK_CHATRELAY_CHANNEL"}
)

// ResolveToken returns the relay bot token: FAK_CHATRELAY_TOKEN env, then a
// FAK_CHATRELAY_TOKEN= line in .env.slack.local, then a fallback to the scoreboard bot
// token. Returns "" if none is found.
func ResolveToken() string {
	return slackenv.Resolve(tokenEnvs[0], scoreboard.ResolveToken)
}

// ResolveChannel returns the relay channel id from FAK_CHATRELAY_CHANNEL, then a
// FAK_CHATRELAY_CHANNEL= line in .env.slack.local. Returns "" if none found so the caller
// requires an explicit --channel (no silent fall-through to another channel's default).
func ResolveChannel() string {
	return slackenv.Lookup(channelEnvs[0]).Value
}
