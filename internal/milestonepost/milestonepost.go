// Package milestonepost posts the milestone tracking report — fak's WITNESSED
// maturity CLIMB plus the epic ROADMAP — to a single Slack "milestones" channel, so
// the fleet has one durable place where "how far has the project climbed, and is it
// moving?" gets an honest answer on a cadence. It is the OUTBOUND milestone half of
// fak's Slack surface, the twin of internal/cachevaluepost, internal/blockerpost,
// internal/benchpost, internal/grafanapost, and internal/dojopost: a local agent or
// CI folds the milestone report (internal/milestonereport) into ONE channel card.
//
// The fold is PURE: a milestonereport.Report in, a Card out — no clock, no I/O, no
// network — mirroring cachevaluepost.Fold. The CLI (cmd/fak/milestone.go) collects
// the report and posts the card via the shared scoreboard chat.postMessage transport.
// This package never authors a number; it only renders the report it is handed.
//
// What it is NOT: there is no inbound listener and no remote shell. fak does not take
// an order from a Slack message here — it POSTS a report it already folded locally.
// The lab control bridge (SLACK_BOT_TOKEN) is a separate concern in a separate
// workspace; this never shares that token.
//
// Resolution order (token, channel) mirrors the cachevalue/bench/blockers
// .env.slack.local idiom so one gitignored file configures every workspace:
//
//	FAK_MILESTONE_TOKEN    then a FAK_MILESTONE_TOKEN=   line in .env.slack.local,
//	                       then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                       so one bot token serves both channels in a shared workspace.
//	FAK_MILESTONE_CHANNEL  then a FAK_MILESTONE_CHANNEL= line in .env.slack.local,
//	                       then the built-in #milestones default (ChannelDefault).
//
// Like cachevalue/blockers/grafana/dojo, the channel id is a PUBLIC, non-secret
// default — a channel id, not a credential — so a local agent posts with zero config.
// It deliberately does NOT inherit FAK_SCOREBOARD_CHANNEL (that is #scoreboard's own
// default; reusing it would misroute a milestone card there).
package milestonepost

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/milestonereport"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is the #milestones channel in the scoreboard Slack workspace. It is
// a PUBLIC channel id (not a secret): the @agent bot is a member and posts here with
// FAK_SCOREBOARD_TOKEN. Override with --channel or FAK_MILESTONE_CHANNEL — NOT
// FAK_SCOREBOARD_CHANNEL, which is the scoreboard CLI's own default (#scoreboard).
// This mirrors cachevaluepost/blockerpost/grafanapost/dojopost's posture: the channel
// id is public, only the token is secret.
const ChannelDefault = "C0BDYFRSW6S"

// tokenEnvs / channelEnvs are the dedicated milestone keys. The token resolver adds a
// scoreboard fallback (below); the channel resolver falls back to the public default,
// never to FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_MILESTONE_TOKEN"}
	channelEnvs = []string{"FAK_MILESTONE_CHANNEL"}
)

// Card is one milestones-channel post: the folded report plus who posted it. The
// render (Text/Blocks) reads only the report, so the card is deterministic given the
// report; Source rides in the context line. It satisfies the shared slackCard
// interface (Text/Blocks), exactly like cachevaluepost.Card and blockerpost.Blocker.
type Card struct {
	Report milestonereport.Report // the folded milestone report this card renders
	Source string                 // who posted: "ci" | "agent" | <hostname> (optional)
}

// Fold builds the channel Card from a folded milestone report. It is pure: it copies
// the report in and renders from it, so a feeder folds once and posts. Source is set
// by the caller after the fold (mirroring cachevaluepost: the CLI stamps it).
func Fold(report milestonereport.Report) Card {
	return Card{Report: report}
}

// ResolveToken applies the documented order: FAK_MILESTONE_TOKEN env, then a
// FAK_MILESTONE_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard
// token. Returns "" if none found. The fallback matches cachevaluepost/grafanapost:
// the milestones channel lives in the same Slack workspace as #scoreboard, so one bot
// token serves both. It falls back ONLY to the scoreboard token, never to the lab
// SLACK_BOT_TOKEN (scoreboard.ResolveToken already refuses that cross-workspace
// fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_MILESTONE_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the milestones channel id from FAK_MILESTONE_CHANNEL, then a
// FAK_MILESTONE_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute a
// milestone card whenever an operator sources .env.slack.local.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_MILESTONE_CHANNEL"); v != "" {
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
