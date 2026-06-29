// Package cachevaluepost posts the cache-effectiveness P&L roll-up — fak's WITNESSED
// kernel cache-value trend — to a single Slack "cache-value" channel, so the fleet has
// one durable place where "is fak's cache method paying off, and is it trending up or
// down?" gets an honest, dogfooded answer on a cadence. It is the OUTBOUND cache-value
// half of fak's Slack surface, the twin of internal/blockerpost, internal/benchpost,
// internal/grafanapost, and internal/dojopost: a local agent or CI folds the rolled-up
// cache-value report (internal/cachevaluereport) into ONE channel card.
//
// This is rung E (the notification output) of epic #1301 — the surface the channel
// C0BDSB81XDZ exists for. The fold is PURE: a cachevaluereport.Report in, a Card out —
// no clock, no I/O, no network — mirroring blockerpost.FoldIssues. The CLI
// (cmd/fak/cachevalue.go) collects the report and posts the card via the shared
// scoreboard chat.postMessage transport.
//
// What it renders is TRACK 1 only — the WITNESSED in-kernel KV-prefix reuse trend the
// report folds. The OBSERVED provider-$ track (epic #1301 Track 2) joins the report
// upstream and rides through here unchanged once it lands; this package never authors a
// number, it only renders the report it is handed.
//
// HONESTY FENCE (#1066): the report self-labels its publishable value family
// (marginal-over-tuned-warm-KV; the vs-naive 1/(1-reuse) multiple is structurally
// excluded). The card carries that fence verbatim into the channel so a reader can never
// mistake the realized reuse for the forbidden multiple — see render.go.
//
// What it is NOT: there is no inbound listener and no remote shell. fak does not take an
// order from a Slack message here — it POSTS a report it already folded locally. The lab
// control bridge (SLACK_BOT_TOKEN) is a separate concern in a separate workspace; this
// never shares that token.
//
// Resolution order (token, channel) mirrors the scoreboard/bench/blockers
// .env.slack.local idiom so one gitignored file configures every workspace:
//
//	FAK_CACHEVALUE_TOKEN    then a FAK_CACHEVALUE_TOKEN=   line in .env.slack.local,
//	                        then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                        so one bot token serves both channels in a shared workspace.
//	FAK_CACHEVALUE_CHANNEL  then a FAK_CACHEVALUE_CHANNEL= line in .env.slack.local,
//	                        then the built-in #cache-value default (ChannelDefault).
//
// Like blockers/grafana/dojo, the channel id is a PUBLIC, non-secret default — a channel
// id, not a credential — so a local agent posts with zero config. It deliberately does
// NOT inherit FAK_SCOREBOARD_CHANNEL (that is #scoreboard's own default; reusing it would
// misroute a cache-value card there).
package cachevaluepost

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is the #cache-value rollup channel in the scoreboard Slack workspace
// (team T0BDEJF1HGB). It is a PUBLIC channel id (not a secret): the @agent bot is a
// member and posts here with FAK_SCOREBOARD_TOKEN. Override with --channel or
// FAK_CACHEVALUE_CHANNEL — NOT FAK_SCOREBOARD_CHANNEL, which is the scoreboard CLI's own
// default (#scoreboard). This mirrors blockerpost/grafanapost/dojopost's posture: the
// channel id is public, only the token is secret.
const ChannelDefault = "C0BDSB81XDZ"

// tokenEnvs / channelEnvs are the dedicated cache-value keys. The token resolver adds a
// scoreboard fallback (below); the channel resolver falls back to the public default,
// never to FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_CACHEVALUE_TOKEN"}
	channelEnvs = []string{"FAK_CACHEVALUE_CHANNEL"}
)

// Card is one cache-value-channel post: the folded report plus who posted it. The render
// (Text/Blocks) reads only the report, so the card is deterministic given the report;
// Source rides in the context line. It satisfies the shared slackCard interface
// (Text/Blocks), exactly like blockerpost.Blocker and benchpost.Post.
type Card struct {
	Report cachevaluereport.Report // the rolled-up Track-1 trend this card renders
	Source string                  // who posted: "ci" | "agent" | <hostname> (optional)
}

// Fold builds the channel Card from a rolled-up cache-value report. It is pure: it copies
// the report in and renders from it, so a feeder folds once and posts. Source is set by
// the caller after the fold (mirroring blockerpost: the CLI stamps it), so the fold stays
// a single-argument report→card transform per the issue's `Fold(report) Card` contract.
func Fold(report cachevaluereport.Report) Card {
	return Card{Report: report}
}

// ResolveToken applies the documented order: FAK_CACHEVALUE_TOKEN env, then a
// FAK_CACHEVALUE_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard token
// (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate and matches blockerpost/grafanapost: the cache-value channel
// lives in the same Slack workspace as #scoreboard, so one bot token serves both. It
// falls back ONLY to the scoreboard token, never to the lab SLACK_BOT_TOKEN
// (scoreboard.ResolveToken already refuses that cross-workspace fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_CACHEVALUE_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the cache-value channel id from FAK_CACHEVALUE_CHANNEL, then a
// FAK_CACHEVALUE_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute a
// cache-value card whenever an operator sources .env.slack.local. The surface owns its
// own default, so it lands with zero config.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_CACHEVALUE_CHANNEL"); v != "" {
		return v
	}
	return ChannelDefault
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by delegating
// to internal/slackenv — the single shared, tested resolver every Slack surface uses.
func envFileValue(key string) string {
	return slackenv.FileValue(key)
}
