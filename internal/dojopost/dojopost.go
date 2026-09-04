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
//
// Invariants and Guards:
//
// Invariant: dojo post formatting and token/channel resolution are fail-closed and deterministic.
// Guard: token resolution strictly isolates lab secrets, preventing SLACK_BOT_TOKEN fallback.
// Guard: channel resolution defaults safely to the public CI/CD reporting sink without misrouting to scoreboard channels.
package dojopost

import (
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault is the CI/CD reporting sink (scoreboard.CICDReportChannel) — dojo is
// one of the CI/CD reporting feeders folded onto that single channel. It is a PUBLIC
// channel id (not a secret): the @agent bot is a member and posts here with
// FAK_SCOREBOARD_TOKEN. Override this surface with --channel or FAK_DOJO_CHANNEL — NOT
// FAK_SCOREBOARD_CHANNEL, which is the scoreboard CLI's own default (#scoreboard) — or
// repoint the whole reporting family with FAK_CICD_REPORT_CHANNEL.
const ChannelDefault = scoreboard.CICDReportChannel

// tokenEnvs is the dedicated dojo token key; the resolver adds a scoreboard fallback
// (below). channelEnvs is the dedicated dojo channel key; the channel resolver adds
// the public ChannelDefault, never the generic FAK_SCOREBOARD_CHANNEL.
var (
	tokenEnvs   = []string{"FAK_DOJO_TOKEN"}
	channelEnvs = []string{"FAK_DOJO_CHANNEL"}
)

// Resolved records a dojo Slack setting plus where it came from. The value can be a
// channel id or a token; callers must redact token values before displaying them.
type Resolved struct {
	Value              string
	Source             string
	ScoreboardFallback bool
}

// ResolveToken applies the documented order: FAK_DOJO_TOKEN env, then a
// FAK_DOJO_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard token
// (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate: the dojo channel lives in the same Slack workspace as
// #scoreboard, so one bot token serves both. It falls back only to the scoreboard
// token, NEVER to the lab SLACK_BOT_TOKEN (scoreboard.ResolveToken already refuses
// that fall-through).
func ResolveToken() string {
	return ResolveTokenWithSource().Value
}

// ResolveTokenWithSource is ResolveToken plus the diagnostic source label. It lets
// `fak dojo post` report whether it used the dedicated dojo token or the noisier
// scoreboard fallback instead of silently masking a dojo-token misconfiguration.
func ResolveTokenWithSource() Resolved {
	for _, e := range tokenEnvs {
		if r := slackenv.Lookup(e); r.Set() {
			return Resolved{Value: r.Value, Source: sourceLabel(r)}
		}
	}
	if r := slackenv.Lookup("FAK_SCOREBOARD_TOKEN"); r.Set() {
		return Resolved{
			Value:              r.Value,
			Source:             "scoreboard-fallback (" + sourceLabel(r) + ")",
			ScoreboardFallback: true,
		}
	}
	return Resolved{Source: "unset"}
}

// ResolveChannel returns the dojo channel id from FAK_DOJO_CHANNEL, then a
// FAK_DOJO_CHANNEL= line in .env.slack.local, then the public ChannelDefault. It
// deliberately does NOT fall through to FAK_SCOREBOARD_CHANNEL — that env var is the
// scoreboard CLI's default target (#scoreboard), so reusing it here would misroute
// the dojo surface whenever an operator sources .env.slack.local. The dojo surface
// owns its own default, so it lands with zero config.
func ResolveChannel() string {
	return ResolveChannelWithSource().Value
}

// ResolveChannelWithSource is ResolveChannel plus the source label used in post
// results. The built-in default is deliberately visible because a recreated/migrated
// Slack channel should be fixed at the resolver boundary, not hidden behind a plain
// chat.postMessage failure.
func ResolveChannelWithSource() Resolved {
	for _, e := range channelEnvs {
		if r := slackenv.Lookup(e); r.Set() {
			return Resolved{Value: r.Value, Source: sourceLabel(r)}
		}
	}
	return Resolved{Value: scoreboard.ResolveCICDReportChannel(), Source: "built-in default"}
}

func sourceLabel(r slackenv.Resolved) string {
	if r.Source == slackenv.SourceUnset {
		return "unset"
	}
	return string(r.Source) + ":" + r.Key
}
