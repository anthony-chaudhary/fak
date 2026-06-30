// Package dispatchpost posts the RESULT of a background code-dispatch run — the
// thing `fak loop run -- <cmd>` produces — to a Slack "dispatch" channel. It is the
// OUTBOUND result half of fak's background-dispatch surface, the twin of
// internal/benchpost and internal/scoreboard: when a slow background dispatch (a
// nightly agent, a /loop worker, a cron-driven fix loop) finishes, this folds the
// outcome into one channel post so a human watching the dispatch channel sees what
// the run did — without tailing a ledger.
//
// The load-bearing distinction from the other feeders: the other Slack feeders post
// STATUS (a number the moment it changes). This posts a DISPATCH RESULT, and it is
// WITNESSED from git, not self-reported. The card's "shipped" line is built from the
// HEAD delta the run actually produced (HeadBefore..HeadAfter), so a dispatch that
// exited 0 but committed nothing is reported as "OK (no commit)" rather than letting
// a green exit code masquerade as a landed change. That mirrors the repo's honesty
// contract (dos verify / commit-audit): a claim is only as strong as its witness.
//
// What it is NOT: there is no inbound listener and no remote shell. fak does not take
// a dispatch order from a Slack message here — it POSTS the outcome of a dispatch it
// already ran locally. The lab control bridge (SLACK_BOT_TOKEN) is a separate concern
// and a separate workspace; this never shares that token.
//
// Transport is reused verbatim from internal/scoreboard (a plain chat.postMessage
// client, no third-party deps). Only token/channel resolution and the result fold are
// new here.
//
// Resolution order (token, channel) mirrors the scoreboard/bench/bridge
// .env.slack.local idiom so one gitignored file configures every workspace:
//
//	FAK_DISPATCH_TOKEN    then a FAK_DISPATCH_TOKEN=   line in .env.slack.local,
//	                      then FALLBACK to the scoreboard token (FAK_SCOREBOARD_TOKEN)
//	                      so one bot token serves both channels in a shared workspace.
//	FAK_DISPATCH_CHANNEL  then a FAK_DISPATCH_CHANNEL= line in .env.slack.local.
//
// The channel id is NEVER hard-coded in tracked source (a real id is a gitignored
// value, per the scrubbing convention) — ResolveChannel returns "" when unset so the
// caller skips the post rather than posting to a wrong default.
package dispatchpost

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// tokenEnvs / channelEnvs are the dedicated dispatch keys. The token resolver adds a
// scoreboard fallback (below); the channel resolver does not — a dispatch result must
// go to the dispatch channel, never silently to #scoreboard.
var (
	tokenEnvs   = []string{"FAK_DISPATCH_TOKEN"}
	channelEnvs = []string{"FAK_DISPATCH_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_DISPATCH_TOKEN env, then a
// FAK_DISPATCH_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard
// token (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found.
//
// The fallback is deliberate and matches benchpost: the dispatch channel commonly
// lives in the same Slack workspace as #scoreboard, so one bot token serves both. It
// falls back ONLY to the scoreboard token, never to the lab SLACK_BOT_TOKEN
// (scoreboard.ResolveToken already refuses that fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_DISPATCH_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the dispatch channel id from FAK_DISPATCH_CHANNEL, then a
// FAK_DISPATCH_CHANNEL= line in .env.slack.local. Returns "" if none found so a caller
// can SKIP the post (the real id is never a tracked default) rather than hard-fail —
// a background dispatch must not error just because no channel is wired.
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_DISPATCH_CHANNEL")
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by delegating
// to internal/slackenv — the single shared, tested resolver now used by every Slack
// surface (the byte-identical per-package walk-up that used to live here is gone).
func envFileValue(key string) string {
	return slackenv.FileValue(key)
}
