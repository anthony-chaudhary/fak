package marketing

import (
	"os"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// resolve.go — the #marketing channel/token resolution, mirroring internal/benchpost so one
// gitignored .env.slack.local configures every workspace. The channel id is NEVER a tracked
// default (the scrubbing convention): ResolveChannel returns "" when unset so a caller
// requires an explicit --channel. The token falls back to the scoreboard token (one bot
// serves the workspace), but the channel never falls back — a marketing post must go to the
// marketing channel, never silently to #scoreboard.

var (
	tokenEnvs   = []string{"FAK_MARKETING_TOKEN"}
	channelEnvs = []string{"FAK_MARKETING_CHANNEL"}
)

// ResolveToken applies the documented order: FAK_MARKETING_TOKEN env, then a
// FAK_MARKETING_TOKEN= line in .env.slack.local, then a FALLBACK to the scoreboard token
// (FAK_SCOREBOARD_TOKEN / its .env.slack.local line). Returns "" if none found. It falls
// back only to the scoreboard token, never to the lab SLACK_BOT_TOKEN (scoreboard.ResolveToken
// already refuses that fall-through).
func ResolveToken() string {
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_MARKETING_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the marketing channel id from FAK_MARKETING_CHANNEL, then a
// FAK_MARKETING_CHANNEL= line in .env.slack.local. Returns "" if none found so a caller can
// require an explicit --channel (the real id is never a tracked default).
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_MARKETING_CHANNEL")
}

// envFileValue resolves key from .env.slack.local, walked up from the cwd, by delegating
// to internal/slackenv — the single shared, tested resolver now used by every Slack
// surface (the byte-identical per-package walk-up that used to live here is gone).
func envFileValue(key string) string {
	return slackenv.FileValue(key)
}
