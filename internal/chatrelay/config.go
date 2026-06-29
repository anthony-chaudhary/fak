package chatrelay

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
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
	for _, e := range tokenEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	if v := envFileValue("FAK_CHATRELAY_TOKEN"); v != "" {
		return v
	}
	return scoreboard.ResolveToken()
}

// ResolveChannel returns the relay channel id from FAK_CHATRELAY_CHANNEL, then a
// FAK_CHATRELAY_CHANNEL= line in .env.slack.local. Returns "" if none found so the caller
// requires an explicit --channel (no silent fall-through to another channel's default).
func ResolveChannel() string {
	for _, e := range channelEnvs {
		if v := strings.TrimSpace(os.Getenv(e)); v != "" {
			return v
		}
	}
	return envFileValue("FAK_CHATRELAY_CHANNEL")
}

// envFileValue walks up from the cwd looking for .env.slack.local and returns the value of
// the first `KEY=...` line for key (an optional `export ` prefix is tolerated). Mirrors the
// scoreboard/blockerpost resolver so one gitignored file configures every workspace.
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
