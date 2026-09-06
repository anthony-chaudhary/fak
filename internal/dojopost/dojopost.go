// Package dojopost formats and publishes fak DOJO calibration rollups and trends to Slack.
// Token and channel resolution use local environment or configuration files with safe defaults.
package dojopost

import (
	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/slackenv"
)

// ChannelDefault defines the default reporting destination channel ID for CI/CD runs.
const ChannelDefault = scoreboard.CICDReportChannel

var (
	tokenEnvs   = []string{"FAK_DOJO_TOKEN"}
	channelEnvs = []string{"FAK_DOJO_CHANNEL"}
)

// Resolved captures a resolved Slack configuration setting and its resolution source.
type Resolved struct {
	Value              string
	Source             string
	ScoreboardFallback bool
}

// ResolveToken locates the Slack authentication token from environment or config files.
func ResolveToken() string {
	return ResolveTokenWithSource().Value
}

// ResolveTokenWithSource locates the Slack authentication token and records source provenance.
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

// ResolveChannel identifies the target Slack channel ID from environment, file, or default.
func ResolveChannel() string {
	return ResolveChannelWithSource().Value
}

// ResolveChannelWithSource identifies the target Slack channel ID and records resolution source.
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
