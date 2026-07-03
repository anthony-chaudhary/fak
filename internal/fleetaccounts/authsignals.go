package fleetaccounts

import (
	"regexp"
	"strings"
)

// Ported from tools/fleet_session_signals.py: the auth-block taxonomy the runtime-status
// fold uses to classify a blocker's text into a kind + a human reason. Kept here so this
// package has no dependency on the Python module.

var accessWallRE = regexp.MustCompile(`(?i)organization has disabled Claude subscription access|` +
	`Claude subscription access .*disabled|` +
	`Use an Anthropic API key instead|` +
	`ask your admin to enable access`)

// usageCapRE matches a self-recovering USAGE/OVERAGE cap in a blocker's text: a session/weekly/
// usage-limit phrasing, or a reset window ("resets ...", "try again at ..."). This is the honest
// discriminator that keeps a recovering cap from being misfiled as a permanent access wall — the
// same text (e.g. a subscription banner) can carry BOTH the org-disable wording accessWallRE
// matches AND a reset, and a reset means the account comes back on its own. HONEST FENCE: this
// taxonomy is TEXT-ONLY — the probe layer (`claude -p`) never sees the anthropic-ratelimit-unified
// headers, so a cap with no reset word in its text cannot be distinguished here; the header-aware
// path lives in internal/agent/internal/resume.
var usageCapRE = regexp.MustCompile(`(?i)session limit|weekly limit|usage limit|usage cap|` +
	`overage|resets? (at|in|on) |try again (at|in|after)|/usage-credits`)

// authBlockKind classifies a blocker's text: usage | credit | access | auth. usage is checked
// FIRST because a recovering cap can carry text that also matches the permanent access-wall regex
// (a subscription banner naming both the org-disable wording and a reset) — and a cap that comes
// back at its reset must never be filed as a permanent wall that excludes the seat from dispatch
// and pages a human. A "usage" kind flows to a recoverable Throttled/CapacityBlockedUntil state
// downstream (capacity.go), where "access" is treated as permanently blocked.
func authBlockKind(text string) string {
	lower := strings.ToLower(text)
	if usageCapRE.MatchString(text) {
		return "usage"
	}
	if strings.Contains(lower, "credit balance is too low") {
		return "credit"
	}
	if accessWallRE.MatchString(text) {
		return "access"
	}
	return "auth"
}

// authBlockReason returns the human reason matching authBlockKind.
func authBlockReason(text string) string {
	switch authBlockKind(text) {
	case "usage":
		return "usage/overage cap (recovers at reset)"
	case "credit":
		return "credit balance too low"
	case "access":
		return "Claude subscription access disabled"
	default:
		return "auth/login required"
	}
}
