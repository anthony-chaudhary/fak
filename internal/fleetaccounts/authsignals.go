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

// authBlockKind classifies a blocker's text: credit | access | auth.
func authBlockKind(text string) string {
	lower := strings.ToLower(text)
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
	case "credit":
		return "credit balance too low"
	case "access":
		return "Claude subscription access disabled"
	default:
		return "auth/login required"
	}
}
