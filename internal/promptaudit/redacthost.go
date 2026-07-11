package promptaudit

// This file adds the REDACTION lever on top of the #1691 stego scanner and the
// #1692 provenance layer. The scanner/provenance layers answer "is a hidden
// hostname channel present, and who produced it?"; they never rewrite the bytes.
// When a control-metadata segment ALSO carries the literal gateway host it was
// classifying against — the base-URL-classifier path the article warns about,
// where a provider shim splices the ANTHROPIC_BASE_URL host into model-visible
// control metadata — an operator needs to be able to scrub that host before the
// segment is logged or echoed, WITHOUT losing a known-good host they explicitly
// trust. RedactHostnames is that scrub: host-structural, allowlist-aware, and
// non-mutating (it returns a redacted copy; Raw is never touched).

import (
	"regexp"
	"strings"
)

// hostRedaction is the fixed placeholder RedactHostnames substitutes for a
// non-allowlisted gateway hostname. It is deliberately visible so an operator
// can see that a host was present and scrubbed, without leaking which host it
// was.
const hostRedaction = "[redacted-host]"

// gatewayHostRe matches a dotted DNS hostname: one or more labels separated by
// '.', ending in an alphabetic top-level label of 2+ letters. Requiring an
// ALPHABETIC final label is what keeps redaction from eating the date-sentence
// channels the Scan layer owns: a date token ("2026-06-30", "2026/06/30") uses
// '-' or '/' as its separator and ends in digits, and a bare IPv4 literal ends
// in digits too, so neither matches here. That leaves RedactHostnames targeting
// real gateway hosts (api.anthropic.com, a custom ANTHROPIC_BASE_URL host) only.
var gatewayHostRe = regexp.MustCompile(`\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}\b`)

// RedactHostnames removes gateway hostnames from a rendered control-metadata
// segment while keeping any host on the allowlist, and returns the redacted
// copy (seg is never mutated). It is the sanitizer for the base-URL-classifier
// leak: a provider shim can splice the ANTHROPIC_BASE_URL host into control
// metadata, and this scrubs that host to hostRedaction before the segment is
// surfaced, unless the host — or a parent domain of it — is explicitly
// allowlisted.
//
// Matching is host-STRUCTURAL, not a fixed denylist: every dotted DNS hostname
// in seg is a candidate. A candidate is KEPT iff it equals an allowlist entry
// (compared case-insensitively) or is a subdomain of one (the host ends in
// "."+entry); every other candidate is replaced by hostRedaction.
func RedactHostnames(seg string, allowlist []string) string {
	allow := normalizeAllowlist(allowlist)
	return gatewayHostRe.ReplaceAllStringFunc(seg, func(host string) string {
		if hostAllowed(host, allow) {
			return host
		}
		return hostRedaction
	})
}

// normalizeAllowlist lowercases and trims the allowlist entries and drops any
// empties or a leading '.', so `.anthropic.com`, `anthropic.com`, and
// ` Anthropic.com ` all mean the same allowlisted domain.
func normalizeAllowlist(allowlist []string) []string {
	out := make([]string, 0, len(allowlist))
	for _, a := range allowlist {
		a = strings.ToLower(strings.TrimSpace(a))
		a = strings.TrimPrefix(a, ".")
		if a != "" {
			out = append(out, a)
		}
	}
	return out
}

// hostAllowed reports whether host is exactly an allowlist entry or a subdomain
// of one, compared case-insensitively. A trailing FQDN dot on host is ignored.
func hostAllowed(host string, allow []string) bool {
	h := strings.ToLower(strings.TrimSuffix(host, "."))
	for _, a := range allow {
		if h == a || strings.HasSuffix(h, "."+a) {
			return true
		}
	}
	return false
}
