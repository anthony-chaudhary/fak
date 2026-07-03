package gateway

import (
	"regexp"
	"strings"
)

// forbidden_detail.go records a SCRUBBED, bounded snapshot of the most recent PERSISTENT 403's
// upstream body for the operator-only /debug/vars drilldown. The gap it closes: a 403's raw body
// is the one signal that tells org-disabled apart from model-not-permitted apart from an abuse
// gate, but it is deliberately withheld from the downstream client (the #82/#346 trust boundary),
// the --debug-stats FAILED line is payload-free by design, and the audit journal records only
// tool verdicts — so after the 2026-07-03 gem8 transient-403 storm there was no way to tell WHY
// the denial happened. /debug/vars is loopback-only (an operator surface, not the client), so the
// body may live here — but only after a conservative scrub, because a hostile or careless
// upstream could echo a credential fragment into an error body and this must never persist one.

// forbiddenDetailMax bounds the stored, scrubbed 403 detail. The upstream body is already
// truncated to 200 bytes at the agent seam; this is the final ceiling on what /debug/vars keeps.
const forbiddenDetailMax = 240

// tokenShapedRe matches the credential shapes an upstream error body might echo: an Anthropic
// key/token prefix and its trailing chars (sk-ant-…, sk-…), a Bearer token, and any long
// unbroken high-entropy run (base64/hex-ish, 24+ chars) that could be a secret. Each match is
// replaced with a fixed redaction marker so the KIND of denial survives while the secret does not.
var tokenShapedRe = regexp.MustCompile(`(?i)(sk-[a-z0-9-]{8,}|bearer\s+[a-z0-9._-]{8,}|[a-z0-9+/_-]{24,})`)

// scrubForbiddenDetail returns a bounded, secret-free rendering of a 403 upstream body safe to
// store on the loopback-only /debug/vars surface. It collapses whitespace (a JSON body is noisy
// on one line), redacts every token-shaped run, and truncates to forbiddenDetailMax. An empty or
// whitespace-only body returns "" (nothing to show). It is conservative by construction: it can
// only ever REMOVE information, never add it, so a false positive costs a little context, never a
// leak.
func scrubForbiddenDetail(body string) string {
	s := strings.Join(strings.Fields(body), " ")
	if s == "" {
		return ""
	}
	s = tokenShapedRe.ReplaceAllString(s, "[redacted]")
	if len(s) > forbiddenDetailMax {
		s = s[:forbiddenDetailMax] + "…"
	}
	return s
}
