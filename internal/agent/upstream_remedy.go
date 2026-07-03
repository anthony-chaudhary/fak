package agent

// upstream_remedy.go — the single source of truth that maps an upstream failure
// (HTTP status + response body) to the ONE remedy that can actually fix it, so the
// caller never applies a DECEIVING remedy: a fix that pattern-matches the surface
// error but cannot address its real cause.
//
// WHY THIS EXISTS. Before this, the remedy for an upstream failure was implied by the
// HTTP status number alone (retryableStatus keyed retries; a 401 keyed the auth-refresh
// arm; every other 4xx surfaced terminally). The 403 family was the worst offender: a
// single body discriminator (forbiddenBodyIsPermanent) collapsed a whole population of
// distinct denials into a binary retry-or-give-up, and the give-up path surfaced a
// message that told the operator to "run /login" — futile for an ORG-SCOPED OAuth
// disable, where every re-login mints another token for the same walled org and the
// upstream denies all of them. The credential is valid; the ORGANIZATION is walled. The
// only remedy that works is failing over to a DIFFERENT account whose org still permits
// the auth path.
//
// classifyUpstream draws the distinctions the status number cannot: which 403s are a
// transient abuse-gate (back off), which are an org/region/credit wall (fail over to a
// permitted sibling account), which are a model-entitlement refusal (switch model), and
// which are genuinely terminal. It is PURE (status + body in, remedy out) so it is
// trivially unit-testable against the exact bodies observed on the wire, and it reuses
// the existing predicates (retryableStatus, forbiddenBodyIsPermanent) rather than
// re-deriving them, so there is one taxonomy, not two that can drift.

import (
	"net/http"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/accountobs"
)

// UpstreamRemedy is the closed set of automated responses to an upstream failure. Exactly
// one applies to any given (status, body); the caller dispatches on it instead of
// re-inspecting the raw status/body at each call site.
type UpstreamRemedy int

const (
	// RemedyTerminal: no automated fix — surface the classified error with actionable
	// guidance (a malformed request, an unknown model, a static bad key, a bare
	// entitlement refusal with no failover target). The default for anything unrecognized.
	RemedyTerminal UpstreamRemedy = iota
	// RemedyRefreshToken: the credential expired but a fresh one is (or will shortly be)
	// on disk — re-read it and re-send. The 401 rotating-subscription self-heal.
	RemedyRefreshToken
	// RemedyBackoff: a transient overload/rate-limit/abuse-gate that clears on its own —
	// wait and retry (429/5xx/529, and an unlabeled 403 that may be a capacity flap).
	RemedyBackoff
	// RemedyFailoverAccount: the credential is valid but its ORG/region/billing is walled —
	// no retry or re-login on THIS account can help; switch to a different account whose org
	// still permits the request. The org-OAuth-disabled 403 is the canonical case.
	RemedyFailoverAccount
	// RemedySwitchModel: the account is fine but not entitled to THIS model/feature — the
	// same credential on a permitted model succeeds. (Detected here; execution is a
	// follow-on — until then a SwitchModel with no switch target degrades to Terminal.)
	RemedySwitchModel
)

// String renders the remedy as a short stable label for logs, notify hooks, and metrics.
// It is deliberately NOT the raw upstream body — the label crosses trust boundaries the
// body must not (see http.go's fixed-literal message invariant).
func (r UpstreamRemedy) String() string {
	switch r {
	case RemedyRefreshToken:
		return "refresh_token"
	case RemedyBackoff:
		return "backoff"
	case RemedyFailoverAccount:
		return "failover_account"
	case RemedySwitchModel:
		return "switch_model"
	default:
		return "terminal"
	}
}

// classifyUpstream maps an upstream (status, body, headers) to the single remedy that can
// address it. The order matters: the transient/overload family is decided by status first (a
// 429 is always a backoff regardless of body); a 401 is always the token-refresh arm (the
// existing 401 self-heal stays authoritative); and only the 403/402 family reads the body AND
// the headers, because that is the family whose remedy genuinely depends on WHICH denial it is.
//
// THE HEADERS ARE THE HONEST DISCRIMINATOR, not the body. A Claude subscription 403 body can say
// "OAuth authentication is currently not allowed for this organization" for TWO very different
// conditions: (a) a genuine standing org-OAuth-disable (the credential cannot serve at all —
// failover to a permitted account is right), OR (b) a USAGE/OVERAGE rejection where the org has
// overage disabled and THIS request would tip a rolling window past its cap (the credential is
// otherwise fine — `unified-status: allowed`, sub-100% utilization — and RECOVERS on its own at
// the 5h/7d reset). The body text is identical; only the anthropic-ratelimit-unified-* /
// -overage-* headers tell them apart. Failing over on (b) is WRONG — it burns a second account's
// bucket for a condition the original account clears at reset — so a 403 carrying an active
// usage/overage rejection is routed to Backoff (the cap-aware wait toward the named reset, via
// the classifyLimit429/unifiedResetFor machinery), NOT FailoverAccount. Only a 403 whose org body
// carries NO usage/overage rejection is treated as a true org wall and failed over.
//
// Region/credit walls and model-entitlement refusals keep their body-only classification (they
// carry no rolling-reset semantics). An unlabeled 403 stays a possibly-transient abuse gate
// (Backoff), matching forbiddenRetryState's own transient arm.
func classifyUpstream(status int, body []byte, h http.Header) UpstreamRemedy {
	if retryableStatus(status) {
		return RemedyBackoff
	}
	switch status {
	case http.StatusUnauthorized: // 401 — the rotating-credential refresh arm owns this.
		return RemedyRefreshToken
	case http.StatusForbidden, http.StatusPaymentRequired: // 403, 402
		switch {
		case usageOrOverageRejected(h):
			// A usage/overage rejection surfaced as a 403 (org overage disabled, a rolling
			// window at its cap). The credential is fine and recovers at the named reset — wait
			// for it, never fail over to a different account's bucket. This MUST precede the
			// org-disable check: the same body text covers both, and the header is the only
			// signal that separates a self-recovering cap from a standing org wall.
			return RemedyBackoff
		case orgOAuthDisabled(body), regionWalled(body), creditExhausted(body):
			// Account-scoped wall with NO usage/overage signal: this credential's org/region/
			// billing is genuinely denied, but a sibling account on a permitted org is not.
			return RemedyFailoverAccount
		case modelNotEntitled(body):
			// Model-scoped: the account is fine, the model is not on it. Switch model.
			return RemedySwitchModel
		case forbiddenBodyIsPermanent(body):
			// A permanent denial we recognize as permanent but cannot map to a failover
			// target — surface it terminally with the actionable message.
			return RemedyTerminal
		default:
			// An unlabeled 403 may be a server-side abuse/capacity flap (the gem8 storm) —
			// exactly as recoverable as a 529. Let the bounded transient arm ride it out.
			return RemedyBackoff
		}
	default:
		return RemedyTerminal
	}
}

// usageOrOverageRejected reports whether a 403/402 response's rate-limit headers show a
// USAGE/OVERAGE rejection — a rolling-window cap the account recovers from at its named reset,
// NOT a standing wall. It is the honest discriminator that separates the two conditions the
// org-disable body text conflates. It is true when the provider relayed either:
//   - anthropic-ratelimit-unified-overage-status: rejected  (overage disabled and this request
//     would exceed the cap — the exact header the live day30 probe showed), or
//   - any unified window (top-level or 5h/7d) with status "rejected" (the window itself is over
//     its limit right now).
//
// A response with no such header (a genuine org OAuth-disable carries none) returns false, so the
// caller treats it as a real account wall. Reusing the accountobs parser keeps ONE header
// taxonomy: the same leaf that feeds the cap-aware wait (unifiedResetFor) decides this.
func usageOrOverageRejected(h http.Header) bool {
	if len(h) == 0 {
		return false
	}
	// The overage-status header is not a per-window field the Unified() parser surfaces, so read
	// it directly (case-insensitively, as http.Header canonicalizes keys). "rejected" means the
	// account hit its cap with overage disabled — a self-recovering usage boundary.
	if strings.EqualFold(strings.TrimSpace(h.Get("Anthropic-Ratelimit-Unified-Overage-Status")), "rejected") {
		return true
	}
	// Any unified window reporting status "rejected" (top-level, 5h, or 7d) is likewise a
	// usage-cap rejection with a reset, not a permission wall.
	t := accountobs.New()
	t.Observe(http.StatusForbidden, h)
	for _, w := range t.Snapshot().Unified() {
		if strings.EqualFold(strings.TrimSpace(w.Status), "rejected") {
			return true
		}
	}
	return false
}

// bodyContainsAny reports whether the lowercased body contains any of the signatures. The
// match is lowercase-substring over the (already truncated) body, so it is robust to the
// surrounding JSON envelope and to case differences in the provider's prose.
func bodyContainsAny(body []byte, sigs ...string) bool {
	b := strings.ToLower(string(body))
	for _, s := range sigs {
		if strings.Contains(b, s) {
			return true
		}
	}
	return false
}

// orgOAuthDisabled reports whether a 403 body carries the ORGANIZATION-scoped OAuth-disable
// signature — the canonical deceiving error. The credential authenticates fine; the org it
// belongs to has subscription/OAuth inference turned off upstream, so no re-login on this
// account can clear it (a fresh token is minted for the SAME walled org). The witnessed
// live body is `"OAuth authentication is currently not allowed for this organization."`;
// the two substrings below match it and the closely-related "subscription access disabled"
// phrasing without over-matching an unrelated permission error.
func orgOAuthDisabled(body []byte) bool {
	return bodyContainsAny(body,
		"not allowed for this organization",     // the exact witnessed live body
		"oauth authentication is currently not", // the same denial, prefix-anchored
		"organization has disabled",             // the sibling "subscription access disabled" phrasing
	)
}

// IsOrgOAuthDisabled is the exported witness for the canonical deceiving 403 — the
// organization-scoped OAuth/subscription disable. The gateway's client-facing error message
// uses it to tell this specific denial apart from a generic entitlement 403, so it can name the
// REAL cause (re-login is futile; the org is walled) instead of the misleading "run /login". It
// reads the same signature as the internal classifier, so there is one taxonomy, not two.
func IsOrgOAuthDisabled(body []byte) bool { return orgOAuthDisabled(body) }

// regionWalled reports whether a 403 body names a GEOGRAPHIC entitlement wall — a retry
// never clears it, and a re-login on the same account lands in the same region, so the fix
// is a sibling account permitted in a supported region.
func regionWalled(body []byte) bool {
	return bodyContainsAny(body, "unsupported_region", "region is not")
}

// creditExhausted reports whether a 403/402 body names a billing wall (the account is out
// of credit). The same request on a funded sibling account succeeds, so it is a failover
// case rather than a terminal one.
func creditExhausted(body []byte) bool {
	return bodyContainsAny(body, "credit balance is too low", "insufficient_quota")
}

// modelNotEntitled reports whether a 403 body names a MODEL/feature entitlement refusal —
// the account is fine, this specific model is not on its plan. The same credential on a
// permitted model succeeds, so the remedy is a model switch, not an account failover.
func modelNotEntitled(body []byte) bool {
	return bodyContainsAny(body,
		"not have access",      // "...does not have access to model..."
		"does not have access", // same, explicit
		"not entitled",         // entitlement refusal
		"not allowed to use",   // model/feature not on the plan
	)
}
