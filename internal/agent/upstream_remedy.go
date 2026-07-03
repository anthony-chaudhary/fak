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

// classifyUpstream maps an upstream (status, body) to the single remedy that can address
// it. The order matters: the transient/overload family is decided by status first (a 429
// is always a backoff regardless of body); a 401 is always the token-refresh arm (the
// existing 401 self-heal stays authoritative); and only the 403 family reads the body,
// because that is the family whose remedy genuinely depends on WHICH denial it is. A body
// signature that names an account-scoped wall (org/region/credit) yields FailoverAccount;
// one that names a model-scoped entitlement yields SwitchModel; a permanent-but-unclassified
// denial is Terminal; anything else on a 403 (no permanent signature) is a possibly-transient
// abuse gate and gets Backoff, matching forbiddenRetryState's own transient arm.
func classifyUpstream(status int, body []byte) UpstreamRemedy {
	if retryableStatus(status) {
		return RemedyBackoff
	}
	switch status {
	case http.StatusUnauthorized: // 401 — the rotating-credential refresh arm owns this.
		return RemedyRefreshToken
	case http.StatusForbidden, http.StatusPaymentRequired: // 403, 402
		switch {
		case orgOAuthDisabled(body), regionWalled(body), creditExhausted(body):
			// Account-scoped wall: this credential's org/region/billing is denied, but a
			// sibling account on a permitted org/region/plan is not. Fail over.
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
