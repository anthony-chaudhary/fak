package agent

import (
	"net/http"
	"testing"
)

// The exact 403 body captured from the live gateway's /debug/vars last_forbidden_detail
// when day30@'s org had OAuth disabled — the ground-truth string the classifier must map
// to RemedyFailoverAccount, and the canonical "deceiving error" (its message reads as if a
// re-login would help, but re-login is futile for an org-scoped wall).
const liveOrgOAuthDisabledBody = `{"type":"error","error":{"type":"permission_error",` +
	`"message":"OAuth authentication is currently not allowed for this organization."},"request_id":"[redacted]"}`

func TestClassifyUpstream(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		want   UpstreamRemedy
	}{
		{"org-oauth-disabled (live body)", http.StatusForbidden, liveOrgOAuthDisabledBody, RemedyFailoverAccount},
		{"org-subscription-disabled", http.StatusForbidden,
			`{"error":{"message":"Your organization has disabled Claude subscription access"}}`, RemedyFailoverAccount},
		{"region walled", http.StatusForbidden,
			`{"error":{"type":"unsupported_region","message":"this region is not supported"}}`, RemedyFailoverAccount},
		{"credit exhausted (403)", http.StatusForbidden,
			`{"error":{"message":"Your credit balance is too low"}}`, RemedyFailoverAccount},
		{"credit exhausted (402)", http.StatusPaymentRequired,
			`{"error":{"message":"credit balance is too low"}}`, RemedyFailoverAccount},
		{"model not entitled", http.StatusForbidden,
			`{"error":{"message":"your account does not have access to model X"}}`, RemedySwitchModel},
		{"not entitled to use", http.StatusForbidden,
			`{"error":{"message":"you are not allowed to use this model"}}`, RemedySwitchModel},
		{"bare permission_error (no failover target)", http.StatusForbidden,
			`{"error":{"type":"permission_error","message":"forbidden"}}`, RemedyTerminal},
		{"transient/unlabeled 403 (abuse gate)", http.StatusForbidden,
			`{"error":{"message":"request temporarily blocked"}}`, RemedyBackoff},
		{"401 refresh", http.StatusUnauthorized, `{"error":{"message":"authentication_error"}}`, RemedyRefreshToken},
		{"429 backoff", http.StatusTooManyRequests, `{"error":{"message":"rate_limit_error"}}`, RemedyBackoff},
		{"529 overloaded backoff", statusOverloaded, `{"error":{"message":"overloaded"}}`, RemedyBackoff},
		{"503 backoff", http.StatusServiceUnavailable, ``, RemedyBackoff},
		{"404 terminal", http.StatusNotFound, `{"error":{"message":"model not found"}}`, RemedyTerminal},
		{"400 terminal", http.StatusBadRequest, `{"error":{"message":"invalid_request_error"}}`, RemedyTerminal},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := classifyUpstream(c.status, []byte(c.body)); got != c.want {
				t.Fatalf("classifyUpstream(%d, %q) = %v, want %v", c.status, c.body, got, c.want)
			}
		})
	}
}

// The org-OAuth-disabled body is classified PERMANENT by the pre-existing
// forbiddenBodyIsPermanent (via "permission_error"), which is why the bounded transient-403
// arm correctly wastes no retries on it — but that permanence must now route to a FAILOVER,
// not a terminal give-up. This pins both facts so a future edit to either predicate that
// broke the pairing would fail here.
func TestOrgOAuthDisabled_IsPermanentAndFailover(t *testing.T) {
	body := []byte(liveOrgOAuthDisabledBody)
	if !orgOAuthDisabled(body) {
		t.Fatal("orgOAuthDisabled must match the live captured body")
	}
	if !forbiddenBodyIsPermanent(body) {
		t.Fatal("forbiddenBodyIsPermanent must still see the live body as permanent (no wasted transient retries)")
	}
	if got := classifyUpstream(http.StatusForbidden, body); got != RemedyFailoverAccount {
		t.Fatalf("the live org-disabled body must classify as failover, got %v", got)
	}
}

func TestUpstreamRemedy_String(t *testing.T) {
	want := map[UpstreamRemedy]string{
		RemedyTerminal:        "terminal",
		RemedyRefreshToken:    "refresh_token",
		RemedyBackoff:         "backoff",
		RemedyFailoverAccount: "failover_account",
		RemedySwitchModel:     "switch_model",
	}
	for r, s := range want {
		if r.String() != s {
			t.Errorf("%d.String() = %q, want %q", r, r.String(), s)
		}
	}
}
