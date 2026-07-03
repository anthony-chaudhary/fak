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
			// No rate-limit headers: the body-only classification path (a genuine org wall
			// carries no usage/overage headers, which is exactly why the header check is the
			// discriminator for the ambiguous org-disable body).
			if got := classifyUpstream(c.status, []byte(c.body), nil); got != c.want {
				t.Fatalf("classifyUpstream(%d, %q, nil) = %v, want %v", c.status, c.body, got, c.want)
			}
		})
	}
}

// TestClassifyUpstreamUsageVsOrgWall pins the load-bearing distinction the first shipped version
// got WRONG: the SAME org-disable 403 body means "fail over" only when there is NO usage/overage
// rejection in the headers. With an overage-rejected header (the live day30 signal: overage
// disabled, a rolling window at its cap, the account otherwise `allowed` and recovering at reset),
// the very same body is a USAGE CAP → Backoff (wait for the reset), never a failover to another
// account's bucket.
func TestClassifyUpstreamUsageVsOrgWall(t *testing.T) {
	orgBody := []byte(liveOrgOAuthDisabledBody)

	t.Run("org-disable body with NO rate-limit headers => failover (true org wall)", func(t *testing.T) {
		if got := classifyUpstream(http.StatusForbidden, orgBody, http.Header{}); got != RemedyFailoverAccount {
			t.Fatalf("bare org-disable 403 = %v, want failover", got)
		}
	})

	t.Run("org-disable body WITH overage-status rejected => backoff (usage cap, self-recovers)", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
		h.Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")
		h.Set("Anthropic-Ratelimit-Unified-Overage-Disabled-Reason", "org_level_disabled")
		h.Set("Anthropic-Ratelimit-Unified-7d-Utilization", "0.54")
		if got := classifyUpstream(http.StatusForbidden, orgBody, h); got != RemedyBackoff {
			t.Fatalf("overage-rejected org-disable 403 = %v, want backoff (must NOT fail over on a self-recovering usage cap)", got)
		}
	})

	t.Run("org-disable body WITH a unified window status rejected => backoff", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-5h-Status", "rejected")
		h.Set("Anthropic-Ratelimit-Unified-5h-Reset", "1783120200")
		if got := classifyUpstream(http.StatusForbidden, orgBody, h); got != RemedyBackoff {
			t.Fatalf("window-rejected org-disable 403 = %v, want backoff", got)
		}
	})

	t.Run("a genuinely allowed window on an org-disable body still fails over", func(t *testing.T) {
		// Headers present but NOT a rejection (status allowed, no overage-rejected): this is the
		// true org wall, not a cap — so failover remains correct.
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
		h.Set("Anthropic-Ratelimit-Unified-7d-Utilization", "0.10")
		if got := classifyUpstream(http.StatusForbidden, orgBody, h); got != RemedyFailoverAccount {
			t.Fatalf("allowed-window org-disable 403 = %v, want failover", got)
		}
	})
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
	// With NO rate-limit headers (a genuine org wall carries none), the live org body is a failover.
	if got := classifyUpstream(http.StatusForbidden, body, nil); got != RemedyFailoverAccount {
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
