package resume

import (
	"net/http"
	"testing"
)

func TestClassifyLimitTextBareFableLimit(t *testing.T) {
	reason, ok := ClassifyLimitText("You've reached your Fable 5 limit. Run /usage-credits to continue.")
	if !ok || reason != LimitUsage {
		t.Fatalf("ClassifyLimitText = (%q,%v), want (%q,true)", reason, ok, LimitUsage)
	}
}

// TestClassifyLimitResponseWithHeader pins the header-aware path that catches a usage/overage cap
// arriving as a 403 — the population the 429-only body classifier misses. The org-flavored 403
// body is unclassifiable by text; only the overage/unified-window headers reveal it is a
// self-recovering cap, and which window (session vs weekly).
func TestClassifyLimitResponseWithHeader(t *testing.T) {
	orgBody := []byte(`{"type":"error","error":{"type":"permission_error","message":"OAuth authentication is currently not allowed for this organization."}}`)

	t.Run("403 org body + overage-status rejected => usage_limit (was invisible to the 429-only path)", func(t *testing.T) {
		// The body-only classifier sees nothing (403, and no usage text).
		if _, ok := ClassifyLimitResponse(http.StatusForbidden, orgBody); ok {
			t.Fatal("the body-only ClassifyLimitResponse must NOT classify a 403 org body as a limit")
		}
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")
		cls, ok := ClassifyLimitResponseWithHeader(http.StatusForbidden, orgBody, h)
		if !ok || cls.Reason != LimitUsage {
			t.Fatalf("403 overage cap = (%q,%v), want (%q,true)", cls.Reason, ok, LimitUsage)
		}
	})

	t.Run("403 with a rejected 5h window => session_limit", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-5h-Status", "rejected")
		cls, ok := ClassifyLimitResponseWithHeader(http.StatusForbidden, orgBody, h)
		if !ok || cls.Reason != LimitSession {
			t.Fatalf("5h-window cap = (%q,%v), want (%q,true)", cls.Reason, ok, LimitSession)
		}
	})

	t.Run("403 with a rejected 7d window => weekly_limit", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-7d-Status", "rejected")
		cls, ok := ClassifyLimitResponseWithHeader(http.StatusForbidden, orgBody, h)
		if !ok || cls.Reason != LimitWeekly {
			t.Fatalf("7d-window cap = (%q,%v), want (%q,true)", cls.Reason, ok, LimitWeekly)
		}
	})

	t.Run("403 org body with NO overage header => not a limit (a genuine wall)", func(t *testing.T) {
		if _, ok := ClassifyLimitResponseWithHeader(http.StatusForbidden, orgBody, http.Header{}); ok {
			t.Fatal("a 403 org body with no overage header must NOT be classified a limit")
		}
	})

	t.Run("429 with no overage header falls back to the body path unchanged", func(t *testing.T) {
		body := []byte(`{"error":{"message":"rate_limit_error"}}`)
		cls, ok := ClassifyLimitResponseWithHeader(http.StatusTooManyRequests, body, http.Header{})
		if !ok || cls.Reason != LimitRate {
			t.Fatalf("429 fallback = (%q,%v), want (%q,true)", cls.Reason, ok, LimitRate)
		}
	})
}
