package accountobs

import (
	"net/http"
	"testing"
	"time"
)

// TestUsageOverageRejection pins the shared cap-vs-wall header discriminator against the exact
// live day30 signal and the window-status shapes, so the one taxonomy both internal/agent and
// internal/resume consume stays correct.
func TestUsageOverageRejection(t *testing.T) {
	reset5h := time.Now().Add(90 * time.Minute).Unix()

	t.Run("no headers => not a rejection (a genuine wall carries none)", func(t *testing.T) {
		if got := UsageOverageRejection(nil); got.Rejected {
			t.Fatalf("nil headers must not be a rejection: %+v", got)
		}
		if got := UsageOverageRejection(http.Header{}); got.Rejected {
			t.Fatalf("empty headers must not be a rejection: %+v", got)
		}
	})

	t.Run("overage-status rejected (the live day30 signal) => rejected", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
		h.Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")
		h.Set("Anthropic-Ratelimit-Unified-Overage-Disabled-Reason", "org_level_disabled")
		got := UsageOverageRejection(h)
		if !got.Rejected {
			t.Fatalf("overage-status rejected must be a rejection: %+v", got)
		}
	})

	t.Run("overage rejected with a top-level reset => carries the reset", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Overage-Status", "rejected")
		h.Set("Anthropic-Ratelimit-Unified-Reset", itoa64(reset5h))
		got := UsageOverageRejection(h)
		if !got.Rejected || !got.HaveReset {
			t.Fatalf("overage rejection with a reset must carry HaveReset: %+v", got)
		}
	})

	t.Run("a 5h window status rejected => rejected, window 5h, with its reset", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-5h-Status", "rejected")
		h.Set("Anthropic-Ratelimit-Unified-5h-Reset", itoa64(reset5h))
		got := UsageOverageRejection(h)
		if !got.Rejected || got.Window != "5h" || !got.HaveReset {
			t.Fatalf("a rejected 5h window must yield window=5h with a reset: %+v", got)
		}
	})

	t.Run("all windows allowed => not a rejection", func(t *testing.T) {
		h := http.Header{}
		h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
		h.Set("Anthropic-Ratelimit-Unified-5h-Status", "allowed")
		h.Set("Anthropic-Ratelimit-Unified-7d-Utilization", "0.54")
		if got := UsageOverageRejection(h); got.Rejected {
			t.Fatalf("all-allowed windows must not be a rejection: %+v", got)
		}
	})
}

func itoa64(n int64) string {
	// small local helper so the test does not depend on strconv import churn
	neg := n < 0
	if neg {
		n = -n
	}
	if n == 0 {
		return "0"
	}
	var b [20]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}
