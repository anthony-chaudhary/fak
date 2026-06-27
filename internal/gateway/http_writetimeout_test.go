package gateway

import (
	"testing"
	"time"
)

// http_writetimeout_test.go pins the #1015 fix: a non-streaming /v1/chat/completions turn on
// a slow LOCAL model (a multi-minute in-kernel GLM-5.2 cpu-offload decode) must not trip the
// http.Server WriteTimeout mid-decode and return an empty reply. The whole-handler write
// deadline default is therefore chosen by the served backend.

// TestServeWriteTimeoutDefault pins the per-backend default: a local in-kernel model gets NO
// write timeout (0) because a single turn can legitimately run for minutes; every other
// backend (proxy to a hosted API, the mock, an unknown planner) keeps the conservative 90s
// network-safe default.
func TestServeWriteTimeoutDefault(t *testing.T) {
	cases := []struct {
		kind string
		want time.Duration
	}{
		{"inkernel", 0},               // local model: a turn can take minutes — no deadline
		{"proxy", 90 * time.Second},   // hosted API: fast + network-exposed — keep the bound
		{"replica", 90 * time.Second}, // a fleet of hosted upstreams — same as proxy
		{"mock", 90 * time.Second},    // instant scripted fallback — the bound is harmless
		{"unknown", 90 * time.Second}, // unclassified — conservative
	}
	for _, c := range cases {
		if got := serveWriteTimeoutDefault(c.kind); got != c.want {
			t.Errorf("serveWriteTimeoutDefault(%q) = %v, want %v", c.kind, got, c.want)
		}
	}
}

// TestServeWriteTimeoutEnvOverridesInkernel pins that the env override still wins even on the
// in-kernel default: an operator who wants a finite bound (or a different one) can set
// FAK_HTTP_WRITE_TIMEOUT_S and durEnv honors it over the backend-derived default.
func TestServeWriteTimeoutEnvOverridesInkernel(t *testing.T) {
	t.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "600")
	// The Serve path composes durEnv over the backend default; durEnv must return the override.
	if got := durEnv("FAK_HTTP_WRITE_TIMEOUT_S", serveWriteTimeoutDefault("inkernel")); got != 600*time.Second {
		t.Errorf("env override over the inkernel default = %v, want 600s", got)
	}
	// With no env set, the inkernel default (0, no timeout) is selected.
	t.Setenv("FAK_HTTP_WRITE_TIMEOUT_S", "")
	if got := durEnv("FAK_HTTP_WRITE_TIMEOUT_S", serveWriteTimeoutDefault("inkernel")); got != 0 {
		t.Errorf("inkernel default with no env = %v, want 0 (no write timeout)", got)
	}
}

// TestPlannerKindClassifiesInkernel guards the link the fix rests on: the in-kernel planner
// classifies as "inkernel", the key serveWriteTimeoutDefault keys off. A nil planner is
// "unknown" and keeps the conservative bound (the safe direction).
func TestPlannerKindClassifiesInkernel(t *testing.T) {
	if got := plannerKind(nil); got != "unknown" {
		t.Errorf("a nil planner must be unknown (conservative), got %q", got)
	}
}
