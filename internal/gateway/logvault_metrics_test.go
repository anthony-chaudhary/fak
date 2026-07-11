package gateway

import (
	"strings"
	"testing"
)

// TestLogvaultMetricsProvider proves the /metrics seam for vault observability
// (#2455): a host-injected provider's pre-rendered fak_logvault_* text appears in
// the scrape, and detaching it (nil) removes it — the same pull contract the
// fak_harness_* family uses.
func TestLogvaultMetricsProvider(t *testing.T) {
	srv := newTestServer(t)

	// No provider: the family must be absent (a box without a vault emits nothing).
	if got := srv.renderMetrics(); strings.Contains(got, "fak_logvault_") {
		t.Fatalf("no provider should render no fak_logvault_* family, got:\n%s", got)
	}

	srv.SetLogvaultMetricsProvider(func() string {
		return "# HELP fak_logvault_verify_mismatches WITNESSED test\n# TYPE fak_logvault_verify_mismatches gauge\nfak_logvault_verify_mismatches 3\n"
	})
	got := srv.renderMetrics()
	if !strings.Contains(got, "fak_logvault_verify_mismatches 3") {
		t.Fatalf("provider text missing from scrape:\n%s", got)
	}

	// A provider that has nothing to report adds nothing rather than an empty block.
	srv.SetLogvaultMetricsProvider(func() string { return "" })
	if got := srv.renderMetrics(); strings.Contains(got, "fak_logvault_") {
		t.Fatalf("empty provider should add no family, got:\n%s", got)
	}

	// Detaching removes the seam entirely.
	srv.SetLogvaultMetricsProvider(nil)
	if got := srv.renderMetrics(); strings.Contains(got, "fak_logvault_") {
		t.Fatalf("nil provider should detach the family, got:\n%s", got)
	}
}
