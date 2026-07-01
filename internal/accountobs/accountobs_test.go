package accountobs

import (
	"fmt"
	"net/http"
	"strings"
	"testing"
	"time"
)

func TestObserveCapturesOnlyRateLimitFamilies(t *testing.T) {
	tr := New()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
	h.Set("X-Ratelimit-Requests-Remaining", "42")
	h.Set("Authorization", "Bearer sk-ant-secret") // must never be captured
	h.Set("Content-Type", "application/json")
	tr.Observe(200, h)

	s := tr.Snapshot()
	if s.Responses != 1 || s.LastStatus != 200 {
		t.Fatalf("responses/status = %d/%d, want 1/200", s.Responses, s.LastStatus)
	}
	if got := s.Headers["anthropic-ratelimit-unified-status"]; got != "allowed" {
		t.Fatalf("unified-status = %q, want allowed", got)
	}
	if got := s.Headers["x-ratelimit-requests-remaining"]; got != "42" {
		t.Fatalf("x-ratelimit remaining = %q, want 42", got)
	}
	for k := range s.Headers {
		if strings.Contains(k, "authorization") || strings.Contains(k, "content-type") {
			t.Fatalf("captured non-ratelimit header %q", k)
		}
	}
}

func TestObserveLatestValueWinsAndCounts429(t *testing.T) {
	tr := New()
	h1 := http.Header{}
	h1.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.10")
	tr.Observe(200, h1)
	h2 := http.Header{}
	h2.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.34")
	tr.Observe(429, h2)

	s := tr.Snapshot()
	if s.Responses != 2 || s.RateLimited != 1 {
		t.Fatalf("responses/rateLimited = %d/%d, want 2/1", s.Responses, s.RateLimited)
	}
	ws := s.Unified()
	if len(ws) != 1 || ws[0].Name != "5h" || !ws[0].HaveUtilization {
		t.Fatalf("unified windows = %+v", ws)
	}
	if ws[0].UtilizationPct != 34 {
		t.Fatalf("5h utilization = %v, want 34 (latest wins, fraction→pct)", ws[0].UtilizationPct)
	}
}

func TestUnifiedParsesWindowsStatusAndReset(t *testing.T) {
	reset := time.Date(2026, 7, 1, 17, 0, 0, 0, time.UTC)
	tr := New()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-Status", "allowed_warning")
	h.Set("Anthropic-Ratelimit-Unified-Reset", fmt.Sprint(reset.Unix()))
	h.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "34") // already-percent spelling
	h.Set("Anthropic-Ratelimit-Unified-5h-Status", "allowed")
	h.Set("Anthropic-Ratelimit-Unified-7d-Utilization", "0.12") // fraction spelling
	tr.Observe(200, h)

	ws := tr.Snapshot().Unified()
	if len(ws) != 3 {
		t.Fatalf("want 3 windows (top-level, 5h, 7d), got %+v", ws)
	}
	top, w5h, w7d := ws[0], ws[1], ws[2]
	if top.Name != "" || top.Status != "allowed_warning" || !top.HaveReset || !top.Reset.Equal(reset) {
		t.Fatalf("top-level window = %+v", top)
	}
	if w5h.Name != "5h" || w5h.UtilizationPct != 34 || w5h.Status != "allowed" {
		t.Fatalf("5h window = %+v", w5h)
	}
	if w7d.Name != "7d" || w7d.UtilizationPct != 12 {
		t.Fatalf("7d window = %+v", w7d)
	}
}

func TestFamiliesParsesAPIKeyTriples(t *testing.T) {
	tr := New()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Requests-Limit", "5000")
	h.Set("Anthropic-Ratelimit-Requests-Remaining", "4990")
	h.Set("Anthropic-Ratelimit-Requests-Reset", "2026-07-01T17:00:00Z")
	h.Set("Anthropic-Ratelimit-Input-Tokens-Remaining", "399000")
	// Unified headers must not leak into the family view.
	h.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.34")
	tr.Observe(200, h)

	fs := tr.Snapshot().Families()
	if len(fs) != 2 {
		t.Fatalf("want 2 families, got %+v", fs)
	}
	inTok, req := fs[0], fs[1] // sorted: input-tokens < requests
	if req.Name != "requests" || req.Limit != 5000 || req.Remaining != 4990 || !req.HaveReset {
		t.Fatalf("requests family = %+v", req)
	}
	if want := time.Date(2026, 7, 1, 17, 0, 0, 0, time.UTC); !req.Reset.Equal(want) {
		t.Fatalf("requests reset = %v, want %v", req.Reset, want)
	}
	if inTok.Name != "input-tokens" || !inTok.HaveRemaining || inTok.Remaining != 399000 || inTok.HaveLimit {
		t.Fatalf("input-tokens family = %+v", inTok)
	}
}

func TestReportEmptyWithoutResponses(t *testing.T) {
	if got := New().Snapshot().Report(time.Now()); got != "" {
		t.Fatalf("zero-response report = %q, want empty", got)
	}
}

func TestReportRendersWindowsWithProvenanceLabel(t *testing.T) {
	now := time.Date(2026, 7, 1, 16, 18, 0, 0, time.UTC)
	tr := New()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-Status", "allowed")
	h.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.34")
	h.Set("Anthropic-Ratelimit-Unified-5h-Reset", fmt.Sprint(now.Add(42*time.Minute).Unix()))
	h.Set("Anthropic-Ratelimit-Unified-7d-Utilization", "0.12")
	tr.Observe(200, h)

	got := tr.Snapshot().Report(now)
	for _, want := range []string{
		"OBSERVED provider-relayed", // the conflation-score provenance label
		"5h window 34% used",
		"in 42m",
		"7d window 12% used",
		"status allowed",
		"1 upstream response(s)",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("report %q missing %q", got, want)
		}
	}
	if strings.Contains(got, "429") {
		t.Fatalf("report %q mentions 429 with none observed", got)
	}
}

func TestReportHonestWhenNoHeadersRelayed(t *testing.T) {
	tr := New()
	tr.Observe(200, http.Header{})
	got := tr.Snapshot().Report(time.Now())
	if !strings.Contains(got, "provider relayed no rate-limit headers") {
		t.Fatalf("report = %q, want honest no-headers note", got)
	}
	if strings.Contains(got, "OBSERVED") {
		t.Fatalf("report %q labels values it does not have", got)
	}
}

func TestPrometheusTextEmitsOnlyPresentAxes(t *testing.T) {
	tr := New()
	h := http.Header{}
	h.Set("Anthropic-Ratelimit-Unified-5h-Utilization", "0.34")
	tr.Observe(429, h)

	got := tr.Snapshot().PrometheusText()
	for _, want := range []string{
		`fak_account_ratelimit_utilization_pct{window="5h"} 34`,
		"fak_account_upstream_responses_total 1",
		"fak_account_rate_limited_responses_total 1",
		"OBSERVED provider-relayed", // HELP provenance label
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("metrics %q missing %q", got, want)
		}
	}
	for _, absent := range []string{"fak_account_ratelimit_remaining", "fak_account_ratelimit_limit", "fak_account_ratelimit_reset_unix_seconds"} {
		if strings.Contains(got, absent) {
			t.Fatalf("metrics emit absent axis %q:\n%s", absent, got)
		}
	}
	if New().Snapshot().PrometheusText() != "" {
		t.Fatal("zero-response snapshot must render no metrics")
	}
}

func TestNilTrackerIsSafe(t *testing.T) {
	var tr *Tracker
	tr.Observe(200, http.Header{}) // must not panic
	s := tr.Snapshot()
	if s.Responses != 0 || s.Report(time.Now()) != "" {
		t.Fatalf("nil tracker snapshot = %+v", s)
	}
}
