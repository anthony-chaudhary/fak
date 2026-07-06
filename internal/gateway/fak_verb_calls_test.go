package gateway

import (
	"strings"
	"testing"
	"time"
)

// TestFakVerbCallsCounterRenders asserts the #3093 unused-substrate producer: the
// fak_mcp_verb_calls_total counter starts at 0, increments per observed fak-verb call, and
// renders on the guard-family scrape the Stop-hook polls. Uses a bare gatewayMetrics (no
// Server / engine registry) so it is independent of package test order.
func TestFakVerbCallsCounterRenders(t *testing.T) {
	m := newGatewayMetrics(time.Unix(0, 0))

	// Fresh metrics: the counter is present and 0 (so a clean stop can tell "0 verbs used"
	// apart from "metric absent").
	var b0 strings.Builder
	m.writeDenyAllMetrics(&b0)
	if !strings.Contains(b0.String(), "fak_mcp_verb_calls_total 0") {
		t.Fatalf("fresh metrics did not show fak_mcp_verb_calls_total 0:\n%s", b0.String())
	}

	// Three admitted verb calls.
	m.observeFakVerbCall()
	m.observeFakVerbCall()
	m.observeFakVerbCall()

	var b3 strings.Builder
	m.writeDenyAllMetrics(&b3)
	if !strings.Contains(b3.String(), "fak_mcp_verb_calls_total 3") {
		t.Fatalf("after 3 calls, metrics did not show fak_mcp_verb_calls_total 3:\n%s", b3.String())
	}
}

// TestFakVerbCallsNilSafe asserts the observer + snapshot are nil-safe (a nil metrics
// receiver must never panic — the same posture the other gatewayMetrics observers take).
func TestFakVerbCallsNilSafe(t *testing.T) {
	var m *gatewayMetrics
	m.observeFakVerbCall() // must not panic
	if got := m.fakVerbCallsSnapshot(); got != 0 {
		t.Fatalf("nil snapshot = %d, want 0", got)
	}
}
