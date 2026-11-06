package gateway

import (
	"strings"
	"testing"
)

// TestWriteHarnessMetricsNoFabricatedToolCallSeries pins T1 of the
// step-level telemetry map: the gateway must never publish the fabricated
// fak_harness_tool_* constants. Real host samples arrive only through
// SetHarnessMetricsProvider.
func TestWriteHarnessMetricsNoFabricatedToolCallSeries(t *testing.T) {
	var m gatewayMetrics
	var b strings.Builder
	m.writeHarnessMetrics(&b)
	if strings.Contains(b.String(), "fak_harness_tool_") {
		t.Fatalf("fabricated fak_harness_tool_* series present in /metrics render:\n%s", b.String())
	}
}

// TestWriteHarnessMetricsNoProviderRendersNothing pins the absence side of the
// same contract byte-for-byte: with no guarded session attached (no provider
// installed), writeHarnessMetrics contributes nothing to /metrics -- absence,
// not zero-valued series.
func TestWriteHarnessMetricsNoProviderRendersNothing(t *testing.T) {
	var m gatewayMetrics
	var b strings.Builder
	m.writeHarnessMetrics(&b)
	if got := b.String(); got != "" {
		t.Fatalf("writeHarnessMetrics with no provider must render nothing, got %q", got)
	}
}
