package gateway

import (
	"testing"
	"time"
)

// Prefill tok/s divides the tokens actually prefilled by prefill time, so a
// cache-read prompt must not count its cached tokens in the numerator.
func TestPrefillNumeratorExcludesCacheHitTokens(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInferenceTimed(1000, 20, 800, 0, "stop", 3*time.Second, time.Second)
	if m.inferPrefillPromptTokens != 200 {
		t.Fatalf("prefill numerator = %d, want 200 uncached tokens", m.inferPrefillPromptTokens)
	}
}

func TestPrefillNumeratorKeepsCacheWriteTokens(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInferenceTimed(1000, 20, 0, 300, "stop", 3*time.Second, time.Second)
	if m.inferPrefillPromptTokens != 1000 {
		t.Fatalf("prefill numerator = %d, want 1000 (a cache write was prefilled)", m.inferPrefillPromptTokens)
	}
}

// output_tokens_per_second is deliberately blended: its denominator is the whole
// turn, prefill included. The decode-only rate must exclude the prefill seconds.
func TestDecodeRateExcludesPrefillSeconds(t *testing.T) {
	m := newGatewayMetrics(time.Now())
	m.observeInferenceTimed(1000, 100, 0, 0, "stop", 5*time.Second, 4*time.Second)
	v := inferenceVarsFromSnapshot(m.inferenceSnapshotData(), 0)
	if v.OutputTokensPerSecond != 20 {
		t.Fatalf("output_tokens_per_second = %v, want 20 (100 tokens over the 5 s blended turn)", v.OutputTokensPerSecond)
	}
	if v.DecodeTokensPerSecond != 100 {
		t.Fatalf("decode_tokens_per_second = %v, want 100 (100 tokens over the 1 s decode phase)", v.DecodeTokensPerSecond)
	}
}
