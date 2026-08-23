package gateway

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

func TestFreshMinimumPrefixCalibrationSteersAnchorWireDecision(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet","max_tokens":64,"system":[{"type":"text","text":"` + strings.Repeat("stable policy ", 80) + `"}],"messages":[{"role":"user","content":"hi"}]}`)
	decode := func(t *testing.T) *agent.AnthropicMessagesRequest {
		t.Helper()
		req, err := agent.DecodeAnthropicMessagesRequest(raw)
		if err != nil {
			t.Fatal(err)
		}
		return req
	}

	fallback := anthropicPassthroughServer(0)
	fallback.vcacheAnchor = true
	fallbackReq := decode(t)
	if !fallback.maybeAnchorAnthropicRaw(fallbackReq, "") || bytes.Equal(fallbackReq.Raw, raw) {
		t.Fatal("missing calibration must preserve the default-on anchor decision")
	}

	calibrated := anthropicPassthroughServer(0)
	calibrated.vcacheAnchor = true
	calibrated.vcacheCalibration = &VCacheRuntimeCalibration{
		Provider: "anthropic", Model: "claude-sonnet", Source: "probe:test",
		MinPrefixTokens: 10_000, MinPrefixMeasured: true,
	}
	calibratedReq := decode(t)
	if calibrated.maybeAnchorAnthropicRaw(calibratedReq, "") {
		t.Fatal("request below measured minimum prefix must not author cache_control")
	}
	if !bytes.Equal(calibratedReq.Raw, raw) {
		t.Fatal("calibration refusal must leave the provider request byte-identical")
	}
}

func TestUnmeasuredMinimumPrefixDoesNotSteerAnchor(t *testing.T) {
	cal := &VCacheRuntimeCalibration{MinPrefixTokens: 10_000, MinPrefixMeasured: false}
	if !cal.admitsAnchor("claude", 1) {
		t.Fatal("an assumed minimum prefix must not alter runtime behavior")
	}
}

func TestModelSpecificCalibrationDoesNotSteerAnotherModel(t *testing.T) {
	cal := &VCacheRuntimeCalibration{Model: "claude-sonnet", MinPrefixTokens: 10_000, MinPrefixMeasured: true}
	if !cal.admitsAnchor("claude-opus", 1) {
		t.Fatal("model-specific constants must not steer another wire model")
	}
}

func TestFreshReadMultiplierChangesRuntimePricingDecision(t *testing.T) {
	base := CachePricing{InputPerMTokUSD: 10}
	usage := CacheUsage{CacheReadTokens: 1_000_000}
	if got := base.CostUSD(usage); got != 1 {
		t.Fatalf("default cache read cost=%v, want 1", got)
	}
	cal := &VCacheRuntimeCalibration{ReadMult: 0.25, ReadMultMeasured: true}
	if got := cal.ApplyCachePricing(base).CostUSD(usage); got != 2.5 {
		t.Fatalf("calibrated cache read cost=%v, want 2.5", got)
	}
	assumed := &VCacheRuntimeCalibration{ReadMult: 0.25}
	if got := assumed.ApplyCachePricing(base).CostUSD(usage); got != 1 {
		t.Fatalf("unmeasured multiplier changed pricing: %v", got)
	}
}

func TestFreshMeasuredTTLSteersAnthropicTierDecision(t *testing.T) {
	raw := []byte(`{"model":"claude-sonnet","max_tokens":64,"system":[{"type":"text","text":"stable policy","cache_control":{"type":"ephemeral"}}],"messages":[{"role":"user","content":"hi"}]}`)
	decode := func(t *testing.T) *agent.AnthropicMessagesRequest {
		t.Helper()
		req, err := agent.DecodeAnthropicMessagesRequest(raw)
		if err != nil {
			t.Fatal(err)
		}
		return req
	}

	fallback := anthropicPassthroughServer(0)
	fallback.cacheTTL1H.Store(true)
	fallbackReq := decode(t)
	if !fallback.maybeUpgradeAnthropicCacheTTL1H(fallbackReq) || !bytes.Contains(fallbackReq.Raw, []byte(`"ttl":"1h"`)) {
		t.Fatalf("missing calibration must preserve the explicit 1h tier:\n%s", fallbackReq.Raw)
	}

	calibrated := anthropicPassthroughServer(0)
	calibrated.cacheTTL1H.Store(true)
	calibrated.vcacheCalibration = &VCacheRuntimeCalibration{
		Provider: "anthropic", Model: "claude-sonnet", Source: "probe:test",
		TTLMillis: int64((2 * time.Hour) / time.Millisecond), TTLMeasured: true,
	}
	calibratedReq := decode(t)
	if calibrated.maybeUpgradeAnthropicCacheTTL1H(calibratedReq) {
		t.Fatal("measured provider retention above one hour must suppress the paid 1h tier")
	}
	if !bytes.Equal(calibratedReq.Raw, raw) {
		t.Fatalf("5m default-tier request changed:\n%s", calibratedReq.Raw)
	}
}

func TestUntrustedTTLCalibrationPreservesStaticTierDecision(t *testing.T) {
	cases := []struct {
		name  string
		cal   *VCacheRuntimeCalibration
		model string
	}{
		{name: "missing", model: "claude-sonnet"},
		{name: "unmeasured", model: "claude-sonnet", cal: &VCacheRuntimeCalibration{TTLMillis: int64((2 * time.Hour) / time.Millisecond)}},
		{name: "invalid", model: "claude-sonnet", cal: &VCacheRuntimeCalibration{TTLMillis: 0, TTLMeasured: true}},
		{name: "model mismatch", model: "claude-opus", cal: &VCacheRuntimeCalibration{Model: "claude-sonnet", TTLMillis: int64((2 * time.Hour) / time.Millisecond), TTLMeasured: true}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.cal.wantsExplicitOneHourTTL(tc.model) {
				t.Fatal("untrusted calibration must preserve the static 1h decision")
			}
		})
	}
}

func TestMeasuredWriteMultipliersChangeServedSpendByTier(t *testing.T) {
	base := CachePricing{InputPerMTokUSD: 1_000_000}
	cal := &VCacheRuntimeCalibration{
		Write5mMult: 1.4, Write5mMeasured: true,
		Write1hMult: 2.2, Write1hMeasured: true,
	}
	pricing := cal.ApplyCachePricing(base)
	if got := pricing.CostUSD(CacheUsage{CacheCreationTokens: 10, WriteTTL: CacheTTL5m}); got != 14 {
		t.Fatalf("5m calibrated spend = %g, want 14", got)
	}
	if got := pricing.CostUSD(CacheUsage{CacheCreationTokens: 10, WriteTTL: CacheTTL1h}); got != 22 {
		t.Fatalf("1h calibrated spend = %g, want 22", got)
	}
	unmeasured := (&VCacheRuntimeCalibration{Write5mMult: 9, Write1hMult: 9}).ApplyCachePricing(base)
	if got := unmeasured.CostUSD(CacheUsage{CacheCreationTokens: 10, WriteTTL: CacheTTL5m}); got != 12.5 {
		t.Fatalf("unmeasured 5m spend = %g, want static 12.5", got)
	}
}
