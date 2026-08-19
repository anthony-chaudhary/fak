package gateway

import (
	"bytes"
	"strings"
	"testing"

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
