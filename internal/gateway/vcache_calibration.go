package gateway

import "strings"

// VCacheRuntimeCalibration is the fresh measured subset wired into live request
// decisions. Minimum-prefix evidence can suppress uneconomic cache-control
// authoring, measured retention can avoid an unnecessary paid 1h Anthropic tier,
// and measured read pricing can replace the static accounting multiplier.
type VCacheRuntimeCalibration struct {
	Provider          string
	Model             string
	Source            string
	TS                string
	TTLMillis         int64
	TTLMeasured       bool
	MinPrefixTokens   int64
	MinPrefixMeasured bool
	ReadMult          float64
	ReadMultMeasured  bool
}

func cloneVCacheRuntimeCalibration(in *VCacheRuntimeCalibration) *VCacheRuntimeCalibration {
	if in == nil {
		return nil
	}
	out := *in
	return &out
}

func (c *VCacheRuntimeCalibration) matchesModel(model string) bool {
	return c == nil || strings.TrimSpace(c.Model) == "" || strings.EqualFold(strings.TrimSpace(c.Model), strings.TrimSpace(model))
}

func (c *VCacheRuntimeCalibration) admitsAnchor(model string, estimatedTokens int) bool {
	if !c.matchesModel(model) {
		return true
	}
	if c == nil || !c.MinPrefixMeasured || c.MinPrefixTokens <= 0 {
		return true
	}
	return int64(estimatedTokens) >= c.MinPrefixTokens
}

// wantsExplicitOneHourTTL decides whether Anthropic needs FAK's paid 1h tier.
// A measured provider retention at or above one hour already satisfies that
// operating target, so authoring an explicit 1h TTL would only increase writes.
// Missing, invalid, unmeasured, or model-mismatched evidence preserves the
// existing managed-cache default.
func (c *VCacheRuntimeCalibration) wantsExplicitOneHourTTL(model string) bool {
	if !c.matchesModel(model) || c == nil || !c.TTLMeasured || c.TTLMillis <= 0 {
		return true
	}
	return c.TTLMillis < 60*60*1000
}

// ApplyCachePricing overlays only fresh measured pricing fields. Static model
// prices and write multipliers remain unchanged when calibration lacks evidence.
func (c *VCacheRuntimeCalibration) ApplyCachePricing(p CachePricing) CachePricing {
	if c != nil && c.ReadMultMeasured && c.ReadMult > 0 {
		p.CacheReadMultiplier = c.ReadMult
	}
	return p
}
