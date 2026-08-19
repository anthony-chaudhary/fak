package gateway

import "strings"

// VCacheRuntimeCalibration is the fresh measured subset wired into live request
// decisions. MinPrefixTokens is currently the first true steering input: it can
// suppress cache-control authoring when the request cannot reach the provider's
// measured cacheability floor. The remaining measured constants are carried now
// so pricing and TTL decisions can consume the same provenance without another
// configuration shape change.
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

func (c *VCacheRuntimeCalibration) admitsAnchor(model string, estimatedTokens int) bool {
	if c != nil && strings.TrimSpace(c.Model) != "" && !strings.EqualFold(strings.TrimSpace(c.Model), strings.TrimSpace(model)) {
		return true
	}
	if c == nil || !c.MinPrefixMeasured || c.MinPrefixTokens <= 0 {
		return true
	}
	return int64(estimatedTokens) >= c.MinPrefixTokens
}

// ApplyCachePricing overlays only fresh measured pricing fields. Static model
// prices and write multipliers remain unchanged when calibration lacks evidence.
func (c *VCacheRuntimeCalibration) ApplyCachePricing(p CachePricing) CachePricing {
	if c != nil && c.ReadMultMeasured && c.ReadMult > 0 {
		p.CacheReadMultiplier = c.ReadMult
	}
	return p
}
