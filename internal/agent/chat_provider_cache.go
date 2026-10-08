package agent

import (
	"bytes"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func (p *HTTPPlanner) attachProviderCacheTelemetry(comp *Completion, reqBody []byte, provider Provider) {
	if comp == nil || comp.ProviderCache != nil {
		return
	}
	cached := comp.Usage.CachedPromptTokens()
	if cached <= 0 {
		return
	}
	endpoint, reasoning, toolSet, region := p.providerVaryAxes(reqBody)
	entry := cachemeta.FromProviderCache(cachemeta.ProviderCache{
		Provider:       string(provider),
		ModelID:        p.ModelID,
		CachedTokens:   int64(cached),
		PromptTokens:   int64(comp.Usage.PromptTokens),
		SerializerID:   cachemeta.DigestBytes(reqBody),
		BreakpointMode: "implicit",
		Endpoint:       endpoint,
		ReasoningMode:  reasoning,
		ToolSetID:      toolSet,
		Region:         region,
		Owner:          "agent.HTTPPlanner",
	})
	comp.ProviderCache = &entry
}

// providerVaryAxes derives the provider-prefix cache-Vary axes that silently
// break the implicit cache, so they shape the cache-family identity rather than
// blend two request shapes into one hit rate. Endpoint and reasoning mode are
// the GLM-5.2 (Z.AI) axes from GLM52-HOSTED-CACHE-COHERENCE-2026-06-19.md §A2;
// tool set and region/affinity are the remaining two from the cache-frontier
// default-enablement plan (item 7, #1525). Best-effort and additive: an axis it
// cannot determine is left empty (no identity contribution).
func (p *HTTPPlanner) providerVaryAxes(reqBody []byte) (endpoint, reasoning, toolSet, region string) {
	// Endpoint: the Z.AI Coding-Plan route carries a "coding" segment in either
	// the model id (zai-coding-plan/glm-5.2) or the base URL (.../coding/paas/...).
	if strings.Contains(p.ModelID, "coding") || strings.Contains(p.BaseURL, "/coding/") {
		endpoint = "coding"
	}
	// Reasoning mode: read reasoning_effort, then thinking.type, from the extra
	// body the operator threaded in. The implicit cache is sensitive to these.
	if len(p.ExtraBody) > 0 {
		var obj map[string]json.RawMessage
		if json.Unmarshal(p.ExtraBody, &obj) == nil {
			if raw, ok := obj["reasoning_effort"]; ok {
				var s string
				if json.Unmarshal(raw, &s) == nil && s != "" {
					reasoning = s
				}
			}
			if reasoning == "" {
				if raw, ok := obj["thinking"]; ok {
					var th struct {
						Type string `json:"type"`
					}
					if json.Unmarshal(raw, &th) == nil && th.Type != "" {
						reasoning = th.Type
					}
				}
			}
		}
	}
	toolSet = toolSetDigest(reqBody)
	region = regionFromBaseURL(p.BaseURL)
	return endpoint, reasoning, toolSet, region
}

// toolSetDigest returns a STABLE digest of the request's tool set, or "" when the
// request carries no tools. The tool definitions sit in the provider's cacheable
// PREFIX — Anthropic folds the tool schema into the cached system block, an
// OpenAI-compatible body prefixes the `tools` array ahead of the messages — so a
// silent tool-set change breaks the implicit cache. Hashing ONLY the tools (not
// the whole request body, which the per-request SerializerID already covers)
// yields the stable cache-FAMILY axis: two turns that share the same tools share
// the digest, and adding/removing/reordering a tool is recorded as a distinct
// cache-write rather than an invisible miss. Both Anthropic Messages and OpenAI
// Chat name the field `tools` at the top level.
func toolSetDigest(reqBody []byte) string {
	if len(reqBody) == 0 {
		return ""
	}
	var obj map[string]json.RawMessage
	if json.Unmarshal(reqBody, &obj) != nil {
		return ""
	}
	raw, ok := obj["tools"]
	if !ok {
		return ""
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || string(trimmed) == "null" || string(trimmed) == "[]" {
		return ""
	}
	return cachemeta.DigestBytes(trimmed)
}

// regionFromBaseURL best-effort extracts a cloud region/affinity token from an
// endpoint host where the provider encodes it there — e.g. AWS Bedrock's
// "bedrock-runtime.us-east-1.amazonaws.com". A provider prompt cache is warm only
// in the region/zone that wrote it, so a request routed elsewhere is a distinct
// COLD family, not a hit-rate dip. It returns "" for the hosted endpoints that do
// not name a region in the host (api.anthropic.com, api.openai.com, Z.AI), so
// region stays an honest "where known" axis rather than a guess: only a
// well-formed AWS-style geo-direction-number label is recognized.
func regionFromBaseURL(baseURL string) string {
	host := baseURL
	if i := strings.Index(host, "://"); i >= 0 {
		host = host[i+3:]
	}
	if i := strings.IndexAny(host, "/?#"); i >= 0 {
		host = host[:i]
	}
	for _, label := range strings.Split(host, ".") {
		if isAWSRegionToken(label) {
			return label
		}
	}
	return ""
}

// isAWSRegionToken reports whether s is a well-formed AWS-style region label of
// the shape <geo>-<direction>-<number> (e.g. "us-east-1", "ap-southeast-2"). The
// check is intentionally tight so an ordinary host label is never misread as a
// region.
func isAWSRegionToken(s string) bool {
	parts := strings.Split(s, "-")
	if len(parts) != 3 {
		return false
	}
	geo, dir, num := parts[0], parts[1], parts[2]
	if len(geo) < 2 || len(geo) > 3 || !isLowerAlpha(geo) {
		return false
	}
	if dir == "" || !isLowerAlpha(dir) {
		return false
	}
	if num == "" || !isDigits(num) {
		return false
	}
	return true
}

func isLowerAlpha(s string) bool {
	for _, r := range s {
		if r < 'a' || r > 'z' {
			return false
		}
	}
	return true
}

func isDigits(s string) bool {
	for _, r := range s {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
