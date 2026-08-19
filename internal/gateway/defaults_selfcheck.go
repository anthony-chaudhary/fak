package gateway

import (
	"bytes"
	"encoding/json"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/agent"
)

// DefaultsSelfcheckResult is deterministic behavioral evidence for the default
// managed-context and provider-cache mechanisms. Unsupported rows are explicit.
type DefaultsSelfcheckResult struct {
	CompactHistory    bool   `json:"compact_history"`
	StaleReadElision  bool   `json:"stale_read_elision"`
	ColdToolDeferral  bool   `json:"cold_tool_deferral"`
	VCacheAnchor      bool   `json:"vcache_anchor"`
	MinimumPrefixGate bool   `json:"minimum_prefix_gate"`
	ReadPricing       bool   `json:"read_pricing"`
	WritePricing      bool   `json:"write_pricing"`
	TTLTierSteering   bool   `json:"ttl_tier_steering"`
	VCacheSignals     bool   `json:"vcache_signals"`
	OpenAIColdTools   string `json:"openai_cold_tools"`
}

// RunDefaultsSelfcheck executes the same transforms and accounting seams used by
// live gateway requests, with deterministic provider-shaped fixtures and no key.
func RunDefaultsSelfcheck() (DefaultsSelfcheckResult, error) {
	result := DefaultsSelfcheckResult{OpenAIColdTools: "unsupported: Anthropic ToolSearch has no witnessed OpenAI-compatible discovery seam"}

	compactRaw := defaultsCompactBody(120)
	compactReq, err := agent.DecodeAnthropicMessagesRequest(compactRaw)
	if err != nil {
		return result, err
	}
	compactServer := &Server{planner: &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}, compactHistoryBudget: 1200, compactAnchorHead: true, logf: func(string, ...any) {}}
	compactFired, _ := compactServer.compactAnthropicRawWithReason(compactReq, 10, "defaults-check")
	result.CompactHistory = compactFired && len(compactReq.Raw) < len(compactRaw)

	staleRaw := defaultsStaleReadBody()
	staleReq, err := agent.DecodeAnthropicMessagesRequest(staleRaw)
	if err != nil {
		return result, err
	}
	staleServer := &Server{planner: &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}, elideStaleReads: true, metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}}
	result.StaleReadElision = staleServer.maybeElideStaleReads(staleReq, "defaults-check") && !bytes.Contains(staleReq.Raw, []byte("STALE-BODY-CONTENT"))

	deferred := deferColdToolsInBody(defaultsToolBody(), defaultHotToolSet, func(b []byte) error { _, e := agent.DecodeAnthropicMessagesRequest(b); return e })
	result.ColdToolDeferral = deferred.Changed && deferred.ColdCount == 1 && bytes.Contains(deferred.Body, []byte(`"defer_loading":true`))

	anchorRaw := []byte(`{"model":"claude","max_tokens":64,"system":[{"type":"text","text":"stable policy"}],"messages":[{"role":"user","content":"hello"}]}`)
	anchorReq, err := agent.DecodeAnthropicMessagesRequest(anchorRaw)
	if err != nil {
		return result, err
	}
	anchorServer := &Server{planner: &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}, vcacheAnchor: true, metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}}
	result.VCacheAnchor = anchorServer.maybeAnchorAnthropicRaw(anchorReq, "defaults-check") && bytes.Contains(anchorReq.Raw, []byte("cache_control"))

	gatedReq, err := agent.DecodeAnthropicMessagesRequest(anchorRaw)
	if err != nil {
		return result, err
	}
	gated := &Server{planner: &agent.HTTPPlanner{Provider: agent.ProviderAnthropic}, vcacheAnchor: true, metrics: newGatewayMetrics(time.Now()), logf: func(string, ...any) {}, vcacheCalibration: &VCacheRuntimeCalibration{MinPrefixTokens: 100000, MinPrefixMeasured: true}}
	result.MinimumPrefixGate = !gated.maybeAnchorAnthropicRaw(gatedReq, "defaults-check") && bytes.Equal(gatedReq.Raw, anchorRaw)

	cal := (&VCacheRuntimeCalibration{ReadMult: .2, ReadMultMeasured: true, Write5mMult: 1.4, Write5mMeasured: true, Write1hMult: 2.2, Write1hMeasured: true}).ApplyCachePricing(CachePricing{InputPerMTokUSD: 1_000_000})
	result.ReadPricing = cal.CostUSD(CacheUsage{CacheReadTokens: 10}) == 2
	result.WritePricing = cal.CostUSD(CacheUsage{CacheCreationTokens: 10, WriteTTL: CacheTTL5m}) == 14 && cal.CostUSD(CacheUsage{CacheCreationTokens: 10, WriteTTL: CacheTTL1h}) == 22
	result.TTLTierSteering = !(&VCacheRuntimeCalibration{TTLMillis: int64((2 * time.Hour) / time.Millisecond), TTLMeasured: true}).wantsExplicitOneHourTTL("claude")

	m := newGatewayMetrics(time.Now())
	m.observeVCacheTurn("openai", 1, 100, 400, 0)
	turns, _ := m.vcacheTurnsSnapshot()
	result.VCacheSignals = len(turns) == 1 && turns[0].CacheRead == 400
	return result, nil
}

func defaultsCompactBody(n int) []byte {
	messages := make([]map[string]any, 0, n)
	messages = append(messages, map[string]any{"role": "user", "content": []map[string]any{{"type": "text", "text": "cached head", "cache_control": map[string]any{"type": "ephemeral"}}}})
	for i := 1; i < n; i++ {
		role := "user"
		if i%2 == 0 {
			role = "assistant"
		}
		messages = append(messages, map[string]any{"role": role, "content": []map[string]any{{"type": "text", "text": strings.Repeat("conversation body ", 40)}}})
	}
	b, _ := json.Marshal(map[string]any{"model": "claude", "max_tokens": 64, "system": "policy", "messages": messages})
	return b
}
func defaultsStaleReadBody() []byte {
	cc := map[string]any{"type": "ephemeral"}
	big := strings.Repeat("STALE-BODY-CONTENT ", 700)
	b, _ := json.Marshal(map[string]any{"model": "claude", "max_tokens": 64, "system": []map[string]any{{"type": "text", "text": "policy", "cache_control": cc}}, "messages": []map[string]any{
		{"role": "user", "content": []map[string]any{{"type": "text", "text": "head", "cache_control": cc}}},
		{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "r", "name": "Read", "input": map[string]any{"file_path": "x.go"}}}},
		{"role": "user", "content": []map[string]any{{"type": "tool_result", "tool_use_id": "r", "content": big}}},
		{"role": "assistant", "content": []map[string]any{{"type": "tool_use", "id": "e", "name": "Edit", "input": map[string]any{"file_path": "x.go"}}}},
		{"role": "user", "content": "edited"}, {"role": "assistant", "content": "tail"}, {"role": "user", "content": "tail"}, {"role": "assistant", "content": "tail"}, {"role": "user", "content": "tail"},
	}})
	return b
}
func defaultsToolBody() []byte {
	return []byte(`{"model":"claude","max_tokens":64,"messages":[{"role":"user","content":"hi"}],"tools":[{"name":"Read","input_schema":{"type":"object"}},{"name":"mcp__custom__cold","input_schema":{"type":"object"},"cache_control":{"type":"ephemeral"}}]}`)
}
