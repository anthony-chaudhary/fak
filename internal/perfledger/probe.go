package perfledger

import "strings"

// ProbeMaxPromptTokens is the largest whole prompt (uncached + cached) a turn can
// carry and still count as a liveness probe. Router and health probes send a
// one-line prompt with no system prompt (4 uncached + 21 cached tokens on the
// Halos); a real agent or chat turn always carries a system prompt far above it.
// On two Halos, probes were 66-68% of perf rows; their warm, sub-second TTFT made
// the partial regime p50 read 148 ms when served turns took ~7 s (fak-private#3151).
const ProbeMaxPromptTokens = 32

// IsProbe reports whether the row is a liveness/probe turn that Summarize keeps
// out of the served-traffic statistics: a row the sender marked synthetic, or a
// tiny prompt the server has already cached, because a probe repeats the same
// one-liner. A tiny cold prompt without the marker is still counted as served (a
// user's first short turn is real traffic).
func (r Record) IsProbe() bool {
	if r.Synthetic {
		return true
	}
	total := r.PromptTokens + r.CachedTokens
	return r.CachedTokens > 0 && total <= ProbeMaxPromptTokens
}

// Request headers a caller uses to name itself. ProbeHeader marks a synthetic
// turn (readiness canary, health probe) with any non-empty value; ClientHeader
// names the harness when its User-Agent does not.
const (
	ProbeHeader  = "X-Fak-Probe"
	ClientHeader = "X-Fak-Client"
)

// Record.Client vocabulary: a closed set so the ledger never stores a raw
// User-Agent. "" is a row written before the field existed.
const (
	ClientPi         = "pi"
	ClientOpenCode   = "opencode"
	ClientClaudeCode = "claude_code"
	ClientCodex      = "codex"
	ClientFak        = "fak"
	ClientProbe      = "probe"
	ClientSDK        = "sdk"
	ClientHTTPLib    = "http_lib"
	ClientUnknown    = "unknown"
	ClientOther      = "other"
)

// Clients is the closed Record.Client vocabulary in render order.
var Clients = []string{
	ClientPi, ClientOpenCode, ClientClaudeCode, ClientCodex, ClientFak, ClientProbe,
	ClientSDK, ClientHTTPLib, ClientUnknown, ClientOther,
}

// clientUAMarkers maps a lowercase User-Agent substring to its class; the first
// match wins, so harness markers precede the SDK and library markers they embed.
var clientUAMarkers = []struct{ marker, client string }{
	{"fak-router-canary", ClientProbe},
	{"opencode", ClientOpenCode},
	{"claude-cli", ClientClaudeCode},
	{"claude-code", ClientClaudeCode},
	{"codex", ClientCodex},
	{"pi-coding-agent", ClientPi},
	{"pi-ai", ClientPi},
	{"fak", ClientFak},
	{"ai-sdk", ClientSDK},
	{"openai/", ClientSDK},
	{"anthropic/", ClientSDK},
	{"go-http-client", ClientHTTPLib},
	{"curl/", ClientHTTPLib},
	{"python-requests", ClientHTTPLib},
	{"python-httpx", ClientHTTPLib},
	{"aiohttp", ClientHTTPLib},
}

// ClassifyClient maps a request's probe marker, explicit client header and
// User-Agent onto the closed Record.Client vocabulary and reports whether the
// turn is synthetic. A probe marker always wins; an explicit header outside the
// vocabulary reads "other", never its raw value.
func ClassifyClient(probe, client, userAgent string) (string, bool) {
	if strings.TrimSpace(probe) != "" {
		return ClientProbe, true
	}
	if c := strings.ToLower(strings.TrimSpace(client)); c != "" {
		for _, known := range Clients {
			if c == known {
				return known, known == ClientProbe
			}
		}
		return ClientOther, false
	}
	ua := strings.ToLower(strings.TrimSpace(userAgent))
	if ua == "" {
		return ClientUnknown, false
	}
	for _, m := range clientUAMarkers {
		if strings.Contains(ua, m.marker) {
			return m.client, m.client == ClientProbe
		}
	}
	return ClientOther, false
}
