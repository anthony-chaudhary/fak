// Adapted from ai-dynamo/aiperf src/aiperf/server_metrics/accumulator.py

package gateway

// ServerMetricKind is the Prometheus type a neutral server metric is read as.
// Counter values are raw cumulative samples (callers take the delta or rate);
// the KV-cache gauge is a 0..1 fraction, not a percentage.
type ServerMetricKind int

const (
	ServerMetricGauge ServerMetricKind = iota
	ServerMetricCounter
)

// Neutral server-metric names, independent of the serving backend.
const (
	ServerMetricKVCacheUsage     = "kv_cache_usage"
	ServerMetricNumRunning       = "num_running"
	ServerMetricNumWaiting       = "num_waiting"
	ServerMetricNumPreemptions   = "num_preemptions"
	ServerMetricPromptTokens     = "prompt_tokens"
	ServerMetricGenerationTokens = "generation_tokens"
)

// ServerMetricChain maps one neutral metric to backend metric names in precedence order.
type ServerMetricChain struct {
	Neutral string
	Kind    ServerMetricKind
	Names   []string
}

// Counter names are the family form without the "_total" suffix, which is how
// Prometheus text parsers key counter families.
var serverMetricChains = []ServerMetricChain{
	{ServerMetricKVCacheUsage, ServerMetricGauge, []string{"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "sglang:token_usage"}},
	{ServerMetricNumRunning, ServerMetricGauge, []string{"vllm:num_requests_running", "sglang:num_running_reqs"}},
	{ServerMetricNumWaiting, ServerMetricGauge, []string{"vllm:num_requests_waiting", "sglang:num_queue_reqs"}},
	{ServerMetricNumPreemptions, ServerMetricCounter, []string{"vllm:num_preemptions", "sglang:num_retracted_reqs"}},
	{ServerMetricPromptTokens, ServerMetricCounter, []string{"vllm:prompt_tokens", "sglang:prompt_tokens"}},
	{ServerMetricGenerationTokens, ServerMetricCounter, []string{"vllm:generation_tokens", "sglang:generation_tokens"}},
}

// ServerMetricChains returns a copy of every neutral chain, vLLM names before SGLang names.
func ServerMetricChains() []ServerMetricChain {
	out := make([]ServerMetricChain, len(serverMetricChains))
	for i, c := range serverMetricChains {
		c.Names = append([]string(nil), c.Names...)
		out[i] = c
	}
	return out
}

// ServerMetricSample is one scraped metric family value with its Prometheus type.
type ServerMetricSample struct {
	Kind  ServerMetricKind
	Value float64
}

// ServerMetricResolution is a neutral value and the backend metric that supplied it.
type ServerMetricResolution struct {
	Neutral string
	Source  string
	Value   float64
}

// ResolveServerMetric returns the value of the first chain entry present in
// scraped with the chain's kind. The kind check matters because stripping
// "_total" can make a counter share its name with a gauge (SGLang's
// num_retracted_reqs). A missing chain reports ok=false rather than a zero
// value, so an absent backend metric is never mistaken for an idle server.
func ResolveServerMetric(neutral string, scraped map[string]ServerMetricSample) (ServerMetricResolution, bool) {
	chain, ok := serverMetricChainFor(neutral)
	if !ok {
		return ServerMetricResolution{}, false
	}
	for _, name := range chain.Names {
		if s, present := scraped[name]; present && s.Kind == chain.Kind {
			return ServerMetricResolution{Neutral: neutral, Source: name, Value: s.Value}, true
		}
	}
	return ServerMetricResolution{}, false
}

// ResolveServerMetrics resolves every neutral chain, omitting the missing ones.
func ResolveServerMetrics(scraped map[string]ServerMetricSample) map[string]ServerMetricResolution {
	out := make(map[string]ServerMetricResolution, len(serverMetricChains))
	for _, c := range serverMetricChains {
		if r, ok := ResolveServerMetric(c.Neutral, scraped); ok {
			out[c.Neutral] = r
		}
	}
	return out
}

func serverMetricChainFor(neutral string) (ServerMetricChain, bool) {
	for _, c := range serverMetricChains {
		if c.Neutral == neutral {
			return c, true
		}
	}
	return ServerMetricChain{}, false
}
