package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"time"
)

// classifyFanoutSweep returns the sweep scope and any violations. Missing
// canonical values alone make a smoke subset; a value outside the canonical set
// makes the sweep non-canonical.
func classifyFanoutSweep(sweep []int) (string, []string) {
	canonical := make(map[int]bool, len(CanonicalFanoutSweep))
	for _, n := range CanonicalFanoutSweep {
		canonical[n] = true
	}
	var violations []string
	present := make(map[int]bool)
	for _, n := range sweep {
		if !canonical[n] {
			violations = append(violations, fmt.Sprintf("fanout sweep value N=%d is outside the canonical sweep %v", n, CanonicalFanoutSweep))
		}
		present[n] = true
	}
	if len(violations) > 0 || len(sweep) == 0 {
		if len(sweep) == 0 {
			violations = append(violations, "fanout sweep is empty")
		}
		return FanoutSweepNonCanonical, violations
	}
	for _, n := range CanonicalFanoutSweep {
		if !present[n] {
			return FanoutSweepSmokeSubset, nil
		}
	}
	return FanoutSweepFull, nil
}

// fanoutRequestBody builds the route and JSON body for one subagent request.
// cache_prompt is llama.cpp's opt-in to slot prefix reuse; servers that do not
// know it ignore it.
func fanoutRequestBody(shape, model, prefix, task string, decode int) (string, map[string]any) {
	body := map[string]any{
		"model":        model,
		"max_tokens":   decode,
		"stream":       true,
		"temperature":  0.0,
		"cache_prompt": true,
	}
	if shape == RequestShapeChatSystem {
		body["messages"] = []map[string]string{
			{"role": "system", "content": prefix},
			{"role": "user", "content": task},
		}
		return "/v1/chat/completions", body
	}
	body["prompt"] = prefix + "\n" + task
	return "/v1/completions", body
}

// defaultHTTPReuseObserver scrapes the server's Prometheus text exposition into
// cumulative counter families. It is the production reuseObserver.
func defaultHTTPReuseObserver(ctx context.Context, endpoint string) (map[string]int64, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(endpoint, "/")+"/metrics", nil)
	if err != nil {
		return nil, err
	}
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("GET /metrics: HTTP status %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return nil, err
	}
	return parsePrometheusCounters(string(body)), nil
}

// reuseCounterSpec pairs a reuse counter family with the prompt-token counter
// that gives its denominator. On the fak gateway the in-kernel KV-prefix prompt
// counter includes the reused tokens; the proxy-path inference prompt counter
// is normalized to exclude the provider's cache_read tokens.
type reuseCounterSpec struct {
	Reused               string
	Prompt               string
	PromptIncludesReused bool
	Path                 string
}

// knownReuseCounters is in preference order. The two gateway families are
// mutually exclusive by path (the provider counter reads 0 in-kernel, the
// KV-prefix counter reads 0 on the proxy path); the harness reads exactly one
// family per cell so a token is never counted twice.
var knownReuseCounters = []reuseCounterSpec{
	{
		Reused:               "fak_gateway_kv_prefix_reused_tokens",
		Prompt:               "fak_gateway_kv_prefix_prompt_tokens",
		PromptIncludesReused: true,
		Path:                 "in-kernel kv prefix",
	},
	{
		Reused:               "fak_gateway_inference_cached_prompt_tokens",
		Prompt:               "fak_gateway_inference_prompt_tokens",
		PromptIncludesReused: false,
		Path:                 "proxy upstream cache_read",
	},
}

// reuseMetricSuffixes bind any other server's reuse counter (no denominator).
var reuseMetricSuffixes = []string{"reused_tokens", "cached_prompt_tokens"}

// parsePrometheusCounters sums every sample per metric family from a Prometheus
// text exposition, keyed by name with any `_total` suffix trimmed. Comment,
// HELP and TYPE lines are skipped; unparseable values are ignored.
func parsePrometheusCounters(body string) map[string]int64 {
	out := make(map[string]int64)
	for _, line := range strings.Split(body, "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		name := line
		if i := strings.IndexAny(line, "{ \t"); i >= 0 {
			name = line[:i]
		}
		name = strings.TrimSuffix(name, "_total")
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		v, err := strconv.ParseFloat(fields[len(fields)-1], 64)
		if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
			continue
		}
		out[name] += int64(v)
	}
	return out
}

// selectReuseDelta picks the one reuse counter family that moved between the
// two scrapes and returns its delta. Known gateway families are preferred in
// order; any other `*reused_tokens` / `*cached_prompt_tokens` family is a
// fallback. When no family moved, the family whose paired prompt counter moved
// (the active serving path) is returned with a measured zero, else the first
// family present in both scrapes.
func selectReuseDelta(before, after map[string]int64) (*FanoutReuseScrape, error) {
	var candidates []*FanoutReuseScrape
	seen := make(map[string]bool)
	consider := func(spec reuseCounterSpec) error {
		seen[spec.Reused] = true
		b, okB := before[spec.Reused]
		a, okA := after[spec.Reused]
		if !okB && !okA {
			return nil
		}
		if okB != okA {
			return fmt.Errorf("%w: %s", ErrReuseScrapeIncomplete, spec.Reused)
		}
		if a < b {
			return fmt.Errorf("%w: %s %d -> %d", ErrReuseCounterReset, spec.Reused, b, a)
		}
		sc := &FanoutReuseScrape{Counter: spec.Reused, Path: spec.Path, Before: b, After: a, Delta: a - b}
		if spec.Prompt != "" {
			pb, okPB := before[spec.Prompt]
			pa, okPA := after[spec.Prompt]
			if okPB && okPA && pa >= pb {
				sc.PromptCounter = spec.Prompt
				sc.PromptTokenDelta = pa - pb
				sc.OfferedTokens = sc.PromptTokenDelta
				if !spec.PromptIncludesReused {
					sc.OfferedTokens += sc.Delta
				}
			}
		}
		candidates = append(candidates, sc)
		return nil
	}
	for _, spec := range knownReuseCounters {
		if err := consider(spec); err != nil {
			return nil, err
		}
	}
	var others []string
	for name := range after {
		if seen[name] {
			continue
		}
		for _, suf := range reuseMetricSuffixes {
			if strings.HasSuffix(name, suf) {
				others = append(others, name)
				break
			}
		}
	}
	sort.Strings(others)
	for _, name := range others {
		if err := consider(reuseCounterSpec{Reused: name, Path: "generic"}); err != nil {
			return nil, err
		}
	}
	if len(candidates) == 0 {
		return nil, ErrReuseCounterAbsent
	}
	for _, c := range candidates {
		if c.Delta > 0 {
			return c, nil
		}
	}
	for _, c := range candidates {
		if c.OfferedTokens > 0 {
			return c, nil
		}
	}
	return candidates[0], nil
}

// applyCellReuse folds the selected counter delta into the cell: the measured
// reuse, hit rate and multiplier, and the divergence verdict against the
// analytic estimate. An error is an evidence gap (source unobserved, no
// divergence flag), never a silent zero that reads as a measured result.
func applyCellReuse(res *FanoutArmResult, scrape *FanoutReuseScrape, err error) {
	if err != nil || scrape == nil {
		res.ReuseObservationSource = FanoutReuseSourceUnobserved
		res.ObservedReuseTokens = 0
		res.ObservedHitRate = 0
		res.ObservedReuseMultiplier = 0
		res.ReuseScrape = nil
		return
	}
	res.ReuseObservationSource = FanoutReuseSourcePrometheus
	res.ReuseScrape = scrape
	res.ObservedReuseTokens = scrape.Delta
	switch {
	case scrape.OfferedTokens > 0:
		res.ObservedHitRate = float64(scrape.Delta) / float64(scrape.OfferedTokens)
		if computed := scrape.OfferedTokens - scrape.Delta; computed > 0 {
			res.ObservedReuseMultiplier = float64(scrape.OfferedTokens) / float64(computed)
		}
	case res.TotalPromptTokens > 0:
		offered := res.TotalPromptTokens * int64(max(res.Trials, 1))
		res.ObservedHitRate = float64(scrape.Delta) / float64(offered)
	}
	if res.PrefixHitRate > 0 && scrape.Delta == 0 {
		res.ReuseDivergence = true
	}
}

type propsProbe struct {
	refusal string
	source  string
	detail  string
}

// probeServedProps reads one /props surface. An empty refusal means the
// response reported a positive per-slot context.
func probeServedProps(ctx context.Context, base, label string) (servedPropsWire, propsProbe) {
	var props servedPropsWire
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, strings.TrimRight(base, "/")+"/props", nil)
	if err != nil {
		return props, propsProbe{FanoutServeRefuseProbeError, label + " (unbuildable request)", fmt.Sprintf("could not build %s request: %v", label, err)}
	}
	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return props, propsProbe{FanoutServeRefuseProbeError, label + " (transport error)", fmt.Sprintf("%s unreachable at %s: %v", label, base, err)}
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return props, propsProbe{FanoutServeRefuseCapacityUnobserved, fmt.Sprintf("%s (HTTP %d)", label, resp.StatusCode), fmt.Sprintf("%s returned HTTP %d", label, resp.StatusCode)}
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 4<<20)).Decode(&props); err != nil {
		return props, propsProbe{FanoutServeRefuseCapacityUnobserved, label + " (malformed JSON)", fmt.Sprintf("%s was not decodable: %v", label, err)}
	}
	source := label + " " + propsSlotCapacityField
	if props.perSlot() <= 0 {
		return props, propsProbe{FanoutServeRefuseCapacityUnobserved, source, fmt.Sprintf("%s reported no per-slot context; the reference command must set -c/--ctx-size", label)}
	}
	return props, propsProbe{source: source}
}
