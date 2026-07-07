package engine

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
)

// vllm_cache_observe.go — the vLLM prefix-cache observation adapter, cache-frontier
// "Next 50" item 33 (#1551), External-engines lane under epic #1490. It reads whatever
// prefix-reuse signal a vLLM upstream exposes on its public Prometheus surface
// (vllm:prefix_cache_queries / vllm:prefix_cache_hits) and maps it onto the wire-neutral
// engine.CacheCapability contract (#1550): observed prefix reuse when the signal is
// present, or an explicit "unavailable" verdict when it is not — never a fabricated
// reuse number. It mirrors the llama.cpp session-cache observer (#1553) so the three
// External-engine adapters keep one mapping shape.
//
// Design memo (gen/second-next — architectural option, never a default exposure):
//
//   PROMOTION EVIDENCE. This is the first per-engine observation adapter that turns a
//   vLLM upstream's prefix-reuse counters into the CacheCapability contract, so a
//   vLLM-fronted session can now report observed prefix reuse OR an explicit
//   "unavailable" — the evidence source the later item-39 value comparison and item-40
//   /debug/vars surfacing read. It re-witnesses the #1549 inventory row (vLLM =
//   passive-observe).
//
//   DEMOTION / RETIREMENT EVIDENCE. The adapter observes ONLY — it never warms, evicts,
//   or clones (that is item 37, gated on this capability row). When a vLLM upstream stops
//   exposing the prefix-cache counters, the adapter demotes itself to CacheUnknown
//   ("unavailable") rather than reading the zero-valued counters as a measured 0% reuse;
//   nothing downstream may read an active class from it.
//
//   INVALIDATING ASSUMPTION. It assumes vLLM's OpenAI-frontend Prometheus surface names
//   the prefix-reuse signal vllm:prefix_cache_queries / vllm:prefix_cache_hits (the same
//   names vllm.go's ParseVLLMPrometheus already reads). A vLLM build that renames or
//   drops those counters makes the present path unreachable — the observation then
//   honestly reports unavailable, and this mapping (and the #1549 inventory row) must be
//   re-witnessed against that build.

// VLLMPrefixCacheSignal is the decoded vLLM prefix-cache observation: whether the
// upstream's Prometheus surface exposed the prefix-cache counters at all, and, when it
// did, the observed query/hit totals. Present is the ONLY thing that separates "observed
// reuse" from "unavailable": the counters MAY be present and zero (a live surface
// reporting no reuse yet), which is a passive-observe observation, not an absent signal.
type VLLMPrefixCacheSignal struct {
	// Present is true when the scrape exposed vllm:prefix_cache_queries or
	// vllm:prefix_cache_hits. A present-but-zero counter is still Present.
	Present bool
	// Queries and Hits are the observed prefix-cache counter totals, summed across the
	// scrape's label sets (a multi-rank worker emits one line per data-parallel rank).
	// Meaningful only when Present; both zero otherwise, and never emitted as a reuse
	// number then.
	Queries float64
	Hits    float64
	// Note explains a signal-ABSENT observation (metrics endpoint disabled, counters not
	// in the build). Empty when Present.
	Note string
}

// ObserveVLLMPrefixCache scans a vLLM Prometheus scrape for the prefix-reuse counters
// and returns the decoded signal. It marks the signal Present the moment either counter
// appears — even at zero — so a live-but-cold cache is distinguished from a build that
// never exposes the counters. It reuses the same parse helper and counter names as
// vllm.go's ParseVLLMPrometheus so the observation and the serving-metrics view never
// drift.
func ObserveVLLMPrefixCache(metricsText string) VLLMPrefixCacheSignal {
	var sig VLLMPrefixCacheSignal
	for _, line := range strings.Split(metricsText, "\n") {
		name, value, ok := parsePromSample(line)
		if !ok {
			continue
		}
		switch name {
		case "vllm:prefix_cache_queries":
			sig.Present = true
			sig.Queries += value
		case "vllm:prefix_cache_hits":
			sig.Present = true
			sig.Hits += value
		}
	}
	if !sig.Present {
		sig.Note = "no vllm:prefix_cache_{queries,hits} counters in the scrape"
	}
	return sig
}

// Capability maps the decoded vLLM prefix-cache signal onto the wire-neutral
// engine.CacheCapability contract. A present counter surface is CachePassiveObserve (fak
// observes symptoms; vLLM's public control plane is whole-prefix reset only,
// SupportsExactSpan=false — never an active class); an absent one is the fail-closed
// CacheUnknown with an explicit "unavailable" Evidence string. Both carry
// ProvenanceProvider (an observed provider counter, not a kernel witness) and
// ColdPathCorrect true (observation never changes the request's cold path — a miss still
// sends full required context).
func (s VLLMPrefixCacheSignal) Capability() CacheCapability {
	out := CacheCapability{
		Engine:          VLLMEngineID,
		Provenance:      ProvenanceProvider,
		ColdPathCorrect: true,
	}
	if s.Present {
		out.Verdict = CachePassiveObserve
		out.Evidence = fmt.Sprintf(
			"vllm:prefix_cache_hits=%s over vllm:prefix_cache_queries=%s (observed prefix reuse); passive-observe, whole-prefix reset only, no exact-span/warm control surface",
			promFloat(s.Hits), promFloat(s.Queries))
		return out
	}
	out.Verdict = CacheUnknown
	note := s.Note
	if note == "" {
		note = "no prefix-cache counters exposed"
	}
	out.Evidence = "vLLM upstream exposed no prefix-reuse signal (" + note + "); unavailable, no reuse number fabricated"
	return out
}

// VLLMCacheObserver reads a vLLM upstream's prefix-cache reuse signal from its public
// Prometheus surface and reports it as a wire-neutral engine.CacheCapability. It
// satisfies CacheCapabilityProducer behind the item-32 seam (#1550), so the gateway can
// report a vLLM-fronted session's cache capability without importing this package.
type VLLMCacheObserver struct {
	cfg    VLLMConfig
	client *http.Client
	signal VLLMPrefixCacheSignal
	read   bool
}

// NewVLLMCacheObserver builds an observer over a vLLM worker's public /metrics surface.
func NewVLLMCacheObserver(cfg VLLMConfig) *VLLMCacheObserver {
	return &VLLMCacheObserver{cfg: cfg, client: defaultHTTPClient(cfg.Client)}
}

// Observe scrapes the vLLM Prometheus endpoint once and caches the prefix-cache signal
// for CacheCapability. A transport error is returned; a NON-200 status (metrics
// disabled) is NOT an error — it resolves to the unavailable/no-evidence signal, because
// a disabled endpoint is a legitimate "engine exposes no cache surface" observation
// rather than a failed read. Safe to call again to refresh.
func (o *VLLMCacheObserver) Observe(ctx context.Context) (VLLMPrefixCacheSignal, error) {
	metricsURL, err := deriveMetricsURL(o.cfg.MetricsURL, o.cfg.BaseURL, "vllm", "FAK_VLLM_METRICS_URL", true)
	if err != nil {
		return VLLMPrefixCacheSignal{}, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, metricsURL, nil)
	if err != nil {
		return VLLMPrefixCacheSignal{}, err
	}
	if o.cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+o.cfg.APIKey)
	}
	resp, err := o.client.Do(req)
	if err != nil {
		return VLLMPrefixCacheSignal{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		sig := VLLMPrefixCacheSignal{Note: fmt.Sprintf("/metrics returned %d (endpoint disabled)", resp.StatusCode)}
		o.signal, o.read = sig, true
		return sig, nil
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return VLLMPrefixCacheSignal{}, err
	}
	sig := ObserveVLLMPrefixCache(string(raw))
	o.signal, o.read = sig, true
	return sig, nil
}

// CacheCapability satisfies engine.CacheCapabilityProducer. Before Observe has resolved a
// signal it FAILS CLOSED to the unknown/unavailable label rather than inferring a
// positive — the same fail-closed default the contract's zero verdict has.
func (o *VLLMCacheObserver) CacheCapability() CacheCapability {
	if !o.read {
		return VLLMPrefixCacheSignal{Note: "not yet observed"}.Capability()
	}
	return o.signal.Capability()
}

// VLLMCacheObserver produces a wire-neutral CacheCapability behind the item-32 seam.
var _ CacheCapabilityProducer = (*VLLMCacheObserver)(nil)
