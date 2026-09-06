package engine

import (
	"context"
	"fmt"
	"net/http"
	"strings"
)

// sglang_cache_observe.go — the SGLang radix/prefix-cache observation adapter,
// cache-frontier "Next 50" item 34 (#1552), External-engines lane under epic #1490. It
// reads whatever radix/prefix-reuse signal an SGLang upstream exposes on its public
// Prometheus surface (sglang:cache_hit_rate / sglang:prefix_cache_hit_rate — the single
// hit-rate gauge SGLang reports, NOT vLLM-style query/hit counters) and maps it onto the
// wire-neutral engine.CacheCapability contract (#1550): observed radix reuse when the
// gauge is present, or an explicit "unavailable" verdict when it is not — never a
// fabricated reuse number. It mirrors the vLLM prefix-cache observer (#1551) so the
// External-engine adapters keep one mapping shape and report on identical score axes.
//
// Design memo (gen/second-next — architectural option, never a default exposure):
//
//   PROMOTION EVIDENCE. This is the SGLang sibling of the vLLM observation adapter: it
//   turns an SGLang upstream's RadixAttention hit-rate gauge into the CacheCapability
//   format, so an SGLang-fronted session can now report observed radix reuse OR an
//   explicit "unavailable" on the SAME axes as vLLM (#1551) and llama.cpp (#1553) — the
//   evidence source the later item-39 same-geometry value comparison and item-40
//   /debug/vars surfacing read. It re-witnesses the #1549 inventory row (SGLang =
//   passive-observe).
//
//   DEMOTION / RETIREMENT EVIDENCE. The adapter observes ONLY — it never warms, evicts,
//   or clones (that is item 37, gated on this capability row), and it adds NO duplicate
//   flush logic: SGLang's whole-prefix flush_cache control stays owned by enginecache
//   (SupportsExactSpan=false for EngineSGLang), so this never claims an active/exact-span
//   class. When an SGLang upstream stops exposing the radix hit-rate gauge, the adapter
//   demotes itself to CacheUnknown ("unavailable") rather than reading an absent gauge as
//   a measured 0% reuse; nothing downstream may read an active class from it.
//
//   INVALIDATING ASSUMPTION. It assumes SGLang's Prometheus surface names the radix
//   reuse signal sglang:cache_hit_rate / sglang:prefix_cache_hit_rate (the same names
//   sglang.go's ParseSGLangPrometheus already reads and normalizes) and that the gauge is
//   a 0..1 ratio (or a 0..100 percentage it normalizes). An SGLang build that renames or
//   drops that gauge makes the present path unreachable — the observation then honestly
//   reports unavailable, and this mapping (and the #1549 inventory row) must be
//   re-witnessed against that build.

// SGLangRadixCacheSignal is the decoded SGLang radix/prefix-cache observation: whether the
// upstream's Prometheus surface exposed the radix hit-rate gauge at all, and, when it did,
// the observed hit ratio. Present is the ONLY thing that separates "observed reuse" from
// "unavailable": the gauge MAY be present and zero (a live surface reporting no reuse
// yet), which is a passive-observe observation, not an absent signal.
type SGLangRadixCacheSignal struct {
	// Present is true when the scrape exposed sglang:cache_hit_rate or
	// sglang:prefix_cache_hit_rate. A present-but-zero gauge is still Present.
	Present bool
	// HitRatio is the observed radix/prefix cache-hit ratio in 0..1, normalized from a
	// percentage when the gauge reports 1..100 (the same normalization
	// ParseSGLangPrometheus applies). Meaningful only when Present; zero otherwise, and
	// never emitted as a reuse number then.
	HitRatio float64
	// Note explains a signal-ABSENT observation (metrics endpoint disabled, gauge not in
	// the build). Empty when Present.
	Note string
}

// ObserveSGLangRadixCache scans an SGLang Prometheus scrape for the radix/prefix-cache
// hit-rate gauge and returns the decoded signal. It marks the signal Present the moment
// the gauge appears — even at zero — so a live-but-cold radix cache is distinguished from
// a build that never exposes the gauge. It reuses the same gauge names and 0..100→0..1
// normalization as sglang.go's ParseSGLangPrometheus so the observation and the
// serving-metrics view never drift.
func ObserveSGLangRadixCache(metricsText string) SGLangRadixCacheSignal {
	var sig SGLangRadixCacheSignal
	for _, line := range strings.Split(metricsText, "\n") {
		name, value, ok := parsePromSample(line)
		if !ok {
			continue
		}
		switch name {
		case "sglang:cache_hit_rate", "sglang:prefix_cache_hit_rate":
			sig.Present = true
			v := value
			if v > 1 && v <= 100 {
				v = v / 100
			}
			sig.HitRatio = v
		}
	}
	if !sig.Present {
		sig.Note = "no sglang:{cache,prefix_cache}_hit_rate gauge in the scrape"
	}
	return sig
}

// Capability maps the decoded SGLang radix-cache signal onto the wire-neutral
// engine.CacheCapability contract. A present gauge is CachePassiveObserve (fak observes
// symptoms; SGLang's public control plane is whole-prefix flush_cache reset only,
// SupportsExactSpan=false — never an active class); an absent one is the fail-closed
// CacheUnknown with an explicit "unavailable" Evidence string. Both carry
// ProvenanceProvider (an observed provider gauge, not a kernel witness) and ColdPathCorrect
// true (observation never changes the request's cold path — a radix miss still sends full
// required context).
func (s SGLangRadixCacheSignal) Capability() CacheCapability {
	out := CacheCapability{
		Engine:          SGLangEngineID,
		Provenance:      ProvenanceProvider,
		ColdPathCorrect: true,
	}
	if s.Present {
		out.Verdict = CachePassiveObserve
		out.Evidence = fmt.Sprintf(
			"sglang radix/prefix-cache hit_rate=%s (observed radix reuse); passive-observe, whole-prefix flush_cache reset only, no exact-span/warm control surface",
			promFloat(s.HitRatio))
		return out
	}
	out.Verdict = CacheUnknown
	note := s.Note
	if note == "" {
		note = "no radix hit-rate gauge exposed"
	}
	out.Evidence = "SGLang upstream exposed no radix-reuse signal (" + note + "); unavailable, no reuse number fabricated"
	return out
}

// SGLangCacheObserver reads an SGLang upstream's radix/prefix-cache reuse signal from its
// public Prometheus surface and reports it as a wire-neutral engine.CacheCapability. It
// satisfies CacheCapabilityProducer behind the item-32 seam (#1550), so the gateway can
// report an SGLang-fronted session's cache capability without importing this package.
type SGLangCacheObserver struct {
	cfg    SGLangConfig
	client *http.Client
	signal SGLangRadixCacheSignal
	read   bool
}

// NewSGLangCacheObserver builds an observer over an SGLang worker's public /metrics surface.
func NewSGLangCacheObserver(cfg SGLangConfig) *SGLangCacheObserver {
	return &SGLangCacheObserver{cfg: cfg, client: defaultHTTPClient(cfg.Client)}
}

// Observe scrapes the SGLang Prometheus endpoint once and caches the radix-cache signal
// for CacheCapability. A transport error is returned; a NON-200 status (metrics disabled)
// is NOT an error — it resolves to the unavailable/no-evidence signal, because a disabled
// endpoint is a legitimate "engine exposes no cache surface" observation rather than a
// failed read. SGLang's HTTP root is not an OpenAI /v1 frontend, so no /v1 suffix is
// stripped when the /metrics URL is derived (matching sglang.go's metricsURL). Safe to
// call again to refresh.
func (o *SGLangCacheObserver) Observe(ctx context.Context) (SGLangRadixCacheSignal, error) {
	metricsURL, err := deriveMetricsURL(o.cfg.MetricsURL, o.cfg.BaseURL, "sglang", "FAK_SGLANG_METRICS_URL", false)
	if err != nil {
		return SGLangRadixCacheSignal{}, err
	}
	raw, disabled, err := scrapeMetricsText(ctx, o.client, metricsURL, o.cfg.APIKey)
	if err != nil {
		return SGLangRadixCacheSignal{}, err
	}
	sig := SGLangRadixCacheSignal{Note: disabled}
	if disabled == "" {
		sig = ObserveSGLangRadixCache(raw)
	}
	o.signal, o.read = sig, true
	return sig, nil
}

// CacheCapability satisfies engine.CacheCapabilityProducer. Before Observe has resolved a
// signal it FAILS CLOSED to the unknown/unavailable label rather than inferring a
// positive — the same fail-closed default the contract's zero verdict has.
func (o *SGLangCacheObserver) CacheCapability() CacheCapability {
	if !o.read {
		return SGLangRadixCacheSignal{Note: "not yet observed"}.Capability()
	}
	return o.signal.Capability()
}

// SGLangCacheObserver produces a wire-neutral CacheCapability behind the item-32 seam.
var _ CacheCapabilityProducer = (*SGLangCacheObserver)(nil)
