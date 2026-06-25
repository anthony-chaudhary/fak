package vcachegov

import (
	"crypto/sha256"
	"encoding/hex"
	"strings"
)

// affinity.go is the cross-shard affinity router and its two safety guards (issue
// #720, acceptance 3, from design §9 + Law D3). A provider's KV cache is
// per-server: a prefix warm on shard A is cold on shard B, and the load balancer
// may route the next request anywhere. vCache biases chained requests onto ONE
// shard by setting a stable affinity key consistently across the warm and every
// recall, so the prefix the warm wrote is the one the recall reads.
//
// Two hazards the router guards against:
//
//   - AUTOSCALE REHASH invalidates the whole warm set at once. When the provider
//     reshard, every key behind one shard misses together — a CORRELATED p(hit)
//     collapse. Continuing to issue cheap recalls that now all miss is pure cost
//     with no benefit, so the detector signals RE-WARM (re-issue the heartbeat
//     prefills under the new sharding) instead.
//
//   - SELF-TRIGGERED REHASH. A warming burst concentrated on one shard can itself
//     trip the provider's autoscaler and invalidate the very set vCache just built.
//     The burst cap throttles warms per window so vCache never engineers its own
//     invalidation.
//
// Affinity is a BIAS, never a guarantee (Law D3): the provider's load balancer can
// still route elsewhere, and a recycle silently evaporates the set. Correctness
// never depends on the route (Law A2) — a miss is only ever a cost/latency hit.

// AffinityKey derives the cross-shard routing hint for one prefix chain: a stable
// hash of (tenant ‖ chain ‖ region). It MUST be set identically on the warm
// request and on every recall so chained requests land on the same warm shard
// (OpenAI prompt_cache_key / Anthropic prefix-hash routing). The components:
//
//   - tenant — namespacing the key per tenant keeps one tenant's warm set from
//     colliding with another's on a shared shard (and bounds the cross-tenant
//     timing side-channel of Law D4 by making cross-tenant membership distinct).
//   - chain — the prefix-chain identity, so all siblings of one anchor share a key
//     and read the warmth the anchor's first request wrote.
//   - region — pinning the egress region into the key prevents a cross-region hop
//     from silently routing to a shard that has never seen the prefix.
//
// The empty/zero tuple hashes to a stable value (so a misrouted caller still gets
// a deterministic key rather than none at all), and the separator guards against
// component-concatenation collisions (tenant "ab"+chain "c" vs "a"+"bc").
func AffinityKey(tenant, chain, region string) string {
	h := sha256.New()
	_, _ = h.Write([]byte(strings.TrimSpace(tenant)))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(strings.TrimSpace(chain)))
	_, _ = h.Write([]byte{0})
	_, _ = h.Write([]byte(strings.TrimSpace(region)))
	return hex.EncodeToString(h.Sum(nil))
}

// AffinityHeader is the truncation of AffinityKey a provider routing hint will
// actually accept (OpenAI's prompt_cache_key is bounded; an over-long key is
// silently dropped, collapsing affinity). 32 hex chars (128 bits) is short enough
// for any documented hint field and long enough that two distinct chains colliding
// is a cryptographic impossibility.
func AffinityHeader(tenant, chain, region string) string {
	k := AffinityKey(tenant, chain, region)
	if len(k) > 32 {
		return k[:32]
	}
	return k
}

// HitSample is one affinity key's accumulated hit/miss telemetry over the current
// observation window. The rehash detector reads the correlated collapse of these
// across keys, not any single key's drift.
type HitSample struct {
	Hits   uint64
	Misses uint64
}

// Observe records a hit (cache_read > 0) or miss (cache_read == 0) for one key.
// A miss here is the Law-A1 "believed-warm entry that read 0" signal — the only
// ground truth about warmth the implicit regime ever gets — folded per affinity
// key so the detector can see whether misses are correlated across keys.
func (hs *HitSample) Observe(hit bool) {
	if hit {
		hs.Hits++
	} else {
		hs.Misses++
	}
}

// Samples returns the total observations in the sample (hits + misses).
func (hs *HitSample) Samples() uint64 { return hs.Hits + hs.Misses }

// HitRate returns the sample's hit probability in [0,1]; 0 when empty.
func (hs *HitSample) HitRate() float64 {
	if hs.Samples() == 0 {
		return 0
	}
	return float64(hs.Hits) / float64(hs.Samples())
}

// RehashDetector watches the per-affinity-key hit samples for a CORRELATED p(hit)
// collapse — the signature of an autoscale rehash that invalidated the whole warm
// set at once (Law D3). A single key drifting is normal (eviction, a stale
// belief); MANY keys collapsing together is the rehash, and the right response is
// to re-warm under the new sharding rather than keep issuing cheap recalls that now
// all miss. ShouldRewarm fires when:
//
//   - at least MinKeys distinct affinity keys have each accumulated at least
//     MinSamplesPerKey observations (enough signal to trust the rate), AND
//   - the FRACTION of those keys whose hit rate fell below CollapseThreshold
//     exceeds CorrelatedFraction (e.g. >half of the warm set collapsed together).
//
// All thresholds are caller-set so a calibration (M1) can tune them per provider;
// the defaults (constructed by NewRehashDetector) reproduce the doc's posture.
type RehashDetector struct {
	samples            map[string]*HitSample
	MinKeys            int     // distinct keys required to call a collapse "correlated"
	MinSamplesPerKey   uint64  // observations per key before its rate is trusted
	CollapseThreshold  float64 // a key below this hit rate has "collapsed"
	CorrelatedFraction float64 // fraction of qualified keys collapsed to signal rehash
}

// NewRehashDetector builds a detector with the default thresholds: a collapse is
// "correlated" when more than half of at least 3 keys (each with ≥5 observations)
// drop below a 0.5 hit rate. These are starting hypotheses to be calibrated per
// provider (M1), not ground truth (Law D2).
func NewRehashDetector() *RehashDetector {
	return &RehashDetector{
		samples:            map[string]*HitSample{},
		MinKeys:            3,
		MinSamplesPerKey:   5,
		CollapseThreshold:  0.5,
		CorrelatedFraction: 0.5,
	}
}

// Observe folds one hit/miss for the given affinity key into the detector.
func (d *RehashDetector) Observe(key string, hit bool) {
	hs := d.samples[key]
	if hs == nil {
		hs = &HitSample{}
		d.samples[key] = hs
	}
	hs.Observe(hit)
}

// ShouldRewarm reports whether the observed hit telemetry shows a correlated
// collapse worth a re-warm. Pure over the accumulated samples; caller resets with
// Reset (typically right after issuing the re-warm, to start a fresh window).
func (d *RehashDetector) ShouldRewarm() bool {
	if d.MinKeys <= 0 || d.MinSamplesPerKey == 0 {
		return false
	}
	qualified := 0
	collapsed := 0
	for _, hs := range d.samples {
		if hs.Samples() < d.MinSamplesPerKey {
			continue
		}
		qualified++
		if hs.HitRate() < d.CollapseThreshold {
			collapsed++
		}
	}
	if qualified < d.MinKeys {
		return false
	}
	return float64(collapsed)/float64(qualified) > d.CorrelatedFraction
}

// Reset clears the observation window. Call after acting on a ShouldRewarm signal
// so the next window measures the re-warmed set, not the collapsed one.
func (d *RehashDetector) Reset() {
	d.samples = map[string]*HitSample{}
}

// BurstCap throttles the warming rate per wall-clock window so a concentrated warm
// burst cannot trip the provider's autoscaler and invalidate vCache's own warm set
// (Law D3, last clause). It is a simple fixed-window counter: at most MaxPerWindow
// warms are admitted inside any WindowMillis span; the (WindowMillis+1)th warm in
// the same window is refused and the caller defers it to the next. This is the
// counterpart of the budget's "how many" (PlanWarmBudget) — it bounds "how fast".
type BurstCap struct {
	MaxPerWindow int
	WindowMillis int64
	windowStart  int64
	admitted     int
	started      bool
}

// NewBurstCap builds a cap admitting maxPerWindow warms per windowMillis. The
// defaults a caller reaches for are a small fraction of the tier's headroom rate
// scaled to a short window (e.g. 5 warms / 10s), enough to make progress without
// concentrating load on one shard.
func NewBurstCap(maxPerWindow int, windowMillis int64) *BurstCap {
	return &BurstCap{MaxPerWindow: maxPerWindow, WindowMillis: windowMillis}
}

// Allow admits one warm at nowMillis if the cap allows, advancing the window when
// nowMillis has moved past it. Returns true if the warm may proceed and false if
// the caller must defer it (the load-bearing "don't self-trigger the rehash" gate).
// A dedicated `started` flag distinguishes "first call" from "window starts at t=0"
// so a cap whose first observation lands at nowMillis==0 does not reset on every
// subsequent call within that window.
func (b *BurstCap) Allow(nowMillis int64) bool {
	if b.MaxPerWindow <= 0 || b.WindowMillis <= 0 {
		return false
	}
	if !b.started || nowMillis-b.windowStart >= b.WindowMillis {
		b.started = true
		b.windowStart = nowMillis
		b.admitted = 0
	}
	if b.admitted >= b.MaxPerWindow {
		return false
	}
	b.admitted++
	return true
}
