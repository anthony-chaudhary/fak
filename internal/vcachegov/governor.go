package vcachegov

import (
	"math"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// governor.go is the pin/lazy/evict classifier — the heart of the vCache Governor
// (issue #720, acceptance 1). For each cacheable prefix it answers one question:
// given the prefix's arrival rate λ, its TTL T, and the latency / rate-limit value
// of keeping it warm, what should the steady-state policy DO with it?
//
// The policy is the reconciled form from the design note's §10 panel, NOT the
// first-pass §5.4 rule. The first-pass "pin iff λR > P/T" was the expensive-recall
// special case; the canonical result is that on PURE cache dollars lazy rebuild
// WEAKLY DOMINATES for every λ (the savings gap collapses to zero only as λT→∞).
// Pinning a prefix with a continuous heartbeat is justified ONLY by something pure
// dollars ignore:
//
//   - latency value L  — the cold first-token-TTFT a heartbeat pulls off the
//     critical path, priced in per-request cost units; and
//   - rate-limit shadow price μ — a warmed prefix absorbs arrivals at the cheap
//     cache-read price instead of burning scarce RPM/TPM headroom, so keeping it
//     warm relaxes the rate-limit constraint everything else fights under.
//
// Those tip the balance back toward pinning exactly when
//
//	λT > ln((w + μ + L) / L)          (the §10 pin cutoff)
//
// where w is the cache-WRITE cost multiplier for the chosen TTL (1.25× for the
// Anthropic 5m tier, 2.0× for the 1h tier). With w=1.25, L=1, μ=0 the cutoff is
// ln(2.25) = 0.811 — the doc's "≈0.81 → ~1 req/TTL"; under rate pressure (μ>0) it
// rises toward ln(4.25)=1.45, the doc's "~1.2–1.5". This package computes that
// cutoff exactly (PinThreshold) so a test can pin it to the decimal.
//
// The classifier is pure and deterministic; the caller injects nowMillis, the same
// testable posture as cachemeta.Lifecycle.Advance.

// TTLWindows pins the two Anthropic prompt-cache TTLs the Governor chooses between.
// They are the constants the §5 economics table and the §10 reconciliation use; a
// future M1 calibration may override them per provider, but the defaults reproduce
// every number in the design note.
const (
	TTL5MinutesMillis int64 = 5 * 60 * 1000
	TTL1HourMillis    int64 = 60 * 60 * 1000
)

// Write-cost multipliers (w) for the two TTL windows, straight from the §5 table:
// the 5m cache-write is 1.25× a fresh input token; opting into the 1h tier doubles
// the write to 2.0× but buys ~7.5× cheaper HOLD per hour (2·P vs ~15·P) — see
// PreferLongTTL.
const (
	WriteMult5Minutes float64 = 1.25
	WriteMult1Hour    float64 = 2.0
)

// GovernorDecision is the steady-state verdict the Governor returns for one prefix.
type GovernorDecision string

const (
	// DecisionRideNatural — λT ≥ 1: ≥1 request arrives per TTL window, so natural
	// traffic refreshes the prefix before it can expire. Do nothing: no pin, no
	// dedicated pre-warm. This is the dominant, free win (§4c).
	DecisionRideNatural GovernorDecision = "ride_natural"
	// DecisionHeartbeatPin — λT < 1 but the pin cutoff (§10) is met: natural traffic
	// alone would let the prefix lapse between bursts, and the latency / rate-limit
	// value of keeping it warm clears the bar. Maintain it with a heartbeat prefill
	// once per TTL window.
	DecisionHeartbeatPin GovernorDecision = "heartbeat_pin"
	// DecisionLazyRebuild — λT < 1 and the pin cutoff is NOT met: on pure cache
	// dollars lazy rebuild weakly dominates, so let the prefix lapse and pay the
	// recall on the next miss. The default for everything that is not hot and not
	// valuable enough to pin.
	DecisionLazyRebuild GovernorDecision = "lazy_rebuild"
	// DecisionEvict — cold: no arrivals and idle past the TTL. Drop it from the
	// manifest; a future recall re-admits it as a fresh entry.
	DecisionEvict GovernorDecision = "evict"
	// DecisionNoCache — the prefix carries secret/PII/regulated content (Law D4)
	// that must NEVER be warmed into a provider prefix cache. It takes the no-cache
	// path: the full prefix is always re-sent; no breakpoint precedes a secret byte.
	DecisionNoCache GovernorDecision = "no_cache"
	// DecisionExplicitCache — the prefix is regulated (not a bare secret) and may be
	// cached ONLY through a deletion-capable surface (e.g. Gemini CachedContent),
	// never through an implicit/auto prefix cache. Implicit warming is refused here;
	// the caller must route the explicit-cache-with-deletion path itself.
	DecisionExplicitCache GovernorDecision = "explicit_cache"
)

// PrefixStats is everything the classifier needs about one prefix. It is projected
// from the cachemeta substrate: ArrivalRatePerSec and LastAccessMillis come from a
// cachemeta.Lifecycle (see PrefixStatsFromLifecycle), and the TTL/write fields from
// the provider's TierTTL. The latency/rate-limit fields (L, μ) are the M1
// calibration outputs that turn pure dollars into the full pin decision.
type PrefixStats struct {
	// ArrivalRatePerSec is λ — the prefix's observed arrival rate. Reuse
	// cachemeta.Lifecycle.AccessRatePerSec; the classifier multiplies by the TTL
	// window to get expected arrivals per TTL (λT).
	ArrivalRatePerSec float64
	// TTLMillis is T — the TTL window the prefix would be held under (5m or 1h).
	TTLMillis int64
	// WriteMult is w — the cache-write cost multiplier for TTLMillis (1.25 for 5m,
	// 2.0 for 1h). Used in the §10 pin cutoff.
	WriteMult float64
	// LatencyValue is L — the per-request cost-unit value of pulling the cold
	// first-token TTFT off the critical path by keeping this prefix warm. L<=0 means
	// "pure cache dollars" and DISABLES pinning (lazy weakly dominates for all λ).
	LatencyValue float64
	// RateShadowPrice is μ — the per-request cost-unit value of the rate-limit
	// headroom a warmed prefix frees up by absorbing arrivals at the cache-read
	// price. μ>=0; raising it lifts the pin cutoff toward ~1.2–1.5.
	RateShadowPrice float64
	// Secret is the prefix's Law-D4 content classification. A non-cacheable prefix
	// short-circuits to NoCache/ExplicitCache before the λT math runs.
	Secret SecretClassification
	// LastAccessMillis is the most recent real access (a Touch). Used with NowMillis
	// to decide coldness (evict).
	LastAccessMillis int64
	// NowMillis is the caller-injected clock, the same posture as Lifecycle.Advance.
	NowMillis int64
}

// PrefixStatsFromLifecycle projects the cachemeta-driven fields of PrefixStats from
// a cachemeta.Lifecycle — the bridge that lets the Governor rank and classify the
// SAME entries the warm set already tracks. ArrivalRatePerSec reuses the lifecycle's
// lifetime reuse intensity; LastAccessMillis reuses its most-recent Touch. The
// economics fields (TTL, write cost, L, μ) and the secret classification are the
// caller's to supply — they come from M1 calibration and the canonicalizer, not
// from the lifecycle record.
func PrefixStatsFromLifecycle(lc cachemeta.Lifecycle, ttlMillis int64, writeMult, latencyValue, rateShadowPrice float64, secret SecretClassification, nowMillis int64) PrefixStats {
	return PrefixStats{
		ArrivalRatePerSec: lc.AccessRatePerSec(nowMillis),
		TTLMillis:         ttlMillis,
		WriteMult:         writeMult,
		LatencyValue:      latencyValue,
		RateShadowPrice:   rateShadowPrice,
		Secret:            secret,
		LastAccessMillis:  lc.LastAccessMillis,
		NowMillis:         nowMillis,
	}
}

// PinThreshold returns the λT above which heartbeat-pinning beats lazy rebuild,
// exactly as the §10 reconciliation prescribes: pin iff
//
//	λT > ln((w + μ + L) / L).
//
// With the Anthropic-5m write cost w=1.25, L=1, μ=0 this is ln(2.25)=0.8109 (the
// doc's "≈0.81 → ~1 req/TTL"); under rate pressure (μ=2, L=1) it is ln(4.25)=1.447
// (the doc's "~1.2–1.5"). L<=0 is the pure-cache-dollars regime where lazy weakly
// dominates for ALL λ, so the threshold is +Inf and pinning never fires.
func PinThreshold(writeMult, latencyValue, rateShadowPrice float64) float64 {
	if latencyValue <= 0 {
		return math.Inf(1)
	}
	return math.Log((writeMult + rateShadowPrice + latencyValue) / latencyValue)
}

// Classify returns the Governor verdict for one prefix. The decision tree, top to
// bottom:
//
//  1. Secret/regulated (Law D4) short-circuits BEFORE any economics: a secret
//     prefix is never warmed, a regulated prefix only via a deletion-capable
//     surface. No λT math can override content safety.
//  2. Cold — no arrivals and idle past the TTL — evicts from the manifest.
//  3. λT ≥ 1 — natural traffic refreshes the prefix within every TTL window, so
//     ride it for free (no pin, no dedicated warm).
//  4. λT < 1 — lazy rebuild weakly dominates on pure dollars; pin ONLY when the
//     §10 cutoff (latency value L / rate-limit shadow price μ) is cleared.
//  5. Otherwise lazy rebuild: let it lapse, pay the recall on the next miss.
//
// Correctness never depends on the outcome (Law A2): whatever the verdict, the
// caller must always be able to re-send the full prefix; a pin is only ever a
// cost/latency win, never a license to elide resent context.
func Classify(s PrefixStats) GovernorDecision {
	if !Warmable(s.Secret) {
		if s.Secret == SecretRegulated {
			return DecisionExplicitCache
		}
		return DecisionNoCache
	}
	ttlSec := float64(s.TTLMillis) / 1000.0
	lambdaT := s.ArrivalRatePerSec * ttlSec
	idleMillis := s.NowMillis - s.LastAccessMillis
	if lambdaT <= 0 && s.TTLMillis > 0 && idleMillis > s.TTLMillis {
		return DecisionEvict
	}
	if lambdaT >= 1 {
		return DecisionRideNatural
	}
	if lambdaT > PinThreshold(s.WriteMult, s.LatencyValue, s.RateShadowPrice) {
		return DecisionHeartbeatPin
	}
	return DecisionLazyRebuild
}

// PreferLongTTL reports whether the 1h TTL beats the 5m TTL for a prefix the
// Governor has decided to heartbeat-pin (issue #720 acceptance 5). The 1h opt-in
// trades a 2× write (w1h=2.0) for ~7.5× cheaper HOLD per hour — one 2·P write
// instead of twelve 1.25·P heartbeats (=15·P) — and 12× fewer heartbeat requests
// burning rate-limit headroom. It wins whenever the planning horizon spans enough
// 5m windows that the heartbeat stream outcosts the single 1h write, AND the prefix
// is bursty enough that its idle gaps exceed the 5m TTL (otherwise natural traffic
// holds it under 5m for free and the pricier 1h write is wasted).
//
// Concretely: 1h is preferred iff pinned AND gapMillis > TTL5MinutesMillis AND
// ceil(horizonMillis / TTL5MinutesMillis) * WriteMult5Minutes > WriteMult1Hour.
// For a 1h horizon that is 12 * 1.25 = 15 > 2 → prefer 1h; for a 5m horizon 1 *
// 1.25 = 1.25 < 2 → keep 5m. A non-pinned prefix (ride-natural or lazy) has no
// heartbeat stream to amortize, so the cheaper 5m write (or no write at all) wins.
func PreferLongTTL(pinned bool, horizonMillis, gapMillis int64) bool {
	if !pinned {
		return false
	}
	if gapMillis <= TTL5MinutesMillis {
		return false
	}
	if horizonMillis <= 0 {
		return false
	}
	heartbeats5m := int64(math.Ceil(float64(horizonMillis) / float64(TTL5MinutesMillis)))
	return float64(heartbeats5m)*WriteMult5Minutes > WriteMult1Hour
}

// IsPinned reports whether a decision keeps a prefix warm with a heartbeat stream
// (the decisions whose hold cost the 1h trade-off is meaningful for). RideNatural
// holds warm for free via natural traffic, so it is not "pinned" in this sense.
func (d GovernorDecision) IsPinned() bool { return d == DecisionHeartbeatPin }
