package compute

import (
	"math"
	"testing"
)

// TestKVEvictionCostPinnedReducesToBaseWithoutPinOrHint proves #2673's headline reduction:
// with no pin (PinBoost == 0), no hint (HintNone), and neither Pinned nor Leased,
// KVEvictionCostPinned is byte-identical to KVEvictionCost — a strict generalization, never a
// divergence, on every input the existing callers already produce.
func TestKVEvictionCostPinnedReducesToBaseWithoutPinOrHint(t *testing.T) {
	cases := []KVSpanStats{
		{Tokens: 100, Bytes: 400, Hits: 0},
		{Tokens: 100, Bytes: 400, Hits: 3},
		{Tokens: 10, Bytes: 40, Hits: 0},
		{Tokens: 512, Bytes: 128, Hits: 7, LastUsed: 42}, // demoted/quantized tier
		{Tokens: 100, Bytes: 0, Hits: 2},                 // unknown footprint → +Inf fail-open
	}
	for _, s := range cases {
		base, pinned := KVEvictionCost(s), KVEvictionCostPinned(s)
		if base != pinned && !(math.IsInf(base, 1) && math.IsInf(pinned, 1)) {
			t.Fatalf("no pin/hint must reduce to KVEvictionCost: base=%v pinned=%v for %+v", base, pinned, s)
		}
	}
}

// TestKVEvictionCostPinnedFullTTLPinIsUnevictable is witness (a): a fresh full-TTL pin keeps
// its span unevictable — today's hard-pin behavior — expressed through the economics term. A
// span carrying a full-TTL PinBoost outscores a cold one-shot span, so the picker chooses the
// cold span as victim, not the boosted one.
func TestKVEvictionCostPinnedFullTTLPinIsUnevictable(t *testing.T) {
	boosted := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 1, PinBoost: PinBoostFromTTL(1000, 1000)}
	cold := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 2} // newer, but no boost
	if KVEvictionCostPinned(boosted) <= KVEvictionCostPinned(cold) {
		t.Fatalf("a full-TTL pin must be more expensive to lose than a cold span: boosted=%v cold=%v",
			KVEvictionCostPinned(boosted), KVEvictionCostPinned(cold))
	}
	if got := PickEvictionVictimPinned([]KVSpanStats{boosted, cold}); got != 1 {
		t.Fatalf("victim must be the cold span (idx 1), not the full-TTL pinned span; got %d", got)
	}
}

// TestKVEvictionCostPinnedExpiringPinReleasesGracefully is witness (b): a near-expired pin
// (PinBoost → 0) on a cold span is now evictable, where the old boolean Pinned would hold it
// forever. As the pin's TTL decays the span rejoins the victim pool once its boost falls
// below a colder competitor's cost — the graceful release a hard boolean cannot express.
func TestKVEvictionCostPinnedExpiringPinReleasesGracefully(t *testing.T) {
	// The expiring pin has almost no TTL left, so its boost is tiny.
	expiring := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 1, PinBoost: PinBoostFromTTL(1, 1_000_000)}
	// A genuinely hotter span (frequently reused) — should be kept over the expiring pin.
	hot := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 50, LastUsed: 2}
	if KVEvictionCostPinned(expiring) >= KVEvictionCostPinned(hot) {
		t.Fatalf("a near-expired pin must be cheaper to lose than a hot span: expiring=%v hot=%v",
			KVEvictionCostPinned(expiring), KVEvictionCostPinned(hot))
	}
	if got := PickEvictionVictimPinned([]KVSpanStats{hot, expiring}); got != 1 {
		t.Fatalf("victim must be the expiring pin (idx 1), not the hot span; got %d", got)
	}
	// Contrast with a HARD Pinned span, which stays an absolute exclusion regardless of how
	// hot the competitor is — the boolean holds it forever.
	hardPinned := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 1, Pinned: true}
	if got := PickEvictionVictimPinned([]KVSpanStats{hardPinned, hot}); got != 1 {
		t.Fatalf("a hard Pinned span is never a victim: want idx 1 (the hot span), got %d", got)
	}
}

// TestKVEvictionCostPinnedLeasedIsAbsoluteExclusion is witness (c): an in-flight Leased span
// stays an absolute exclusion (+Inf) regardless of boost or hint — the correctness floor.
// Reclaiming a span being served would corrupt an active decode, so economics never apply.
func TestKVEvictionCostPinnedLeasedIsAbsoluteExclusion(t *testing.T) {
	leased := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, Leased: true, PinBoost: 0, Hint: HintEphemeral}
	if got := KVEvictionCostPinned(leased); !math.IsInf(got, 1) {
		t.Fatalf("a Leased span must score +Inf regardless of boost/hint; got %v", got)
	}
	// Even with an ephemeral hint (which lowers keep-value) and zero boost, a leased span is
	// never chosen while an unleased candidate exists.
	victim := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, LastUsed: 5}
	if got := PickEvictionVictimPinned([]KVSpanStats{leased, victim}); got != 1 {
		t.Fatalf("victim must be the unleased span (idx 1), never the leased one; got %d", got)
	}
	// A Pinned span is likewise +Inf — the hard-pin limit.
	pinned := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, Pinned: true}
	if got := KVEvictionCostPinned(pinned); !math.IsInf(got, 1) {
		t.Fatalf("a Pinned span must score +Inf (the PinBoost→+Inf limit); got %v", got)
	}
}

// TestKVEvictionCostPinnedHintIsBoundedPrior is witness (d): an ephemeral hint LOWERS a
// span's keep-value and a precious hint RAISES it, but both are bounded so neither defeats a
// genuinely hotter/colder span — the adjudication guard against an agent gaming the cache.
func TestKVEvictionCostPinnedHintIsBoundedPrior(t *testing.T) {
	base := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 2}
	none := KVEvictionCostPinned(base)
	precious := KVEvictionCostPinned(KVSpanStats{Tokens: 10, Bytes: 40, Hits: 2, Hint: HintPrecious})
	ephemeral := KVEvictionCostPinned(KVSpanStats{Tokens: 10, Bytes: 40, Hits: 2, Hint: HintEphemeral})
	if !(ephemeral < none && none < precious) {
		t.Fatalf("hint must order ephemeral < none < precious: e=%v n=%v p=%v", ephemeral, none, precious)
	}
	// Bounded, direction 1: an ephemeral hint on a genuinely HOT span still outranks a much
	// colder unhinted span — the hint cannot evict a hot span.
	hotEphemeral := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 100, Hint: HintEphemeral}
	coldPlain := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0}
	if KVEvictionCostPinned(hotEphemeral) <= KVEvictionCostPinned(coldPlain) {
		t.Fatalf("ephemeral hint must not defeat a genuinely hotter span: hotEph=%v coldPlain=%v",
			KVEvictionCostPinned(hotEphemeral), KVEvictionCostPinned(coldPlain))
	}
	// Bounded, direction 2: a precious hint on a genuinely COLD span still loses to a much
	// hotter unhinted span — the hint cannot pin a cold span into permanence.
	coldPrecious := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 0, Hint: HintPrecious}
	hotPlain := KVSpanStats{Tokens: 10, Bytes: 40, Hits: 100}
	if KVEvictionCostPinned(coldPrecious) >= KVEvictionCostPinned(hotPlain) {
		t.Fatalf("precious hint must not defeat a genuinely colder span: coldPrec=%v hotPlain=%v",
			KVEvictionCostPinned(coldPrecious), KVEvictionCostPinned(hotPlain))
	}
}

// TestPinBoostFromTTLMonotoneDecay proves PinBoostFromTTL's contract: monotone
// non-decreasing in remainingMs, full strength at/above the lease, zero at/below expiry, and
// zero for a non-lease — the consistent TTL → boost curve every caller shares.
func TestPinBoostFromTTLMonotoneDecay(t *testing.T) {
	const lease = 1000
	full := PinBoostFromTTL(lease, lease)
	half := PinBoostFromTTL(lease/2, lease)
	tiny := PinBoostFromTTL(1, lease)
	if !(full > half && half > tiny && tiny > 0) {
		t.Fatalf("boost must decay monotonically as TTL shrinks: full=%v half=%v tiny=%v", full, half, tiny)
	}
	if got := PinBoostFromTTL(0, lease); got != 0 {
		t.Fatalf("expired pin (remaining 0) must have zero boost; got %v", got)
	}
	if got := PinBoostFromTTL(-5, lease); got != 0 {
		t.Fatalf("negative remaining must have zero boost; got %v", got)
	}
	if got := PinBoostFromTTL(lease*2, lease); got != full {
		t.Fatalf("remaining beyond the lease must clamp to full strength; got %v want %v", got, full)
	}
	if got := PinBoostFromTTL(lease, 0); got != 0 {
		t.Fatalf("no lease (leaseMs 0) must have zero boost; got %v", got)
	}
	if got := PinBoostFromTTL(lease, -1); got != 0 {
		t.Fatalf("negative lease must have zero boost; got %v", got)
	}
	// Half-TTL is exactly half strength — the linear decay curve.
	if math.Abs(half-full/2) > 1e-9 {
		t.Fatalf("half-TTL must be half strength (linear decay): half=%v full/2=%v", half, full/2)
	}
}
