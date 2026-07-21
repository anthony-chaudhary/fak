package gateway

// tier_overlap_credit.go — the tier-weighted overlap credit term for the fleet
// residency scorer (issue #5272, epic #50, borrow: NVIDIA Dynamo selector, INSPIRE
// / clean-room Go over an Apache-2.0 source).
//
// CacheAwarePolicy.score (residency_router.go) credits a worker's held prefix
// overlap as a PLAIN block count — overlap-length times inverse-load — treating
// every held block as equally valuable regardless of which memory tier holds it.
// But a block sitting on-device is free to reuse, while the same block evicted down
// to disk must be paged up first. Dynamo's selector credits an overlap hit by how
// expensive it is to reach: an on-device block is full credit, a host-pinned block
// a fraction, a disk block a smaller fraction, a shared remote block in between.
//
// This file supplies ONLY the pure term: given the held overlap a worker has for a
// request AND the memory tier each held run sits in, it sums a tier-weighted credit
// (higher credit = better placement). It REPLACES the raw block-count in the score
// (it does not add on top), so folding it in never double-counts the same overlap.
// Deterministic and wall-clock-free — no network, no GPU, no time source.

import "math"

// OverlapTier names which memory tier a held overlap run currently sits in. A run on
// a faster tier is cheaper to reuse, so it earns more credit for the same length.
type OverlapTier int

const (
	// TierDevice is the fastest tier: the overlap is resident on-device, free to
	// reuse with no page-up. Full credit.
	TierDevice OverlapTier = iota
	// TierHost is host-pinned memory: reusable after a device copy-up, so it earns a
	// fraction of the device credit.
	TierHost
	// TierShared is an external shared remote span: reachable over the fabric, priced
	// between host and disk.
	TierShared
	// TierDisk is the slowest tier: the overlap was evicted to disk or SSD and must be
	// paged up, so it earns the smallest fraction.
	TierDisk
)

// TierWeights is the per-tier credit multiplier the scorer folds a held overlap
// through. Each weight scales one tier's held block count into its contribution.
// Defaults follow the borrowed selector: device 1.0, host 0.75, shared 0.5,
// disk 0.25 — a faster tier is worth strictly more per held block.
type TierWeights struct {
	Device float64
	Host   float64
	Shared float64
	Disk   float64
}

// DefaultTierWeights returns the standing multipliers (device 1.0 / host 0.75 /
// shared 0.5 / disk 0.25). Documented so an operator can retune the tier trade-off
// without reading the code.
func DefaultTierWeights() TierWeights {
	return TierWeights{Device: 1.0, Host: 0.75, Shared: 0.5, Disk: 0.25}
}

// valid reports whether every weight is a finite, non-negative number. A bad
// (NaN/Inf) or negative weight fails the whole set closed, so a misconfigured tier
// table can never turn a held overlap into a negative or nonsense credit.
func (w TierWeights) valid() bool {
	for _, v := range []float64{w.Device, w.Host, w.Shared, w.Disk} {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 {
			return false
		}
	}
	return true
}

// weightFor returns the multiplier for a known tier and ok=false for an unknown
// tier value, so an out-of-range tag contributes nothing rather than a wrong credit.
func (w TierWeights) weightFor(t OverlapTier) (float64, bool) {
	switch t {
	case TierDevice:
		return w.Device, true
	case TierHost:
		return w.Host, true
	case TierShared:
		return w.Shared, true
	case TierDisk:
		return w.Disk, true
	default:
		return 0, false
	}
}

// TierHeldOverlap is one contiguous held run of a request's prefix: how many blocks
// the worker holds (Blocks) and which memory tier holds them (Tier). A worker's
// total overlap for a request is a list of these runs, one per tier the overlap
// spans across.
type TierHeldOverlap struct {
	Tier   OverlapTier
	Blocks int
}

// TierWeightedOverlapCredit sums the tier-weighted credit of a worker's held overlap:
// for each held run, its block count times its tier weight. Properties the witness
// pins:
//   - Monotone in overlap: more held blocks in the same tier yield at least as much
//     credit (never less).
//   - Faster tier wins: the same held length earns more credit on a faster tier than
//     on a slower one, since the faster tier's weight is larger.
//   - Zero base: an empty held list (or all-empty runs) earns zero credit.
//   - Fail closed: a bad or negative weight table yields zero credit; an unknown tier
//     tag or a non-positive block count contributes nothing.
func TierWeightedOverlapCredit(w TierWeights, held []TierHeldOverlap) float64 {
	if !w.valid() {
		return 0
	}
	total := 0.0
	for _, h := range held {
		if h.Blocks <= 0 {
			continue
		}
		wt, ok := w.weightFor(h.Tier)
		if !ok {
			continue
		}
		total += float64(h.Blocks) * wt
	}
	return total
}

// ScoreWithTierWeightedOverlap folds a tier-weighted overlap credit into the same
// inverse-load shape the residency scorer already uses: credit / (1 + load). It
// stands in for the raw overlap-length term rather than adding onto it, so a request
// is never credited twice for the same held overlap. A non-positive credit (a cold
// worker, or a fail-closed term) scores zero; a negative load clamps to zero so load
// only ever discounts.
func ScoreWithTierWeightedOverlap(credit float64, load int) float64 {
	if credit <= 0 || math.IsNaN(credit) || math.IsInf(credit, 0) {
		return 0
	}
	if load < 0 {
		load = 0
	}
	return credit / float64(1+load)
}
