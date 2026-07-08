package accounts

// headroomtier.go — the ONE definition of the headroom-score tier contract the whole
// seat-selection subsystem reads. A per-account headroom score (see RotationHeadroom) is not
// a continuous quota; its SIGN is a three-way offerability tier, and that tier is what every
// consumer actually branches on:
//
//	> 0   OFFERABLE — available, not throttled (the (1,2] band cmd/fak/accounts_headroom.go
//	                  assigns: OfferableBase plus a within-tier least-loaded-first bonus).
//	== 0  UNKNOWN   — no runtime signal (UnknownScore); neither proven room nor proven wall.
//	< 0   WALLED    — usage-throttled or blocked (the [-1,0) band: WalledBase plus a
//	                  within-tier soonest-reset-first bonus).
//
// Before this file the boundaries lived only as a prose comment in accounts_headroom.go plus
// raw sign comparisons re-derived independently in three read helpers (hasRoom,
// rotationSeatRuntimeWalled, headroomLabel) and the -1/+1/0 band bases restated as literals at
// each producer site. Classify + the band anchors + HeadroomLabel give producer and
// interpreter one source of truth across the package seam, so a band retune or a new consumer
// can never silently disagree about where a tier begins.

// The band anchors: the base score each tier is built from. The producers in
// cmd/fak/accounts_headroom.go add a within-tier sub-band bonus that never leaves the tier (so
// an OFFERABLE bucket stays strictly > 0 and a WALLED one strictly < 0). These consts are those
// bases, kept here beside the sign contract they must satisfy rather than as bare literals at
// each site. They are untyped so they compose in float64 arithmetic without conversion.
const (
	// WalledBase is the floor of the WALLED band: a usage-throttled/blocked bucket scores
	// WalledBase plus a reset-soonness bonus in [0,1), so it stays strictly < 0.
	WalledBase = -1.0
	// UnknownScore is the neutral UNKNOWN tier: no runtime signal, neither preferred nor
	// penalised. It is also the walled/non-walled boundary the producers clamp a usage-capped
	// row against.
	UnknownScore = 0.0
	// OfferableBase is the base of the OFFERABLE band: an available bucket scores OfferableBase
	// plus a load bonus in (0,1], so it stays strictly > 0.
	OfferableBase = 1.0
)

// Tier is the three-way offerability class a headroom score encodes — the closed vocabulary a
// consumer should branch on instead of re-deriving a sign comparison.
type Tier int

const (
	// TierWalled — score < 0: usage-throttled or blocked; never rotate onto it.
	TierWalled Tier = iota - 1 // -1
	// TierUnknown — score == 0: no runtime signal; launchable but not proven to have room.
	TierUnknown // 0
	// TierOfferable — score > 0: available with room; the only tier proven safe to prefer.
	TierOfferable // 1
)

// Classify maps a concrete headroom score to its Tier by sign — the single boundary rule
// (< 0 walled, == 0 unknown, > 0 offerable). It takes a concrete float64, NOT a *float64: a
// nil pointer means "no signal", which is a distinct caller-level state (neither walled nor
// offerable). The two consumers that hold a *float64 keep their own nil-guard and call Classify
// only once they have a value, so the nil case is never misread as a tier.
func Classify(score float64) Tier {
	switch {
	case score < 0:
		return TierWalled
	case score > 0:
		return TierOfferable
	default:
		return TierUnknown
	}
}

// HeadroomLabel renders a headroom score as a short, honest word ("room" / "walled" /
// "unknown") for operator one-liners. It keys off the Tier, never the fraction, so the label
// reads as an offerability tier rather than leaking a false-precision quota number. It is the
// single home for the sign→word mapping; cmd/fak surfaces call through it so the vocabulary
// cannot drift from Classify.
func HeadroomLabel(score float64) string {
	switch Classify(score) {
	case TierOfferable:
		return "room"
	case TierWalled:
		return "walled"
	default:
		return "unknown"
	}
}
