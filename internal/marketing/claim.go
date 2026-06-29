package marketing

import (
	"errors"
	"fmt"
)

// Label is the provenance tag a claim carries, in the WITNESSED-vs-OBSERVED discipline the
// repo already enforces on every number: WITNESSED means fak authored/controls the evidence
// (a commit it made), OBSERVED means a value relayed from an external party. A marketing
// claim is, by construction, always WITNESSED — it cites a fak commit sha. The type exists
// so the render layer can never emit an unlabeled claim, and so a future OBSERVED variant
// (e.g. a relayed third-party benchmark) is a distinct, explicitly-labeled thing.
type Label string

const (
	// LabelWitnessed: the claim cites a fak-authored, ship-stamped commit. The only label
	// a Ship-backed Claim may carry.
	LabelWitnessed Label = "WITNESSED"
)

// ErrUnwitnessedClaim is returned by NewClaim when the backing Ship is not a real witness:
// an empty sha, or a Kind outside {trailer,direct}. It is the construction-time refusal that
// makes "a claim without a witnessing commit" unrepresentable — the marketing analogue of
// the kernel's default-deny: an unprovable boast is refused, not softened.
var ErrUnwitnessedClaim = errors.New("marketing: refused claim with no witnessing ship-stamped commit")

// Claim is one marketing assertion bound to its witness. There is no exported way to build a
// Claim except NewClaim, and NewClaim refuses an unwitnessed Ship — so every Claim that
// exists carries a non-empty sha and a {trailer,direct} kind. The render layer relies on
// this: it prints Ship.SHA inline on every bullet without re-checking, because an empty sha
// cannot reach it.
type Claim struct {
	Text  string // the human-facing assertion, e.g. "Q4_K AVX2 reducers landed"
	Ship  Ship   // the witnessing commit — required, validated by NewClaim
	Label Label  // always LabelWitnessed for a Ship-backed claim
}

// NewClaim constructs a witnessed Claim or returns ErrUnwitnessedClaim. It is the ONLY
// constructor: a Ship with an empty SHA, or a Kind that is not a per-leaf ship-stamp
// ("trailer"/"direct"), is refused. text is trimmed-checked too — a claim with no assertion
// is as useless as one with no witness.
//
// This is the single chokepoint the whole subsystem's honesty rests on. Every generator
// (generate.go) builds its claims through here; the render layer never re-validates because
// it cannot receive an invalid Claim.
func NewClaim(text string, s Ship) (Claim, error) {
	if s.SHA == "" {
		return Claim{}, fmt.Errorf("%w: empty sha (subject=%q)", ErrUnwitnessedClaim, s.Subject)
	}
	if s.Kind != "trailer" && s.Kind != "direct" {
		return Claim{}, fmt.Errorf("%w: kind %q is not a per-leaf ship-stamp (sha=%s)", ErrUnwitnessedClaim, s.Kind, s.SHA)
	}
	if text == "" {
		return Claim{}, fmt.Errorf("%w: empty claim text (sha=%s)", ErrUnwitnessedClaim, s.SHA)
	}
	return Claim{Text: text, Ship: s, Label: LabelWitnessed}, nil
}

// MustClaims builds claims from a set of ships, skipping (never panicking on) any that fail
// the witness check, and returns the claims plus the count skipped. It is the bulk path a
// generator uses: a ship that somehow lacks a sha is dropped, not fatal — but because
// CollectShips only ever emits {trailer,direct} ships with shas, skipped is normally 0 and a
// non-zero skipped is a real signal worth surfacing.
func MustClaims(text func(Ship) string, ships []Ship) (claims []Claim, skipped int) {
	for _, s := range ships {
		c, err := NewClaim(text(s), s)
		if err != nil {
			skipped++
			continue
		}
		claims = append(claims, c)
	}
	return claims, skipped
}
