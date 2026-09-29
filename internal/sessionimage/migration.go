// migration.go — the WITNESS-FENCE sibling: the fail-closed ACR fence adapter a restore
// consults before re-firing an effect after an agent-checkpoint-restore (#12815). It is a
// deliberately self-contained successor over witness.go's trunk-native `(*Image).Witness()`.
//
// The semantic-rollback failure it defends against re-executes a side effect (a
// payment, a DB write) the predecessor already performed, because the restoring
// framework cannot tell ALREADY-COMPLETED from NEEDS-REPLAYING. This adapter reads
// the restored image's integrity-checked keep-bits — the taskmgr.VerifiedDone rung
// persisted into witness.json — and turns them into a closed dispatch decision.
//
// It is fail-closed by construction: an unreadable completion-evidence journal yields
// an explicit ACRFenceRefuse, never an empty done set that would authorize a duplicate
// effect. That reader-layer guarantee comes from Image.Witness() (landed via #12822),
// which fails closed on a missing indexed part, a version mismatch, or a digest that
// does not match the bytes; this adapter only lifts its entries into a decision and
// folds any read error into a refusal.
package sessionimage

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/taskmgr"
)

// WitnessedDoneSet reads the image's integrity-checked keep-bits and returns the set of
// effect ids the predecessor ALREADY witnessed as VerifiedDone. Unreadable evidence is
// returned to the caller so it cannot be mistaken for a valid empty done set.
//
// Only a VerifiedDone rung enters the set. A present-but-unverified record
// (VerifiedRefused / VerifiedUnavailable / the zero VerifiedUnknown) is deliberately NOT
// treated as done — mirroring witness.go's VerifiedDone, a duplicate is skipped only on
// positive evidence, never on an absent or contradicted one.
func (img *Image) WitnessedDoneSet() (map[string]struct{}, error) {
	if img == nil || strings.TrimSpace(img.Dir) == "" {
		return nil, fmt.Errorf("sessionimage: ACR fence: image has no evidence directory")
	}
	entries, err := img.Witness()
	if err != nil {
		return nil, fmt.Errorf("sessionimage: ACR fence: read witness evidence: %w", err)
	}
	done := make(map[string]struct{}, len(entries))
	for _, e := range entries {
		if e.Record.VerifiedState == taskmgr.VerifiedDone {
			done[e.EffectID] = struct{}{}
		}
	}
	return done, nil
}

// ACRFenceVerdict is the closed dispatch decision. Refusal is the ZERO value so a nil or
// uninitialized guard cannot authorize an effect.
type ACRFenceVerdict uint8

const (
	ACRFenceRefuse        ACRFenceVerdict = iota // zero value: evidence unreadable / guard unready
	ACRFenceFire                                 // no keep-bit: the effect may run
	ACRFenceSkipDuplicate                        // a valid VerifiedDone keep-bit: do NOT replay
)

// String renders the verdict for logs and receipts; an out-of-range value reads UNKNOWN.
func (v ACRFenceVerdict) String() string {
	switch v {
	case ACRFenceRefuse:
		return "REFUSE"
	case ACRFenceFire:
		return "FIRE"
	case ACRFenceSkipDuplicate:
		return "SKIP_DUPLICATE"
	default:
		return "UNKNOWN"
	}
}

// ACRFenceGuard is a restore-side gate over a predecessor image's keep-bits. It is
// fail-closed: a zero or nil guard refuses, and a guard built from unreadable evidence
// carries its refusal error so an accidental dispatch still cannot fire a duplicate.
type ACRFenceGuard struct {
	done    map[string]struct{}
	refusal error
	ready   bool
}

// NewACRFenceGuard builds the guard from a restored image's keep-bits. It returns a guard
// even on error; that guard stays fail-closed (REFUSE) if a caller accidentally dispatches.
func NewACRFenceGuard(img *Image) (*ACRFenceGuard, error) {
	done, err := img.WitnessedDoneSet()
	if err != nil {
		return &ACRFenceGuard{refusal: err}, err
	}
	return &ACRFenceGuard{done: done, ready: true}, nil
}

// Decide returns SKIP_DUPLICATE for an effect already witnessed done, FIRE for an effect
// with no keep-bit, and REFUSE for a nil/zero/evidence-failed guard.
func (g *ACRFenceGuard) Decide(effectID string) ACRFenceVerdict {
	if g == nil || !g.ready || g.refusal != nil {
		return ACRFenceRefuse
	}
	if _, ok := g.done[effectID]; ok {
		return ACRFenceSkipDuplicate
	}
	return ACRFenceFire
}

// DecideAll maps Decide over a batch in order; the result is parallel to ids.
func (g *ACRFenceGuard) DecideAll(ids []string) []ACRFenceVerdict {
	out := make([]ACRFenceVerdict, len(ids))
	for i, id := range ids {
		out[i] = g.Decide(id)
	}
	return out
}
