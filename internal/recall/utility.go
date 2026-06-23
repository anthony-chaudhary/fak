package recall

// Outcome-utility second-phase recall (#540).
//
// fak recall is provenance/quarantine-gated but, before this leaf, NOT outcome-
// utility-weighted: Recall() ranked benign pages by lexical overlap alone, so a
// semantically on-topic but historically useless (or misleading) page ranked
// identically to one that had repeatedly helped. This file closes the loop from a
// WITNESSED task outcome back to which experiences get recalled next time — MemRL's
// Intent-Experience-Utility idea, re-scoped with the fak/honesty twist: utility is
// learned from WITNESSED outcomes (never the agent's self-report), and provenance
// gates the LEARNING, not just the read.
//
// The kernel twist MemRL has no analog for: a witness-refuted or re-quarantined page
// can never accrue positive utility, and any utility it had is zeroed when its
// witness is revoked (the dream seal path, dream.go). Utility re-ranks WITHIN the
// already-provenance-clean candidate set (recall.go Recall phase 2); it can never
// resurrect a sealed page or override a deny. Fail-closed: an unwitnessed (self-
// asserted) outcome is refused, and unknown utility = neutral, matching
// abi.FallbackDeny.

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/vdso"
)

// UtilityMax bounds the per-page learned utility scalar. It is the headroom phase-2
// re-ranking has over phase-1 relevance: utility re-orders candidates across a
// bounded relevance gap (so a repeatedly-helpful page can outrank a more-lexically-
// matching but useless one), but a single witnessed outcome can never let a barely-
// relevant page dominate a strongly-relevant one. Neutral is the floor (0), so an
// uncredited or demoted page is never ranked below a never-seen page.
const UtilityMax = 4.0

// ErrUnwitnessed is returned by Credit when the offered outcome carries no external
// witness — a self-asserted success. Utility may be learned ONLY from a witnessed
// outcome (a git-verified `(fak <leaf>)` ship, or a graded ResultAdmitter verdict),
// so a self-report is refused fail-closed (matching abi.FallbackDeny posture). It
// is distinct from ErrSealed, which the trust/quarantine gates raise.
var ErrUnwitnessed = errors.New("recall: utility credit refused — outcome is self-asserted, not witnessed")

// Outcome is a task result offered as utility evidence for the pages that produced
// it. The honesty gate lives in its Witness field: the credit is applied ONLY if
// Witness names a live external trust authority (the SAME evidence the dos verify
// referee already trusts — a git-witnessed ship leaf, or a graded admitter verdict
// id). An empty Witness is a self-report and is refused (ErrUnwitnessed); a revoked
// Witness is refused (ErrSealed). The store thus cannot up-weight the memories a
// poisoned/stale/unverified session used.
type Outcome struct {
	// Witness is the external trust token the outcome was graded under. Empty ==
	// self-asserted == refused. It is checked against the live vDSO revocation
	// ledger, the same authority recall.Resolve consults on page-in.
	Witness string
	// Reward is the bounded utility delta the witnessed outcome carries. Positive
	// for a verified success (the working set helped), negative to demote a working
	// set that appeared in a verified failure. Utility is clamped to [0, UtilityMax]
	// after the delta, so a negative reward demotes toward neutral but never below.
	Reward float64
}

// Credit applies a WITNESSED outcome's reward to the utility of the working-set
// pages in `steps`. It is the closed loop from a verified task outcome back to the
// retrieval policy: a page earns utility credit only if it was in the working set of
// a session whose outcome was independently witnessed.
//
// Honesty + provenance gates (all fail-closed):
//   - A self-asserted outcome (empty Witness) is refused: ErrUnwitnessed.
//   - A revoked-witness outcome is refused: ErrSealed. A poisoned/stale session
//     cannot up-weight the memories it used.
//   - Per page, the LEARNING is provenance-gated: a quarantined page, or one whose
//     own admission witness is revoked, is SKIPPED — it can never accrue positive
//     utility. (Retention is handled by the dream seal path, which zeroes utility on
//     revocation/re-quarantine; see sealPage in dream.go.)
//
// On success it returns the number of pages actually credited. The caller persists
// the learned utility with Session.Persist.
func (s *Session) Credit(steps []int, out Outcome) (int, error) {
	if out.Witness == "" {
		return 0, ErrUnwitnessed
	}
	if vdso.Default.Revoked(out.Witness) {
		return 0, fmt.Errorf("%w: outcome witness %q is revoked — a refuted session cannot up-weight its memories", ErrSealed, out.Witness)
	}
	credited := 0
	for _, step := range steps {
		if step < 0 || step >= len(s.Manifest.Pages) {
			continue
		}
		p := &s.Manifest.Pages[step]
		// Provenance gates the LEARNING: a sealed page, or one whose admission
		// witness is refuted, can never accrue positive utility — skip it.
		if p.Quarantined {
			continue
		}
		if p.Witness != "" && vdso.Default.Revoked(p.Witness) {
			continue
		}
		p.Utility = clampUtility(p.Utility + out.Reward)
		credited++
	}
	return credited, nil
}

// clampUtility holds a learned utility scalar in [0, UtilityMax]. Neutral (0) is the
// fail-closed floor (unknown/never-credited == neutral, matching abi.FallbackDeny),
// and UtilityMax bounds how far phase-2 utility can re-rank over phase-1 relevance.
func clampUtility(u float64) float64 {
	if u < 0 {
		return 0
	}
	if u > UtilityMax {
		return UtilityMax
	}
	return u
}
