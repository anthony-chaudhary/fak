package hooks

import (
	"sort"

	"github.com/anthony-chaudhary/fak/internal/devindex"
)

// gate_verbtier.go — the whole-tree gate that catches a NEW dispatched cmd/fak verb before
// it reaches the shared trunk with no verb-tier row. internal/devindex's
// TestVerbTierCoverageIsTotal already reds CI when a live dispatch token resolves to no tier,
// but that test only fires AFTER the offending commit is on the trunk and `go test ./...`
// runs — so on a hot, many-session trunk the ratchet reds the release-gating fast subset
// minutes after the push and holds the release cadence on CI_BASE_RED until a human adds the
// row. This gate surfaces the same VERB_UNTIERED gap one boundary earlier, in `fak hygiene`,
// so a contributor sees it BEFORE the trunk goes red (epic #2653).
//
// It is the verb-tier twin of gate_tierdeclared.go: like that gate it reads the SINGLE source
// of truth rather than a private copy. Unlike gate_tierdeclared (which parses the tier-table
// FILE because architest is a peer tier), devindex is the same tier-1 layer hooks already
// imports (gate_baredevspelling.go), so this gate resolves tiers through the very function the
// CI ratchet uses — devindex.TierOf — and parses the dispatch switch through the very parser
// the ratchet uses — devindex.DispatchVerbs. There is no second authority to drift: the gate
// and TestVerbTierCoverageIsTotal compute the same verdict from the same two functions, one
// boundary apart.

// mainGoFile is the dispatch surface whose verbs must each carry a tier.
const mainGoFile = "cmd/fak/main.go"

// reasonVerbUntiered is the hygiene finding class for a dispatched verb with no tier row. It
// mirrors the ambiguity TestVerbTierCoverageIsTotal reds on ("dispatched verbs with NO tier"),
// and reads parallel to the leaf ratchet's UNTIERED_LEAF (its staged sibling of TIER_DECLARED).
const reasonVerbUntiered = "VERB_UNTIERED"

// gateVerbTierTree emits a VERB_UNTIERED finding for every verb the cmd/fak/main.go dispatch
// switch routes that devindex.TierOf cannot resolve to a tier — the same verdict
// TestVerbTierCoverageIsTotal computes, one boundary earlier. Returns ErrCouldNotRun when
// main.go is unreadable or the dispatch switch yields no tokens (the parser shape changed):
// fail open, exit 2 → the devindex TEST still catches the drift in CI as the backstop, and the
// gate never emits a false VERB_UNTIERED against an unreadable source.
func gateVerbTierTree(t *TrackedTree) ([]Finding, error) {
	body, exists := t.FileBytes(mainGoFile)
	if !exists {
		return nil, ErrCouldNotRun
	}
	verbs := devindex.DispatchVerbs(body)
	if len(verbs) == 0 {
		return nil, ErrCouldNotRun // no dispatch tokens parsed — the switch shape moved; fail open
	}
	var findings []Finding
	for _, verb := range verbs {
		if _, ok := devindex.TierOf(verb); ok {
			continue
		}
		findings = append(findings, Finding{
			Gate: reasonVerbUntiered,
			File: mainGoFile,
			Detail: "dispatched verb " + verb + " has no tier — classify it in ONE tier block of " +
				"internal/devindex/tiers.go (" + reasonVerbUntiered + "). Most verbs are dev; a new " +
				"product-surface verb also bumps the frontdoor ceiling in the same commit. Until the " +
				"row lands, TestVerbTierCoverageIsTotal reds `go test ./...` and holds the release cadence.",
		})
	}
	sort.Slice(findings, func(i, j int) bool { return findings[i].Detail < findings[j].Detail })
	return findings, nil
}
