package main

// dispatch_knownbad.go -- the W4 scope-hold seam of the blast-radius containment epic
// (#2712 / #2716). It sits in the cmd shell ON PURPOSE: the routing FOLD lives in the
// (lease-held) internal/dispatchtick package, but the KNOWN-BAD HOLD is a runtime overlay
// the dispatch verbs apply to the built RouterPayload, reading the live internal/knownbad
// ledger (#2713) and each issue's already-declared paths. So `fak dispatch route` /
// `fak dispatch skipped` skip ONLY the issues whose tree intersects a live known-bad
// signature, while every disjoint issue keeps dispatching -- "progress, not stall".
//
// The distinction that matters (a #2716 confusion risk): BLOCKED_BY_KNOWN_BAD is DYNAMIC
// (a live-ledger intersection that clears when the signature is released) and holds only
// the intersecting SUBSET; BLOCKED_BY_HUMAN is STATIC (a router label) and is a different
// row with a different next-action. They never share a bucket.

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
	"github.com/anthony-chaudhary/fak/internal/knownbad"
)

// reasonBlockedByKnownBad is the closed-vocabulary skip reason a SkippedIssue carries when
// it was held back from dispatch because its declared paths intersect a LIVE known-bad
// signature. It is registered in dos.toml [reasons.BLOCKED_BY_KNOWN_BAD] so the skip is a
// structured refusal verifiable with `dos man wedge BLOCKED_BY_KNOWN_BAD --explain`, not free text.
const reasonBlockedByKnownBad = "BLOCKED_BY_KNOWN_BAD"

// holdKnownBadForRoute is the impure wrapper the dispatch verbs call: it loads the live
// known-bad ledger from the workspace and applies the hold to the freshly built payload.
// It FAILS OPEN -- a missing ledger (nothing recorded yet) or a read error must never
// stall dispatch, so on any problem the payload is returned unchanged. The clock is read
// here (the only impurity); the fold in applyKnownBadHold takes `now` as data. The read
// goes through the stat-keyed cache (#3471): this runs on EVERY route/tick, so an
// unchanged ledger must not be re-read and re-parsed each time.
func holdKnownBadForRoute(root string, payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	records, err := readKnownBadLedgerCached(filepath.Join(root, knownbad.DefaultLedgerRel))
	if err != nil || len(records) == 0 {
		return payload
	}
	return applyKnownBadHold(payload, records, time.Now().Unix())
}

// applyKnownBadHold is the pure scope-hold fold: given a routed payload, the live
// known-bad records, and a clock, it moves every routed issue whose declared paths
// intersect a LIVE signature out of its dispatch lane and into the skipped set with reason
// BLOCKED_BY_KNOWN_BAD, and leaves every disjoint issue exactly where it was. Same inputs
// -> same payload out, so the scope-hold is a witness a test (or a peer) can re-derive.
//
// Only issues carrying declared Paths can intersect: an issue the router placed by
// scope/label/keyword alone (no repo path) has nothing to compare and dispatches normally
// -- the honest behavior, since a hold needs a tree to hold against.
func applyKnownBadHold(payload dispatchtick.RouterPayload, records []knownbad.Record, nowUnix int64) dispatchtick.RouterPayload {
	if len(records) == 0 {
		return payload
	}
	// Identify the held issues and the signature each one hit, in payload.Issues order so
	// the result is deterministic. stepByNum lets us re-derive per-lane step budgets after
	// removal without importing dispatchtick's unexported routeStepBudget.
	held := map[int]knownbad.Record{}
	stepByNum := map[int]int{}
	var heldRoutes []dispatchtick.IssueRoute
	for _, iss := range payload.Issues {
		stepByNum[iss.Number] = routeIssueSteps(iss)
		if iss.Lane == "" || len(iss.Paths) == 0 {
			continue
		}
		if _, done := held[iss.Number]; done {
			continue
		}
		matches := knownbad.Match(records, knownbad.Query{TreeGlobs: iss.Paths}, nowUnix)
		if len(matches) == 0 {
			continue
		}
		held[iss.Number] = matches[0]
		heldRoutes = append(heldRoutes, iss)
	}
	if len(held) == 0 {
		return payload
	}
	// The rest -- lane rebuild, candidate drop, skipped rows, counts -- is the shared
	// dispatch-hold rewrite (dispatch_hold.go). The known-bad hint names the signature the
	// issue's own declared paths hit, so it reads both the record and iss.Paths.
	return applyDispatchHold(payload, held, heldRoutes, stepByNum, reasonBlockedByKnownBad,
		func(iss dispatchtick.IssueRoute) string {
			return knownBadNextAction(held[iss.Number], iss.Paths)
		})
}

// routeIssueSteps mirrors dispatchtick's routeStepBudget (which is unexported): an issue's
// declared expected steps, or 1 when it declared none.
func routeIssueSteps(iss dispatchtick.IssueRoute) int {
	if iss.ExpectedSteps > 0 {
		return iss.ExpectedSteps
	}
	return 1
}

// knownBadNextAction is the "what unblocks this" hint a held row carries: the live
// signature id + its reason class, and the fixer-election (W5) / auto-release (W6) exits
// that clear it. Electing the fixer is out of scope for W4, so it is named as the next
// step rather than resolved here.
func knownBadNextAction(sig knownbad.Record, paths []string) string {
	tree := strings.Join(paths, ",")
	if tree == "" {
		tree = strings.Join(sig.TreeGlobs, ",")
	}
	return fmt.Sprintf("held: %s intersects live known-bad %s (%s); elect a fixer (W5) or wait for auto-release (W6)",
		tree, sig.Signature, sig.ReasonClass)
}

// knownBadBlockedSkipped selects the BLOCKED_BY_KNOWN_BAD rows out of the router's skipped
// set -- the dynamic, ledger-driven holds, distinct from the static human-blocked rows.
func knownBadBlockedSkipped(router dispatchtick.RouterPayload) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0)
	for _, s := range router.SkippedHumanBlocked {
		if s.Reason == reasonBlockedByKnownBad {
			out = append(out, s)
		}
	}
	return out
}

// renderSkippedKnownBadCard folds the known-bad-held issues into one Slack card block,
// kept SEPARATE from the human-blocked card so an operator sees the two hold classes as
// distinct rows. Empty is rendered by the caller (it only appends this block when there is
// at least one hold), so this always has rows.
func renderSkippedKnownBadCard(issues []dispatchtick.SkippedIssue, repoURL string) string {
	var b strings.Builder
	fmt.Fprintf(&b, ":construction: *%d issue(s) held by a live known-bad* — scope-held from dispatch until the signature clears", len(issues))
	shown := issues
	if len(shown) > skippedMaxRows {
		shown = shown[:skippedMaxRows]
	}
	for _, s := range shown {
		fmt.Fprintf(&b, "\n• %s", skippedIssueRow(s, repoURL))
	}
	if len(issues) > skippedMaxRows {
		fmt.Fprintf(&b, "\n… and %d more", len(issues)-skippedMaxRows)
	}
	return b.String()
}
