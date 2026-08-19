package main

// dispatch_prereq.go -- the dependency soft-hold seam. Like the known-bad hold (dispatch_knownbad.go)
// it is a runtime overlay the dispatch verbs apply to the built RouterPayload, NOT part of the
// (lease-held) routing fold. It reads the "depends-on:/blocked-by: #N" edges the router already
// parsed onto each IssueRoute.BlockedBy and, reusing the tested internal/dispatchorder engine, holds
// back any dispatchable leaf whose prerequisite is still an OPEN candidate this tick. A held leaf is
// removed from its lane (so PickTargetIssue cannot select it) and surfaced in the skipped set with
// reason BLOCKED_BY_OPEN_PREREQ -- legible, not silently dropped.
//
// The hold itself is single-tick and self-clearing. A tiny durable snapshot records only which
// dependency edges were held on the prior pass, allowing prerequisite closure to produce one bounded
// newly-unblocked pickup signal; failure to read or write that snapshot leaves ordinary routing intact.

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dispatchorder"
	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

// reasonBlockedByOpenPrereq is the closed-vocabulary skip reason a SkippedIssue carries when it was
// held back because a prerequisite it named ("depends-on:/blocked-by: #N") is still an open candidate
// this tick. Registered in dos.toml [reasons.BLOCKED_BY_OPEN_PREREQ] so the skip is a structured,
// refusal verifiable with `dos man wedge BLOCKED_BY_OPEN_PREREQ --explain`, not free text.
const reasonBlockedByOpenPrereq = "BLOCKED_BY_OPEN_PREREQ"

// holdOpenPrereqForRoute is the pure dependency soft-hold fold: given a routed payload, it moves every
// DISPATCHABLE issue (in a lane) whose BlockedBy names a prerequisite still open this tick out of its
// lane and into the skipped set with reason BLOCKED_BY_OPEN_PREREQ, and leaves every other issue where
// it was. It is payload-only (no root/ledger/clock), so same payload in -> same payload out.
//
// The presence universe -- what counts as "still open" -- is payload.Issues ∪ payload.SkippedHumanBlocked,
// i.e. the full open set the router saw: an open-but-undispatchable prerequisite (human-blocked,
// known-bad-held, unrouted) still holds its dependent, because it is not CLOSED. Only a prerequisite
// absent from both (closed -> gone) fails open. The fail-open and 2-cycle-safe invariants come from the
// shared dispatchorder engine (BlockedByOpenPrereq); this wrapper only projects the payload into that
// engine's Candidate set and folds the verdict back.
func holdOpenPrereqForRoute(payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	// Build the candidate set: every routed/unrouted issue carries its BlockedBy edges; every
	// already-skipped-but-still-open issue contributes PRESENCE only (SkippedIssue has no body/edges),
	// so a dependent of a skipped-open prerequisite is still held.
	cands := make([]dispatchorder.Candidate, 0, len(payload.Issues)+len(payload.SkippedHumanBlocked))
	for _, iss := range payload.Issues {
		cands = append(cands, dispatchorder.Candidate{ID: strconv.Itoa(iss.Number), BlockedBy: iss.BlockedBy})
	}
	for _, sk := range payload.SkippedHumanBlocked {
		cands = append(cands, dispatchorder.Candidate{ID: strconv.Itoa(sk.Number)})
	}
	blocked := dispatchorder.BlockedByOpenPrereq(cands)
	if len(blocked) == 0 {
		return payload
	}

	// Identify the held issues in payload.Issues order (deterministic). Only a DISPATCHABLE issue
	// (in a lane) is held -- an unrouted issue is not pickable anyway, so holding it would be noise.
	// The map value is the issue's still-open prerequisite ids, which is also this hold's evidence:
	// a number is present here only when `open` was non-empty, so presence == held.
	held := map[int][]string{}
	stepByNum := map[int]int{}
	var heldRoutes []dispatchtick.IssueRoute
	for _, iss := range payload.Issues {
		stepByNum[iss.Number] = routeIssueSteps(iss)
		if iss.Lane == "" {
			continue
		}
		open := blocked[strconv.Itoa(iss.Number)]
		if _, done := held[iss.Number]; len(open) == 0 || done {
			continue
		}
		held[iss.Number] = open
		heldRoutes = append(heldRoutes, iss)
	}
	if len(held) == 0 {
		return payload
	}
	// The rest -- lane rebuild, candidate drop, skipped rows, counts -- is the shared
	// dispatch-hold rewrite (dispatch_hold.go).
	return applyDispatchHold(payload, held, heldRoutes, stepByNum, reasonBlockedByOpenPrereq,
		func(iss dispatchtick.IssueRoute) string {
			return openPrereqNextAction(held[iss.Number])
		})
}

// openPrereqNextAction is the "what unblocks this" hint a held row carries: the open prerequisite(s)
// the issue named, and the self-clearing exit (they close -> this dispatches next tick).
func openPrereqNextAction(open []string) string {
	refs := make([]string, len(open))
	for i, n := range open {
		refs[i] = "#" + n
	}
	joined := strings.Join(refs, ", ")
	return fmt.Sprintf("held: prerequisite %s still open; dispatches once %s closes", joined, joined)
}

// openPrereqBlockedSkipped selects the BLOCKED_BY_OPEN_PREREQ rows out of the router's skipped set --
// the dependency holds, distinct from the static human-blocked and dynamic known-bad rows.
func openPrereqBlockedSkipped(router dispatchtick.RouterPayload) []dispatchtick.SkippedIssue {
	out := make([]dispatchtick.SkippedIssue, 0)
	for _, s := range router.SkippedHumanBlocked {
		if s.Reason == reasonBlockedByOpenPrereq {
			out = append(out, s)
		}
	}
	return out
}

const dispatchPrereqStateSchema = "fak-dispatch-prereq-state/1"

type dispatchPrereqState struct {
	Schema     string              `json:"schema"`
	Held       map[string][]string `json:"held"`
	ReadySince map[string]int64    `json:"ready_since,omitempty"`
}

func dispatchPrereqStatePath(root string) string {
	return filepath.Join(root, ".fak", "dispatch", "prereq-held.json")
}

// reconcilePrereqRelease compares the prior pass's durable dependency holds with the current
// pre-hold open graph. A dependent that remains open after every prior prerequisite disappears
// gets exactly one newly-unblocked pass: this call writes the current hold set, so the next call
// no longer sees the transition. State I/O fails open; ordinary routing must never depend on it.
func reconcilePrereqRelease(root string, payload dispatchtick.RouterPayload) dispatchtick.RouterPayload {
	return reconcilePrereqReleaseAt(root, payload, time.Now().Unix())
}

func reconcilePrereqReleaseAt(root string, payload dispatchtick.RouterPayload, nowUnix int64) dispatchtick.RouterPayload {
	prior := readDispatchPrereqState(dispatchPrereqStatePath(root))
	open := make(map[string]bool, len(payload.Issues))
	for _, issue := range payload.Issues {
		open[strconv.Itoa(issue.Number)] = true
	}
	current := currentDispatchPrereqState(payload)
	current.ReadySince = make(map[string]int64, len(payload.Issues))
	newly := make([]int, 0)
	for _, issue := range payload.Issues {
		issueID := strconv.Itoa(issue.Number)
		if _, held := current.Held[issueID]; held {
			continue
		}
		readySince := prior.ReadySince[issueID]
		if blockers, wasHeld := prior.Held[issueID]; wasHeld {
			blocked := false
			for _, blocker := range blockers {
				if open[blocker] {
					blocked = true
					break
				}
			}
			if !blocked {
				readySince = nowUnix
				newly = append(newly, issue.Number)
			}
		}
		if readySince <= 0 {
			p := dispatchIssueProvenanceFor(root, issue.Number)
			readySince = dispatchReadySince(0, p.UpdatedUnix, p.CreatedUnix)
		}
		if readySince > 0 {
			current.ReadySince[issueID] = readySince
		}
	}
	sort.Ints(newly)
	payload.NewlyUnblocked = newly
	payload.PrereqHeldCount = len(current.Held)
	_ = writeDispatchPrereqState(dispatchPrereqStatePath(root), current)
	return payload
}

func currentDispatchPrereqState(payload dispatchtick.RouterPayload) dispatchPrereqState {
	open := make(map[string]bool, len(payload.Issues))
	for _, issue := range payload.Issues {
		open[strconv.Itoa(issue.Number)] = true
	}
	held := map[string][]string{}
	for _, issue := range payload.Issues {
		var blockers []string
		for _, blocker := range issue.BlockedBy {
			if open[blocker] {
				blockers = append(blockers, blocker)
			}
		}
		if len(blockers) > 0 {
			sort.Strings(blockers)
			held[strconv.Itoa(issue.Number)] = blockers
		}
	}
	return dispatchPrereqState{Schema: dispatchPrereqStateSchema, Held: held}
}

func dispatchPrereqTransitionPending(root string) bool {
	return len(readDispatchPrereqState(dispatchPrereqStatePath(root)).Held) > 0
}

func readDispatchPrereqState(path string) dispatchPrereqState {
	state := dispatchPrereqState{
		Schema: dispatchPrereqStateSchema, Held: map[string][]string{}, ReadySince: map[string]int64{},
	}
	b, err := os.ReadFile(path)
	if err != nil || json.Unmarshal(b, &state) != nil || state.Schema != dispatchPrereqStateSchema || state.Held == nil {
		return dispatchPrereqState{
			Schema: dispatchPrereqStateSchema, Held: map[string][]string{}, ReadySince: map[string]int64{},
		}
	}
	if state.ReadySince == nil {
		state.ReadySince = map[string]int64{}
	}
	return state
}

func writeDispatchPrereqState(path string, state dispatchPrereqState) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	b, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return err
	}
	b = append(b, '\n')
	tmp, err := os.CreateTemp(filepath.Dir(path), ".prereq-held-*.tmp")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer os.Remove(tmpName)
	if err := tmp.Chmod(0o600); err != nil {
		tmp.Close()
		return err
	}
	if _, err := tmp.Write(b); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	return os.Rename(tmpName, path)
}
