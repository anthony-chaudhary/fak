// affordance.go — the tier-gated affordance-hint injection (#5380, parent #2741,
// epic #2721).
//
// THE MEASURED GAP. The replay-graded commit_stamp contrast in
// docs/benchmarks/conceptbench-findings-2026-07-24.md localizes the small arm's
// fall-off twice over: claude-3-5-haiku emitted a subject-only "all tests pass"
// claim over an EMPTY diff (CLAIM_UNWITNESSED) AND omitted the `(fak <leaf>)`
// ship stamp (stamp=none), while claude-opus-4-8 passed the SAME frame. The frame
// conveys the whole contract as one terse prose clause, so what the contrast
// localizes is a DISCOVERABILITY gap in the fak affordance — a thing this repo can
// change and re-measure — not (only) a raw-ability gap that needs a model swap.
//
// WHAT THE HINT SAYS. Restating the rule would be nearly worthless: the terse
// clause already states it, and the arm still missed it. The hint instead hands the
// arm the three things it cannot derive from the frame:
//
//   - the EXACT trailer template for the concept's own lane, spelled out rather
//     than described;
//   - the OFFLINE CHECK that decides gradeability BEFORE the commit is taken
//     (tools/check_commit_msg.py --message "<subject>"; exit 0 means the witness
//     can grade the subject). This is the load-bearing line: an arm that can run
//     the check does not have to have memorized which verbs the witness accepts,
//     and the check is the same code the real witness mirrors;
//   - the report-`not yet` rule, which is what makes an unfinished episode honest
//     instead of the unwitnessed claim the findings recorded.
//
// WHAT GATES IT. Hint ANDs four conditions, and each one is load-bearing:
//
//  1. Enabled — the operator's opt-in. gen/next dogfoods before default, so a run
//     that does not ask for the injection is byte-identical to today's.
//  2. Tier == TierSmall — the arm is the weaker one. This condition is NOT
//     operator-overridable on purpose: the promotion evidence #5380 asks for is
//     "the small arm moved, the frontier arm unchanged", and that contrast is only
//     sound if no flag can reach the frontier prompt. TierUnrated is refused for
//     the same reason — an id with no rating on record is not PROVEN weak, and
//     injecting on a guess would quietly rewrite an arm the report may be reading
//     as a control.
//  3. Concept == ConceptCommitStamp — this text is that concept's contract. A hint
//     that fires on every concept is a hint an arm learns to skip.
//  4. Leaf != "" — the template names a real lane. A hint carrying an empty or
//     guessed leaf would TEACH THE WRONG STAMP, which is worse than silence, so an
//     underivable leaf fails closed.
package conceptbench

import (
	"fmt"
	"sort"
	"strings"
)

// ArmTier is the model-strength band the affordance-hint injection reads. It is a
// deliberate TWO-band split plus an explicit unknown: the only contrast the replay
// findings actually measured is frontier vs small, and a finer ladder would imply a
// calibration this benchmark has not done.
type ArmTier string

const (
	// TierFrontier — an arm the findings recorded passing the concept unaided. It
	// never receives the hint: it is the control the promotion evidence reads
	// against.
	TierFrontier ArmTier = "frontier"
	// TierSmall — the weaker band the findings localized the fall-off to.
	TierSmall ArmTier = "small"
	// TierUnrated — no rating on record (an unknown id, or a RegisterRaw endpoint
	// whose strength the registry has no basis to rate). Refused like a frontier
	// arm, so an unrated id can never be silently treated.
	TierUnrated ArmTier = "unrated"
)

// TierOf reports a registered model's strength band, TierUnrated when the id is
// unknown or carries no rating. Matching is case-insensitive, matching Resolve.
func (r *Registry) TierOf(model string) ArmTier {
	e, ok := r.entries[strings.ToLower(strings.TrimSpace(model))]
	if !ok || e.tier == "" {
		return TierUnrated
	}
	return e.tier
}

// AffordanceAsk is one arm's request for the in-band hint: which concept's
// contract, which lane leaf a correct stamp must name, the arm's strength band,
// and whether the operator opted the injection in at all. Hint decides; the zero
// value is a refusal.
type AffordanceAsk struct {
	Concept Concept // the concept whose contract the hint would echo
	Leaf    string  // the lane a correct `(fak <leaf>)` stamp must name
	Tier    ArmTier // the arm's strength band (from Registry.TierOf)
	Enabled bool    // the operator's opt-in; gen/next keeps it off by default
}

// stampAffordanceText is the commit-stamp concept's hint body. It is a template
// over (concept, leaf, leaf): the trailer is spelled out for the concept's own
// lane rather than described, and the middle paragraph names the checkable step
// instead of paraphrasing the rule it checks.
const stampAffordanceText = `[fak affordance hint — %s, weaker-tier arm]

Subject shape. Copy it exactly, including the trailer:
    type(scope): <verb> <what> (fak %s)
The "(fak %s)" trailer IS the ship stamp: a subject without it stays NOT_SHIPPED,
and the leaf inside it must name the lane the committed paths live in.

Check your own subject BEFORE you commit. The check is offline and instant:
    python tools/check_commit_msg.py --message "<your subject line>"
Exit 0 means the witness can grade that subject. Exit 1 prints the reason — most
often the first word after the colon is not one the witness recognizes as a verb.
Rewrite the subject and run the check again. The witness ABSTAINs permanently on a
subject it cannot grade, and a landed subject cannot be rewritten.

Report only what your diff shows. A subject announcing finished work over a diff
that does not carry that work grades CLAIM_UNWITNESSED, which is a worse outcome
than an honest partial. If the work is not finished, say "not yet", name the
evidence you do have, and name the next checkable step.`

// Hint returns the in-band affordance hint and true when every gate admits it, and
// ("", false) otherwise. The four gates and why each one is there are set out in
// this file's header comment; the short version is that the hint fires only for an
// opted-in run, on the weaker tier, for the commit-stamp concept, with a lane leaf
// the template can name honestly.
func (a AffordanceAsk) Hint() (string, bool) {
	if !a.Enabled {
		return "", false
	}
	if a.Tier != TierSmall {
		return "", false
	}
	if a.Concept != ConceptCommitStamp {
		return "", false
	}
	leaf := strings.TrimSpace(a.Leaf)
	if leaf == "" {
		return "", false
	}
	return fmt.Sprintf(stampAffordanceText, a.Concept, leaf, leaf), true
}

// Frame returns prompt with the hint appended when the gates admit it, and prompt
// UNCHANGED — byte for byte — when they do not. That exactness is the property the
// frontier control rests on: a refused ask must leave no trace in the frame.
func (a AffordanceAsk) Frame(prompt string) string {
	hint, ok := a.Hint()
	if !ok {
		return prompt
	}
	return strings.TrimRight(prompt, "\n") + "\n\n" + hint + "\n"
}

// LeafOfPaths derives the lane leaf a set of repo-relative paths implies, by the
// repo's own directory convention: internal/<leaf>/... and cmd/<leaf>/... name the
// leaf in their second segment (the ship lint accepts a cmd/<dir> shim's directory
// name as the leaf), and any other nested path names it in its first segment.
//
// It returns "" when no path implies a leaf — an empty set, or root-level files
// only. A caller must treat "" as "do not inject": stamping a guessed leaf into the
// hint would teach the arm a stamp the real lint then refuses. Paths are read in
// sorted order so the answer does not depend on map iteration.
func LeafOfPaths(paths []string) string {
	ordered := append([]string(nil), paths...)
	sort.Strings(ordered)
	for _, p := range ordered {
		seg := strings.Split(strings.TrimPrefix(strings.ReplaceAll(strings.TrimSpace(p), "\\", "/"), "./"), "/")
		if len(seg) < 2 || seg[0] == "" {
			continue
		}
		if seg[0] == "internal" || seg[0] == "cmd" {
			if seg[1] != "" {
				return strings.ToLower(seg[1])
			}
			continue
		}
		return strings.ToLower(seg[0])
	}
	return ""
}
