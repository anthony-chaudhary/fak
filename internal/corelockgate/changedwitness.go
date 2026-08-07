package corelockgate

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// changedwitness.go — `changed:<path>`: the half of the claim grammar that work
// which is not yet TRACKED could not previously reach.
//
// THE GAP. The hard-self core lock is consulted BEFORE any `git add`
// (internal/safecommit/safecommit.go, precommitGates step 4 — ahead of the staging
// step whose own comment says "before any `git add`"; internal/workerworktree asks
// it before anything is applied). So `committed:<path>` — which resolves through
// `git ls-files --error-unmatch -- :/<path>` against the PRE-COMMIT index — is
// REFUTED for a file this very change ADDS. A maintainer whose core-lock change is
// purely additive therefore had exactly one way to name their own new file:
// `path:<path>`, the filesystem-existence rung. And `path:` proves only that a file
// is on disk, which is why the record contains four maintenance clearances citing an
// absolute temp file the clearing agent had written itself. The grammar offered the
// honest additive maintainer a choice between a claim that could not name their work
// (`committed:` on a neighbouring tracked file — commits 58a8924131 and e24c78a028
// each did exactly that, both single-file ADDs) and one that names anything at all.
//
// THE VERB. `changed:<path>` is CONFIRMED only when the named path is a member of
// the pathset this gate was handed to classify — the change actually under way.
//
// WHY IT IS NOT FORGEABLE THE WAY `path:` IS. The candidate set is not the
// claimant's assertion; it is git's, obtained by the CALLER through its own git seam
// before the claim is ever read: `git status --porcelain -- <requested pathspec>` on
// the `fak commit` path, `git diff --name-only <base>` on the worker-worktree land.
// Creating a file does not put it in that set. An absolute path outside the
// repository can never be a member (and is refuted by name, so the temp-file shape
// is called out rather than silently missed). A file created INSIDE the repository
// but left out of the commit's pathspec is not a member either, because the status
// call is scoped to the paths the commit will actually carry. The only way to
// satisfy `changed:` is to make the cited path part of the change — at which point it
// LANDS ON TRUNK in the same commit, under the author's sign-off, permanently
// visible in `git show --name-only` and recorded in the decision note's `tree`. A
// forged witness is no longer a file in a temp directory that evaporates; it is a
// public artifact in the very commit it cleared.
//
// WHY IT IS RESOLVED HERE AND NOT IN internal/witness. `committed:` is a SHARED rung:
// internal/hooks/gate_fileadmission.go, cmd/fak/dispatch_tick_witness.go,
// internal/agent/turn.go and internal/workflow/journal.go all raise claims through the
// same resolver, and none of them has a changed pathset at all — a change-relative
// question is not merely stricter there, it is meaningless. So this verb is not added
// to the shared resolver's grammar. It is resolved at the ONE consult point that owns
// a changed set, and the shared resolver keeps its existing fail-closed posture for
// `changed:` everywhere else: an unrecognized kind ABSTAINS, and an abstain is not
// clearance. Nothing about `committed:` changes for anybody.
//
// WHAT IT DELIBERATELY DOES NOT DO. It does not turn an uncorrelated `committed:`
// into a refusal. Enforcement is the step AFTER this one: the point of shipping the
// grammar first is that a rule refusing "the witness does not name the change" would,
// today, refuse the honest additive maintainer who had no way to name it. With this
// verb in place that excuse is gone, and enforcement becomes a policy decision rather
// than a lockout.

// ChangedWitnessKind is the claim kind resolved from the gate's own changed pathset.
// It is exported so a caller can name it in a remedy sentence without re-spelling
// the literal.
const ChangedWitnessKind = "changed"

// isChangedWitnessClaim reports whether a claim is the gate-local `changed:` verb,
// and returns its path argument. It parses with the same splitter the correlation
// uses, so the verb is recognized for exactly the spellings the correlation reads.
func isChangedWitnessClaim(claim string) (arg string, ok bool) {
	kind, arg, ok := splitCorrelationClaim(claim)
	if !ok || kind != ChangedWitnessKind {
		return "", false
	}
	return arg, true
}

// resolveChangedWitness decides a `changed:<path>` claim against the changed pathset
// the gate is classifying, and returns the outcome plus a cause sentence for the
// refusal detail (empty when CONFIRMED).
//
// ABSTAIN-OVER-REFUTE, the same care correlate.go and the resolver take: a claim
// that names nothing at all, or one offered when the gate holds no changed set, is an
// ABSTAIN — "I cannot judge this" — never an accusation. Only a claim that can be
// POSITIVELY shown to name something outside the change is REFUTED. Both keep the
// lock closed, so the distinction changes no verdict; it changes what the refusal
// tells the maintainer, and it keeps the vocabulary honest for the day a reader
// mines these details.
//
// Membership is EXACT (modulo spelling: separators, "./", git's ":/" magic prefix,
// surrounding quotes and case, all normalized by normalizeRepoPath). A containing
// DIRECTORY is not enough — correlate.go is generous about directories because its
// job is to avoid falsely accusing an honest maintainer, whereas this verb's job is
// to make a maintainer name the artifact, and "internal/adjudicator" is a restatement
// of the refusal rather than a witness to anything. That case is refused with a
// pointed cure instead of a bare mismatch.
func resolveChangedWitness(arg string, changed []string) (abi.WitnessOutcome, string) {
	want := normalizeRepoPath(arg)
	if want == "" {
		return abi.WitnessAbstain, fmt.Sprintf("%s: names an empty path, so no membership question can be asked", ChangedWitnessKind)
	}
	if isOutsideRepoPath(want) {
		return abi.WitnessRefuted, fmt.Sprintf(
			"%s: %q is an absolute path outside the repository, so it cannot be part of this change — a self-authored file on disk is not a witness; name a repo-relative path this commit carries",
			ChangedWitnessKind, arg)
	}
	if len(changed) == 0 {
		return abi.WitnessAbstain, fmt.Sprintf("%s: the gate holds no changed pathset, so no path can be shown to be part of this change", ChangedWitnessKind)
	}
	var containedExample string
	for _, c := range changed {
		got := normalizeRepoPath(c)
		if got == "" {
			continue
		}
		if got == want || strings.EqualFold(got, want) {
			return abi.WitnessConfirmed, ""
		}
		if containedExample == "" && strings.HasPrefix(strings.ToLower(got), strings.ToLower(want)+"/") {
			containedExample = c
		}
	}
	if containedExample != "" {
		return abi.WitnessRefuted, fmt.Sprintf(
			"%s: %q is a directory containing the changed path %q, not a changed path itself; name the file",
			ChangedWitnessKind, arg, containedExample)
	}
	return abi.WitnessRefuted, fmt.Sprintf(
		"%s: %q is not one of the %d path(s) this change carries [%s]",
		ChangedWitnessKind, arg, len(changed), samplePaths(changed))
}
