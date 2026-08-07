package corelockgate

import (
	"fmt"
	"strings"
)

// correlate.go — WITNESS/CHANGE CORRELATION: the question the hard-self core lock
// was never actually asking.
//
// THE DEFECT. CheckCoreLockHardSelf clears a core-locked change when the offered
// maintenance witness resolves CONFIRMED. But the resolver answers a question about
// the CLAIM ALONE. `committed:<path>` runs `git ls-files --error-unmatch -- :/<path>`
// (internal/witness, case "committed"), which asks "is this path tracked ANYWHERE in
// this repository?" — never "is this path part of the change being cleared". So
// `committed:README.md` resolves CONFIRMED and opens the strictest gate in the
// system, and so does any other one of the repository's tracked paths. The call site
// makes it structural rather than accidental: CheckCoreLockHardSelf resolves with a
// nil *abi.ToolCall, so the resolver is handed no description of the change at all
// and COULD NOT correlate even if its rung wanted to.
//
// The correlation is nonetheless computable right here, with no new evidence and no
// new git call: CoreLockCheck.Changed is the very pathset the gate just classified.
// This file computes it, so a witness that points somewhere else is NAMED as such
// instead of clearing the lock silently.
//
// WHY IT LIVES AT THE CALL SITE, NOT IN THE RESOLVER. `committed:` is a shared rung
// of the claim grammar. The file-admission hook, the dispatch-tick witness, the
// agent turn journal and the workflow journal all raise it in contexts that have no
// "changed pathset" whatsoever, where a correlation constraint is not merely
// stricter but MEANINGLESS. Redefining the rung for every caller in order to repair
// one caller would break rungs that are behaving correctly. The gate that owns a
// changed set is the gate that asks the extra question.
//
// WHY IT OBSERVES AND DOES NOT (YET) REFUSE. Making the mismatch blocking on the
// same commit that first measures it would refuse live work mid-flight, and — on a
// surface whose lock has NO environment escape — a correlation rule that is even
// slightly too tight locks every maintainer out of internal/adjudicator/** with no
// way back. So the correlation is recorded on the append-only decision note first.
// Enforcement is a one-line change at the single place this is consulted, once the
// recorded distribution shows the rule refuses nothing honest.
//
// ABSTAIN-OVER-REFUTE, the same care the resolver takes. Only a claim that can be
// POSITIVELY shown to name something outside the change is Uncorrelated. A claim
// whose kind names no repository path, or one offered against an empty changed set,
// is Indeterminate — "no evidence either way" is not evidence of a mismatch. A
// containing directory counts as correlated, and matching is generous about
// separators, "./" prefixes, git's ":/" magic prefix and case, so an honest
// maintainer naming one of their own changed files can never be accused by a
// spelling difference.

// CorrelationOutcome is the tri-state answer to "does this maintenance witness claim
// name any part of the change it is clearing?".
type CorrelationOutcome string

const (
	// CorrelationIndeterminate means no correlation is computable: the claim names
	// no repository path (ancestor:/commit:/grep: and friends), or there was no
	// changed pathset to compare against. It is the abstain — never an accusation.
	CorrelationIndeterminate CorrelationOutcome = "indeterminate"
	// CorrelationCorrelated means the claim names a path that is part of the change
	// (or a directory containing one). This is the honest maintainer's case and it
	// must always pass trivially.
	CorrelationCorrelated CorrelationOutcome = "correlated"
	// CorrelationUncorrelated means the claim names a repository path that is
	// positively NOT part of the change — the weakness this file exists to make
	// visible. It is only ever produced when the comparison could actually be made.
	CorrelationUncorrelated CorrelationOutcome = "uncorrelated"
)

// WitnessCorrelation is one correlation reading: the outcome, the claim kind it was
// computed for, and a human sentence naming why. Reason carries repository paths
// only — the same paths the decision note's `tree` field already records.
type WitnessCorrelation struct {
	Outcome CorrelationOutcome
	Kind    string
	Reason  string
}

// String is the compact form recorded on the append-only decision note, e.g.
// "uncorrelated: the claim names internal/adjudicator/decide.go, which is not among
// the 3 changed path(s) ...". An empty outcome (a zero value that was never
// computed) stringifies to "" so it is omitted from the record rather than
// recorded as a false reading.
func (w WitnessCorrelation) String() string {
	if w.Outcome == "" {
		return ""
	}
	if w.Reason == "" {
		return string(w.Outcome)
	}
	return string(w.Outcome) + ": " + w.Reason
}

// pathClaimKinds are the claim kinds whose argument IS a repository path, and so the
// only kinds for which "does it name the change?" is a meaningful question.
//
// Every other kind is deliberately absent. ancestor:/commit:/grep: resolve against
// COMMIT HISTORY, and this gate runs before the change is a commit at all (the
// `fak commit` path asks it before any `git add`; the worker-worktree land asks it
// before anything is applied), so no history-shaped claim can name the change under
// gate even in principle — including the degenerate `ancestor:HEAD`, which is
// constant-true because a commit is its own ancestor. That is a real weakness, but it
// is a weakness of the CLAIM GRAMMAR offered to this gate rather than a mismatch this
// function can measure, so it abstains and says so instead of manufacturing a verdict.
var pathClaimKinds = map[string]bool{"committed": true, "path": true}

// CorrelateWitness reports whether a maintenance witness claim names any part of the
// changed pathset it is being offered to clear. It is pure: no git, no filesystem, no
// clock — the correlation is a set question over inputs the gate already holds.
func CorrelateWitness(claim string, changed []string) WitnessCorrelation {
	kind, arg, ok := splitCorrelationClaim(claim)
	if !ok {
		return WitnessCorrelation{
			Outcome: CorrelationIndeterminate,
			Reason:  "the claim is empty or is not in kind:arg form, so it names no path to correlate",
		}
	}
	if !pathClaimKinds[kind] {
		return WitnessCorrelation{
			Outcome: CorrelationIndeterminate,
			Kind:    kind,
			Reason: fmt.Sprintf("claim kind %q names no repository path; it resolves against commit history, "+
				"which cannot contain the change because this gate runs before that change is a commit", kind),
		}
	}
	if len(changed) == 0 {
		return WitnessCorrelation{
			Outcome: CorrelationIndeterminate,
			Kind:    kind,
			Reason:  "the gate was given no changed pathset, so there is nothing to correlate against",
		}
	}
	want := normalizeRepoPath(arg)
	if want == "" {
		return WitnessCorrelation{
			Outcome: CorrelationIndeterminate,
			Kind:    kind,
			Reason:  "the claim names an empty path, so no correlation is computable",
		}
	}
	if isOutsideRepoPath(want) {
		// Positively outside: a repo-relative changed set cannot contain an absolute
		// path, so this is a measurement, not an abstain. It is also the shape a
		// self-authored "read-back attestation" temp file takes.
		return WitnessCorrelation{
			Outcome: CorrelationUncorrelated,
			Kind:    kind,
			Reason: fmt.Sprintf("the claim names %q, an absolute path outside the repository, so it cannot be any of the %d changed path(s)",
				arg, len(changed)),
		}
	}
	for _, c := range changed {
		got := normalizeRepoPath(c)
		if got == "" {
			continue
		}
		if got == want || strings.EqualFold(got, want) {
			return WitnessCorrelation{
				Outcome: CorrelationCorrelated,
				Kind:    kind,
				Reason:  fmt.Sprintf("the claim names %q, which is part of the change", arg),
			}
		}
		if strings.HasPrefix(got, want+"/") || strings.HasPrefix(strings.ToLower(got), strings.ToLower(want)+"/") {
			return WitnessCorrelation{
				Outcome: CorrelationCorrelated,
				Kind:    kind,
				Reason:  fmt.Sprintf("the claim names %q, a directory containing the changed path %q", arg, c),
			}
		}
	}
	return WitnessCorrelation{
		Outcome: CorrelationUncorrelated,
		Kind:    kind,
		Reason: fmt.Sprintf("the claim names %q, which is not among the %d changed path(s) [%s]",
			arg, len(changed), samplePaths(changed)),
	}
}

// splitCorrelationClaim parses "kind:arg" the same way the resolver's own claim
// parser does, so the correlation is computed for exactly the claim the resolver
// resolved. It is duplicated rather than imported because internal/witness is a
// tier-2 mechanism and this is a tier-1 foundation leaf; the parse is four lines and
// the alternative is the upward import the layered-DAG gate refuses.
func splitCorrelationClaim(claim string) (kind, arg string, ok bool) {
	claim = strings.TrimSpace(claim)
	i := strings.IndexByte(claim, ':')
	if i <= 0 || i == len(claim)-1 {
		return "", "", false
	}
	return strings.ToLower(claim[:i]), strings.TrimSpace(claim[i+1:]), true
}

// normalizeRepoPath puts a claim argument and a git-reported changed path into one
// comparable spelling: backslashes to forward slashes, collapsed repeats, git's
// repo-root ":/" magic prefix and any "./" prefix removed, quotes and trailing
// slashes stripped. It is deliberately forgiving — every normalization here can only
// turn a spurious "uncorrelated" into a "correlated", never the reverse.
func normalizeRepoPath(p string) string {
	p = strings.TrimSpace(p)
	p = strings.Trim(p, `"`)
	p = strings.ReplaceAll(p, `\`, "/")
	for strings.Contains(p, "//") {
		p = strings.ReplaceAll(p, "//", "/")
	}
	p = strings.TrimPrefix(p, ":/")
	for strings.HasPrefix(p, "./") {
		p = strings.TrimPrefix(p, "./")
	}
	return strings.TrimSuffix(p, "/")
}

// isOutsideRepoPath reports whether a normalized path is absolute, and therefore
// cannot name a member of a repo-relative changed set. It covers the POSIX leading
// slash and the Windows drive prefix; git's ":/" magic prefix has already been
// stripped by normalizeRepoPath and is NOT absolute.
func isOutsideRepoPath(p string) bool {
	if strings.HasPrefix(p, "/") {
		return true
	}
	return len(p) >= 2 && p[1] == ':' && isASCIILetter(p[0])
}

func isASCIILetter(b byte) bool {
	return (b >= 'a' && b <= 'z') || (b >= 'A' && b <= 'Z')
}

// samplePaths renders a bounded excerpt of the changed set for the reason sentence,
// so a wide commit cannot turn one decision note line into a pathset dump.
func samplePaths(changed []string) string {
	const max = 4
	if len(changed) <= max {
		return strings.Join(changed, ", ")
	}
	return strings.Join(changed[:max], ", ") + fmt.Sprintf(", … (+%d more)", len(changed)-max)
}
