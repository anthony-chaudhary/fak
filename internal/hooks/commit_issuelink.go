package hooks

import "regexp"

// commit_issuelink.go — the PRE-commit issue-link lint (#312). The ship-stamp lint
// (commitstamp.go) answers "can `dos verify` bind this commit to a unit of WORK (a leaf)?".
// This answers the sibling question the issue-closure audit asks: "can `tools/issue_closure_audit.py`
// bind this commit to the ISSUE it resolves?" — because a close with no resolving commit reads as
// CLAIMED_CLOSED, and the measured closure_rate sits at ~0.196 (114 TRUE_RESOLVED vs 468
// CLAIMED_CLOSED) precisely because ~61% of dispatched commits carry no bindable `#N`.
//
// The grammar here is a VERBATIM mirror of the auditor's binding regexes (tools/issue_closure_audit.py
// _RESOLVE_RE / _ISSUE_NOUN_RE / the subject-ref rung) so author-time and audit-time agree on what
// "resolving" means — the same gate↔referee parity discipline the ship-stamp lint keeps with
// commit_stamp_doctor.py. A commit the preview calls resolving is one the auditor will count.
//
// Default behaviour is ADVISORY (a Note): not every commit closes an issue, and a hard block on
// every mid-feature commit would be wrong. The blocking form is OPT-IN (`--require-issue`), for the
// dispatch-worker contract where a worker spawned to resolve `#N` MUST stamp it.

var (
	// resolveVerbRE — the GitHub-closing VERB form, mirror of the auditor's _RESOLVE_RE:
	// close|fix|fixe|resolve, optional s/d, then `#N`. Matches in subject OR body.
	resolveVerbRE = regexp.MustCompile(`(?i)\b(?:close|fixe?|resolve)[sd]?\s+#(\d+)\b`)
	// issueNounRE — the repo's house NOUN form, mirror of the auditor's _ISSUE_NOUN_RE:
	// `issue #N` / `issues #N, #M`, written in commit bodies when a change resolves a ticket.
	issueNounRE = regexp.MustCompile(`(?i)\bissues?\b[\s:]*((?:#\d+[\s,]*(?:and\s+)?)+)`)
	// anyRefRE — any `#N` reference, mirror of the auditor's _REF_RE (a negative-lookbehind in
	// Python; Go's RE2 has no lookbehind, so the boundary is enforced manually in issueRefs).
	anyRefRE = regexp.MustCompile(`#(\d+)\b`)
)

// issueLink is the verdict over a proposed commit message's issue references: which `#N` it names
// anywhere, and whether ANY of them is in a RESOLVING position the closure auditor will bind to
// (a closing-verb form, the house noun form, or a `#N` in the subject line). A bare body mention
// (`see #123`) is a reference but NOT resolving — exactly the MENTION the auditor declines to count.
type issueLinkResult struct {
	refs      []int // every distinct #N in the whole message, in first-seen order
	resolving bool  // at least one #N is in a resolving position
}

// lintIssueLink scans a full commit message (subject + body) for issue references and decides
// whether the message carries a bindable resolving link, using the auditor's exact grammar.
func lintIssueLink(message string) issueLinkResult {
	subject := firstSubjectLine(message)
	res := issueLinkResult{refs: issueRefs(message)}

	// Rung 1: any closing-verb form anywhere (subject or body).
	if resolveVerbRE.MatchString(message) {
		res.resolving = true
		return res
	}
	// Rung 2: the house noun form anywhere.
	if issueNounRE.MatchString(message) {
		res.resolving = true
		return res
	}
	// Rung 3: a bare `#N` in the SUBJECT line counts (the auditor's subject-ref rung). A bare
	// `#N` only in the BODY is a MENTION, not resolving.
	if len(issueRefs(subject)) > 0 {
		res.resolving = true
	}
	return res
}

// issueRefs returns every distinct issue number referenced in s, in first-seen order. The boundary
// before `#` is enforced so a fragment like `abc#12` (part of a token) is not read as a ref, matching
// the auditor's `(?<![\w-])#` negative-lookbehind under RE2 (which has no lookbehind).
func issueRefs(s string) []int {
	var out []int
	seen := map[int]bool{}
	for _, loc := range anyRefRE.FindAllStringSubmatchIndex(s, -1) {
		start := loc[0] // index of '#'
		if start > 0 {
			c := s[start-1]
			if c == '_' || c == '-' || isWordByte(c) {
				continue // part of a larger token, not an issue ref
			}
		}
		n := 0
		for _, b := range s[loc[2]:loc[3]] {
			n = n*10 + int(b-'0')
		}
		if !seen[n] {
			seen[n] = true
			out = append(out, n)
		}
	}
	return out
}

func isWordByte(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')
}
