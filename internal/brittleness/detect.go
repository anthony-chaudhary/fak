package brittleness

import (
	"fmt"
	"sort"
	"strings"
)

// Commit is one landed commit as the git-history detectors need it: its short SHA,
// its subject line, and the set of files it touched. The CLI shell fills these from
// `git log`; tests construct them directly, which is why the detectors are pure over
// this type. (A local type, deliberately not shared with internal/antipattern's
// Commit, so this leaf depends only on pkg/scorecard and stays clean-checkout-green.)
type Commit struct {
	SHA     string   `json:"sha"`
	Subject string   `json:"subject"`
	Files   []string `json:"files"`
}

// DetectFromCommits runs every git-history brittleness detector over a window of
// landed commits and returns the merged, worst-first finding list. It is pure over
// its input; the CLI shell owns the single `git log` read that produces the window.
// An empty window yields no findings -- the card degrades honestly when history is
// unavailable, exactly like antipattern's REDUNDANT_REWORK.
func DetectFromCommits(commits []Commit) []Finding {
	var out []Finding
	out = append(out, DetectRecurringFixes(commits)...)
	out = append(out, DetectReverts(commits)...)
	SortFindings(out)
	return out
}

// DetectRecurringFixes finds RECURRING_FIX: a file touched by TWO OR MORE fix/revert
// commits within the window. The mechanism it names is "the earlier fix got lucky" --
// a symptom-patch that did not hold, so the same file keeps coming back under a fresh
// `fix(...)`. It is the regression sibling of antipattern's REDUNDANT_REWORK, which
// keys on the same CLAIM; this keys on the FIX/REVERT commit TYPE and file recurrence,
// a distinct axis (a file re-fixed under three different claims is still a brittle seam).
//
// PRECISION OVER RECALL (the orphanscan discipline): only fix/revert-typed commits
// count (a feature that iterates a file across commits is healthy development, not
// brittleness), and the threshold is >= 2 so a single fix never flags. Weight is the
// number of distinct fix commits touching the file -- the worst-first recurrence key --
// and Fresh captures every one of those SHAs so the root-cause hunt starts from the
// full chain, not a re-derivation. Order-independent.
func DetectRecurringFixes(commits []Commit) []Finding {
	// file -> ordered, de-duplicated SHAs of the fix/revert commits that touched it.
	fixSHAs := map[string][]string{}
	seen := map[string]map[string]bool{}
	for _, c := range commits {
		if !isFixLike(c.Subject) {
			continue
		}
		for _, f := range c.Files {
			if f == "" {
				continue
			}
			if seen[f] == nil {
				seen[f] = map[string]bool{}
			}
			if seen[f][c.SHA] {
				continue
			}
			seen[f][c.SHA] = true
			fixSHAs[f] = append(fixSHAs[f], c.SHA)
		}
	}

	var out []Finding
	for file, shas := range fixSHAs {
		if len(shas) < 2 {
			continue
		}
		fresh := append([]string(nil), shas...)
		sort.Strings(fresh)
		out = append(out, Finding{
			Class:  ClassRecurringFix,
			Ref:    file,
			Detail: fmt.Sprintf("re-fixed by %d fix/revert commits in the window", len(shas)),
			Weight: len(shas),
			Fresh:  fresh,
		})
	}
	SortFindings(out)
	return out
}

// DetectReverts finds REVERTED_LANDING: a commit whose subject is a revert -- either a
// conventional `revert(scope): ...` type or git's default `Revert "..."` subject. The
// reverted work "landed then had to be undone"; the revert SHA is captured Fresh so the
// eventual re-land starts from the record of why it was pulled, not a cold re-derivation.
func DetectReverts(commits []Commit) []Finding {
	var out []Finding
	for _, c := range commits {
		if !isRevert(c.Subject) {
			continue
		}
		out = append(out, Finding{
			Class:  ClassRevertedLanding,
			Ref:    c.SHA,
			Detail: fmt.Sprintf("revert landed: %q", strings.TrimSpace(c.Subject)),
			Weight: 1,
			Fresh:  []string{c.SHA},
		})
	}
	SortFindings(out)
	return out
}

// isFixLike reports whether a subject is a fix- or revert-typed commit: a
// conventional `fix(...)` / `revert(...)` prefix, or a git default `Revert "..."`.
// Only these count toward RECURRING_FIX -- a `feat`/`refactor` touching the same file
// is healthy iteration, not a symptom-patch that failed to hold.
func isFixLike(subject string) bool {
	return hasConventionalType(subject, "fix") || isRevert(subject)
}

// isRevert reports whether a subject is a revert (conventional `revert` type or git's
// default `Revert "<subject>"`).
func isRevert(subject string) bool {
	s := strings.TrimSpace(subject)
	if strings.HasPrefix(s, "Revert \"") || strings.HasPrefix(strings.ToLower(s), "revert:") {
		return true
	}
	return hasConventionalType(s, "revert")
}

// hasConventionalType reports whether an (any-case) subject opens with the given
// conventional-commit type -- `type:`, `type(scope):`, or a breaking `type!:` /
// `type(scope)!:` -- for the given lowercase type token. It matches only the prefix
// before the first colon, so a "fix" appearing later in the subject never trips it.
func hasConventionalType(subject, typ string) bool {
	s := strings.ToLower(strings.TrimSpace(subject))
	colon := strings.IndexByte(s, ':')
	if colon <= 0 || colon > 40 {
		return false
	}
	head := strings.TrimSuffix(s[:colon], "!")
	if paren := strings.IndexByte(head, '('); paren >= 0 {
		if !strings.HasSuffix(head, ")") {
			return false
		}
		head = head[:paren]
	}
	return head == typ
}
