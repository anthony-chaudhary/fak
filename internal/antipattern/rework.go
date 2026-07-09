package antipattern

import (
	"fmt"
	"sort"
	"strings"
)

// Commit is one landed commit as the redundant-rework detector needs it: its short SHA,
// its subject line, and the set of files it touched. The CLI fills these from `git log`;
// tests construct them directly, which is why DetectRedundantRework is pure over this type.
type Commit struct {
	SHA     string   `json:"sha"`
	Subject string   `json:"subject"`
	Files   []string `json:"files"`
}

// DetectRedundantRework finds REDUNDANT_REWORK across a window of landed commits: two or
// more commits that redid the SAME unit of work. It is the post-hoc repetition detector no
// other seam covers -- write-time issue dedup and pre-spawn dispatch dedup stop redundant
// work before it starts, but nothing flagged two commits that each claimed to do the same
// thing after both landed.
//
// PRECISION OVER RECALL (the orphanscan discipline). Iterating on one file across several
// commits is normal, healthy development, not rework -- flagging it would make the detector
// noise nobody trusts. A cluster is reported only when BOTH signals agree:
//
//  1. SAME CLAIM: the commit subjects normalize to the same bag-of-words key (the
//     conventional-commit type/scope/issue-ref/stamp stripped, stopwords dropped, tokens
//     sorted). A real multi-step feature reads "add X" / "wire X" / "test X" -- distinct
//     keys that never cluster. Two "add cache eviction" subjects DO.
//  2. OVERLAPPING FILES: at least two commits in the same-key cluster touch a common file.
//     Same claim on disjoint files is two different jobs that happen to share words; same
//     claim on overlapping files is the same job done twice.
//
// Commits with a claim key too generic to be distinctive (fewer than two salient tokens)
// are never clustered. Order-independent: commits may arrive newest- or oldest-first.
func DetectRedundantRework(commits []Commit) []Finding {
	// Bucket commit indices by their normalized claim key (skip un-clusterable keys).
	buckets := map[string][]int{}
	keys := make([]string, len(commits))
	for i, c := range commits {
		key := normalizeClaim(c.Subject)
		keys[i] = key
		if key == "" {
			continue
		}
		buckets[key] = append(buckets[key], i)
	}

	var out []Finding
	for key, idxs := range buckets {
		if len(idxs) < 2 {
			continue
		}
		// Keep only commits connected to another cluster member by a shared file: the
		// second, corroborating signal. A commit in the cluster that shares no file with
		// any sibling is a same-words-different-job coincidence and is dropped.
		connected, shared := connectedByFile(commits, idxs)
		if len(connected) < 2 {
			continue
		}
		var shas []string
		for _, i := range connected {
			shas = append(shas, commits[i].SHA)
		}
		sort.Strings(shas)
		sort.Strings(shared)
		detail := fmt.Sprintf(
			"%d commits claim the same work %q touching shared file(s) %s (%s) -- likely redone or a no-op re-claim",
			len(connected), key, strings.Join(shared, ", "), strings.Join(shas, ", "))
		out = append(out, Finding{
			Class:  ClassRedundantRework,
			Ref:    shas[0],
			Detail: detail,
			Weight: len(connected),
		})
	}
	// Deterministic order regardless of Go map iteration: worst-first, then by ref.
	SortFindings(out)
	return out
}

// connectedByFile returns the sub-cluster of commit indices each of which shares at least
// one touched file with another member, plus the sorted set of files shared by two or more
// members. It is the file-overlap corroboration for a same-claim cluster.
func connectedByFile(commits []Commit, idxs []int) (connected []int, shared []string) {
	// Count how many cluster members touch each file.
	fileHits := map[string][]int{}
	for _, i := range idxs {
		seen := map[string]bool{}
		for _, f := range commits[i].Files {
			if f == "" || seen[f] {
				continue
			}
			seen[f] = true
			fileHits[f] = append(fileHits[f], i)
		}
	}
	inCluster := map[int]bool{}
	sharedSet := map[string]bool{}
	for f, hits := range fileHits {
		if len(hits) < 2 {
			continue
		}
		sharedSet[f] = true
		for _, i := range hits {
			inCluster[i] = true
		}
	}
	for _, i := range idxs {
		if inCluster[i] {
			connected = append(connected, i)
		}
	}
	sort.Ints(connected)
	for f := range sharedSet {
		shared = append(shared, f)
	}
	return connected, shared
}

// reworkStopwords are subject tokens too common to distinguish one unit of work from
// another. The conventional-commit TYPE (feat/fix/...) is stripped separately by
// normalizeClaim, so verbs like "add"/"fix" that also appear inline are dropped here.
var reworkStopwords = map[string]bool{
	"a": true, "an": true, "the": true, "to": true, "for": true, "of": true, "in": true,
	"on": true, "and": true, "or": true, "with": true, "into": true, "from": true, "by": true,
	"add": true, "adds": true, "added": true, "fix": true, "fixes": true, "fixed": true,
	"update": true, "updates": true, "updated": true, "wire": true, "wires": true, "wired": true,
	"support": true, "make": true, "use": true, "via": true, "when": true, "that": true, "this": true,
}

// normalizeClaim reduces a commit subject to a canonical bag-of-words claim key: the
// conventional-commit prefix, any trailing `(fak <leaf>)` stamp and `(#123)` issue refs
// stripped, lowercased, split on non-alphanumerics, stopwords and 1-char tokens dropped,
// then the unique remaining tokens sorted and joined. Returns "" when fewer than two
// salient tokens survive -- a key too generic to cluster on safely.
func normalizeClaim(subject string) string {
	s := strings.ToLower(strings.TrimSpace(subject))
	s = stripConventionalPrefix(s)

	tokens := strings.FieldsFunc(s, func(r rune) bool {
		return !((r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'))
	})
	uniq := map[string]bool{}
	var kept []string
	for _, t := range tokens {
		if len(t) < 2 || reworkStopwords[t] {
			continue
		}
		// Drop a bare issue-ref remnant (e.g. "2105") and pure-digit noise.
		if isAllDigits(t) {
			continue
		}
		if uniq[t] {
			continue
		}
		uniq[t] = true
		kept = append(kept, t)
	}
	if len(kept) < 2 {
		return ""
	}
	sort.Strings(kept)
	return strings.Join(kept, " ")
}

// stripConventionalPrefix removes a leading `type(scope):` or `type:` conventional-commit
// prefix (with an optional `!` breaking-change marker) from an already-lowercased subject.
// It removes only the prefix; the salient nouns after the colon are what cluster.
func stripConventionalPrefix(s string) string {
	colon := strings.IndexByte(s, ':')
	if colon <= 0 || colon > 40 {
		return s
	}
	head := s[:colon]
	// head must look like a type or type(scope), optionally with a trailing '!'.
	head = strings.TrimSuffix(head, "!")
	if paren := strings.IndexByte(head, '('); paren >= 0 {
		if !strings.HasSuffix(head, ")") {
			return s
		}
		head = head[:paren]
	}
	for _, r := range head {
		if !(r >= 'a' && r <= 'z') {
			return s // not a bare type token; leave the subject untouched
		}
	}
	if head == "" {
		return s
	}
	return strings.TrimSpace(s[colon+1:])
}

func isAllDigits(t string) bool {
	if t == "" {
		return false
	}
	for _, r := range t {
		if r < '0' || r > '9' {
			return false
		}
	}
	return true
}
