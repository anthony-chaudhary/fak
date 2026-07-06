// Package issuededup is the write-time near-duplicate gate shared by every issue
// producer (#2504): given a candidate {title, body} and a backlog index built from
// cached, read-only `gh issue list` output, it returns advisory dup-risk verdicts
// {issue_number, similarity, matched_on} via simhash embed + TopK over the title
// and title+body axes. It is the pre-FILE twin of dispatchtick's pre-SPAWN gate
// (#1756): producers keep their exact seen-caches as the first gate; this is the
// second gate that catches a paraphrase those caches cannot see.
//
// Advisory only: verdicts always carry the matched issue's number so a producer
// (or `fak issue contract`) can show the twin — nothing here drops, closes, or
// files anything, and nothing here talks to the network.
//
// Tier: foundation (1) — see internal/architest. This package may import only
// packages whose tier is <= 1 (it imports simhash, tier 1); an upward import
// fails the architest gate. Deliberately NOT wired into internal/vdso/neardup.go:
// runtime call dedup refuses fuzzy signals; this layer is off the hot path.
package issuededup
