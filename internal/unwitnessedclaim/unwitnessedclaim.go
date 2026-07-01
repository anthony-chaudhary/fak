// Package unwitnessedclaim is a closed, pure checker for one narrow drift: an
// issue whose latest comment reads as a self-reported completion claim
// ("done", "fixed", "shipped", ...) while the issue itself is still open --
// meaning no commit's ship-stamp ancestry ("Fixes #N") ever landed to close
// it. The witness-gated dispatch contract (this repo's dos-dispatch skill:
// "non-commit closes ... are operator moves an agent may propose, never
// execute") means a claim sitting unanswered like this is exactly the drift
// an operator would otherwise have to notice by eye.
//
// The checker never closes the issue and never invents a missing witness:
// when it flags a claim, the rendered comment names the claim's own words
// back, states what corroboration is still missing, and names the concrete
// next recovery action (link the commit/PR, or say the work isn't done). It
// is silent whenever there is nothing concrete to name: a closed issue (its
// own ancestry already witnessed whatever happened), an issue with no
// comments, or one whose latest comment doesn't read as a completion claim.
//
// The package is deterministic and stdlib-only: it takes pre-read Input
// facts (issue open/closed, its comments) and evaluates them into a Report.
// All I/O -- reading the issue via `gh issue view`, posting the comment via
// `gh issue comment` -- is the caller's job.
package unwitnessedclaim

import (
	"fmt"
	"regexp"
	"strings"
)

// claimRE matches a closed vocabulary of completion-claim phrasing, case
// insensitive, on a word boundary (so "unfixed" does not match "fixed").
var claimRE = regexp.MustCompile(`(?i)\b(done|fixed|completed|shipped|resolved|implemented|finished)\b`)

// Comment is one pre-read issue comment fact, in chronological order (the
// same order `gh issue view --json comments` returns).
type Comment struct {
	Author string `json:"author"`
	Body   string `json:"body"`
}

// Input is the pre-gathered state of one issue: whether it is still open,
// and its comments oldest-first -- the caller's job via `gh issue view
// --json state,comments`.
type Input struct {
	IssueNumber int       `json:"issue_number"`
	Open        bool      `json:"open"`
	Comments    []Comment `json:"comments"`
}

// Report is the closed result of evaluating one issue: whether its latest
// comment is an unwitnessed completion claim, and if so, the comment body a
// caller may post (never auto-posted by this package, and never a close).
type Report struct {
	Flagged      bool   `json:"flagged"`
	ClaimAuthor  string `json:"claim_author,omitempty"`
	ClaimSnippet string `json:"claim_snippet,omitempty"`
	CommentBody  string `json:"comment_body,omitempty"`
}

// Evaluate inspects the issue's latest comment only -- an earlier claim that
// was since superseded by other discussion is not re-flagged. A closed issue
// is never flagged (its own ancestry already witnessed whatever happened,
// whatever a comment said); an issue with no comments, or whose latest
// comment does not read as a completion claim, is never flagged either.
func Evaluate(in Input) Report {
	if !in.Open || len(in.Comments) == 0 {
		return Report{}
	}
	last := in.Comments[len(in.Comments)-1]
	m := claimRE.FindString(last.Body)
	if m == "" {
		return Report{}
	}
	snippet := snippetAround(last.Body, m)
	return Report{
		Flagged:      true,
		ClaimAuthor:  last.Author,
		ClaimSnippet: snippet,
		CommentBody:  renderComment(in.IssueNumber, last.Author, snippet),
	}
}

// snippetAround trims a comment body to a short window around the matched
// claim word, so the rendered comment quotes the relevant part rather than a
// whole essay.
func snippetAround(body, match string) string {
	body = strings.TrimSpace(body)
	if len(body) <= 140 {
		return body
	}
	i := strings.Index(strings.ToLower(body), strings.ToLower(match))
	start := i - 60
	if start < 0 {
		start = 0
	}
	end := i + len(match) + 60
	if end > len(body) {
		end = len(body)
	}
	prefix, suffix := "", ""
	if start > 0 {
		prefix = "…"
	}
	if end < len(body) {
		suffix = "…"
	}
	return prefix + strings.TrimSpace(body[start:end]) + suffix
}

// renderComment is the deterministic comment body: it names the claim back,
// states the missing witness, and gives the next recovery action. It never
// asks anyone to close the issue -- only ancestry does that.
func renderComment(issueNumber int, author, snippet string) string {
	return fmt.Sprintf(
		"This issue is still open, so the platform's own ancestry check never witnessed a fix -- no commit's \"Fixes #%d\" trailer has landed on trunk.\n\n"+
			"@%s's comment reads as a completion claim:\n\n"+
			"> %s\n\n"+
			"Missing witness: a commit that closes this by ancestry (a `Fixes #%d` trailer in its body), or a linked PR/artifact a third party can independently verify.\n\n"+
			"Next recovery action: point to the commit/PR that ships this so it can close by ancestry, or say plainly the work isn't actually done yet. This comment does not close the issue -- only ancestry does that.",
		issueNumber, author, snippet, issueNumber)
}
