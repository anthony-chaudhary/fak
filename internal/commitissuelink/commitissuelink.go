// Package commitissuelink is a closed, pure checker for one narrow drift: a
// commit that reads as real, tracked work (it carries this repo's own
// ship-stamp trailer, e.g. "(fak audit)") but whose SUBJECT line omits a
// scannable #N issue reference -- the thing an operator (or another agent)
// eyeballing `git log --oneline` actually reads to tell what a commit closed.
//
// A commit is not flagged just for lacking #N in its subject: many real
// commits in this repo carry the issue link only in a body trailer
// ("Fixes #1612"), which GitHub's auto-close still honors. That is not
// wrong, just harder to scan by eye -- so this package still reports it as a
// Finding, but attaches the GuessedIssue pulled straight from that body
// trailer whenever one exists. When no body trailer exists either, the
// GuessedIssue is left empty rather than guessed from a bare number
// elsewhere in the message: a wrong guess is worse than no guess, so only a
// trailer already written in Fixes/Closes/Resolves #N form is trusted.
//
// The package is deterministic and stdlib-only: it takes pre-read Commit
// facts (SHA, subject, body) and folds them into a Report. All git plumbing
// -- resolving a revision range, reading `git log` -- is the caller's job.
package commitissuelink

import (
	"fmt"
	"regexp"
	"strings"
)

// subjectIssueRE matches a #N issue reference occurring in the subject line.
var subjectIssueRE = regexp.MustCompile(`#(\d+)`)

// bodyTrailerRE matches a GitHub auto-close trailer (Fixes/Closes/Resolves
// #N) anywhere in the commit body, case-insensitive.
var bodyTrailerRE = regexp.MustCompile(`(?i)\b(?:fixes|closes|resolves)\s+#(\d+)`)

// leafTrailerRE matches this repo's own ship-stamp trailer, e.g. "(fak
// audit)". Its presence is the signal that a commit is real, tracked leaf
// work -- the kind CLAUDE.md requires end with this trailer -- as opposed to
// an untracked aside (a typo fix, a WIP snapshot) that was never meant to
// close an issue and so is not held to the same #N-in-subject bar.
var leafTrailerRE = regexp.MustCompile(`\(fak [a-zA-Z0-9_-]+\)`)

// Commit is one pre-read git commit fact. The caller reads these via `git
// log`; this package touches no git plumbing of its own.
type Commit struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject"`        // the first line of the commit message
	Body    string `json:"body,omitempty"` // everything after the first blank line, "" if none
}

// Finding is a commit that carries the ship-stamp trailer -- so it reads as
// real, tracked work -- but whose subject line has no scannable #N.
type Finding struct {
	SHA          string `json:"sha"`
	Subject      string `json:"subject"`
	GuessedIssue string `json:"guessed_issue,omitempty"` // from a body "Fixes/Closes/Resolves #N" trailer; "" if none
}

// Report is the closed result of scanning a commit range for subjects
// missing a usable #N issue reference.
type Report struct {
	Scanned  int       `json:"scanned"`
	Findings []Finding `json:"findings"`
}

const (
	ReasonMissingIssueLink         = "missing_issue_link"
	ReasonFailedAudit              = "failed_audit"
	ReasonStaleSHA                 = "stale_sha"
	ReasonInsufficientDiffEvidence = "insufficient_diff_evidence"
)

// CommitLinkedIssue is a pre-read issue/commit/audit fact for the close-gate
// witness bucket. It lets callers surface issues that already mention a
// resolving commit but cannot be witnessed-closed yet. The fold is pure: callers
// decide how to read GitHub, git ancestry, and dos commit-audit.
type CommitLinkedIssue struct {
	Number       int    `json:"number"`
	Title        string `json:"title,omitempty"`
	SHA          string `json:"sha"`
	Subject      string `json:"subject,omitempty"`
	Body         string `json:"body,omitempty"`
	AuditVerdict string `json:"audit_verdict,omitempty"`
	AuditWitness string `json:"audit_witness,omitempty"`
	Reachable    *bool  `json:"reachable,omitempty"`
}

type UnresolvedFinding struct {
	Number int    `json:"number"`
	SHA    string `json:"sha,omitempty"`
	Reason string `json:"reason"`
	Detail string `json:"detail"`
}

type UnresolvedReport struct {
	Scanned  int                 `json:"scanned"`
	Findings []UnresolvedFinding `json:"findings"`
}

// Fold scans commits for the ship-stamp trailer without a subject-line #N. A
// commit whose subject already carries any #N is never a finding, regardless
// of the trailer or the body.
func Fold(commits []Commit) Report {
	rep := Report{Scanned: len(commits)}
	for _, c := range commits {
		if subjectIssueRE.MatchString(c.Subject) {
			continue
		}
		if !leafTrailerRE.MatchString(c.Subject) && !leafTrailerRE.MatchString(c.Body) {
			continue
		}
		f := Finding{SHA: c.SHA, Subject: c.Subject}
		if m := bodyTrailerRE.FindStringSubmatch(c.Body); m != nil {
			f.GuessedIssue = m[1]
		}
		rep.Findings = append(rep.Findings, f)
	}
	return rep
}

func FoldUnresolvedCommitLinkedIssues(rows []CommitLinkedIssue) UnresolvedReport {
	rep := UnresolvedReport{Scanned: len(rows)}
	for _, row := range rows {
		reason, detail, ok := unresolvedReason(row)
		if !ok {
			continue
		}
		rep.Findings = append(rep.Findings, UnresolvedFinding{
			Number: row.Number,
			SHA:    shortSHA(row.SHA),
			Reason: reason,
			Detail: detail,
		})
	}
	return rep
}

func unresolvedReason(row CommitLinkedIssue) (reason, detail string, ok bool) {
	if row.Number > 0 && !commitTextNamesIssue(row, row.Number) {
		return ReasonMissingIssueLink, fmt.Sprintf("commit text does not name #%d", row.Number), true
	}
	if row.Reachable != nil && !*row.Reachable {
		return ReasonStaleSHA, "commit is not reachable from the audited trunk", true
	}
	verdict := strings.ToUpper(strings.TrimSpace(row.AuditVerdict))
	witness := strings.TrimSpace(row.AuditWitness)
	if verdict != "" && verdict != "OK" {
		return ReasonFailedAudit, "commit-audit verdict=" + verdict, true
	}
	if verdict == "OK" && witness != "diff-witnessed" && witness != "data-witnessed" {
		if witness == "" {
			witness = "missing"
		}
		return ReasonInsufficientDiffEvidence, "commit-audit witness=" + witness, true
	}
	return "", "", false
}

func commitTextNamesIssue(row CommitLinkedIssue, number int) bool {
	re := regexp.MustCompile(fmt.Sprintf(`#%d\b`, number))
	return re.MatchString(row.Subject) || re.MatchString(row.Body)
}

func shortSHA(sha string) string {
	if len(sha) > 12 {
		return sha[:12]
	}
	return sha
}
