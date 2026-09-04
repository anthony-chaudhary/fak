// Package ghspam scans untrusted GitHub issue and PR comments for known abuse
// patterns, including fake patch/fix lures and malicious release archive links.
package ghspam

import (
	"regexp"
	"sort"
	"strings"
)

// Schema identifies the versioned JSON report payload for spam scans.
const Schema = "fak.gh_spam_comments/v1"

var releaseArchiveRE = regexp.MustCompile(`(?i)https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/releases/download/[^\s)]+?\.(zip|rar|7z|exe|msi|bat|cmd|ps1|scr|dll)(?:[?#][^\s)]*)?`)

// patchLureNounRE and patchLureActionRE together recognize the "fake patch/fix"
// lure family: an untrusted comment naming a patch/fix/crack payload AND an
// imperative download/extract action or a throwaway file host. Both signals must
// fire so a genuine outsider bug report that merely says "fix" does not match.
var (
	patchLureNounRE   = regexp.MustCompile(`(?i)\b(patch|fix|crack|keygen|activator|loader|nulled|cracked)\b`)
	patchLureActionRE = regexp.MustCompile(`(?i)(\bdownload\b|\binstall\b|\bgrab\b|password\s+is|\bunzip\b|\bextract\b|mega\.nz|mediafire|anonfiles|gofile\.io|drive\.google\.com|bit\.ly)`)
)

// Family is a reusable GitHub-comment abuse match family. Each inspects one
// untrusted comment body and, on a match, returns the detail (archive URL, lure
// phrase, ...) the Finding carries for an operator or scheduled report. Families
// are ordered; the first to fire on a comment owns it, so a comment is reported
// once under its most specific family. New abuse patterns are added here rather
// than by hardcoding a single incident.
type Family struct {
	Reason string
	match  func(body string) (detail string, ok bool)
}

// Families returns the ordered abuse match families the sweeper scans for. Order
// is significant: the most specific/high-confidence family comes first.
func Families() []Family {
	return []Family{
		{Reason: "untrusted_github_release_archive_link", match: matchReleaseArchive},
		{Reason: "fake_patch_fix_lure_phrasing", match: matchFakePatchLure},
	}
}

func matchReleaseArchive(body string) (string, bool) {
	url := releaseArchiveRE.FindString(body)
	if url == "" {
		return "", false
	}
	return strings.TrimRight(url, ".,;:"), true
}

func matchFakePatchLure(body string) (string, bool) {
	noun := patchLureNounRE.FindString(body)
	action := patchLureActionRE.FindString(body)
	if noun == "" || action == "" {
		return "", false
	}
	return strings.ToLower(strings.TrimSpace(noun)) + "/" + strings.ToLower(strings.TrimSpace(action)), true
}

// User identifies the author of a GitHub comment.
type User struct {
	Login string `json:"login"`
}

// Comment represents a GitHub issue or pull request comment payload.
type Comment struct {
	ID                int64  `json:"id"`
	NodeID            string `json:"node_id"`
	HTMLURL           string `json:"html_url"`
	User              User   `json:"user"`
	AuthorAssociation string `json:"author_association"`
	CreatedAt         string `json:"created_at"`
	UpdatedAt         string `json:"updated_at"`
	Body              string `json:"body"`
}

// Options configures insider trust filters and exemption lists during analysis.
type Options struct {
	TrustedAssociations []string
	TrustedUsers        []string
}

// Finding records a flagged abusive comment and its matching detection rule.
type Finding struct {
	ID                int64  `json:"id"`
	NodeID            string `json:"node_id"`
	HTMLURL           string `json:"html_url"`
	User              string `json:"user"`
	AuthorAssociation string `json:"author_association"`
	CreatedAt         string `json:"created_at"`
	ArchiveURL        string `json:"archive_url,omitempty"`
	Match             string `json:"match"`
	Reason            string `json:"reason"`
	Body              string `json:"body"`
}

// Action records the outcome of an automated response to a flagged comment,
// such as minimizing an abusive comment via the GitHub API.
type Action struct {
	NodeID          string `json:"node_id"`
	HTMLURL         string `json:"html_url"`
	OK              bool   `json:"ok"`
	Minimized       bool   `json:"minimized,omitempty"`
	MinimizedReason string `json:"minimized_reason,omitempty"`
	Error           string `json:"error,omitempty"`
}

// Counts summarizes scan and action totals across a set of comments.
type Counts struct {
	Scanned        int `json:"scanned"`
	TrustedSkipped int `json:"trusted_skipped"`
	Matched        int `json:"matched"`
	Applied        int `json:"applied,omitempty"`
	Failed         int `json:"failed,omitempty"`
}

// Report is the structured output containing scan counts, findings, and applied actions.
type Report struct {
	Schema   string    `json:"schema"`
	Mode     string    `json:"mode"`
	Repo     string    `json:"repo,omitempty"`
	Counts   Counts    `json:"counts"`
	Findings []Finding `json:"findings"`
	Actions  []Action  `json:"actions,omitempty"`
}

// DefaultOptions returns recommended trust defaults exempting repository owners,
// collaborators, and organization members.
func DefaultOptions() Options {
	return Options{
		TrustedAssociations: []string{"OWNER", "COLLABORATOR", "MEMBER"},
	}
}

// Analyze scans a batch of comments against configured abuse families, skipping
// trusted authors and returning a structured report ordered by creation time.
func Analyze(comments []Comment, opt Options) Report {
	if len(opt.TrustedAssociations) == 0 {
		opt.TrustedAssociations = DefaultOptions().TrustedAssociations
	}
	trustedAssociations := normalizedSet(opt.TrustedAssociations)
	trustedUsers := normalizedSet(opt.TrustedUsers)

	rep := Report{
		Schema:   Schema,
		Mode:     "dry-run",
		Findings: []Finding{},
	}
	rep.Counts.Scanned = len(comments)
	families := Families()
	for _, c := range comments {
		if trustedComment(c, trustedAssociations, trustedUsers) {
			rep.Counts.TrustedSkipped++
			continue
		}
		for _, fam := range families {
			detail, ok := fam.match(c.Body)
			if !ok {
				continue
			}
			finding := Finding{
				ID:                c.ID,
				NodeID:            c.NodeID,
				HTMLURL:           c.HTMLURL,
				User:              c.User.Login,
				AuthorAssociation: c.AuthorAssociation,
				CreatedAt:         c.CreatedAt,
				Match:             detail,
				Reason:            fam.Reason,
				Body:              oneLine(c.Body),
			}
			if fam.Reason == "untrusted_github_release_archive_link" {
				finding.ArchiveURL = detail
			}
			rep.Findings = append(rep.Findings, finding)
			break
		}
	}
	rep.Counts.Matched = len(rep.Findings)
	sort.Slice(rep.Findings, func(i, j int) bool {
		if rep.Findings[i].CreatedAt == rep.Findings[j].CreatedAt {
			return rep.Findings[i].ID < rep.Findings[j].ID
		}
		return rep.Findings[i].CreatedAt < rep.Findings[j].CreatedAt
	})
	return rep
}

// AppendAction records an action execution result into the report and updates
// applied or failed counters.
func AppendAction(rep *Report, action Action) {
	rep.Actions = append(rep.Actions, action)
	if action.OK {
		rep.Counts.Applied++
	} else {
		rep.Counts.Failed++
	}
}

func normalizedSet(values []string) map[string]bool {
	out := map[string]bool{}
	for _, v := range values {
		v = strings.ToLower(strings.TrimSpace(v))
		if v != "" {
			out[v] = true
		}
	}
	return out
}

func trustedComment(c Comment, trustedAssociations, trustedUsers map[string]bool) bool {
	if trustedAssociations[strings.ToLower(strings.TrimSpace(c.AuthorAssociation))] {
		return true
	}
	return trustedUsers[strings.ToLower(strings.TrimSpace(c.User.Login))]
}

func oneLine(s string) string {
	s = strings.ReplaceAll(s, "\r", " ")
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.ReplaceAll(s, "\t", " ")
	return strings.Join(strings.Fields(s), " ")
}
