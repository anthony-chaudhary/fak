package ghspam

import (
	"regexp"
	"sort"
	"strings"
)

const Schema = "fak.gh_spam_comments/v1"

var releaseArchiveRE = regexp.MustCompile(`(?i)https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/releases/download/[^\s)]+?\.(zip|rar|7z|exe|msi|bat|cmd|ps1|scr|dll)(?:[?#][^\s)]*)?`)

type User struct {
	Login string `json:"login"`
}

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

type Options struct {
	TrustedAssociations []string
	TrustedUsers        []string
}

type Finding struct {
	ID                int64  `json:"id"`
	NodeID            string `json:"node_id"`
	HTMLURL           string `json:"html_url"`
	User              string `json:"user"`
	AuthorAssociation string `json:"author_association"`
	CreatedAt         string `json:"created_at"`
	ArchiveURL        string `json:"archive_url"`
	Reason            string `json:"reason"`
	Body              string `json:"body"`
}

type Action struct {
	NodeID          string `json:"node_id"`
	HTMLURL         string `json:"html_url"`
	OK              bool   `json:"ok"`
	Minimized       bool   `json:"minimized,omitempty"`
	MinimizedReason string `json:"minimized_reason,omitempty"`
	Error           string `json:"error,omitempty"`
}

type Counts struct {
	Scanned        int `json:"scanned"`
	TrustedSkipped int `json:"trusted_skipped"`
	Matched        int `json:"matched"`
	Applied        int `json:"applied,omitempty"`
	Failed         int `json:"failed,omitempty"`
}

type Report struct {
	Schema   string    `json:"schema"`
	Mode     string    `json:"mode"`
	Repo     string    `json:"repo,omitempty"`
	Counts   Counts    `json:"counts"`
	Findings []Finding `json:"findings"`
	Actions  []Action  `json:"actions,omitempty"`
}

func DefaultOptions() Options {
	return Options{
		TrustedAssociations: []string{"OWNER", "COLLABORATOR", "MEMBER"},
	}
}

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
	for _, c := range comments {
		if trustedComment(c, trustedAssociations, trustedUsers) {
			rep.Counts.TrustedSkipped++
			continue
		}
		url := releaseArchiveRE.FindString(c.Body)
		if url == "" {
			continue
		}
		rep.Findings = append(rep.Findings, Finding{
			ID:                c.ID,
			NodeID:            c.NodeID,
			HTMLURL:           c.HTMLURL,
			User:              c.User.Login,
			AuthorAssociation: c.AuthorAssociation,
			CreatedAt:         c.CreatedAt,
			ArchiveURL:        strings.TrimRight(url, ".,;:"),
			Reason:            "untrusted_github_release_archive_link",
			Body:              oneLine(c.Body),
		})
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
