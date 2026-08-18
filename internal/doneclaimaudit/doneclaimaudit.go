package doneclaimaudit

import (
	"regexp"
	"sort"
	"strings"
)

const Schema = "fak.done-claim-audit.v1"

var (
	doneClaimRE = regexp.MustCompile(`(?im)(?:^|\n)\s*(?:[-*]\s*)?(?:shipped|landed|implemented|fixed|resolved|done)\b|\b(?:work|implementation|acceptance)\s+is\s+(?:now\s+)?complete\b`)
	shaRE       = regexp.MustCompile(`(?i)(?:^|[^0-9a-f])([0-9a-f]{7,40})(?:$|[^0-9a-f])`)
	untrackedRE = regexp.MustCompile("(?im)(?:^|\\n)\\s*(?:\\?\\?|untracked(?:\\s+(?:file|path|work))?\\s*[:=-])\\s*`?([^`\\r\\n]+)")
)

type Comment struct {
	Body string `json:"body"`
	URL  string `json:"url,omitempty"`
}

type Issue struct {
	Number   int       `json:"number"`
	Title    string    `json:"title"`
	State    string    `json:"state"`
	URL      string    `json:"url,omitempty"`
	Comments []Comment `json:"comments,omitempty"`
}

type CommitEvidence struct {
	SHA   string   `json:"sha"`
	Paths []string `json:"paths"`
}

type Finding struct {
	Number         int              `json:"number"`
	Title          string           `json:"title"`
	State          string           `json:"state"`
	URL            string           `json:"url,omitempty"`
	CommentURL     string           `json:"comment_url,omitempty"`
	Claim          string           `json:"claim"`
	CommitEvidence []CommitEvidence `json:"commit_evidence,omitempty"`
	UntrackedPaths []string         `json:"untracked_paths,omitempty"`
	Reason         string           `json:"reason"`
}

type Report struct {
	Schema        string    `json:"schema"`
	Verdict       string    `json:"verdict"`
	IssuesScanned int       `json:"issues_scanned"`
	ClaimsScanned int       `json:"claims_scanned"`
	Findings      []Finding `json:"findings"`
}

type CommitPaths func(sha string) ([]string, bool)

// Audit finds completion claims whose own comment shows neither a tracked diff nor
// an explicitly named untracked path. A bare SHA is not evidence unless it resolves
// to a commit with at least one changed path.
func Audit(issues []Issue, commitPaths CommitPaths) Report {
	report := Report{Schema: Schema, Verdict: "OK", IssuesScanned: len(issues)}
	for _, issue := range issues {
		for _, comment := range issue.Comments {
			if !doneClaimRE.MatchString(comment.Body) {
				continue
			}
			report.ClaimsScanned++
			commits := trackedEvidence(comment.Body, commitPaths)
			untracked := untrackedPaths(comment.Body)
			if len(commits) > 0 || len(untracked) > 0 {
				continue
			}
			report.Findings = append(report.Findings, Finding{
				Number: issue.Number, Title: issue.Title, State: issue.State, URL: issue.URL,
				CommentURL: comment.URL, Claim: compactClaim(comment.Body),
				Reason: "completion claim shows no commit with a tracked diff and no explicitly named untracked path",
			})
		}
	}
	if len(report.Findings) > 0 {
		report.Verdict = "ACTION"
	}
	sort.Slice(report.Findings, func(i, j int) bool {
		if report.Findings[i].Number != report.Findings[j].Number {
			return report.Findings[i].Number > report.Findings[j].Number
		}
		return report.Findings[i].CommentURL < report.Findings[j].CommentURL
	})
	return report
}

func trackedEvidence(body string, commitPaths CommitPaths) []CommitEvidence {
	if commitPaths == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []CommitEvidence
	for _, match := range shaRE.FindAllStringSubmatch(body, -1) {
		sha := strings.ToLower(match[1])
		if seen[sha] {
			continue
		}
		seen[sha] = true
		paths, ok := commitPaths(sha)
		if !ok || len(paths) == 0 {
			continue
		}
		out = append(out, CommitEvidence{SHA: sha, Paths: append([]string(nil), paths...)})
	}
	return out
}

func untrackedPaths(body string) []string {
	seen := map[string]bool{}
	var out []string
	for _, match := range untrackedRE.FindAllStringSubmatch(body, -1) {
		path := strings.Trim(strings.TrimSpace(match[1]), "`'\"")
		if path == "" || seen[path] {
			continue
		}
		seen[path] = true
		out = append(out, path)
	}
	sort.Strings(out)
	return out
}

func compactClaim(body string) string {
	claim := strings.Join(strings.Fields(body), " ")
	if len(claim) > 240 {
		return claim[:237] + "..."
	}
	return claim
}
