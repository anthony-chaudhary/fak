package commitlifecycle

import (
	"context"
	"fmt"
	"strings"
)

// GitRunner is the read-only git seam used by InspectAncestry.
type GitRunner func(context.Context, string, ...string) (string, int, error)

// Commit records one witnessed local commit and whether the configured observed
// remote-tracking branch already contains it.
type Commit struct {
	SHA     string `json:"sha"`
	Subject string `json:"subject,omitempty"`
	Shipped bool   `json:"shipped"`
	Issue   string `json:"issue,omitempty"`
	Leaf    string `json:"leaf,omitempty"`
}

// Ancestry is a read-only local-vs-observed-remote report. Stale means the
// remote-tracking ref is absent or cannot be read; callers must not infer SHIPPED.
type Ancestry struct {
	Branch       string   `json:"branch"`
	Remote       string   `json:"remote"`
	RemoteRef    string   `json:"remote_ref"`
	Stale        bool     `json:"stale"`
	Reason       string   `json:"reason,omitempty"`
	Commits      []Commit `json:"commits"`
	Head         string   `json:"head,omitempty"`
	HeadOnRemote bool     `json:"head_on_remote"`
}

// InspectAncestry resolves branch.<name>.remote/merge and reads local commits
// not reachable from the corresponding remote-tracking ref. It never fetches,
// pushes, or writes refs/index/worktree.
func InspectAncestry(ctx context.Context, repo string, run GitRunner) (Ancestry, error) {
	out := Ancestry{Commits: []Commit{}}
	branch, code, err := run(ctx, repo, "symbolic-ref", "--quiet", "--short", "HEAD")
	if err != nil || code != 0 || strings.TrimSpace(branch) == "" {
		return out, fmt.Errorf("resolve current branch")
	}
	out.Branch = strings.TrimSpace(branch)
	out.Remote = config(ctx, repo, run, "branch."+out.Branch+".remote")
	if out.Remote == "" {
		out.Remote = "origin"
	}
	merge := config(ctx, repo, run, "branch."+out.Branch+".merge")
	if merge == "" {
		merge = "refs/heads/" + out.Branch
	}
	remoteBranch := strings.TrimPrefix(merge, "refs/heads/")
	out.RemoteRef = "refs/remotes/" + out.Remote + "/" + remoteBranch
	remoteSHA, code, err := run(ctx, repo, "rev-parse", "--verify", "--quiet", out.RemoteRef)
	if err != nil || code != 0 {
		out.Stale = true
		out.Reason = "observed remote-tracking ref missing; run fak sync check --fetch before shipment decisions"
		return out, nil
	}
	head, hcode, herr := run(ctx, repo, "rev-parse", "HEAD")
	if herr != nil || hcode != 0 {
		return out, fmt.Errorf("read local HEAD")
	}
	out.Head = strings.TrimSpace(head)
	if out.Head != "" && out.Head == strings.TrimSpace(remoteSHA) {
		out.HeadOnRemote = true
	}
	log, code, err := run(ctx, repo, "log", "--format=%H%x09%s", out.RemoteRef+"..HEAD")
	if err != nil || code != 0 {
		return out, fmt.Errorf("read local commit ancestry")
	}
	for _, line := range strings.Split(strings.TrimSpace(log), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		sha, subject, _ := strings.Cut(line, "\t")
		out.Commits = append(out.Commits, Commit{SHA: strings.TrimSpace(sha), Subject: strings.TrimSpace(subject)})
	}
	return out, nil
}

func config(ctx context.Context, repo string, run GitRunner, key string) string {
	out, code, err := run(ctx, repo, "config", "--get", key)
	if err != nil || code != 0 {
		return ""
	}
	return strings.TrimSpace(out)
}

// AncestryRows folds the report into lifecycle rows. A missing/stale remote is
// one operator-gated row; local-only commits each remain COMMITTED_UNPUSHED.
func AncestryRows(a Ancestry) []Row {
	if a.Stale {
		return []Row{gated(Unknown, a.Reason)}
	}
	rows := make([]Row, 0, len(a.Commits)+1)
	if len(a.Commits) == 0 && a.HeadOnRemote && a.Head != "" {
		rows = append(rows, Fold(Facts{LocalCommit: a.Head, LocalOnRemote: true}))
	}
	for _, c := range a.Commits {
		rows = append(rows, Fold(Facts{LocalCommit: c.SHA, LocalOnRemote: c.Shipped}))
	}
	return rows
}
