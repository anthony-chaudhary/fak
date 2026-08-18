package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghspam"
)

type ghSpamRunner func(args []string) ([]byte, error)

func runGHSpamCommentsWith(stdout, stderr io.Writer, argv []string, runner ghSpamRunner) int {
	fs := flag.NewFlagSet("gh-spam-comments", flag.ContinueOnError)
	fs.SetOutput(stderr)
	repo := fs.String("repo", "", "owner/repo to scan (default: infer from GITHUB_REPOSITORY or origin remote)")
	commentsJSON := fs.String("comments-json", "", "read a GitHub issue-comments JSON fixture instead of calling gh")
	since := fs.String("since", "", "only fetch comments updated since this RFC3339 time or duration ago (for example 24h)")
	pageLimit := fs.Int("page-limit", 0, "maximum REST pages to scan (0 = all)")
	apply := fs.Bool("apply", false, "minimize matched comments with GitHub's SPAM classifier")
	asJSON := fs.Bool("json", false, "emit the machine-readable report")
	trustedAssociations := fs.String("trusted-associations", "OWNER,COLLABORATOR,MEMBER", "comma-separated author_association values to ignore")
	var trustedUsers stringList
	fs.Var(&trustedUsers, "trust-user", "GitHub login to ignore even if author_association is untrusted; repeatable")
	if !parseFlags(fs, argv) {
		return 2
	}
	if fs.NArg() != 0 {
		fmt.Fprintf(stderr, "fak gh-spam-comments: unexpected argument %q\n", fs.Arg(0))
		return 2
	}
	if *pageLimit < 0 {
		fmt.Fprintln(stderr, "fak gh-spam-comments: --page-limit must be non-negative")
		return 2
	}

	run := runner
	if run == nil {
		run = runGHForSpamComments
	}

	var comments []ghspam.Comment
	var err error
	resolvedRepo := strings.TrimSpace(*repo)
	if strings.TrimSpace(*commentsJSON) != "" {
		comments, err = readGHSpamCommentsFixture(*commentsJSON)
		if err != nil {
			fmt.Fprintf(stderr, "fak gh-spam-comments: %v\n", err)
			return 2
		}
	} else {
		if resolvedRepo == "" {
			resolvedRepo = inferGitHubRepo()
		}
		if resolvedRepo == "" {
			fmt.Fprintln(stderr, "fak gh-spam-comments: --repo owner/repo is required when the repo cannot be inferred")
			return 2
		}
		resolvedSince, sinceErr := resolveGHSpamSince(*since, time.Now())
		if sinceErr != nil {
			fmt.Fprintf(stderr, "fak gh-spam-comments: %v\n", sinceErr)
			return 2
		}
		comments, err = fetchGHSpamComments(resolvedRepo, resolvedSince, *pageLimit, run)
		if err != nil {
			fmt.Fprintf(stderr, "fak gh-spam-comments: %v\n", err)
			return 1
		}
	}

	opt := ghspam.Options{
		TrustedAssociations: splitCommaList(*trustedAssociations),
		TrustedUsers:        []string(trustedUsers),
	}
	rep := ghspam.Analyze(comments, opt)
	rep.Repo = resolvedRepo
	if *apply {
		rep.Mode = "apply"
		for _, f := range rep.Findings {
			action := minimizeGHSpamComment(f, run)
			ghspam.AppendAction(&rep, action)
		}
	}

	if *asJSON {
		if err := writeIndentedJSONNoEscape(stdout, rep); err != nil {
			fmt.Fprintf(stderr, "fak gh-spam-comments: encode json: %v\n", err)
			return 1
		}
	} else {
		renderGHSpamCommentsReport(stdout, rep, *apply)
	}
	if rep.Counts.Failed > 0 {
		return 1
	}
	return 0
}

func runGHForSpamComments(args []string) ([]byte, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, "gh", args...)
	configureDispatchHelperCommand(cmd)
	cmd.WaitDelay = 10 * time.Second
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	out, err := cmd.Output()
	if err != nil {
		if ctx.Err() == context.DeadlineExceeded {
			return out, fmt.Errorf("gh %s timed out after 60s", strings.Join(args, " "))
		}
		msg := strings.TrimSpace(errBuf.String())
		if msg == "" {
			msg = strings.TrimSpace(string(out))
		}
		return out, fmt.Errorf("gh %s: %w: %s", strings.Join(args, " "), err, msg)
	}
	return out, nil
}

func fetchGHSpamComments(repo, since string, pageLimit int, run ghSpamRunner) ([]ghspam.Comment, error) {
	var out []ghspam.Comment
	for page := 1; ; page++ {
		if pageLimit > 0 && page > pageLimit {
			break
		}
		endpoint := fmt.Sprintf("repos/%s/issues/comments?per_page=100&page=%d", repo, page)
		if since != "" {
			endpoint += "&since=" + url.QueryEscape(since)
		}
		raw, err := run([]string{"api", endpoint})
		if err != nil {
			return nil, err
		}
		var pageRows []ghspam.Comment
		if err := json.Unmarshal(raw, &pageRows); err != nil {
			return nil, fmt.Errorf("decode gh comments page %d: %w", page, err)
		}
		out = append(out, pageRows...)
		if len(pageRows) < 100 {
			break
		}
	}
	return out, nil
}

func minimizeGHSpamComment(f ghspam.Finding, run ghSpamRunner) ghspam.Action {
	action := ghspam.Action{NodeID: f.NodeID, HTMLURL: f.HTMLURL}
	if strings.TrimSpace(f.NodeID) == "" {
		action.Error = "missing node_id"
		return action
	}
	const query = `mutation($id:ID!) { minimizeComment(input:{subjectId:$id, classifier:SPAM}) { minimizedComment { isMinimized minimizedReason } } }`
	raw, err := run([]string{"api", "graphql", "-f", "query=" + query, "-F", "id=" + f.NodeID})
	if err != nil {
		action.Error = err.Error()
		return action
	}
	var resp struct {
		Data struct {
			MinimizeComment struct {
				MinimizedComment struct {
					IsMinimized     bool   `json:"isMinimized"`
					MinimizedReason string `json:"minimizedReason"`
				} `json:"minimizedComment"`
			} `json:"minimizeComment"`
		} `json:"data"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if err := json.Unmarshal(raw, &resp); err != nil {
		action.Error = fmt.Sprintf("decode graphql response: %v", err)
		return action
	}
	if len(resp.Errors) > 0 {
		msgs := make([]string, 0, len(resp.Errors))
		for _, e := range resp.Errors {
			msgs = append(msgs, e.Message)
		}
		action.Error = strings.Join(msgs, "; ")
		return action
	}
	action.OK = true
	action.Minimized = resp.Data.MinimizeComment.MinimizedComment.IsMinimized
	action.MinimizedReason = resp.Data.MinimizeComment.MinimizedComment.MinimizedReason
	return action
}

func readGHSpamCommentsFixture(path string) ([]ghspam.Comment, error) {
	var raw []byte
	var err error
	if path == "-" {
		raw, err = io.ReadAll(os.Stdin)
	} else {
		raw, err = os.ReadFile(path)
	}
	if err != nil {
		return nil, fmt.Errorf("read --comments-json: %w", err)
	}
	var comments []ghspam.Comment
	if err := json.Unmarshal(raw, &comments); err == nil {
		return comments, nil
	}
	var wrapped struct {
		Comments []ghspam.Comment `json:"comments"`
	}
	if err := json.Unmarshal(raw, &wrapped); err != nil {
		return nil, fmt.Errorf("--comments-json must be a GitHub issue-comment array or {comments:[...]}: %w", err)
	}
	return wrapped.Comments, nil
}

func resolveGHSpamSince(s string, now time.Time) (string, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return "", nil
	}
	if d, err := time.ParseDuration(s); err == nil {
		return now.UTC().Add(-d).Format(time.RFC3339), nil
	}
	if _, err := time.Parse(time.RFC3339, s); err != nil {
		return "", fmt.Errorf("--since must be RFC3339 or a duration like 24h")
	}
	return s, nil
}

func inferGitHubRepo() string {
	if repo := strings.TrimSpace(os.Getenv("GITHUB_REPOSITORY")); validOwnerRepo(repo) {
		return repo
	}
	cmd := exec.Command("git", "config", "--get", "remote.origin.url")
	configureDispatchHelperCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return ""
	}
	return repoFromRemoteURL(strings.TrimSpace(string(out)))
}

func repoFromRemoteURL(remote string) string {
	remote = strings.TrimSpace(remote)
	remote = strings.TrimSuffix(remote, ".git")
	switch {
	case strings.HasPrefix(remote, "https://github.com/"):
		return ownerRepoFromPath(strings.TrimPrefix(remote, "https://github.com/"))
	case strings.HasPrefix(remote, "http://github.com/"):
		return ownerRepoFromPath(strings.TrimPrefix(remote, "http://github.com/"))
	case strings.HasPrefix(remote, "git@github.com:"):
		return ownerRepoFromPath(strings.TrimPrefix(remote, "git@github.com:"))
	case strings.HasPrefix(remote, "ssh://git@github.com/"):
		return ownerRepoFromPath(strings.TrimPrefix(remote, "ssh://git@github.com/"))
	default:
		return ""
	}
}

func ownerRepoFromPath(path string) string {
	parts := strings.Split(strings.Trim(path, "/"), "/")
	if len(parts) < 2 {
		return ""
	}
	repo := parts[0] + "/" + parts[1]
	if validOwnerRepo(repo) {
		return repo
	}
	return ""
}

func validOwnerRepo(repo string) bool {
	parts := strings.Split(repo, "/")
	return len(parts) == 2 && strings.TrimSpace(parts[0]) != "" && strings.TrimSpace(parts[1]) != ""
}

func renderGHSpamCommentsReport(w io.Writer, rep ghspam.Report, applied bool) {
	mode := rep.Mode
	if mode == "" {
		mode = "dry-run"
	}
	repo := rep.Repo
	if repo == "" {
		repo = "(fixture)"
	}
	fmt.Fprintf(w, "gh-spam-comments %s -- repo %s\n", mode, repo)
	fmt.Fprintf(w, "scanned=%d trusted_skipped=%d matched=%d", rep.Counts.Scanned, rep.Counts.TrustedSkipped, rep.Counts.Matched)
	if applied {
		fmt.Fprintf(w, " applied=%d failed=%d", rep.Counts.Applied, rep.Counts.Failed)
	}
	fmt.Fprintln(w)
	if len(rep.Findings) == 0 {
		fmt.Fprintln(w, "no matching GitHub comment-spam findings across the scanned abuse families")
		return
	}
	for _, f := range rep.Findings {
		fmt.Fprintf(w, "\n@%s %s %s [%s]\n  %s\n  %s\n", f.User, f.AuthorAssociation, f.CreatedAt, f.Reason, f.HTMLURL, f.Match)
	}
	if !applied {
		fmt.Fprintln(w, "\ndry-run only; rerun with --apply to minimize matched comments as SPAM")
		return
	}
	for _, action := range rep.Actions {
		if action.OK {
			continue
		}
		fmt.Fprintf(w, "\nFAILED %s\n  %s\n", action.HTMLURL, action.Error)
	}
}
