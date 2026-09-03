package main

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

const (
	steerLifecycleStatusMarkerStart = "<!-- fak-worker-worktree-status:v1 -->"
	steerLifecycleStatusMarkerEnd   = "<!-- /fak-worker-worktree-status:v1 -->"
)

type steerLifecycleStatusAction string

const (
	steerLifecycleStatusSkipped steerLifecycleStatusAction = "skipped"
	steerLifecycleStatusCreated steerLifecycleStatusAction = "created"
	steerLifecycleStatusUpdated steerLifecycleStatusAction = "updated"
	steerLifecycleStatusNoop    steerLifecycleStatusAction = "no-op"
)

type steerLifecycleIssueComment struct {
	ID   int64  `json:"id"`
	Body string `json:"body"`
}

type steerLifecycleGHRunner func(args ...string) ([]byte, error)

var steerLifecycleStatusUpsert = ghSteerLifecycleStatusUpsert

var steerLifecycleGHRun steerLifecycleGHRunner = ghSteerLifecycleGHRun

func upsertSteerLifecycleStatus(resolves []string, status workerworktree.StatusProjection) (steerLifecycleStatusAction, error) {
	issue, ok := steerLifecycleIssueAssociation(resolves)
	if !ok {
		return steerLifecycleStatusSkipped, nil
	}
	if status.IssueNumber != 0 && status.IssueNumber != issue {
		return steerLifecycleStatusSkipped, nil
	}
	status.IssueNumber = issue
	return steerLifecycleStatusUpsert(issue, renderSteerLifecycleStatus(status))
}

func steerLifecycleIssueNumber(resolves []string) int {
	issue, ok := steerLifecycleIssueAssociation(resolves)
	if !ok {
		return 0
	}
	return issue
}

func steerLifecycleIssueAssociation(resolves []string) (int, bool) {
	seen := make(map[int]struct{}, len(resolves))
	for _, raw := range resolves {
		n := strings.TrimPrefix(strings.TrimSpace(raw), "#")
		issue, err := strconv.Atoi(n)
		if err != nil || issue <= 0 {
			return 0, false
		}
		seen[issue] = struct{}{}
	}
	if len(seen) != 1 {
		return 0, false
	}
	for issue := range seen {
		return issue, true
	}
	return 0, false
}

func renderSteerLifecycleStatus(status workerworktree.StatusProjection) string {
	evidence := workerworktree.StatusEvidence{IssueNumber: status.IssueNumber, Lane: status.Lane, Session: status.Session}
	evidence.AssociationKnown = evidence.IssueNumber > 0 || evidence.Lane != "" || evidence.Session != ""
	switch status.State {
	case workerworktree.DisplayActive:
		evidence.OwnerLive = true
	case workerworktree.DisplayUnlandedChanges:
		evidence.Dirty = true
	case workerworktree.DisplayCleanupReady:
		evidence.CleanupReady = true
	case workerworktree.DisplayLandedWitnessed:
		if status.Commit != "" {
			evidence.HeadSHA, evidence.BaseSHA, evidence.LandedWitnessed = status.Commit, "independent-base", true
		} else {
			evidence.OwnerLive = true
		}
	}
	status = workerworktree.ProjectStatus(evidence)

	var b strings.Builder
	b.WriteString(steerLifecycleStatusMarkerStart)
	b.WriteString("\n### Worker lifecycle\n")
	if status.IssueNumber > 0 {
		fmt.Fprintf(&b, "- Issue: #%d\n", status.IssueNumber)
	}
	if status.Lane != "" {
		fmt.Fprintf(&b, "- Lane: `%s`\n", status.Lane)
	}
	if status.Session != "" {
		fmt.Fprintf(&b, "- Session: `%s`\n", status.Session)
	}
	fmt.Fprintf(&b, "- State: `%s`\n", status.State)
	if status.Commit != "" {
		fmt.Fprintf(&b, "- Witnessed commit: `%s`\n", status.Commit)
	}
	if status.Complete {
		b.WriteString("- Complete: yes — landed with independent commit evidence.\n")
	} else if status.State == workerworktree.DisplayCleanupReady {
		b.WriteString("- Complete: no — cleanup-ready residue is not landed completion.\n")
	} else {
		b.WriteString("- Complete: no — no independently witnessed landed commit.\n")
	}
	b.WriteString(steerLifecycleStatusMarkerEnd)
	return b.String()
}

func ghSteerLifecycleStatusUpsert(issue int, body string) (steerLifecycleStatusAction, error) {
	if issue <= 0 {
		return steerLifecycleStatusSkipped, nil
	}
	comments, err := fetchSteerLifecycleComments(issue)
	if err != nil {
		return "", err
	}
	matches := make([]steerLifecycleIssueComment, 0, 1)
	for _, comment := range comments {
		starts := strings.Count(comment.Body, steerLifecycleStatusMarkerStart)
		ends := strings.Count(comment.Body, steerLifecycleStatusMarkerEnd)
		if starts == 0 && ends == 0 {
			continue
		}
		if starts != 1 || ends != 1 {
			return "", fmt.Errorf("issue #%d lifecycle status markers are malformed; refusing to write", issue)
		}
		matches = append(matches, comment)
	}
	if len(matches) > 1 {
		return "", fmt.Errorf("issue #%d has multiple lifecycle status markers; refusing to write", issue)
	}
	if len(matches) == 0 {
		endpoint := fmt.Sprintf("repos/{owner}/{repo}/issues/%d/comments", issue)
		if _, err := steerLifecycleGHRun("api", "--method", "POST", endpoint, "-f", "body="+body); err != nil {
			return "", fmt.Errorf("create lifecycle status on issue #%d: %w", issue, err)
		}
		return steerLifecycleStatusCreated, nil
	}
	if matches[0].Body == body {
		return steerLifecycleStatusNoop, nil
	}
	if matches[0].ID <= 0 {
		return "", fmt.Errorf("issue #%d lifecycle status comment has no stable id; refusing to write", issue)
	}
	endpoint := fmt.Sprintf("repos/{owner}/{repo}/issues/comments/%d", matches[0].ID)
	if _, err := steerLifecycleGHRun("api", "--method", "PATCH", endpoint, "-f", "body="+body); err != nil {
		return "", fmt.Errorf("update lifecycle status comment %d on issue #%d: %w", matches[0].ID, issue, err)
	}
	return steerLifecycleStatusUpdated, nil
}

func fetchSteerLifecycleComments(issue int) ([]steerLifecycleIssueComment, error) {
	var comments []steerLifecycleIssueComment
	for page := 1; ; page++ {
		endpoint := fmt.Sprintf("repos/{owner}/{repo}/issues/%d/comments?per_page=100&page=%d", issue, page)
		raw, err := steerLifecycleGHRun("api", endpoint)
		if err != nil {
			return nil, fmt.Errorf("list lifecycle comments on issue #%d page %d: %w", issue, page, err)
		}
		var rows []steerLifecycleIssueComment
		if err := json.Unmarshal(raw, &rows); err != nil {
			return nil, fmt.Errorf("decode lifecycle comments on issue #%d page %d: %w", issue, page, err)
		}
		comments = append(comments, rows...)
		if len(rows) < 100 {
			return comments, nil
		}
	}
}

func ghSteerLifecycleGHRun(args ...string) ([]byte, error) {
	cmd, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, args...)
	defer cancel()
	out, err := cmd.Output()
	if err != nil {
		return out, fmt.Errorf("gh %s: %w", strings.Join(args, " "), err)
	}
	return out, nil
}
