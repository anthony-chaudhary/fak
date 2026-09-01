package main

// `fak steer comment` (#5029): the ANNOTATE rung of the steering ladder — the
// weakest and most-used steering act, and the one most steering should
// terminate at.
//
// In a PR world the review comment costs nothing, blocks nothing, and lands
// where the work already lives. Continuous merge removed that place, so an
// operator reading a forming unit had nowhere to put a thought: the reasoning
// either evaporated or ended up in a Slack thread disconnected from the intent.
// The unit already knows its closure-grade binding (`Resolves` → "#N"), so it
// knows exactly where the note belongs.
//
// The verb posts the operator's note to that bound issue through the SAME
// trusted `gh` seam `fak issue create` and the redirect rung already use
// (internal/ghexec — deadlined, prompt-disabled, window-suppressed), so it
// never shells raw `gh` and never trips the reversibility preview-confirm gate.
// The posted body carries the unit's identity — leaf + the exact member SHA set
// and band the operator was reading — so the note is anchored to what was
// actually READ rather than to a unit name that means different commits
// tomorrow. Then it appends a row to the overlay comment ledger so the
// brief/loop can see that a unit received operator attention.
//
// ANNOTATE-ONLY by construction: nothing here touches the band, the ack state,
// git, or the landed commits. The structural fence is
// TestCommentNeverReachesGitMutation in internal/steerpr, plus the architest
// steer-overlay floor that globs every cmd/fak/steer_*.go.

import (
	"encoding/json"
	"fmt"
	"io"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// steerCommentPost is the trusted `gh` seam the annotation posts through
// (#5029): overridable in tests so a test run never reaches the network. The
// default routes ONLY through internal/ghexec — the same trusted seam the
// redirect rung uses — and only ever the `gh issue` verb family: a comment can
// move a GitHub issue's conversation, and can never move git.
var steerCommentPost = ghSteerCommentPost

// runSteerComment annotates a forming unit onto its bound issue. It posts the
// operator's note (anchored to the unit's exact member SHA set and band) to the
// unit's closure-grade "#N" through the trusted gh seam, then appends an
// attributable, append-only row to the overlay comment ledger.
//
// A unit with NO closure-grade binding is REFUSED rather than posted somewhere
// plausible: a unit's Mentions are not a binding, and putting operator
// reasoning on a merely-mentioned issue is worse than not posting at all.
func runSteerComment(stdout, stderr io.Writer, argv []string) int {
	// The unit name may come before the flags or after them; accept both.
	unitArg, argv := splitSteerUnitArg(argv)
	commentFlags := newSteerActorCommand("fak steer comment", stderr, argv, unitArg, "is annotating")
	note := commentFlags.String("m", "", "the operator's note about this unit (required)")
	base := commentFlags.String("base", "", "range base ref (default: origin/<release_branch>)")
	head := commentFlags.String("head", "", "range head ref (default: <release_source> tip)")
	const usage = `usage: fak steer comment <unit> -m "<note>" [--by WHO] [--base REF] [--head REF]`
	unit, root, who, code := commentFlags.resolveUnit(usage, base, head,
		"fak steer comment: unit %q binds no issue (#N), so there is no honest place to post this note — only a subject-bound `Resolves` is closure-grade (a mention is not a binding). `fak steer redirect` can still re-aim it\n")
	if code != 0 {
		return code
	}

	// The unit's closure-grade binding: the annotation lands on the intent's own
	// ticket, anchored to the exact member SHA set the operator was reading.
	rec, err := steerpr.NewComment(unit.Leaf, who, *note, steerpr.UnitSHAs(*unit), unit.Band, unit.Resolves[0], time.Now())
	if err != nil {
		fmt.Fprintf(stderr, "fak steer comment: %v\n", err)
		return 2
	}
	// Post FIRST, ledger after: a note that never landed is not operator
	// attention, and the ledger row records where the posted one went.
	posted, err := steerCommentPost(rec)
	if err != nil {
		fmt.Fprintf(stderr, "fak steer comment: post via gh: %v\n", err)
		return 1
	}
	rec.Posted = strings.TrimSpace(posted)
	status := workerworktree.ProjectStatus(workerworktree.StatusEvidence{IssueNumber: steerLifecycleIssueNumber(unit.Resolves), Lane: unit.Leaf, AssociationKnown: true, OwnerLive: true})
	if _, err := upsertSteerLifecycleStatus(unit.Resolves, status); err != nil {
		fmt.Fprintf(stderr, "fak steer comment: project worker-worktree lifecycle: %v\n", err)
		return 1
	}
	if code, done := appendAndWriteSteerRecord(stdout, stderr, "fak steer comment", rec, func() error {
		return steerpr.AppendComment(steerpr.CommentLedgerPath(root), rec)
	}); done {
		return code
	}
	fmt.Fprintf(stdout, "commented on %s (%d commit(s), band %s) as %s — posted to %s anchored to %d member SHA(s); the band, the ack, and the landed commits are untouched: a comment annotates, it never steers\n",
		unit.Leaf, len(unit.Commits), unit.Band, who, rec.Issue, len(rec.SHAs))
	return 0
}

// ghSteerCommentPost is the default trusted gh seam: it posts the anchored note
// as a comment on the unit's bound issue. Every invocation goes through
// internal/ghexec — deadlined, prompt-disabled, window-suppressed — and only
// ever the `gh issue` verb family: the issue conversation moves, git never
// does. Unlike the redirect seam it never reopens and never files a fresh
// issue: an annotation posts where the unit is already bound or it does not
// post at all.
func ghSteerCommentPost(c steerpr.Comment) (string, error) {
	num := strings.TrimPrefix(strings.TrimSpace(c.Issue), "#")
	if num == "" {
		return "", fmt.Errorf("refusing to post a comment with no bound issue")
	}
	comment, cancel := ghexec.CommandTimeout(nil, ghexec.DefaultTimeout, "issue", "comment", num, "--body", c.Body())
	defer cancel()
	out, err := comment.Output()
	if err != nil {
		return "", fmt.Errorf("gh issue comment %s: %v", num, err)
	}
	return strings.TrimSpace(string(out)), nil
}

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

// steerLifecycleStatusUpsert is separate from steerCommentPost on purpose. The
// operator comment path is an append-only human annotation; lifecycle status is
// a machine-owned marker block that may update only its own exact comment.
var steerLifecycleStatusUpsert = ghSteerLifecycleStatusUpsert

var steerLifecycleGHRun steerLifecycleGHRunner = ghSteerLifecycleGHRun

// upsertSteerLifecycleStatus projects lifecycle state only when the unit has one
// unambiguous closure-grade issue association. Mentions are never accepted by
// this seam, and no association is a quiet no-write result rather than a guess.
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

// renderSteerLifecycleStatus is deterministic and deliberately excludes
// timestamps and filesystem paths. Re-projecting the input makes the completion
// claim fail closed: only landed_witnessed with valid commit evidence is done.
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
