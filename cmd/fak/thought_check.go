package main

// `fak thought-check` is the trusted GitHub edge for issue #9568. The
// internal/issuecheck package owns the pure catalog, review validation, rendering,
// and create/update/no-op/refuse decision. This file owns only bounded `gh api`
// reads and the explicitly armed write. Every write is followed by a fresh,
// byte-exact readback; dry-run is the default.

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/ghexec"
	"github.com/anthony-chaudhary/fak/internal/issuecheck"
)

const (
	thoughtCheckUpsertResultSchema = "fak.issuecheck.upsert-result.v1"
	thoughtCheckVerifyResultSchema = "fak.issuecheck.verify-result.v1"
	thoughtCheckPrepareSchema      = "fak.issuecheck.prepare.v1"
	thoughtCheckCatalogSchema      = "fak.issuecheck.catalog-output.v1"
	thoughtCheckCommentsPerPage    = 100
	thoughtCheckMaxCommentPages    = 100
)

var thoughtCheckRepoRE = regexp.MustCompile(`^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`)

// thoughtCheckRunner is deliberately smaller than exec.Cmd. Tests replace it
// with an ordered GitHub transcript, so no command test can reach the network.
type thoughtCheckRunner func(context.Context, ...string) ([]byte, error)

var thoughtCheckGH thoughtCheckRunner = runThoughtCheckGH

var newThoughtCheckOperationContext = func() (context.Context, context.CancelFunc) {
	return context.WithTimeout(context.Background(), ghexec.DefaultTimeout)
}

type thoughtCheckUpsertResult struct {
	Schema    string  `json:"schema"`
	Issue     int     `json:"issue"`
	Live      bool    `json:"live"`
	Action    string  `json:"action"`
	CommentID int64   `json:"comment_id,omitempty"`
	Verified  bool    `json:"verified"`
	Reason    string  `json:"reason,omitempty"`
	Matches   []int64 `json:"matching_ids,omitempty"`
}

type thoughtCheckVerifyResult struct {
	Schema         string  `json:"schema"`
	Issue          int     `json:"issue"`
	OK             bool    `json:"ok"`
	CommentID      int64   `json:"comment_id,omitempty"`
	IssueDigest    string  `json:"issue_digest,omitempty"`
	CatalogVersion string  `json:"catalog_version,omitempty"`
	Reviewer       string  `json:"reviewer_version,omitempty"`
	Reason         string  `json:"reason,omitempty"`
	Matches        []int64 `json:"matching_ids,omitempty"`
}

type thoughtCheckCatalogResult struct {
	Schema         string             `json:"schema"`
	CatalogVersion string             `json:"catalog_version"`
	Checks         []issuecheck.Check `json:"checks"`
}

type thoughtCheckPrepareResult struct {
	Schema               string                        `json:"schema"`
	Issue                issuecheck.Projection         `json:"issue"`
	IssueDigest          string                        `json:"issue_digest"`
	CatalogVersion       string                        `json:"catalog_version"`
	Checks               []issuecheck.Check            `json:"checks"`
	EvidenceStatusTokens []string                      `json:"evidence_status_tokens"`
	RowTemplate          thoughtCheckReviewRowTemplate `json:"row_template"`
	ReviewTemplate       issuecheck.Review             `json:"review_template"`
}

// These template-only structs intentionally do not use omitempty: prepare's
// output is a mechanical closed-shape authoring contract, including the keys
// that are empty for one or more of the allowed evidence statuses.
type thoughtCheckReviewRowTemplate struct {
	ID         string                       `json:"id"`
	Relevance  string                       `json:"relevance"`
	Assessment string                       `json:"assessment"`
	Evidence   thoughtCheckEvidenceTemplate `json:"evidence"`
	Action     string                       `json:"action"`
}

type thoughtCheckEvidenceTemplate struct {
	Status string   `json:"status"`
	Refs   []string `json:"refs"`
	Gap    string   `json:"gap"`
}

type thoughtCheckIssueJSON struct {
	Number int    `json:"number"`
	Title  string `json:"title"`
	Body   string `json:"body"`
	Labels []struct {
		Name string `json:"name"`
	} `json:"labels"`
}

type thoughtCheckCommentJSON struct {
	ID   int64  `json:"id"`
	Body string `json:"body"`
	User struct {
		Login string `json:"login"`
	} `json:"user"`
}

type thoughtCheckViewerJSON struct {
	Login string `json:"login"`
}

type thoughtCheckRepoJSON struct {
	Owner struct {
		Login string `json:"login"`
		Type  string `json:"type"`
	} `json:"owner"`
}

func cmdThoughtCheck(argv []string) { os.Exit(runThoughtCheck(os.Stdout, os.Stderr, argv)) }

func runThoughtCheck(stdout, stderr io.Writer, argv []string) int {
	if len(argv) == 0 {
		thoughtCheckUsage(stderr)
		return 2
	}
	switch argv[0] {
	case "catalog":
		return runThoughtCheckCatalog(stdout, stderr, argv[1:])
	case "prepare":
		return runThoughtCheckPrepare(stdout, stderr, argv[1:], thoughtCheckGH)
	case "upsert":
		return runThoughtCheckUpsert(stdout, stderr, argv[1:], thoughtCheckGH)
	case "verify":
		return runThoughtCheckVerify(stdout, stderr, argv[1:], thoughtCheckGH)
	default:
		fmt.Fprintf(stderr, "fak thought-check: unknown subcommand %q\n", argv[0])
		thoughtCheckUsage(stderr)
		return 2
	}
}

func thoughtCheckUsage(w io.Writer) {
	fmt.Fprintln(w, "usage: fak thought-check catalog --json")
	fmt.Fprintln(w, "       fak thought-check prepare --issue N [--repo owner/name] --json")
	fmt.Fprintln(w, "       fak thought-check upsert --issue N --input review.json [--repo owner/name] [--live]")
	fmt.Fprintln(w, "       fak thought-check verify --issue N [--repo owner/name] --json")
}

func runThoughtCheckCatalog(stdout, stderr io.Writer, argv []string) int {
	fs := flag.NewFlagSet("fak thought-check catalog", flag.ContinueOnError)
	fs.SetOutput(stderr)
	asJSON := fs.Bool("json", false, "emit the versioned checker catalog as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || !*asJSON {
		fmt.Fprintln(stderr, "fak thought-check catalog: --json is required and no positional arguments are accepted")
		return 2
	}
	result := thoughtCheckCatalogResult{
		Schema: thoughtCheckCatalogSchema, CatalogVersion: issuecheck.CatalogVersion,
		Checks: issuecheck.Catalog(),
	}
	if err := writeThoughtCheckJSON(stdout, result); err != nil {
		fmt.Fprintf(stderr, "fak thought-check catalog: %v\n", err)
		return 1
	}
	return 0
}

func runThoughtCheckPrepare(stdout, stderr io.Writer, argv []string, runner thoughtCheckRunner) int {
	fs := flag.NewFlagSet("fak thought-check prepare", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issueNumber := fs.Int("issue", 0, "GitHub issue number")
	repo := fs.String("repo", "", "GitHub repository as owner/name (default current repository)")
	asJSON := fs.Bool("json", false, "emit the current issue-bound review scaffold as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *issueNumber <= 0 || !*asJSON {
		fmt.Fprintln(stderr, "fak thought-check prepare: --issue N and --json are required")
		return 2
	}
	if err := validateThoughtCheckRepo(*repo); err != nil {
		fmt.Fprintf(stderr, "fak thought-check prepare: %v\n", err)
		return 2
	}
	ctx, cancel := newThoughtCheckOperationContext()
	defer cancel()
	issue, err := fetchThoughtCheckIssue(ctx, runner, *repo, *issueNumber)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check prepare: %v\n", err)
		return 1
	}
	digest, err := issuecheck.IssueDigest(issue)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check prepare: %v\n", err)
		return 1
	}
	result := thoughtCheckPrepareResult{
		Schema: thoughtCheckPrepareSchema, Issue: issuecheck.CanonicalProjection(issue),
		IssueDigest: digest, CatalogVersion: issuecheck.CatalogVersion, Checks: issuecheck.Catalog(),
		EvidenceStatusTokens: []string{issuecheck.EvidenceSupported, issuecheck.EvidencePartial, issuecheck.EvidenceGap},
		RowTemplate: thoughtCheckReviewRowTemplate{
			ID: "", Relevance: "", Assessment: "", Action: "",
			Evidence: thoughtCheckEvidenceTemplate{Status: "", Refs: []string{}, Gap: ""},
		},
		ReviewTemplate: issuecheck.Review{
			Schema: issuecheck.ReviewSchema, IssueNumber: issue.Number,
			IssueBinding: issuecheck.CanonicalIssueBinding(issue), IssueDigest: digest,
			CatalogVersion: issuecheck.CatalogVersion, Rows: []issuecheck.ReviewRow{},
		},
	}
	if err := writeThoughtCheckJSON(stdout, result); err != nil {
		fmt.Fprintf(stderr, "fak thought-check prepare: %v\n", err)
		return 1
	}
	return 0
}

func runThoughtCheckUpsert(stdout, stderr io.Writer, argv []string, runner thoughtCheckRunner) int {
	fs := flag.NewFlagSet("fak thought-check upsert", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issueNumber := fs.Int("issue", 0, "GitHub issue number")
	input := fs.String("input", "", "validated agent review JSON")
	repo := fs.String("repo", "", "GitHub repository as owner/name (default current repository)")
	live := fs.Bool("live", false, "create or update the managed comment; default is dry-run")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *issueNumber <= 0 || strings.TrimSpace(*input) == "" {
		fmt.Fprintln(stderr, "fak thought-check upsert: --issue N and --input review.json are required")
		return 2
	}
	if err := validateThoughtCheckRepo(*repo); err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
		return 2
	}

	reviewRaw, err := os.ReadFile(*input)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: read review: %v\n", err)
		return 1
	}
	var review issuecheck.Review
	if err := decodeThoughtCheckStrict(reviewRaw, &review); err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: malformed review: %v\n", err)
		return 1
	}

	ctx, cancel := newThoughtCheckOperationContext()
	defer cancel()
	issue, err := fetchThoughtCheckIssue(ctx, runner, *repo, *issueNumber)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
		return 1
	}
	// Reject stale or structurally invalid agent output before fetching comments
	// or considering any mutation. ChooseCommentAction repeats this check at the pure
	// boundary; keeping the early check makes the fail-closed edge observable.
	if err := issuecheck.ValidateReview(issue, review); err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: review refused: %v\n", err)
		return 1
	}
	stableOwner, err := fetchThoughtCheckRepoOwner(ctx, runner, *repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
		return 1
	}
	comments, err := fetchThoughtCheckComments(ctx, runner, *repo, *issueNumber, stableOwner)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
		return 1
	}
	if len(comments) == 1 {
		inspection := issuecheck.InspectManagedComment(issue, comments[0].Body)
		if inspection.State != issuecheck.ManagedCommentCurrent && inspection.State != issuecheck.ManagedCommentStale {
			fmt.Fprintf(stderr, "fak thought-check upsert: refusing %s owned production marker comment %d: %s\n", inspection.State, comments[0].ID, inspection.Reason)
			return 1
		}
	}
	plan, err := issuecheck.ChooseCommentAction(issue, review, comments)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: review refused: %v\n", err)
		return 1
	}
	result := thoughtCheckUpsertResult{
		Schema: thoughtCheckUpsertResultSchema, Issue: issue.Number, Live: *live,
		Action: plan.Action, CommentID: plan.CommentID, Reason: plan.Reason,
		Matches: append([]int64(nil), plan.MatchingIDs...),
	}
	if plan.Action == issuecheck.ActionRefuse {
		_ = writeThoughtCheckJSON(stdout, result)
		fmt.Fprintf(stderr, "fak thought-check upsert: refused: %s (matching comment ids %v)\n", plan.Reason, plan.MatchingIDs)
		return 1
	}
	if !*live {
		if err := writeThoughtCheckJSON(stdout, result); err != nil {
			fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
			return 1
		}
		return 0
	}
	if plan.Action != issuecheck.ActionNoop {
		viewer, err := fetchThoughtCheckViewer(ctx, runner)
		if err != nil {
			fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
			return 1
		}
		if viewer != stableOwner {
			fmt.Fprintf(stderr, "fak thought-check upsert: mutation refused: authenticated GitHub actor %q is not stable repository owner %q\n", viewer, stableOwner)
			return 1
		}
	}

	switch plan.Action {
	case issuecheck.ActionNoop:
		// The common postcondition below re-fetches the issue and full comment set.
	case issuecheck.ActionCreate:
		created, err := mutateThoughtCheckComment(ctx, runner, *repo, *issueNumber, 0, plan.Body)
		if err != nil {
			fmt.Fprintf(stderr, "fak thought-check upsert: create: %v\n", err)
			return 1
		}
		result.CommentID = created.ID
	case issuecheck.ActionUpdate:
		updated, err := mutateThoughtCheckComment(ctx, runner, *repo, *issueNumber, plan.CommentID, plan.Body)
		if err != nil {
			fmt.Fprintf(stderr, "fak thought-check upsert: update comment %d: %v\n", plan.CommentID, err)
			return 1
		}
		if updated.ID != plan.CommentID {
			fmt.Fprintf(stderr, "fak thought-check upsert: update returned comment id %d, want %d\n", updated.ID, plan.CommentID)
			return 1
		}
	default:
		fmt.Fprintf(stderr, "fak thought-check upsert: internal error: unknown plan action %q\n", plan.Action)
		return 1
	}
	if err := verifyThoughtCheckPostcondition(ctx, runner, *repo, *issueNumber, stableOwner, result.CommentID, review.IssueDigest); err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %s postcondition: %v\n", plan.Action, err)
		return 1
	}
	result.Verified = true
	if err := writeThoughtCheckJSON(stdout, result); err != nil {
		fmt.Fprintf(stderr, "fak thought-check upsert: %v\n", err)
		return 1
	}
	return 0
}

func runThoughtCheckVerify(stdout, stderr io.Writer, argv []string, runner thoughtCheckRunner) int {
	fs := flag.NewFlagSet("fak thought-check verify", flag.ContinueOnError)
	fs.SetOutput(stderr)
	issueNumber := fs.Int("issue", 0, "GitHub issue number")
	repo := fs.String("repo", "", "GitHub repository as owner/name (default current repository)")
	asJSON := fs.Bool("json", false, "emit verification as JSON")
	if err := fs.Parse(argv); err != nil {
		return 2
	}
	if fs.NArg() != 0 || *issueNumber <= 0 || !*asJSON {
		fmt.Fprintln(stderr, "fak thought-check verify: --issue N and --json are required")
		return 2
	}
	if err := validateThoughtCheckRepo(*repo); err != nil {
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 2
	}
	ctx, cancel := newThoughtCheckOperationContext()
	defer cancel()
	issue, err := fetchThoughtCheckIssue(ctx, runner, *repo, *issueNumber)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 1
	}
	stableOwner, err := fetchThoughtCheckRepoOwner(ctx, runner, *repo)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 1
	}
	comments, err := fetchThoughtCheckComments(ctx, runner, *repo, *issueNumber, stableOwner)
	if err != nil {
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 1
	}
	verification, err := issuecheck.VerifyComment(issue, comments)
	result := thoughtCheckVerifyResult{Schema: thoughtCheckVerifyResultSchema, Issue: issue.Number}
	if err != nil {
		result.Reason = err.Error()
		_ = writeThoughtCheckJSON(stdout, result)
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 1
	}
	if !verification.Valid {
		result.Reason = verification.Reason
		result.Matches = append([]int64(nil), verification.MatchingIDs...)
		_ = writeThoughtCheckJSON(stdout, result)
		fmt.Fprintf(stderr, "fak thought-check verify: %s\n", verification.Reason)
		return 1
	}
	result.OK = true
	result.CommentID = verification.CommentID
	result.IssueDigest = verification.IssueDigest
	result.CatalogVersion = verification.CatalogVersion
	result.Reviewer = verification.ReviewerVersion
	result.Matches = append([]int64(nil), verification.MatchingIDs...)
	if err := writeThoughtCheckJSON(stdout, result); err != nil {
		fmt.Fprintf(stderr, "fak thought-check verify: %v\n", err)
		return 1
	}
	return 0
}

func runThoughtCheckGH(ctx context.Context, args ...string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	cmd := ghexec.Command(ctx, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return nil, fmt.Errorf("gh %s: %w", strings.Join(args, " "), ctxErr)
		}
		return nil, fmt.Errorf("gh %s: %w: %s", strings.Join(args, " "), err, strings.TrimSpace(string(out)))
	}
	return out, nil
}

func fetchThoughtCheckIssue(ctx context.Context, runner thoughtCheckRunner, repo string, number int) (issuecheck.Issue, error) {
	endpoint := thoughtCheckRepoEndpoint(repo) + "/issues/" + strconv.Itoa(number)
	raw, err := runner(ctx, "api", endpoint)
	if err != nil {
		return issuecheck.Issue{}, fmt.Errorf("fetch issue #%d: %w", number, err)
	}
	var got thoughtCheckIssueJSON
	if err := json.Unmarshal(raw, &got); err != nil {
		return issuecheck.Issue{}, fmt.Errorf("decode issue #%d: %w", number, err)
	}
	if got.Number != number || strings.TrimSpace(got.Title) == "" {
		return issuecheck.Issue{}, fmt.Errorf("issue readback mismatch: got #%d title=%q, want #%d with a title", got.Number, got.Title, number)
	}
	labels := make([]string, 0, len(got.Labels))
	for _, label := range got.Labels {
		labels = append(labels, label.Name)
	}
	return issuecheck.Issue{Number: got.Number, Title: got.Title, Body: got.Body, Labels: labels}, nil
}

func fetchThoughtCheckViewer(ctx context.Context, runner thoughtCheckRunner) (string, error) {
	raw, err := runner(ctx, "api", "user")
	if err != nil {
		return "", fmt.Errorf("fetch authenticated GitHub actor: %w", err)
	}
	var got thoughtCheckViewerJSON
	if err := json.Unmarshal(raw, &got); err != nil {
		return "", fmt.Errorf("decode authenticated GitHub actor: %w", err)
	}
	got.Login = strings.TrimSpace(got.Login)
	if got.Login == "" {
		return "", fmt.Errorf("authenticated GitHub actor has an empty login")
	}
	return got.Login, nil
}

func fetchThoughtCheckRepoOwner(ctx context.Context, runner thoughtCheckRunner, repo string) (string, error) {
	raw, err := runner(ctx, "api", thoughtCheckRepoEndpoint(repo))
	if err != nil {
		return "", fmt.Errorf("fetch stable repository owner: %w", err)
	}
	var got thoughtCheckRepoJSON
	if err := json.Unmarshal(raw, &got); err != nil {
		return "", fmt.Errorf("decode stable repository owner: %w", err)
	}
	got.Owner.Login = strings.TrimSpace(got.Owner.Login)
	if got.Owner.Login == "" {
		return "", fmt.Errorf("stable repository owner has an empty login")
	}
	if got.Owner.Type != "User" {
		return "", fmt.Errorf("repository owner %q has unsupported GitHub type %q; configure a stable producer identity before using thought-check on organization-owned repositories", got.Owner.Login, got.Owner.Type)
	}
	return got.Owner.Login, nil
}

func fetchThoughtCheckComments(ctx context.Context, runner thoughtCheckRunner, repo string, number int, owner string) ([]issuecheck.ExistingComment, error) {
	if strings.TrimSpace(owner) == "" {
		return nil, fmt.Errorf("managed-comment owner login must not be empty")
	}
	base := thoughtCheckRepoEndpoint(repo) + "/issues/" + strconv.Itoa(number) + "/comments"
	comments := make([]issuecheck.ExistingComment, 0)
	for page := 1; page <= thoughtCheckMaxCommentPages; page++ {
		endpoint := fmt.Sprintf("%s?per_page=%d&page=%d", base, thoughtCheckCommentsPerPage, page)
		raw, err := runner(ctx, "api", endpoint)
		if err != nil {
			return nil, fmt.Errorf("fetch issue #%d comments page %d: %w", number, page, err)
		}
		var got []thoughtCheckCommentJSON
		if err := json.Unmarshal(raw, &got); err != nil {
			return nil, fmt.Errorf("decode issue #%d comments page %d: %w", number, page, err)
		}
		for _, comment := range got {
			if comment.ID <= 0 {
				return nil, fmt.Errorf("issue #%d comments page %d contains non-positive comment id %d", number, page, comment.ID)
			}
			if !issuecheck.IsManagedComment(comment.Body) {
				continue
			}
			// Repository ownership is the stable witness identity. Foreign exact
			// markers are untrusted issue text: ignore them so they cannot deny service
			// or redirect an update away from the owner's canonical comment.
			if comment.User.Login != owner {
				continue
			}
			comments = append(comments, issuecheck.ExistingComment{ID: comment.ID, Body: comment.Body})
		}
		if len(got) < thoughtCheckCommentsPerPage {
			return comments, nil
		}
	}
	return nil, fmt.Errorf("issue #%d comments exceed safety bound of %d pages", number, thoughtCheckMaxCommentPages)
}

// verifyThoughtCheckPostcondition closes races around every live action. It
// re-fetches both inputs to the pure verifier rather than trusting a mutation
// response or a pre-write snapshot. That catches issue edits, concurrent
// creates, credential-owner ambiguity, stale payloads, and body tampering.
func verifyThoughtCheckPostcondition(ctx context.Context, runner thoughtCheckRunner, repo string, issueNumber int, owner string, wantCommentID int64, wantDigest string) error {
	currentIssue, err := fetchThoughtCheckIssue(ctx, runner, repo, issueNumber)
	if err != nil {
		return fmt.Errorf("re-fetch current issue: %w", err)
	}
	comments, err := fetchThoughtCheckComments(ctx, runner, repo, issueNumber, owner)
	if err != nil {
		return fmt.Errorf("re-fetch complete comment set: %w", err)
	}
	verification, err := issuecheck.VerifyComment(currentIssue, comments)
	if err != nil {
		return fmt.Errorf("verify current managed comment: %w", err)
	}
	if !verification.Valid {
		return fmt.Errorf("verification refused: %s (matching comment ids %v)", verification.Reason, verification.MatchingIDs)
	}
	if verification.CommentID != wantCommentID {
		return fmt.Errorf("verified comment id %d, want %d", verification.CommentID, wantCommentID)
	}
	if verification.IssueDigest != wantDigest {
		return fmt.Errorf("verified issue digest %q, want %q", verification.IssueDigest, wantDigest)
	}
	return nil
}

func mutateThoughtCheckComment(ctx context.Context, runner thoughtCheckRunner, repo string, issueNumber int, commentID int64, body string) (thoughtCheckCommentJSON, error) {
	var endpoint, method string
	if commentID == 0 {
		endpoint = thoughtCheckRepoEndpoint(repo) + "/issues/" + strconv.Itoa(issueNumber) + "/comments"
		method = "POST"
	} else {
		endpoint = thoughtCheckRepoEndpoint(repo) + "/issues/comments/" + strconv.FormatInt(commentID, 10)
		method = "PATCH"
	}
	raw, err := runner(ctx, "api", "--method", method, endpoint, "--raw-field", "body="+body)
	if err != nil {
		return thoughtCheckCommentJSON{}, err
	}
	var got thoughtCheckCommentJSON
	if err := json.Unmarshal(raw, &got); err != nil {
		return thoughtCheckCommentJSON{}, fmt.Errorf("decode %s response: %w", strings.ToLower(method), err)
	}
	if got.ID <= 0 {
		return thoughtCheckCommentJSON{}, fmt.Errorf("%s response has non-positive comment id %d", strings.ToLower(method), got.ID)
	}
	if got.Body != body {
		return thoughtCheckCommentJSON{}, fmt.Errorf("%s response body does not exactly match requested body", strings.ToLower(method))
	}
	return got, nil
}

func thoughtCheckRepoEndpoint(repo string) string {
	if strings.TrimSpace(repo) == "" {
		return "repos/{owner}/{repo}"
	}
	return "repos/" + repo
}

func validateThoughtCheckRepo(repo string) error {
	if strings.TrimSpace(repo) == "" {
		return nil
	}
	if repo != strings.TrimSpace(repo) || !thoughtCheckRepoRE.MatchString(repo) {
		return fmt.Errorf("--repo must be owner/name using letters, digits, dot, underscore, or hyphen")
	}
	return nil
}

func decodeThoughtCheckStrict(raw []byte, dst any) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(dst); err != nil {
		return err
	}
	var extra any
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return fmt.Errorf("unexpected trailing JSON value")
		}
		return fmt.Errorf("trailing JSON: %w", err)
	}
	return nil
}

func writeThoughtCheckJSON(w io.Writer, value any) error {
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	if err := enc.Encode(value); err != nil {
		return fmt.Errorf("encode JSON: %w", err)
	}
	return nil
}
