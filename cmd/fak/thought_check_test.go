package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/issuecheck"
)

type thoughtCheckFakeGH struct {
	issue               issuecheck.Issue
	viewer              string
	repoOwner           string
	repoOwnerType       string
	comments            []thoughtCheckCommentJSON
	calls               [][]string
	contexts            []context.Context
	cancel              context.CancelFunc
	cancelAfterCall     int
	mutationErr         error
	mutationBody        *string
	issueAfterMutation  *issuecheck.Issue
	concurrentComment   *thoughtCheckCommentJSON
	postMutationBody    *string
	postMutationApplied bool
	concurrentInjected  bool
	nextID              int64
	mutationCount       int
	issueReadCount      int
	commentPageCount    int
}

func (f *thoughtCheckFakeGH) run(ctx context.Context, args ...string) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	f.calls = append(f.calls, append([]string(nil), args...))
	f.contexts = append(f.contexts, ctx)
	if f.cancel != nil && f.cancelAfterCall == len(f.calls) {
		defer f.cancel()
	}
	if len(args) < 2 || args[0] != "api" {
		return nil, fmt.Errorf("unexpected gh args: %v", args)
	}
	if len(args) >= 3 && args[1] == "--method" {
		return f.mutate(args)
	}
	endpoint := args[1]
	if endpoint == "user" {
		login := f.viewer
		if login == "" {
			login = "fak-review-bot"
		}
		return json.Marshal(map[string]string{"login": login})
	}
	if endpoint == "repos/owner/repo" || endpoint == "repos/{owner}/{repo}" {
		owner := f.repoOwner
		if owner == "" {
			owner = f.actor()
		}
		ownerType := f.repoOwnerType
		if ownerType == "" {
			ownerType = "User"
		}
		return json.Marshal(map[string]any{"owner": map[string]string{"login": owner, "type": ownerType}})
	}
	if strings.Contains(endpoint, "/comments?per_page=") {
		if f.mutationCount > 0 && !f.postMutationApplied {
			f.postMutationApplied = true
			if f.postMutationBody != nil && len(f.comments) > 0 {
				f.comments[len(f.comments)-1].Body = *f.postMutationBody
			}
		}
		if f.mutationCount > 0 && f.concurrentComment != nil && !f.concurrentInjected {
			f.concurrentInjected = true
			f.comments = append(f.comments, *f.concurrentComment)
		}
		f.commentPageCount++
		page, err := queryInt(endpoint, "page")
		if err != nil {
			return nil, err
		}
		start := (page - 1) * thoughtCheckCommentsPerPage
		if start >= len(f.comments) {
			return json.Marshal([]thoughtCheckCommentJSON{})
		}
		end := start + thoughtCheckCommentsPerPage
		if end > len(f.comments) {
			end = len(f.comments)
		}
		return json.Marshal(f.comments[start:end])
	}
	if strings.HasSuffix(endpoint, "/issues/"+strconv.Itoa(f.issue.Number)) {
		f.issueReadCount++
		current := f.issue
		if f.mutationCount > 0 && f.issueAfterMutation != nil {
			current = *f.issueAfterMutation
		}
		labels := make([]map[string]string, 0, len(current.Labels))
		for _, label := range current.Labels {
			labels = append(labels, map[string]string{"name": label})
		}
		return json.Marshal(map[string]any{
			"number": current.Number, "title": current.Title,
			"body": current.Body, "labels": labels,
		})
	}
	return nil, fmt.Errorf("unexpected gh endpoint %q", endpoint)
}

func (f *thoughtCheckFakeGH) mutate(args []string) ([]byte, error) {
	f.mutationCount++
	if f.mutationErr != nil {
		return nil, f.mutationErr
	}
	if len(args) != 6 || args[4] != "--raw-field" || !strings.HasPrefix(args[5], "body=") {
		return nil, fmt.Errorf("unexpected mutation args: %v", args)
	}
	method, endpoint := args[2], args[3]
	body := strings.TrimPrefix(args[5], "body=")
	responseBody := body
	if f.mutationBody != nil {
		responseBody = *f.mutationBody
	}
	var id int64
	switch method {
	case "POST":
		if !strings.HasSuffix(endpoint, "/issues/"+strconv.Itoa(f.issue.Number)+"/comments") {
			return nil, fmt.Errorf("POST endpoint = %q", endpoint)
		}
		id = f.nextID
		if id <= 0 {
			id = 901
		}
		created := thoughtCheckCommentJSON{ID: id, Body: body}
		created.User.Login = f.actor()
		f.comments = append(f.comments, created)
	case "PATCH":
		var err error
		id, err = endpointID(endpoint)
		if err != nil {
			return nil, err
		}
		found := false
		for i := range f.comments {
			if f.comments[i].ID == id {
				f.comments[i].Body = body
				found = true
			}
		}
		if !found {
			return nil, fmt.Errorf("PATCH comment %d not found", id)
		}
	default:
		return nil, fmt.Errorf("unexpected method %q", method)
	}
	result := thoughtCheckCommentJSON{ID: id, Body: responseBody}
	result.User.Login = f.actor()
	return json.Marshal(result)
}

func (f *thoughtCheckFakeGH) actor() string {
	if f.viewer != "" {
		return f.viewer
	}
	return "fak-review-bot"
}

func thoughtCheckComment(id int64, body, login string) thoughtCheckCommentJSON {
	comment := thoughtCheckCommentJSON{ID: id, Body: body}
	comment.User.Login = login
	return comment
}

func queryInt(endpoint, key string) (int, error) {
	at := strings.IndexByte(endpoint, '?')
	if at >= 0 {
		for _, field := range strings.Split(endpoint[at+1:], "&") {
			parts := strings.SplitN(field, "=", 2)
			if len(parts) == 2 && parts[0] == key {
				return strconv.Atoi(parts[1])
			}
		}
	}
	return 0, fmt.Errorf("endpoint %q missing %s", endpoint, key)
}

func endpointID(endpoint string) (int64, error) {
	raw := endpoint[strings.LastIndex(endpoint, "/")+1:]
	return strconv.ParseInt(raw, 10, 64)
}

func testThoughtCheckIssue() issuecheck.Issue {
	return issuecheck.Issue{
		Number: 17,
		Title:  "fix(cache): stop stale entries after migration",
		Body:   "The migration changes cache keys. Rollback and invalidation evidence are missing.",
		Labels: []string{"bug", "reliability"},
	}
}

func testThoughtCheckReview(t *testing.T, issue issuecheck.Issue) issuecheck.Review {
	t.Helper()
	digest, err := issuecheck.IssueDigest(issue)
	if err != nil {
		t.Fatal(err)
	}
	rows := make([]issuecheck.ReviewRow, 0, 5)
	for i, check := range issuecheck.Catalog()[:5] {
		rows = append(rows, issuecheck.ReviewRow{
			ID:         check.ID,
			Relevance:  fmt.Sprintf("Issue-specific relevance %d: the cache migration changes keys and invalidation behavior.", i+1),
			Assessment: fmt.Sprintf("Assessment %d: the issue does not yet prove the cache behavior across rollback.", i+1),
			Evidence:   issuecheck.Evidence{Status: issuecheck.EvidenceSupported, Refs: []string{"Issue body: migration changes cache keys"}},
			Action:     fmt.Sprintf("Add acceptance criterion %d with a rollback fixture.", i+1),
		})
	}
	return issuecheck.Review{
		Schema: issuecheck.ReviewSchema, IssueNumber: issue.Number,
		IssueBinding: issuecheck.CanonicalIssueBinding(issue), IssueDigest: digest,
		CatalogVersion: issuecheck.CatalogVersion, ReviewerVersion: "test-reviewer/v1", Rows: rows,
	}
}

func writeThoughtCheckReview(t *testing.T, review issuecheck.Review) string {
	t.Helper()
	path := t.TempDir() + string(os.PathSeparator) + "review.json"
	raw, err := json.Marshal(review)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, raw, 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func runThoughtCheckUpsertTest(t *testing.T, fake *thoughtCheckFakeGH, review issuecheck.Review, live bool) (int, string, string) {
	t.Helper()
	path := writeThoughtCheckReview(t, review)
	args := []string{"--issue", strconv.Itoa(fake.issue.Number), "--input", path, "--repo", "owner/repo"}
	if live {
		args = append(args, "--live")
	}
	var out, errb bytes.Buffer
	code := runThoughtCheckUpsert(&out, &errb, args, fake.run)
	return code, out.String(), errb.String()
}

func decodeThoughtCheckUpsertResult(t *testing.T, raw string) thoughtCheckUpsertResult {
	t.Helper()
	var result thoughtCheckUpsertResult
	if err := json.Unmarshal([]byte(raw), &result); err != nil {
		t.Fatalf("decode result: %v\n%s", err, raw)
	}
	return result
}

func installThoughtCheckOperationContext(t *testing.T, ctx context.Context, cancel context.CancelFunc) {
	t.Helper()
	previous := newThoughtCheckOperationContext
	newThoughtCheckOperationContext = func() (context.Context, context.CancelFunc) {
		return ctx, func() {}
	}
	t.Cleanup(func() {
		newThoughtCheckOperationContext = previous
		cancel()
	})
}

func TestThoughtCheckCatalogJSON(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runThoughtCheckCatalog(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got thoughtCheckCatalogResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != thoughtCheckCatalogSchema || got.CatalogVersion != issuecheck.CatalogVersion ||
		len(got.Checks) != 30 || got.Checks[0].ID != "TC-01" || got.Checks[29].ID != "TC-30" {
		t.Fatalf("catalog result = %+v", got)
	}
}

func TestThoughtCheckPrepareBindsScaffoldToCurrentIssue(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue}
	var out, errb bytes.Buffer
	code := runThoughtCheckPrepare(&out, &errb, []string{"--issue", "17", "--repo", "owner/repo", "--json"}, fake.run)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var got thoughtCheckPrepareResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	wantDigest, _ := issuecheck.IssueDigest(issue)
	wantBinding := issuecheck.CanonicalIssueBinding(issue)
	if got.Schema != thoughtCheckPrepareSchema || got.Issue.Number != 17 || got.IssueDigest != wantDigest ||
		got.ReviewTemplate.Schema != issuecheck.ReviewSchema || got.ReviewTemplate.IssueDigest != wantDigest ||
		got.ReviewTemplate.IssueBinding != wantBinding ||
		got.ReviewTemplate.CatalogVersion != issuecheck.CatalogVersion || len(got.ReviewTemplate.Rows) != 0 || len(got.Checks) != 30 ||
		!equalStrings(got.EvidenceStatusTokens, []string{issuecheck.EvidenceSupported, issuecheck.EvidencePartial, issuecheck.EvidenceGap}) {
		t.Fatalf("prepare result = %+v", got)
	}
	var envelope map[string]json.RawMessage
	if err := json.Unmarshal(out.Bytes(), &envelope); err != nil {
		t.Fatal(err)
	}
	var rowShape map[string]json.RawMessage
	if err := json.Unmarshal(envelope["row_template"], &rowShape); err != nil {
		t.Fatal(err)
	}
	if len(rowShape) != 5 || rowShape["id"] == nil || rowShape["relevance"] == nil || rowShape["assessment"] == nil || rowShape["evidence"] == nil || rowShape["action"] == nil {
		t.Fatalf("row template does not have the exact closed shape: %s", envelope["row_template"])
	}
	var evidenceShape map[string]json.RawMessage
	if err := json.Unmarshal(rowShape["evidence"], &evidenceShape); err != nil {
		t.Fatal(err)
	}
	if len(evidenceShape) != 3 || evidenceShape["status"] == nil || evidenceShape["refs"] == nil || evidenceShape["gap"] == nil {
		t.Fatalf("evidence template does not have the exact closed shape: %s", rowShape["evidence"])
	}

	// Prove the scaffold is mechanically completable without inventing keys or
	// status tokens, then accepted by the strict dry-run edge.
	got.ReviewTemplate.ReviewerVersion = "prepare-roundtrip/v1"
	for i, check := range got.Checks[:5] {
		rowRaw, err := json.Marshal(got.RowTemplate)
		if err != nil {
			t.Fatal(err)
		}
		var row issuecheck.ReviewRow
		if err := json.Unmarshal(rowRaw, &row); err != nil {
			t.Fatal(err)
		}
		row.ID = check.ID
		row.Relevance = fmt.Sprintf("Prepared issue-specific relevance %d for the cache migration.", i+1)
		row.Assessment = fmt.Sprintf("Prepared conclusion %d identifies rollback proof as missing.", i+1)
		row.Evidence.Status = got.EvidenceStatusTokens[0]
		row.Evidence.Refs = []string{"Issue body: migration changes cache keys"}
		row.Evidence.Gap = ""
		row.Action = fmt.Sprintf("Add prepared rollback acceptance criterion %d.", i+1)
		got.ReviewTemplate.Rows = append(got.ReviewTemplate.Rows, row)
	}
	code, dryOut, dryErr := runThoughtCheckUpsertTest(t, fake, got.ReviewTemplate, false)
	if code != 0 || decodeThoughtCheckUpsertResult(t, dryOut).Action != issuecheck.ActionCreate {
		t.Fatalf("prepared review dry-run exit=%d stderr=%s stdout=%s", code, dryErr, dryOut)
	}
}

func equalStrings(got, want []string) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

func TestThoughtCheckUpsertDryRunPlansWithoutMutation(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue}
	code, out, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), false)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	result := decodeThoughtCheckUpsertResult(t, out)
	if result.Action != issuecheck.ActionCreate || result.Live || result.Verified || fake.mutationCount != 0 {
		t.Fatalf("result=%+v mutations=%d", result, fake.mutationCount)
	}
}

func TestThoughtCheckUpsertCreateNoopAndPatchSameID(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	desired, err := issuecheck.FormatReviewComment(issue, review)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("create", func(t *testing.T) {
		fake := &thoughtCheckFakeGH{issue: issue, nextID: 7001}
		code, out, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb)
		}
		result := decodeThoughtCheckUpsertResult(t, out)
		if result.Action != issuecheck.ActionCreate || result.CommentID != 7001 || !result.Verified || fake.mutationCount != 1 || fake.issueReadCount != 2 || fake.commentPageCount != 2 {
			t.Fatalf("result=%+v mutations=%d", result, fake.mutationCount)
		}
	})

	t.Run("noop", func(t *testing.T) {
		fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{thoughtCheckComment(81, desired, "fak-review-bot")}}
		code, out, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb)
		}
		result := decodeThoughtCheckUpsertResult(t, out)
		if result.Action != issuecheck.ActionNoop || result.CommentID != 81 || !result.Verified || fake.mutationCount != 0 || fake.issueReadCount != 2 || fake.commentPageCount != 2 {
			t.Fatalf("result=%+v mutations=%d issue_reads=%d comment_pages=%d", result, fake.mutationCount, fake.issueReadCount, fake.commentPageCount)
		}
	})

	t.Run("patch same id", func(t *testing.T) {
		oldReview := review
		oldReview.ReviewerVersion = "test-reviewer/previous"
		oldBody, err := issuecheck.FormatReviewComment(issue, oldReview)
		if err != nil {
			t.Fatal(err)
		}
		fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{thoughtCheckComment(82, oldBody, "fak-review-bot")}}
		code, out, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb)
		}
		result := decodeThoughtCheckUpsertResult(t, out)
		if result.Action != issuecheck.ActionUpdate || result.CommentID != 82 || !result.Verified || fake.mutationCount != 1 || fake.issueReadCount != 2 || fake.commentPageCount != 2 {
			t.Fatalf("result=%+v mutations=%d", result, fake.mutationCount)
		}
		if len(fake.comments) != 1 || fake.comments[0].ID != 82 || fake.comments[0].Body != desired {
			t.Fatalf("patch did not preserve id/body: %+v", fake.comments)
		}
	})

	t.Run("patch structurally valid stale marker after title edit", func(t *testing.T) {
		priorIssue := issue
		priorIssue.Title = "fix(cache): document the migration"
		priorIssue.Body = "The migration changes cache keys; initial scope omitted invalidation details."
		priorReview := testThoughtCheckReview(t, priorIssue)
		priorBody, err := issuecheck.FormatReviewComment(priorIssue, priorReview)
		if err != nil {
			t.Fatal(err)
		}
		if got := issuecheck.InspectManagedComment(issue, priorBody).State; got != issuecheck.ManagedCommentStale {
			t.Fatalf("prior marker state=%q, want stale", got)
		}
		fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{thoughtCheckComment(83, priorBody, "fak-review-bot")}}
		code, out, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 0 {
			t.Fatalf("exit=%d stderr=%s", code, errb)
		}
		result := decodeThoughtCheckUpsertResult(t, out)
		if result.Action != issuecheck.ActionUpdate || result.CommentID != 83 || !result.Verified || fake.mutationCount != 1 {
			t.Fatalf("result=%+v mutations=%d", result, fake.mutationCount)
		}
	})
}

func TestThoughtCheckUpsertIgnoresUnrelatedAndDraftMarkers(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(1, "ordinary discussion", "fak-review-bot"),
		thoughtCheckComment(2, "<!-- fak-issuecheck: top-five-draft -->\nmanual notes", "fak-review-bot"),
		thoughtCheckComment(3, "example prose mentions fak-issuecheck", "issue-author"),
	}}
	code, out, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), false)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	if got := decodeThoughtCheckUpsertResult(t, out).Action; got != issuecheck.ActionCreate {
		t.Fatalf("action=%q, want create", got)
	}
}

func TestThoughtCheckCommentDiscoveryIgnoresForeignProductionMarker(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue, viewer: "owned-bot", comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(10, issuecheck.CommentMarker+"\nowned", "owned-bot"),
		thoughtCheckComment(11, issuecheck.CommentMarker+"\nforeign", "issue-author"),
		thoughtCheckComment(12, "<!-- fak-issuecheck: top-five-draft -->\ndraft", "owned-bot"),
	}}
	if !issuecheck.IsManagedComment(fake.comments[0].Body) {
		t.Fatalf("core did not recognize exact production marker %q", fake.comments[0].Body)
	}
	raw, err := json.Marshal(fake.comments)
	if err != nil {
		t.Fatal(err)
	}
	var roundTrip []thoughtCheckCommentJSON
	if err := json.Unmarshal(raw, &roundTrip); err != nil {
		t.Fatal(err)
	}
	if len(roundTrip) != 3 || roundTrip[0].User.Login != "owned-bot" {
		t.Fatalf("GitHub comment JSON lost author identity: %s -> %+v", raw, roundTrip)
	}
	got, err := fetchThoughtCheckComments(context.Background(), fake.run, "owner/repo", issue.Number, "owned-bot")
	if err != nil || len(got) != 1 || got[0].ID != 10 {
		t.Fatalf("managed comments = %+v err=%v, want only stable owner's marker; raw=%+v", got, err, fake.comments)
	}
}

func TestThoughtCheckUpsertRefusesMutationWhenViewerDiffersFromStableOwner(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	oldReview := review
	oldReview.ReviewerVersion = "owner-review/previous"
	body, err := issuecheck.FormatReviewComment(issue, oldReview)
	if err != nil {
		t.Fatal(err)
	}
	fake := &thoughtCheckFakeGH{issue: issue, repoOwner: "old-bot", viewer: "new-bot", comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(41, body, "old-bot"),
	}}
	code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
	if code != 1 || fake.mutationCount != 0 || !strings.Contains(errb, `actor "new-bot" is not stable repository owner "old-bot"`) {
		t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
	}
}

func TestThoughtCheckUpsertPreservesStableOwnerCommentIDAcrossCredentialCheck(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	oldReview := review
	oldReview.ReviewerVersion = "owner-review/previous"
	oldBody, err := issuecheck.FormatReviewComment(issue, oldReview)
	if err != nil {
		t.Fatal(err)
	}
	fake := &thoughtCheckFakeGH{issue: issue, repoOwner: "repo-owner", viewer: "repo-owner", comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(51, oldBody, "repo-owner"),
		thoughtCheckComment(52, issuecheck.CommentMarker+"\nforeign spoof", "attacker"),
	}}
	code, out, errb := runThoughtCheckUpsertTest(t, fake, review, true)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	result := decodeThoughtCheckUpsertResult(t, out)
	if result.Action != issuecheck.ActionUpdate || result.CommentID != 51 || !result.Verified || fake.mutationCount != 1 {
		t.Fatalf("result=%+v mutations=%d comments=%+v", result, fake.mutationCount, fake.comments)
	}
	if fake.comments[0].ID != 51 || fake.comments[1].ID != 52 || fake.comments[1].Body != issuecheck.CommentMarker+"\nforeign spoof" {
		t.Fatalf("owner continuity or foreign spoof isolation lost: %+v", fake.comments)
	}
	if len(fake.contexts) < 2 {
		t.Fatalf("GitHub contexts=%d, want a multi-request operation", len(fake.contexts))
	}
	for i := 1; i < len(fake.contexts); i++ {
		if fake.contexts[i] != fake.contexts[0] {
			t.Fatalf("GitHub request %d received a different operation context", i+1)
		}
	}
}

func TestThoughtCheckUpsertRefusesDuplicateProductionMarkers(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(20, issuecheck.CommentMarker+"\none", "fak-review-bot"),
		thoughtCheckComment(21, issuecheck.CommentMarker+"\ntwo", "fak-review-bot"),
	}}
	code, out, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), true)
	if code != 1 || fake.mutationCount != 0 || !strings.Contains(errb, "multiple managed") {
		t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
	}
	result := decodeThoughtCheckUpsertResult(t, out)
	if result.Action != issuecheck.ActionRefuse || len(result.Matches) != 2 {
		t.Fatalf("result=%+v", result)
	}
}

func TestThoughtCheckCommentsPaginateBeforePlanning(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	desired, err := issuecheck.FormatReviewComment(issue, review)
	if err != nil {
		t.Fatal(err)
	}
	comments := make([]thoughtCheckCommentJSON, 100)
	for i := range comments {
		comments[i] = thoughtCheckComment(int64(i+1), "unrelated", "fak-review-bot")
	}
	comments = append(comments, thoughtCheckComment(500, desired, "fak-review-bot"))
	fake := &thoughtCheckFakeGH{issue: issue, comments: comments}
	code, out, errb := runThoughtCheckUpsertTest(t, fake, review, false)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb)
	}
	if got := decodeThoughtCheckUpsertResult(t, out); got.Action != issuecheck.ActionNoop || got.CommentID != 500 {
		t.Fatalf("result=%+v", got)
	}
	if fake.commentPageCount != 2 {
		t.Fatalf("comment page calls=%d, want 2", fake.commentPageCount)
	}
}

func TestThoughtCheckUpsertFailsClosedOnMalformedAndStaleReview(t *testing.T) {
	t.Run("malformed unknown field before github", func(t *testing.T) {
		path := t.TempDir() + string(os.PathSeparator) + "bad.json"
		if err := os.WriteFile(path, []byte(`{"schema":"x","surprise":true}`), 0o600); err != nil {
			t.Fatal(err)
		}
		fake := &thoughtCheckFakeGH{issue: testThoughtCheckIssue()}
		var out, errb bytes.Buffer
		code := runThoughtCheckUpsert(&out, &errb, []string{"--issue", "17", "--input", path}, fake.run)
		if code != 1 || len(fake.calls) != 0 || !strings.Contains(errb.String(), "unknown field") {
			t.Fatalf("exit=%d calls=%v stderr=%s", code, fake.calls, errb.String())
		}
	})

	t.Run("stale digest before comments", func(t *testing.T) {
		issue := testThoughtCheckIssue()
		review := testThoughtCheckReview(t, issue)
		fake := &thoughtCheckFakeGH{issue: issue}
		fake.issue.Body += "\nnew scope"
		code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 1 || fake.commentPageCount != 0 || fake.mutationCount != 0 || !strings.Contains(errb, "stale") {
			t.Fatalf("exit=%d pages=%d mutations=%d stderr=%s", code, fake.commentPageCount, fake.mutationCount, errb)
		}
	})
}

func TestThoughtCheckUpsertFailsOnMutationAndReadbackMismatch(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)

	t.Run("mutation failure", func(t *testing.T) {
		fake := &thoughtCheckFakeGH{issue: issue, mutationErr: errors.New("permission denied")}
		code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 1 || fake.mutationCount != 1 || !strings.Contains(errb, "permission denied") {
			t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
		}
	})

	t.Run("readback mismatch", func(t *testing.T) {
		bad := issuecheck.CommentMarker + "\ntampered"
		fake := &thoughtCheckFakeGH{issue: issue, postMutationBody: &bad}
		code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 1 || fake.mutationCount != 1 || !strings.Contains(errb, "postcondition") {
			t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
		}
	})
}

func TestThoughtCheckUpsertPostconditionClosesIssueAndCommentRaces(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	desired, err := issuecheck.FormatReviewComment(issue, review)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("issue edited after create", func(t *testing.T) {
		changed := issue
		changed.Body += "\nConcurrent scope edit."
		fake := &thoughtCheckFakeGH{issue: issue, issueAfterMutation: &changed}
		code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 1 || fake.mutationCount != 1 || fake.issueReadCount != 2 || !strings.Contains(errb, "postcondition") || !strings.Contains(errb, "stale") {
			t.Fatalf("exit=%d mutations=%d issue_reads=%d stderr=%s", code, fake.mutationCount, fake.issueReadCount, errb)
		}
	})

	t.Run("concurrent create produces duplicate", func(t *testing.T) {
		concurrent := thoughtCheckComment(902, desired, "fak-review-bot")
		fake := &thoughtCheckFakeGH{issue: issue, nextID: 901, concurrentComment: &concurrent}
		code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
		if code != 1 || fake.mutationCount != 1 || !strings.Contains(errb, "multiple managed") {
			t.Fatalf("exit=%d mutations=%d stderr=%s comments=%+v", code, fake.mutationCount, errb, fake.comments)
		}
	})
}

func TestThoughtCheckUpsertRefusesMalformedOwnedMarker(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(88, issuecheck.CommentMarker+"\nnot a signed render", "fak-review-bot"),
	}}
	code, _, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), true)
	if code != 1 || fake.mutationCount != 0 || !strings.Contains(errb, "refusing malformed owned production marker") {
		t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
	}
}

func TestThoughtCheckUpsertRefusesVisibleTamperWithValidPayload(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	body, err := issuecheck.FormatReviewComment(issue, review)
	if err != nil {
		t.Fatal(err)
	}
	tampered := strings.Replace(body, "Issue-specific relevance 1", "attacker-supplied visible conclusion", 1)
	if tampered == body {
		t.Fatal("test did not alter visible conclusion")
	}
	if got := issuecheck.InspectManagedComment(issue, tampered); got.State != issuecheck.ManagedCommentMalformed {
		t.Fatalf("tampered marker state=%q reason=%q, want malformed", got.State, got.Reason)
	}
	fake := &thoughtCheckFakeGH{issue: issue, comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(89, tampered, "fak-review-bot"),
	}}
	code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
	if code != 1 || fake.mutationCount != 0 || !strings.Contains(errb, "refusing malformed owned production marker") {
		t.Fatalf("exit=%d mutations=%d stderr=%s", code, fake.mutationCount, errb)
	}
}

func TestThoughtCheckVerifyUsesPureExactCommentVerification(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)
	desired, err := issuecheck.FormatReviewComment(issue, review)
	if err != nil {
		t.Fatal(err)
	}
	fake := &thoughtCheckFakeGH{issue: issue, repoOwner: "repo-owner", viewer: "unrelated-viewer", comments: []thoughtCheckCommentJSON{
		thoughtCheckComment(404, desired, "repo-owner"),
		thoughtCheckComment(405, issuecheck.CommentMarker+"\nforeign spoof", "attacker"),
	}}
	var out, errb bytes.Buffer
	code := runThoughtCheckVerify(&out, &errb, []string{"--issue", "17", "--repo", "owner/repo", "--json"}, fake.run)
	if code != 0 {
		t.Fatalf("exit=%d stderr=%s", code, errb.String())
	}
	var result thoughtCheckVerifyResult
	if err := json.Unmarshal(out.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.OK || result.CommentID != 404 || result.IssueDigest != review.IssueDigest || result.CatalogVersion != issuecheck.CatalogVersion {
		t.Fatalf("result=%+v", result)
	}
}

func TestThoughtCheckOperationsHonorExpiredAggregateDeadline(t *testing.T) {
	issue := testThoughtCheckIssue()
	review := testThoughtCheckReview(t, issue)

	tests := []struct {
		name string
		run  func(*thoughtCheckFakeGH) (int, string)
	}{
		{name: "prepare", run: func(fake *thoughtCheckFakeGH) (int, string) {
			var out, errb bytes.Buffer
			code := runThoughtCheckPrepare(&out, &errb, []string{"--issue", "17", "--repo", "owner/repo", "--json"}, fake.run)
			return code, errb.String()
		}},
		{name: "upsert", run: func(fake *thoughtCheckFakeGH) (int, string) {
			code, _, errb := runThoughtCheckUpsertTest(t, fake, review, true)
			return code, errb
		}},
		{name: "verify", run: func(fake *thoughtCheckFakeGH) (int, string) {
			var out, errb bytes.Buffer
			code := runThoughtCheckVerify(&out, &errb, []string{"--issue", "17", "--repo", "owner/repo", "--json"}, fake.run)
			return code, errb.String()
		}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithDeadline(context.Background(), time.Unix(0, 0))
			installThoughtCheckOperationContext(t, ctx, cancel)
			fake := &thoughtCheckFakeGH{issue: issue}
			code, errb := tt.run(fake)
			if code != 1 || len(fake.calls) != 0 || !strings.Contains(errb, context.DeadlineExceeded.Error()) {
				t.Fatalf("exit=%d calls=%v stderr=%s", code, fake.calls, errb)
			}
		})
	}
}

func TestThoughtCheckUpsertUsesOneContextAcrossAllGitHubRequests(t *testing.T) {
	issue := testThoughtCheckIssue()
	ctx, cancel := context.WithCancel(context.Background())
	installThoughtCheckOperationContext(t, ctx, cancel)
	fake := &thoughtCheckFakeGH{issue: issue, cancel: cancel, cancelAfterCall: 1}
	code, _, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), true)
	if code != 1 || len(fake.calls) != 1 || !strings.Contains(errb, context.Canceled.Error()) {
		t.Fatalf("exit=%d calls=%v stderr=%s", code, fake.calls, errb)
	}
	if len(fake.contexts) != 1 || fake.contexts[0] != ctx {
		t.Fatalf("contexts=%v, want the one operation context", fake.contexts)
	}
}

func TestThoughtCheckRefusesOrganizationOwnerWithoutStableProducerConfig(t *testing.T) {
	issue := testThoughtCheckIssue()
	fake := &thoughtCheckFakeGH{issue: issue, repoOwner: "example-org", repoOwnerType: "Organization"}
	code, _, errb := runThoughtCheckUpsertTest(t, fake, testThoughtCheckReview(t, issue), false)
	if code != 1 || fake.commentPageCount != 0 || fake.mutationCount != 0 || !strings.Contains(errb, "configure a stable producer identity") {
		t.Fatalf("exit=%d comment_pages=%d mutations=%d stderr=%s", code, fake.commentPageCount, fake.mutationCount, errb)
	}
}
