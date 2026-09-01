package main

// Tests for `fak steer comment` (#5029): the ANNOTATE rung posts the anchored note
// through the (test-overridden) trusted gh seam, ledgers the annotation as an
// attributable append-only row, and refuses every case where there is no honest place
// to post — without ledgering anything. The structural no-git-mutation fence lives
// beside the leaf in internal/steerpr/comment_test.go; these cover the verb shell and
// the route, so a rung that is never reachable from `fak steer` fails here.

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
	"github.com/anthony-chaudhary/fak/internal/workerworktree"
)

// steerUnboundUnitLog is a range whose only stamped commit binds NO issue: the gateway
// unit forms, but with an empty Resolves set. It is the log the refusal path needs, and
// prPlanFakeLog cannot serve it — there the feat member carries #1146.
const steerUnboundUnitLog = "\x1ebbb2222222222222222222222222222222222222\x1ffix(gateway): drop duplicate counter (fak gateway)\x1f\x1f\ninternal/gateway/messages.go\n"

// withSteerCommentSeam swaps the trusted gh seam for a recorder so no test ever reaches
// the network, returning the captured records.
func withSteerCommentSeam(t *testing.T, posted string, err error) *[]steerpr.Comment {
	t.Helper()
	var got []steerpr.Comment
	orig := steerCommentPost
	steerCommentPost = func(c steerpr.Comment) (string, error) {
		got = append(got, c)
		return posted, err
	}
	t.Cleanup(func() { steerCommentPost = orig })
	return &got
}

// countCommentRows reads the overlay ledger the way the brief does.
func countCommentRows(t *testing.T, root string) []steerpr.Comment {
	t.Helper()
	return steerpr.LoadComments(steerpr.CommentLedgerPath(root))
}

// The happy path, driven through the REAL `fak steer` router so the #5029 route hunk is
// witnessed too: the note posts to the unit's OWN bound issue, anchored to the exact
// member SHA set and band the operator was reading, and the annotation is ledgered as
// one attributable row carrying where it landed.
func TestSteerCommentPostsToTheBoundIssueAndLedgersTheAnnotation(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	posts := withSteerCommentSeam(t, "https://example.invalid/issues/1146#issuecomment-7", nil)
	origLifecycle := steerLifecycleStatusUpsert
	var lifecycleIssue int
	var lifecycleBody string
	steerLifecycleStatusUpsert = func(issue int, body string) (steerLifecycleStatusAction, error) {
		lifecycleIssue, lifecycleBody = issue, body
		return steerLifecycleStatusCreated, nil
	}
	t.Cleanup(func() { steerLifecycleStatusUpsert = origLifecycle })

	var stdout, stderr bytes.Buffer
	code := runSteer(&stdout, &stderr, []string{"comment", "gateway",
		"-m", "the counter fix reads right; the ready-tick change is the one to look at",
		"--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}

	var row steerpr.Comment
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}
	if row.Schema != steerpr.CommentSchema || row.Leaf != "gateway" || row.By != "op-jane" || row.At == "" {
		t.Fatalf("echoed row = %#v, want an attributable %s row", row, steerpr.CommentSchema)
	}
	// The unit's CLOSURE-GRADE binding, not its mentions: prPlanFakeLog's feat member
	// resolves #1146 and merely mentions #999. Posting to #999 would put operator
	// reasoning on a ticket the unit never claimed to close.
	if row.Issue != "#1146" {
		t.Fatalf("row issue = %q, want the unit's closure-grade #1146 (never the mentioned #999)", row.Issue)
	}
	if len(row.SHAs) != 2 || row.SHAs[0] != steerFeatSHA || row.SHAs[1] != steerFixSHA {
		t.Fatalf("row anchor SHAs = %v, want the unit's exact member set [%s %s]", row.SHAs, steerFeatSHA, steerFixSHA)
	}
	if row.Band != steerpr.BandResidual {
		t.Fatalf("row anchor band = %q, want the unit's band at comment time (RESIDUAL)", row.Band)
	}
	if row.Posted != "https://example.invalid/issues/1146#issuecomment-7" {
		t.Fatalf("row posted = %q, want the seam's reported comment ref", row.Posted)
	}

	// The seam saw ONE record, and the body it would post carries the anchor rather
	// than only the unit's name — a name means different commits tomorrow.
	if len(*posts) != 1 {
		t.Fatalf("gh seam called %d time(s), want exactly 1", len(*posts))
	}
	if lifecycleIssue != 1146 || !strings.Contains(lifecycleBody, "State: `active`") || strings.Contains(lifecycleBody, "Complete: yes") {
		t.Fatalf("lifecycle projection issue=%d body=%q", lifecycleIssue, lifecycleBody)
	}
	body := (*posts)[0].Body()
	for _, want := range []string{"gateway", steerFeatSHA[:7], steerFixSHA[:7], "the counter fix reads right"} {
		if !strings.Contains(body, want) {
			t.Fatalf("posted body missing %q:\n%s", want, body)
		}
	}

	// One countable ledger row, matching what was echoed.
	rows := countCommentRows(t, root)
	if len(rows) != 1 || rows[0].Issue != "#1146" || rows[0].By != "op-jane" {
		t.Fatalf("ledger = %#v, want exactly one attributable row bound to #1146", rows)
	}

	// ANNOTATE-ONLY: the prs view still reads the same band and residual count. A
	// comment that moved either would be a stronger rung wearing this one's name.
	view, err := buildSteerPRsView(root, "baseref", "headref")
	if err != nil {
		t.Fatalf("buildSteerPRsView: %v", err)
	}
	if got := view["residual_count"].(int); got != 1 {
		t.Fatalf("residual_count after commenting = %d, want 1 (unchanged)", got)
	}
}

// A unit that binds NO issue is refused, and nothing is posted or ledgered. This is the
// rung's central rule: a mention is not a binding, so there is no honest place to put
// the note, and guessing a plausible ticket is worse than not annotating at all.
func TestSteerCommentRefusesAnUnboundUnitWithoutPostingOrLedgering(t *testing.T) {
	withSteerFakes(t, steerUnboundUnitLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	posts := withSteerCommentSeam(t, "https://example.invalid/should-not-be-reached", nil)

	var stdout, stderr bytes.Buffer
	code := runSteer(&stdout, &stderr, []string{"comment", "gateway",
		"-m", "nowhere to put this", "--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 (refusal); stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "binds no issue") {
		t.Fatalf("refusal should say the unit binds no issue: %s", stderr.String())
	}
	if len(*posts) != 0 {
		t.Fatalf("a refused comment still reached the gh seam %d time(s)", len(*posts))
	}
	if rows := countCommentRows(t, root); len(rows) != 0 {
		t.Fatalf("a refused comment still ledgered %d row(s)", len(rows))
	}
}

// A post that FAILS ledgers nothing. The verb posts first on purpose — a note that never
// landed is not operator attention — so a ledger row here would tell the brief a unit got
// attention it never got, which is the one lie this ledger exists to prevent.
func TestSteerCommentLedgersNothingWhenThePostFails(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	withSteerCommentSeam(t, "", errors.New("gh issue comment 1146: exit status 1"))

	var stdout, stderr bytes.Buffer
	code := runSteer(&stdout, &stderr, []string{"comment", "gateway",
		"-m", "this note never lands", "--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1 when the post fails; stdout=%s", code, stdout.String())
	}
	if !strings.Contains(stderr.String(), "post via gh") {
		t.Fatalf("stderr should name the failed post: %s", stderr.String())
	}
	if rows := countCommentRows(t, root); len(rows) != 0 {
		t.Fatalf("a failed post still ledgered %d row(s) — the brief would read attention that never happened", len(rows))
	}
}

// The usage refusals: an empty note and an unknown unit both stop before the seam. An
// empty note is a usage error (2) because the operator can fix it; an unknown unit is a
// run error (1) because the range genuinely does not contain it.
func TestSteerCommentRefusesIncompleteInvocations(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	withSteerRoot(t)
	posts := withSteerCommentSeam(t, "https://example.invalid/should-not-be-reached", nil)

	for _, tc := range []struct {
		name string
		argv []string
		want int
	}{
		{"no note", []string{"comment", "gateway", "--by", "op-jane", "--base", "baseref", "--head", "headref"}, 2},
		{"no unit", []string{"comment", "-m", "a note with no unit"}, 2},
		{"unknown unit", []string{"comment", "nosuchleaf", "-m", "a note", "--by", "op-jane", "--base", "baseref", "--head", "headref"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runSteer(&stdout, &stderr, tc.argv); code != tc.want {
				t.Fatalf("exit = %d, want %d; stdout=%s stderr=%s", code, tc.want, stdout.String(), stderr.String())
			}
		})
	}
	if len(*posts) != 0 {
		t.Fatalf("a refused invocation still reached the gh seam %d time(s)", len(*posts))
	}
}

func TestRenderSteerLifecycleStatusIsDeterministicAndCompletionIsWitnessed(t *testing.T) {
	cases := []struct {
		name       string
		status     workerworktree.StatusProjection
		wantState  workerworktree.DisplayState
		wantPhrase string
		complete   bool
	}{
		{
			name: "active",
			status: workerworktree.StatusProjection{
				State: workerworktree.DisplayActive, IssueNumber: 10551, Lane: "workerworktree", Session: "RID-7",
			},
			wantState: workerworktree.DisplayActive,
		},
		{
			name: "unlanded changes",
			status: workerworktree.StatusProjection{
				State: workerworktree.DisplayUnlandedChanges, IssueNumber: 10551, Lane: "workerworktree",
			},
			wantState: workerworktree.DisplayUnlandedChanges,
		},
		{
			name: "cleanup ready is incomplete",
			status: workerworktree.StatusProjection{
				State: workerworktree.DisplayCleanupReady, Complete: true, IssueNumber: 10551, Lane: "workerworktree",
			},
			wantState:  workerworktree.DisplayCleanupReady,
			wantPhrase: "cleanup-ready residue is not landed completion",
		},
		{
			name: "landed without evidence is incomplete",
			status: workerworktree.StatusProjection{
				State: workerworktree.DisplayLandedWitnessed, Complete: true, IssueNumber: 10551, Lane: "workerworktree",
			},
			wantState: workerworktree.DisplayActive,
		},
		{
			name: "landed with evidence is complete",
			status: workerworktree.StatusProjection{
				State: workerworktree.DisplayLandedWitnessed, Complete: false, IssueNumber: 10551, Lane: "workerworktree", Commit: "ABCDEF012345",
			},
			wantState: workerworktree.DisplayLandedWitnessed,
			complete:  true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			body := renderSteerLifecycleStatus(tc.status)
			if again := renderSteerLifecycleStatus(tc.status); again != body {
				t.Fatalf("renderer is not deterministic:\nfirst: %q\nagain: %q", body, again)
			}
			if strings.Count(body, steerLifecycleStatusMarkerStart) != 1 || strings.Count(body, steerLifecycleStatusMarkerEnd) != 1 {
				t.Fatalf("marker bounds = %q", body)
			}
			if !strings.Contains(body, fmt.Sprintf("- State: `%s`", tc.wantState)) {
				t.Fatalf("body missing state %q: %s", tc.wantState, body)
			}
			if tc.complete != strings.Contains(body, "- Complete: yes") {
				t.Fatalf("complete rendering mismatch: %s", body)
			}
			if tc.wantPhrase != "" && !strings.Contains(body, tc.wantPhrase) {
				t.Fatalf("body missing %q: %s", tc.wantPhrase, body)
			}
			if strings.Contains(body, "2026-") || strings.Contains(body, `C:\`) || strings.Contains(body, "/tmp/") {
				t.Fatalf("body leaked a timestamp or path: %s", body)
			}
		})
	}
}

func TestRenderSteerLifecycleStatusScrubsPathIdentities(t *testing.T) {
	body := renderSteerLifecycleStatus(workerworktree.StatusProjection{
		State:       workerworktree.DisplayActive,
		IssueNumber: 10551,
		Lane:        `C:\Users\USER\worker-worktree`,
		Session:     "/tmp/RID-7",
	})
	if strings.Contains(body, "Users") || strings.Contains(body, "/tmp") || strings.Contains(body, "Lane:") || strings.Contains(body, "Session:") {
		t.Fatalf("path identity escaped renderer: %s", body)
	}
}

func TestUpsertSteerLifecycleStatusRequiresOneClosureGradeIssue(t *testing.T) {
	orig := steerLifecycleStatusUpsert
	defer func() { steerLifecycleStatusUpsert = orig }()
	var calls int
	steerLifecycleStatusUpsert = func(issue int, body string) (steerLifecycleStatusAction, error) {
		calls++
		return steerLifecycleStatusCreated, nil
	}
	status := workerworktree.StatusProjection{State: workerworktree.DisplayActive, IssueNumber: 10551}
	for _, resolves := range [][]string{nil, {}, {"#10551", "#10552"}, {"mention #10551"}} {
		action, err := upsertSteerLifecycleStatus(resolves, status)
		if err != nil || action != steerLifecycleStatusSkipped {
			t.Fatalf("resolves %#v = (%q, %v), want skipped", resolves, action, err)
		}
	}
	if calls != 0 {
		t.Fatalf("ambiguous/missing association made %d writes", calls)
	}
	status.IssueNumber = 10552
	if action, err := upsertSteerLifecycleStatus([]string{"#10551"}, status); err != nil || action != steerLifecycleStatusSkipped {
		t.Fatalf("mismatched projection association = (%q, %v)", action, err)
	}
	if calls != 0 {
		t.Fatalf("mismatched association made %d writes", calls)
	}
	status.IssueNumber = 10551
	if action, err := upsertSteerLifecycleStatus([]string{"#10551", "#10551"}, status); err != nil || action != steerLifecycleStatusCreated {
		t.Fatalf("one deduplicated association = (%q, %v)", action, err)
	}
	if calls != 1 {
		t.Fatalf("safe association calls = %d, want 1", calls)
	}
}

func TestGHSteerLifecycleStatusUpsertCreateUpdateAndNoop(t *testing.T) {
	const body = "<!-- fak-worker-worktree-status:v1 -->\nstatus\n<!-- /fak-worker-worktree-status:v1 -->"
	cases := []struct {
		name       string
		comments   string
		want       steerLifecycleStatusAction
		wantWrites [][]string
	}{
		{
			name:       "create",
			comments:   `[]`,
			want:       steerLifecycleStatusCreated,
			wantWrites: [][]string{{"api", "--method", "POST", "repos/{owner}/{repo}/issues/10551/comments", "-f", "body=" + body}},
		},
		{
			name:     "exact body is no-op",
			comments: fmt.Sprintf(`[{"id":77,"body":%q}]`, body),
			want:     steerLifecycleStatusNoop,
		},
		{
			name:       "changed body updates exact comment id",
			comments:   `[{"id":77,"body":"<!-- fak-worker-worktree-status:v1 -->\nold\n<!-- /fak-worker-worktree-status:v1 -->"}]`,
			want:       steerLifecycleStatusUpdated,
			wantWrites: [][]string{{"api", "--method", "PATCH", "repos/{owner}/{repo}/issues/comments/77", "-f", "body=" + body}},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			orig := steerLifecycleGHRun
			defer func() { steerLifecycleGHRun = orig }()
			var writes [][]string
			steerLifecycleGHRun = func(args ...string) ([]byte, error) {
				if len(args) >= 3 && args[1] == "--method" {
					writes = append(writes, append([]string(nil), args...))
					return []byte(`{}`), nil
				}
				return []byte(tc.comments), nil
			}
			got, err := ghSteerLifecycleStatusUpsert(10551, body)
			if err != nil || got != tc.want {
				t.Fatalf("upsert = (%q, %v), want (%q, nil)", got, err, tc.want)
			}
			if !reflect.DeepEqual(writes, tc.wantWrites) {
				t.Fatalf("writes = %#v, want %#v", writes, tc.wantWrites)
			}
			for _, call := range writes {
				if strings.Contains(strings.Join(call, " "), "--edit-last") {
					t.Fatalf("unsafe edit-last call: %#v", call)
				}
			}
		})
	}
}

func TestGHSteerLifecycleStatusUpsertFailsClosedOnMultipleOrMalformedMarkers(t *testing.T) {
	const body = "<!-- fak-worker-worktree-status:v1 -->\nstatus\n<!-- /fak-worker-worktree-status:v1 -->"
	cases := []string{
		`[{"id":1,"body":"<!-- fak-worker-worktree-status:v1 -->x<!-- /fak-worker-worktree-status:v1 -->"},{"id":2,"body":"<!-- fak-worker-worktree-status:v1 -->y<!-- /fak-worker-worktree-status:v1 -->"}]`,
		`[{"id":1,"body":"<!-- fak-worker-worktree-status:v1 --><!-- fak-worker-worktree-status:v1 --><!-- /fak-worker-worktree-status:v1 -->"}]`,
		`[{"id":1,"body":"<!-- /fak-worker-worktree-status:v1 -->"}]`,
	}
	for _, comments := range cases {
		orig := steerLifecycleGHRun
		var writes int
		steerLifecycleGHRun = func(args ...string) ([]byte, error) {
			if len(args) >= 3 && args[1] == "--method" {
				writes++
			}
			return []byte(comments), nil
		}
		_, err := ghSteerLifecycleStatusUpsert(10551, body)
		steerLifecycleGHRun = orig
		if err == nil {
			t.Fatalf("comments %s: expected fail-closed error", comments)
		}
		if writes != 0 {
			t.Fatalf("comments %s: made %d writes", comments, writes)
		}
	}
}
