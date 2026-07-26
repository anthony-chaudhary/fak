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
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
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
