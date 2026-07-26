package main

// Tests for `fak steer comment` (#5029): the verb posts the anchored note to
// the unit's CLOSURE-GRADE bound issue through the (test-overridden) trusted gh
// seam, records the attention on the overlay ledger, and refuses an unbound
// unit instead of posting somewhere plausible. The structural
// no-git-mutation / annotate-only fence lives beside the leaf in
// internal/steerpr/comment_test.go.

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// A stamped unit that binds NO issue: the honest-refusal fixture. The gateway
// unit in prPlanFakeLog binds #1146 in its subject (closure-grade) and merely
// MENTIONS #999 in its body, which is exactly the pair this verb must tell
// apart.
const steerUnboundLog = "\x1eddd4444444444444444444444444444444444444\x1ffeat(dojo): add a calibration pass (fak dojo)\x1f\x1f\ninternal/dojo/pass.go\n"

// withSteerCommentSeam swaps the trusted gh seam for a recorder so no test ever
// reaches the network, returning the captured records.
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

// The annotate rung posts the note to the unit's closure-grade bound issue
// (#1146 from the subject — NOT the merely-mentioned #999), anchored to the
// exact member SHA set and band that were read, then ledgers the attention. And
// the machine numbers stay exactly what they were: a comment annotates, it
// never steers.
func TestSteerCommentPostsToBoundIssueAndLedgersAttention(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	posted := withSteerCommentSeam(t, "https://example.invalid/issues/1146#issuecomment-7", nil)

	var stdout, stderr bytes.Buffer
	code := runSteer(&stdout, &stderr, []string{"comment", "gateway",
		"-m", "the same-tick ready path looks like it double-counts a shed tick",
		"--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}

	var row steerpr.Comment
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}
	if row.Schema != steerpr.CommentSchema || row.Leaf != "gateway" || row.By != "op-jane" || row.At == "" {
		t.Fatalf("echoed row = %#v, want an attributable fak.steerpr.comment.v1 row", row)
	}
	// The binding, not the mention: posting operator reasoning onto #999 would
	// put it on an unrelated ticket.
	if row.Issue != "#1146" {
		t.Fatalf("row bound issue = %q, want the subject-bound #1146 (a body mention like #999 is NOT a binding)", row.Issue)
	}
	if len(row.SHAs) != 2 || row.SHAs[0] != steerFeatSHA || row.SHAs[1] != steerFixSHA {
		t.Fatalf("row anchor SHAs = %v, want the unit's exact member set [%s %s]", row.SHAs, steerFeatSHA, steerFixSHA)
	}
	if row.Band != steerpr.BandResidual {
		t.Fatalf("row anchor band = %q, want the unit's band at comment time (RESIDUAL)", row.Band)
	}
	if row.Posted != "https://example.invalid/issues/1146#issuecomment-7" {
		t.Fatalf("row posted ref = %q, want the seam's posted ref", row.Posted)
	}

	// The seam saw ONE record, bound to #1146, whose body carries the full
	// anchor — the acceptance gate: anchored to a SHA set, not just a name.
	if len(*posted) != 1 {
		t.Fatalf("gh seam called %d time(s), want 1", len(*posted))
	}
	if got := (*posted)[0].Issue; got != "#1146" {
		t.Fatalf("gh seam posted to %q, want #1146", got)
	}
	body := (*posted)[0].Body()
	for _, want := range []string{"double-counts a shed tick", steerFeatSHA, steerFixSHA, string(steerpr.BandResidual), "op-jane"} {
		if !strings.Contains(body, want) {
			t.Errorf("posted comment missing anchor %q:\n%s", want, body)
		}
	}
	if strings.Contains(body, "#999") {
		t.Errorf("posted comment names the merely-mentioned #999:\n%s", body)
	}

	// The attention is on the ledger, countable per unit.
	rows := steerpr.LoadComments(steerpr.CommentLedgerPath(root))
	if got := steerpr.CommentsFor(rows, "gateway"); len(got) != 1 || got[0].Posted != row.Posted {
		t.Fatalf("ledger rows for gateway = %v, want the one posted annotation", got)
	}

	// Annotate-only: the prs view's machine numbers are untouched, and the unit
	// does not render as acked (a comment is not a review).
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if out := stdout.String(); !strings.Contains(out, "1 RESIDUAL") || strings.Contains(out, "acked by") {
		t.Fatalf("a comment must move neither the residual count nor the ack state:\n%s", out)
	}
}

// #5029's acceptance gate at the verb: a unit with no closure-grade binding
// REFUSES rather than posting somewhere plausible. Nothing reaches gh and
// nothing reaches the ledger, and the refusal says why.
func TestSteerCommentRefusesAnUnboundUnit(t *testing.T) {
	withSteerFakes(t, steerUnboundLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	posted := withSteerCommentSeam(t, "#1", nil)

	var stdout, stderr bytes.Buffer
	code := runSteerComment(&stdout, &stderr, []string{"dojo",
		"-m", "this needs a second look", "--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 1 {
		t.Fatalf("unbound exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if s := stderr.String(); !strings.Contains(s, "binds no issue") {
		t.Fatalf("refusal should say the unit binds no issue: %s", s)
	}
	if len(*posted) != 0 {
		t.Fatalf("an unbound unit reached the gh seam %d time(s) — it must never guess an issue", len(*posted))
	}
	if rows := steerpr.LoadComments(steerpr.CommentLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("a refused comment wrote %d ledger row(s): %#v", len(rows), rows)
	}
}

// Refusals: a noteless annotation, an unattributable one, an unknown unit, a
// missing unit, and a seam that failed to post — none may ledger a row, and
// none may reach gh except the seam-failure case itself.
func TestSteerCommentRefusalsLedgerNothing(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	posted := withSteerCommentSeam(t, "#1", nil)

	var stdout, stderr bytes.Buffer
	// No -m: the note is the whole point.
	if code := runSteerComment(&stdout, &stderr, []string{"gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("noteless exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "empty note") {
		t.Fatalf("noteless refusal should say why the note is required: %s", stderr.String())
	}

	// No --by and the faked git yields no config user.name.
	stderr.Reset()
	if code := runSteerComment(&stdout, &stderr, []string{"gateway", "-m", "note", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("unattributable exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "attributable") {
		t.Fatalf("refusal should say attribution is required: %s", stderr.String())
	}

	stderr.Reset()
	if code := runSteerComment(&stdout, &stderr, []string{"no-such-leaf", "-m", "note", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("unknown unit exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no forming unit") {
		t.Fatalf("refusal should name the missing unit: %s", stderr.String())
	}

	stderr.Reset()
	if code := runSteerComment(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("missing unit exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if len(*posted) != 0 {
		t.Fatalf("a refused comment reached the gh seam %d time(s)", len(*posted))
	}

	// The seam fails to post: the note never landed, so NO attention is
	// ledgered — a comment nobody can read is not operator attention.
	withSteerCommentSeam(t, "", errors.New("gh: connection refused"))
	stderr.Reset()
	if code := runSteerComment(&stdout, &stderr, []string{"gateway", "-m", "note", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("seam-failure exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "post via gh") {
		t.Fatalf("seam failure should surface the gh error: %s", stderr.String())
	}

	if rows := steerpr.LoadComments(steerpr.CommentLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("a refused comment wrote %d ledger row(s): %#v", len(rows), rows)
	}
}
