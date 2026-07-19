package main

// Tests for `fak steer redirect` (#5030): the verb files the anchored
// follow-up through the (test-overridden) trusted gh seam, appends the
// countable ledger row, refuses incomplete steers without ledgering anything,
// and — behaviorally — reaches git only through read verbs. The structural
// no-git-mutation fence lives beside the leaf in
// internal/steerpr/redirect_test.go.

import (
	"bytes"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/steerpr"
)

// withSteerRedirectSeam swaps the trusted gh seam for a recorder so no test
// ever reaches the network, returning the captured records.
func withSteerRedirectSeam(t *testing.T, followUp string, err error) *[]steerpr.Redirect {
	t.Helper()
	var got []steerpr.Redirect
	orig := steerRedirectFile
	steerRedirectFile = func(r steerpr.Redirect) (string, error) {
		got = append(got, r)
		return followUp, err
	}
	t.Cleanup(func() { steerRedirectFile = orig })
	return &got
}

// A redirect files the follow-up carrying the note + exact member SHA set +
// band anchor through the gh seam, then ledgers the event as an attributable,
// countable row — and the machine band/residual numbers stay exactly what
// they were: a redirect re-aims the next tick, never the merge.
func TestSteerRedirectFilesFollowUpAndLedgersEvent(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	filed := withSteerRedirectSeam(t, "https://example.invalid/issues/4242", nil)

	var stdout, stderr bytes.Buffer
	code := runSteer(&stdout, &stderr, []string{"redirect", "gateway",
		"-m", "aim the gateway work at the read path, not the write path",
		"--by", "op-jane", "--base", "baseref", "--head", "headref"})
	if code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}

	var row steerpr.Redirect
	if err := json.NewDecoder(strings.NewReader(stdout.String())).Decode(&row); err != nil {
		t.Fatalf("decode echoed row: %v\n%s", err, stdout.String())
	}
	if row.Schema != steerpr.RedirectSchema || row.Leaf != "gateway" || row.By != "op-jane" || row.At == "" {
		t.Fatalf("echoed row = %#v, want an attributable fak.steerpr.redirect.v1 row", row)
	}
	if len(row.SHAs) != 2 || row.SHAs[0] != steerFeatSHA || row.SHAs[1] != steerFixSHA {
		t.Fatalf("row anchor SHAs = %v, want the unit's exact member set [%s %s]", row.SHAs, steerFeatSHA, steerFixSHA)
	}
	if row.Band != steerpr.BandResidual {
		t.Fatalf("row anchor band = %q, want the unit's band at redirect time (RESIDUAL)", row.Band)
	}
	if row.FollowUp != "https://example.invalid/issues/4242" {
		t.Fatalf("row follow-up = %q, want the seam's filed ref", row.FollowUp)
	}

	// The seam saw ONE record whose follow-up body carries the full anchor.
	if len(*filed) != 1 {
		t.Fatalf("gh seam called %d time(s), want 1", len(*filed))
	}
	body := (*filed)[0].FollowUpBody()
	for _, want := range []string{"aim the gateway work", steerFeatSHA, steerFixSHA, string(steerpr.BandResidual)} {
		if !strings.Contains(body, want) {
			t.Errorf("filed follow-up missing anchor %q:\n%s", want, body)
		}
	}

	// The event is on the ledger, countable per unit.
	rows := steerpr.LoadRedirects(steerpr.RedirectLedgerPath(root))
	if got := steerpr.RedirectsFor(rows, "gateway"); len(got) != 1 || got[0].FollowUp != row.FollowUp {
		t.Fatalf("ledger rows for gateway = %v, want the one filed steering event", got)
	}

	// And the prs view's machine numbers are untouched by the steer.
	stdout.Reset()
	stderr.Reset()
	if code := runSteerPRs(&stdout, &stderr, []string{"--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("prs exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if out := stdout.String(); !strings.Contains(out, "1 RESIDUAL") {
		t.Fatalf("a redirect must not move the residual count:\n%s", out)
	}
}

// The redirect path touches git only through READ verbs: every call the verb
// makes through the git seam during a successful redirect is in the read-only
// vocabulary. The advisory affordance never commits, pushes, reverts, or
// resets — the structural twin of this behavioral check is
// TestRedirectNeverReachesGitMutation in internal/steerpr.
func TestSteerRedirectReachesGitOnlyThroughReadVerbs(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	withSteerRoot(t)
	withSteerRedirectSeam(t, "#77", nil)

	readOnly := map[string]bool{"log": true, "rev-parse": true, "config": true, "show": true, "merge-base": true}
	var verbs []string
	inner := releasePRPlanGit
	releasePRPlanGit = func(root string, args ...string) string {
		if len(args) > 0 {
			verbs = append(verbs, args[0])
		}
		return inner(root, args...)
	}
	t.Cleanup(func() { releasePRPlanGit = inner })

	var stdout, stderr bytes.Buffer
	if code := runSteerRedirect(&stdout, &stderr, []string{"gateway", "-m", "re-aim", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", code, stderr.String())
	}
	if len(verbs) == 0 {
		t.Fatal("recorded no git seam calls — the check is vacuous")
	}
	for _, v := range verbs {
		if !readOnly[v] {
			t.Errorf("redirect invoked git seam verb %q — the redirect path must reach git only through reads", v)
		}
	}
}

// Refusals: a noteless steer, an unattributable one, an unknown unit, a
// missing unit, and a seam that failed to file — none may ledger a row, and
// none may call gh except the seam-failure case itself.
func TestSteerRedirectRefusalsLedgerNothing(t *testing.T) {
	withSteerFakes(t, prPlanFakeLog, steerpr.VerdictUnwitnessed)
	root := withSteerRoot(t)
	filed := withSteerRedirectSeam(t, "#1", nil)

	var stdout, stderr bytes.Buffer
	// No -m: the note is the whole point.
	if code := runSteerRedirect(&stdout, &stderr, []string{"gateway", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("noteless exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "where the intent goes next") {
		t.Fatalf("noteless refusal should say why the note is required: %s", stderr.String())
	}

	// No --by and the faked git yields no config user.name.
	stderr.Reset()
	if code := runSteerRedirect(&stdout, &stderr, []string{"gateway", "-m", "note", "--base", "baseref", "--head", "headref"}); code != 2 {
		t.Fatalf("unattributable exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "attributable") {
		t.Fatalf("refusal should say attribution is required: %s", stderr.String())
	}

	stderr.Reset()
	if code := runSteerRedirect(&stdout, &stderr, []string{"no-such-leaf", "-m", "note", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("unknown unit exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "no forming unit") {
		t.Fatalf("refusal should name the missing unit: %s", stderr.String())
	}

	stderr.Reset()
	if code := runSteerRedirect(&stdout, &stderr, nil); code != 2 {
		t.Fatalf("missing unit exit = %d, want 2; stderr=%s", code, stderr.String())
	}
	if len(*filed) != 0 {
		t.Fatalf("a refused redirect reached the gh seam %d time(s)", len(*filed))
	}

	// The seam fails to file: no follow-up landed, so NO steering event is
	// ledgered — a redirect that filed nothing steered nothing.
	withSteerRedirectSeam(t, "", errors.New("gh: connection refused"))
	stderr.Reset()
	if code := runSteerRedirect(&stdout, &stderr, []string{"gateway", "-m", "note", "--by", "op", "--base", "baseref", "--head", "headref"}); code != 1 {
		t.Fatalf("seam-failure exit = %d, want 1; stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "file follow-up via gh") {
		t.Fatalf("seam failure should surface the gh error: %s", stderr.String())
	}

	if rows := steerpr.LoadRedirects(steerpr.RedirectLedgerPath(root)); len(rows) != 0 {
		t.Fatalf("a refused redirect wrote %d ledger row(s): %#v", len(rows), rows)
	}
}
