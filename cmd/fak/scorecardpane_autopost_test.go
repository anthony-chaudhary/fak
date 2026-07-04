package main

import (
	"bytes"
	"encoding/json"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/scoreboard"
	"github.com/anthony-chaudhary/fak/internal/scorecardpane"
	"github.com/anthony-chaudhary/fak/pkg/scorecard"
)

func TestScorecardAutoPostEnabledByFlagOrEnv(t *testing.T) {
	t.Setenv(scoreboardAutoPostEnv, "")
	if !scorecardAutoPostEnabled(true) {
		t.Fatal("explicit --post should enable local autopost")
	}
	if scorecardAutoPostEnabled(false) {
		t.Fatal("autopost must be off by default")
	}
	t.Setenv(scoreboardAutoPostEnv, "1")
	if !scorecardAutoPostEnabled(false) {
		t.Fatal("FAK_SCOREBOARD_AUTOPOST=1 should enable local autopost")
	}
	t.Setenv(scoreboardAutoPostEnv, "false")
	if scorecardAutoPostEnabled(false) {
		t.Fatal("falsey env value should not enable local autopost")
	}
}

func TestScoreboardAutoPostSkipsUnchangedStateBeforeSlackConfig(t *testing.T) {
	root := t.TempDir()
	res := scorecardResultFixture(t, "demo", "demo_debt", "A", 100, 0, "OK")
	up, err := scoreboardUpdateFromResult(res)
	if err != nil {
		t.Fatal(err)
	}
	statePath := filepath.Join(root, filepath.FromSlash(scoreboardAutoPostStateRel))
	state := scoreboardAutoPostState{
		Schema:  scoreboardAutoPostStateSchema,
		Updates: map[string]string{up.ChangeKey(): scoreboardUpdateDigest(up)},
	}
	if err := saveScoreboardAutoPostState(statePath, state); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_SCOREBOARD_CHANNEL", "")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "")
	var stdout, stderr bytes.Buffer
	code := postScorecardResults(&stdout, &stderr, root, []scorecardpane.Result{res})
	if code != 0 {
		t.Fatalf("unchanged autopost should skip before Slack config, code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "skipped demo: no change") {
		t.Fatalf("skip message missing:\n%s", stderr.String())
	}
}

func TestScoreboardUpdateDigestUsesScoreFieldsOnly(t *testing.T) {
	a := scoreboard.Update{Title: "demo", Grade: "A", Score: "100", Debt: "0", Verdict: "OK", Source: "one", Detail: "first"}
	b := scoreboard.Update{Title: "demo", Grade: "A", Score: "100", Debt: "0", Verdict: "OK", Source: "two", Detail: "second"}
	if scoreboardUpdateDigest(a) != scoreboardUpdateDigest(b) {
		t.Fatal("digest should ignore source/detail noise")
	}
	b.Debt = "1"
	if scoreboardUpdateDigest(a) == scoreboardUpdateDigest(b) {
		t.Fatal("digest should change when the scorecard debt changes")
	}
}

func TestScoreboardUpdateFromResultUsesLocalSource(t *testing.T) {
	t.Setenv("FAK_SCOREBOARD_SOURCE", "host-test")
	res := scorecardResultFixture(t, "demo", "demo_debt", "B", 88, 3, "ACTION")
	up, err := scoreboardUpdateFromResult(res)
	if err != nil {
		t.Fatal(err)
	}
	if up.Title != "demo" || up.Source != "host-test" {
		t.Fatalf("update title/source = %q/%q, want demo/host-test", up.Title, up.Source)
	}
	if up.Grade != "B" || up.Score != "88" || up.Debt != "3" || up.Verdict != "ACTION" {
		t.Fatalf("update did not preserve score fields: %+v", up)
	}
}

func TestScoreboardControlPaneUpdateFields(t *testing.T) {
	p := scorecardpane.Payload{
		Schema:    scorecardpane.Schema,
		Verdict:   "ACTION",
		Finding:   "scorecard_debt",
		Reason:    "portfolio debt 12 across 40 scorecards",
		TotalDebt: 12,
		GradeDebt: 3,
	}
	up := scoreboardControlPaneUpdate(p, "host-x")
	if up.Title != "scorecard portfolio" {
		t.Fatalf("title = %q, want scorecard portfolio", up.Title)
	}
	if up.DebtKey != "total_debt" || up.Debt != "12" {
		t.Fatalf("debt = %q/%q, want total_debt/12", up.DebtKey, up.Debt)
	}
	if up.Score != "3" {
		t.Fatalf("score (grade_debt companion) = %q, want 3", up.Score)
	}
	if up.Verdict != "ACTION" || up.Source != "host-x" {
		t.Fatalf("verdict/source = %q/%q, want ACTION/host-x", up.Verdict, up.Source)
	}
	if up.Detail != "portfolio debt 12 across 40 scorecards" {
		t.Fatalf("detail = %q, want the portfolio reason", up.Detail)
	}
	// Detail falls back to the machine finding token when the reason is empty.
	p.Reason = ""
	if up := scoreboardControlPaneUpdate(p, "host-x"); up.Detail != "scorecard_debt" {
		t.Fatalf("detail fallback = %q, want scorecard_debt", up.Detail)
	}
}

// TestAutopostControlPanePostsAndDedups is the local witness the issue asks for: a
// scorecard regen with the opt-in set posts a card to #scoreboard (a fake server here,
// so no live token is needed), a rerun that moved nothing is silent, and the default
// (no flag, no env) never posts.
func TestAutopostControlPanePostsAndDedups(t *testing.T) {
	outboxTestDir(t)
	root := t.TempDir()
	t.Chdir(root)
	t.Setenv("FAK_SCOREBOARD_AUTOPOST", "")
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "C_SCORE")
	t.Setenv("FAK_SCOREBOARD_TOKEN", "xoxb-test")
	t.Setenv("FAK_SCOREBOARD_SOURCE", "host-witness")

	posts := 0
	srv := okSlackServer(t, &posts)
	defer srv.Close()
	orig := newScoreboardPostClient
	newScoreboardPostClient = func(tok string) (*scoreboard.Client, error) {
		return scoreboard.NewClient(tok, scoreboard.WithAPIBase(srv.URL+"/"), scoreboard.WithHTTPClient(srv.Client()))
	}
	defer func() { newScoreboardPostClient = orig }()

	payload := scorecardpane.Payload{Verdict: "ACTION", Finding: "scorecard_debt", Reason: "portfolio debt 12", TotalDebt: 12, GradeDebt: 3}

	// First opt-in regen: posts one card and surfaces the ts.
	var errb bytes.Buffer
	if code := autopostControlPane(&errb, root, payload, true); code != 0 {
		t.Fatalf("first autopost code=%d stderr=%s", code, errb.String())
	}
	if posts != 1 {
		t.Fatalf("posts=%d after first opt-in regen, want 1", posts)
	}
	if !strings.Contains(errb.String(), "posted to C_SCORE ts=") {
		t.Fatalf("no posted-ts witness:\n%s", errb.String())
	}

	// Unchanged rerun: silent (deduped, no new post).
	errb.Reset()
	if code := autopostControlPane(&errb, root, payload, true); code != 0 {
		t.Fatalf("rerun code=%d stderr=%s", code, errb.String())
	}
	if posts != 1 {
		t.Fatalf("posts=%d after unchanged rerun, want still 1 (dedup)", posts)
	}
	if !strings.Contains(errb.String(), "skipped scorecard portfolio: no change") {
		t.Fatalf("unchanged rerun not silent:\n%s", errb.String())
	}

	// Off by default: a moved number with NO opt-in must not post.
	moved := payload
	moved.TotalDebt = 5
	errb.Reset()
	if code := autopostControlPane(&errb, root, moved, false); code != 0 {
		t.Fatalf("default-off code=%d", code)
	}
	if posts != 1 {
		t.Fatalf("posts=%d with autopost off, want still 1", posts)
	}

	// The same moved number WITH opt-in reposts.
	errb.Reset()
	if code := autopostControlPane(&errb, root, moved, true); code != 0 {
		t.Fatalf("moved autopost code=%d stderr=%s", code, errb.String())
	}
	if posts != 2 {
		t.Fatalf("posts=%d after debt moved, want 2", posts)
	}
}

func scorecardResultFixture(t *testing.T, label, debtKey, grade string, score float64, debt int, verdict string) scorecardpane.Result {
	t.Helper()
	payload := scorecard.Payload{
		Schema:  "fak-" + label + "/1",
		OK:      verdict == "OK",
		Verdict: verdict,
		Finding: "fixture",
		Corpus: map[string]any{
			"grade": grade,
			"score": score,
			debtKey: debt,
		},
		KPIs: []scorecard.KPI{{Key: "fixture", Score: score}},
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		t.Fatal(err)
	}
	return scorecardpane.Result{
		Card: scorecardpane.Card{Key: label, Debt: debtKey, Label: label},
		Raw:  raw,
	}
}
