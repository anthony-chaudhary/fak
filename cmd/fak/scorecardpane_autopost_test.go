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
	var stderr bytes.Buffer
	code := postScorecardResults(&stderr, root, []scorecardpane.Result{res})
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
