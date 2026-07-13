package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/loopmgr"
)

func TestLogDojoEpisodeFileWritesScorableInputs(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)

	input := dojo.ScoredInput{
		Prediction: dojo.Prediction{
			Lever:   "vcache-warmth",
			Metric:  "warm_recall",
			Claimed: 1.0,
			Unit:    "fraction",
			Basis:   "test",
		},
		Outcome: dojo.Outcome{
			Realized:   1.0,
			Provenance: dojo.Observed,
			Source:     "test provider cache window",
			Measured:   true,
			Sample:     4,
		},
	}
	if err := logDojoEpisodeFile("guard", []dojo.ScoredInput{input}); err != nil {
		t.Fatalf("logDojoEpisodeFile: %v", err)
	}

	lc, err := dojo.ReadLiveCorpus(filepath.Join(root, filepath.FromSlash(dojo.LiveEpisodesRel)))
	if err != nil {
		t.Fatalf("ReadLiveCorpus: %v", err)
	}
	if lc.Found != 1 || lc.Scorable != 1 {
		t.Fatalf("live corpus found/scorable = %d/%d, want 1/1 (%+v)", lc.Found, lc.Scorable, lc)
	}
	inputs := dojo.ScorableLiveEpisodes(lc)
	if len(inputs) != 1 || inputs[0].Prediction.Metric != "warm_recall" || !inputs[0].Outcome.Measured {
		t.Fatalf("scorable inputs = %+v, want measured warm_recall", inputs)
	}
}

func TestRunDojoLiveFoldsScoredRows(t *testing.T) {
	root := t.TempDir()
	if err := os.Mkdir(filepath.Join(root, ".git"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Chdir(root)
	input := dojo.ScoredInput{
		Prediction: dojo.Prediction{
			Lever:   "vcache-warmth",
			Metric:  "warm_recall",
			Claimed: 1.0,
			Unit:    "fraction",
			Basis:   "test",
		},
		Outcome: dojo.Outcome{
			Realized:   1.0,
			Provenance: dojo.Observed,
			Source:     "test provider cache window",
			Measured:   true,
			Sample:     4,
		},
	}
	if err := logDojoEpisodeFile("serve", []dojo.ScoredInput{input}); err != nil {
		t.Fatalf("logDojoEpisodeFile: %v", err)
	}

	// Seed a loop ledger so the live arm's dispatch-yield fold (#4497) measures:
	// 2 dispatched workers, 1 diff-witnessed close -> verified_ship_rate 0.5.
	t.Setenv("FAK_LOOP_LEDGER", "")
	ledger := filepath.Join(root, ".fak", "loops.jsonl")
	for _, ev := range []loopmgr.Event{
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED"},
		{LoopID: "issue-resolve-dispatch/test", Kind: loopmgr.EventStart, Reason: "SPAWNED"},
		{LoopID: "issue-resolve-progress", Kind: loopmgr.EventEnd, Reason: "OK", Metrics: map[string]int64{"closed_now": 1}},
	} {
		if _, err := loopmgr.Append(ledger, ev); err != nil {
			t.Fatalf("seed loop ledger: %v", err)
		}
	}

	var out, errb bytes.Buffer
	code := runDojoRun(&out, &errb, []string{"--live", "--workspace", root, "--json"})
	if code != 0 {
		t.Fatalf("dojo run --live should fold the scored row, code=%d stderr=%s stdout=%s", code, errb.String(), out.String())
	}
	var got dojoLiveJSON
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatalf("bad live json: %v\n%s", err, out.String())
	}
	if got.Live.Found != 1 || got.Live.Scorable != 1 {
		t.Fatalf("live corpus found/scorable = %d/%d, want 1/1", got.Live.Found, got.Live.Scorable)
	}
	if got.Report.Measured != 2 || got.Report.Unmeasured != 0 {
		t.Fatalf("live report measured/unmeasured = %d/%d, want 2/0 (provider-cache row + dispatch-yield)", got.Report.Measured, got.Report.Unmeasured)
	}
	var yield *dojo.Episode
	for i := range got.Report.Episodes {
		if got.Report.Episodes[i].Lever == "dispatch-yield" {
			yield = &got.Report.Episodes[i]
		}
	}
	if yield == nil {
		t.Fatalf("live report is missing the dispatch-yield episode: %s", out.String())
	}
	if yield.Metric != "verified_ship_rate" || yield.Claimed != 0.5 || yield.Realized != 0.5 {
		t.Fatalf("dispatch-yield episode wrong: %+v", yield)
	}
}
