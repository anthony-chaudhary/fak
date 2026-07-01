package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
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
	if got.Report.Measured != 1 || got.Report.Unmeasured != 0 {
		t.Fatalf("live report measured/unmeasured = %d/%d, want 1/0", got.Report.Measured, got.Report.Unmeasured)
	}
}
