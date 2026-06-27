package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/resume"
)

func TestResumeEpisodesFromBacktest(t *testing.T) {
	rep := resume.BacktestReport{
		Scored:                      1000,
		Accuracy:                    0.977,
		FirstTurnCold:               40,
		FirstTurnColdWriteShareMean: 0.68,
		FirstTurnResumes:            60,
		FirstTurnWarmHit:            18,
	}
	ins := resumeEpisodesFromBacktest(rep)
	if len(ins) != 3 {
		t.Fatalf("a full report should yield 3 metrics, got %d", len(ins))
	}

	got := map[string]dojo.ScoredInput{}
	for _, in := range ins {
		got[in.Prediction.Metric] = in
	}

	acc := got["posture_accuracy"]
	if acc.Prediction.Claimed != 1.0 || acc.Outcome.Realized != 0.977 || acc.Outcome.Sample != 1000 {
		t.Fatalf("posture_accuracy mapping wrong: %+v", acc)
	}
	if acc.Outcome.Provenance != dojo.Observed || !acc.Outcome.Measured {
		t.Fatalf("provider numbers must be OBSERVED + measured: %+v", acc.Outcome)
	}

	cold := got["cold_write_share"]
	if cold.Prediction.Claimed != 1.0 || cold.Outcome.Realized != 0.68 || cold.Outcome.Sample != 40 {
		t.Fatalf("cold_write_share mapping wrong: %+v", cold)
	}

	warm := got["cross_session_warm_hit_rate"]
	if warm.Prediction.Claimed != 0.0 {
		t.Fatalf("warm-hit theory should claim 0.0, got %v", warm.Prediction.Claimed)
	}
	if want := 18.0 / 60.0; warm.Outcome.Realized != want {
		t.Fatalf("warm-hit realized should be 18/60=%.4f, got %.4f", want, warm.Outcome.Realized)
	}
}

func TestResumeEpisodesSkipsEmptyMetrics(t *testing.T) {
	// no scored boundaries and no resumes -> no episodes (never a misleading zero).
	ins := resumeEpisodesFromBacktest(resume.BacktestReport{})
	if len(ins) != 0 {
		t.Fatalf("an empty report should yield no episodes, got %d", len(ins))
	}
}

func TestResumeEpisodesScoreIntoExpectedVerdicts(t *testing.T) {
	rep := resume.BacktestReport{
		Scored:                      1000,
		Accuracy:                    0.977,
		FirstTurnCold:               40,
		FirstTurnColdWriteShareMean: 0.68,
		FirstTurnResumes:            60,
		FirstTurnWarmHit:            18,
	}
	var eps []dojo.Episode
	for _, in := range resumeEpisodesFromBacktest(rep) {
		eps = append(eps, dojo.Score("corpus", in.Prediction, in.Outcome, dojo.DefaultCalibBand()))
	}
	byMetric := map[string]dojo.Episode{}
	for _, e := range eps {
		byMetric[e.Metric] = e
	}
	if byMetric["posture_accuracy"].Verdict != dojo.VerdictCalibrated {
		t.Fatalf("0.977 vs 1.0 should be CALIBRATED, got %s", byMetric["posture_accuracy"].Verdict)
	}
	if byMetric["cold_write_share"].Verdict != dojo.VerdictOverClaim {
		t.Fatalf("0.68 vs 1.0 should be OVER_CLAIM, got %s", byMetric["cold_write_share"].Verdict)
	}
	if byMetric["cross_session_warm_hit_rate"].Verdict != dojo.VerdictUnderClaim {
		t.Fatalf("0.30 vs 0.0 should be UNDER_CLAIM, got %s", byMetric["cross_session_warm_hit_rate"].Verdict)
	}
}

func TestDojoCatalogMatchesEmittedMetrics(t *testing.T) {
	// the static `dojo list` catalog must not drift from the metrics the lever
	// actually emits on a full report.
	rep := resume.BacktestReport{
		Scored: 1, Accuracy: 1, FirstTurnCold: 1, FirstTurnColdWriteShareMean: 1,
		FirstTurnResumes: 1, FirstTurnWarmHit: 0,
	}
	emitted := map[string]bool{}
	for _, in := range resumeEpisodesFromBacktest(rep) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "resume-posture" {
			continue
		}
		for _, m := range lv.Metrics {
			if !emitted[m.Name] {
				t.Fatalf("catalog advertises metric %q the lever never emits", m.Name)
			}
		}
		if len(lv.Metrics) != len(emitted) {
			t.Fatalf("catalog lists %d metrics but the lever emits %d", len(lv.Metrics), len(emitted))
		}
	}
}

func TestRunDojoDispatch(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDojo(&out, &errb, nil); code != 2 {
		t.Fatalf("no subcommand should exit 2, got %d", code)
	}
	out.Reset()
	errb.Reset()
	if code := runDojo(&out, &errb, []string{"bogus"}); code != 2 {
		t.Fatalf("unknown subcommand should exit 2, got %d", code)
	}
	if !strings.Contains(errb.String(), "unknown subcommand") {
		t.Fatalf("expected unknown-subcommand message, got %q", errb.String())
	}
	out.Reset()
	errb.Reset()
	if code := runDojo(&out, &errb, []string{"run"}); code != 2 {
		t.Fatalf("run without --corpus should exit 2, got %d", code)
	}
}

func TestRunDojoList(t *testing.T) {
	var out, errb bytes.Buffer
	if code := runDojoList(&out, &errb, []string{"--json"}); code != 0 {
		t.Fatalf("list --json should exit 0, got %d (%s)", code, errb.String())
	}
	if !strings.Contains(out.String(), "resume-posture") || !strings.Contains(out.String(), "posture_accuracy") {
		t.Fatalf("list --json should describe the resume-posture lever, got %q", out.String())
	}
}
