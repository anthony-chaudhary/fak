package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/metrics"
	"github.com/anthony-chaudhary/fak/internal/resume"
	"github.com/anthony-chaudhary/fak/internal/vcachecal"
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
	if cold.Prediction.Claimed != 0.85 || cold.Outcome.Realized != 0.68 || cold.Outcome.Sample != 40 {
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
		t.Fatalf("0.68 vs 0.85 should be OVER_CLAIM, got %s", byMetric["cold_write_share"].Verdict)
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
	if !strings.Contains(out.String(), "compaction") || !strings.Contains(out.String(), "cache_prefix_preserved") {
		t.Fatalf("list --json should describe the compaction lever, got %q", out.String())
	}
}

func TestCompactionEpisodesFromBacktest(t *testing.T) {
	// Perfect compaction: 100 fired, 0 prefix_mismatch, shed matches billed delta
	rep := CompactionBacktestReport{
		FiredAttempts:       100,
		PrefixMismatchBails: 0,
		ShedTokensSum:       10000,
		InputTokensOffSum:   50000,
		InputTokensOnSum:    40000,
	}
	ins := compactionEpisodesFromBacktest(rep)
	if len(ins) != 2 {
		t.Fatalf("a full report should yield 2 metrics, got %d", len(ins))
	}

	got := map[string]dojo.ScoredInput{}
	for _, in := range ins {
		got[in.Prediction.Metric] = in
	}

	prefix := got["cache_prefix_preserved"]
	if prefix.Prediction.Claimed != 1.0 || prefix.Outcome.Realized != 1.0 || prefix.Outcome.Sample != 100 {
		t.Fatalf("cache_prefix_preserved mapping wrong: %+v", prefix)
	}
	if prefix.Outcome.Provenance != dojo.Witnessed || !prefix.Outcome.Measured {
		t.Fatalf("compaction metrics must be WITNESSED + measured: %+v", prefix.Outcome)
	}

	shed := got["token_shed_ratio"]
	if shed.Prediction.Claimed != 1.0 || shed.Outcome.Realized != 1.0 {
		t.Fatalf("token_shed_ratio mapping wrong for perfect case: %+v", shed)
	}
}

func TestCompactionEpisodesPrefixMismatch(t *testing.T) {
	// Single prefix_mismatch drives cache_prefix_preserved to 0.99 (OVER_CLAIM vs 1.0)
	rep := CompactionBacktestReport{
		FiredAttempts:       100,
		PrefixMismatchBails: 1,
	}
	ins := compactionEpisodesFromBacktest(rep)
	if len(ins) != 1 {
		t.Fatalf("only cache_prefix_preserved should be emitted when no token data, got %d", len(ins))
	}
	prefix := ins[0]
	if prefix.Outcome.Realized != 0.99 {
		t.Fatalf("single prefix_mismatch should yield 0.99 preserved, got %.2f", prefix.Outcome.Realized)
	}
}

func TestCompactionEpisodesOverClaimShed(t *testing.T) {
	// Projected shed 10k, billed delta 5k -> 0.5 ratio (OVER_CLAIM vs 1.0)
	// CalibErr = |0.5 - 1.0| / 1.0 = 0.5 > 0.10, Residual = -0.5 < 0 -> OVER_CLAIM
	// No prefix data, but FiredAttempts > 0 triggers cache_prefix_preserved emission too
	rep := CompactionBacktestReport{
		FiredAttempts:     10,
		ShedTokensSum:     10000,
		InputTokensOffSum: 50000,
		InputTokensOnSum:  45000, // delta = 5k < 10k projected
	}
	ins := compactionEpisodesFromBacktest(rep)
	if len(ins) != 2 {
		t.Fatalf("both metrics should be emitted when FiredAttempts > 0, got %d", len(ins))
	}
	shed := map[string]dojo.ScoredInput{}
	for _, in := range ins {
		shed[in.Prediction.Metric] = in
	}
	if shed["token_shed_ratio"].Outcome.Realized != 0.5 {
		t.Fatalf("projected 10k, billed 5k should yield 0.5 ratio, got %.2f", shed["token_shed_ratio"].Outcome.Realized)
	}
	// cache_prefix_preserved is 1.0 when PrefixMismatchBails = 0
	if shed["cache_prefix_preserved"].Outcome.Realized != 1.0 {
		t.Fatalf("zero prefix_mismatch should yield 1.0 preserved, got %.2f", shed["cache_prefix_preserved"].Outcome.Realized)
	}
}

func TestCompactionEpisodesSkipsEmptyMetrics(t *testing.T) {
	// Empty report -> no episodes
	ins := compactionEpisodesFromBacktest(CompactionBacktestReport{})
	if len(ins) != 0 {
		t.Fatalf("an empty report should yield no episodes, got %d", len(ins))
	}
}

func TestCompactionEpisodesScoreIntoExpectedVerdicts(t *testing.T) {
	// Mix of calibrated (0.99 vs 1.0, CalibErr=0.01 <= 0.10), over-claim (0.5 vs 1.0, CalibErr=0.5 > 0.10, Residual<0)
	rep := CompactionBacktestReport{
		FiredAttempts:       100,
		PrefixMismatchBails: 1,
		ShedTokensSum:       10000,
		InputTokensOffSum:   50000,
		InputTokensOnSum:    45000, // delta = 5k, ratio = 0.5 -> OVER_CLAIM
	}
	var eps []dojo.Episode
	for _, in := range compactionEpisodesFromBacktest(rep) {
		eps = append(eps, dojo.Score("corpus", in.Prediction, in.Outcome, dojo.DefaultCalibBand()))
	}
	byMetric := map[string]dojo.Episode{}
	for _, e := range eps {
		byMetric[e.Metric] = e
	}
	// 0.99 vs 1.0 is CALIBRATED (CalibErr=0.01 <= 0.10)
	if byMetric["cache_prefix_preserved"].Verdict != dojo.VerdictCalibrated {
		t.Fatalf("0.99 vs 1.0 should be CALIBRATED, got %s", byMetric["cache_prefix_preserved"].Verdict)
	}
	// 0.5 vs 1.0 is OVER_CLAIM (CalibErr=0.5 > 0.10, Residual=-0.5 < 0)
	if byMetric["token_shed_ratio"].Verdict != dojo.VerdictOverClaim {
		t.Fatalf("0.5 vs 1.0 should be OVER_CLAIM, got %s", byMetric["token_shed_ratio"].Verdict)
	}
}

func TestDojoCatalogMatchesCompactionEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the compaction lever emits on a full report
	rep := CompactionBacktestReport{
		FiredAttempts:       1,
		PrefixMismatchBails: 0,
		ShedTokensSum:       1000,
		InputTokensOffSum:   10000,
		InputTokensOnSum:    8000,
	}
	emitted := map[string]bool{}
	for _, in := range compactionEpisodesFromBacktest(rep) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "compaction" {
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

func TestVcacheEpisodesFromObserve(t *testing.T) {
	// 10 believed-warm calls, 1 of which billed cache_read=0 (false-warm); 8
	// provider-warm reads, 6 of which the belief called warm (recall 6/8=0.75).
	pe := vcachecal.PredictionError{TrueWarm: 6, FalseWarm: 1, TrueCold: 3, FalseCold: 2}
	ins := vcacheEpisodesFromObserve(pe)
	if len(ins) != 2 {
		t.Fatalf("a full prediction-error should yield 2 metrics, got %d", len(ins))
	}
	got := map[string]dojo.ScoredInput{}
	for _, in := range ins {
		got[in.Prediction.Metric] = in
	}

	fw := got["false_warm_rate"]
	// false-warm rate = FalseWarm/(TrueWarm+FalseWarm) = 1/7.
	if fw.Prediction.Claimed != 0.0 || fw.Outcome.Realized != 1.0/7.0 || fw.Outcome.Sample != 7 {
		t.Fatalf("false_warm_rate mapping wrong: %+v", fw)
	}
	if fw.Outcome.Provenance != dojo.Observed || !fw.Outcome.Measured {
		t.Fatalf("vcache metrics must be OBSERVED + measured: %+v", fw.Outcome)
	}

	wr := got["warm_recall"]
	// recall = TrueWarm/(TrueWarm+FalseCold) = 6/8 = 0.75.
	if wr.Prediction.Claimed != 1.0 || wr.Outcome.Realized != 0.75 || wr.Outcome.Sample != 8 {
		t.Fatalf("warm_recall mapping wrong: %+v", wr)
	}
}

func TestVcacheEpisodesSkipsEmptyMetrics(t *testing.T) {
	if ins := vcacheEpisodesFromObserve(vcachecal.PredictionError{}); len(ins) != 0 {
		t.Fatalf("an empty prediction-error should yield no episodes, got %d", len(ins))
	}
}

func TestAblateEpisodesFromArms(t *testing.T) {
	// OFF sent 100 engine calls; ON sent 30 (70 elided) -> realized elision 0.70.
	on := metrics.Arm{Label: "vdso-on", EngineCalls: 30}
	off := metrics.Arm{Label: "vdso-off", EngineCalls: 100}
	ins := ablateEpisodesFromArms(on, off)
	if len(ins) != 1 {
		t.Fatalf("want 1 ablation episode, got %d", len(ins))
	}
	e := ins[0]
	if e.Prediction.Metric != "engine_call_elision" || e.Prediction.Claimed != 1.0 {
		t.Fatalf("prediction wrong: %+v", e.Prediction)
	}
	if e.Outcome.Realized != 0.70 || e.Outcome.Sample != 100 {
		t.Fatalf("outcome wrong: realized=%v sample=%d, want 0.70/100", e.Outcome.Realized, e.Outcome.Sample)
	}
	if e.Outcome.Provenance != dojo.Witnessed || !e.Outcome.Measured {
		t.Fatalf("ablation outcome must be WITNESSED + measured: %+v", e.Outcome)
	}
}

func TestAblateEpisodesNoEngineCalls(t *testing.T) {
	// An OFF arm that sent zero engine calls has no elision to score.
	if ins := ablateEpisodesFromArms(metrics.Arm{}, metrics.Arm{}); len(ins) != 0 {
		t.Fatalf("zero off engine-calls should yield no episode, got %d", len(ins))
	}
}

func TestDojoCatalogMatchesVcacheEmittedMetrics(t *testing.T) {
	// the static catalog must match the metrics the vcache-warmth lever emits.
	pe := vcachecal.PredictionError{TrueWarm: 1, FalseWarm: 1, TrueCold: 1, FalseCold: 1}
	emitted := map[string]bool{}
	for _, in := range vcacheEpisodesFromObserve(pe) {
		emitted[in.Prediction.Metric] = true
	}
	for _, lv := range dojoLeverCatalog() {
		if lv.Name != "vcache-warmth" {
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
