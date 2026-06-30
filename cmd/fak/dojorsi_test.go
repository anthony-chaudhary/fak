package main

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/dojo"
	"github.com/anthony-chaudhary/fak/internal/dojocal"
	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

func TestRunDojoRSILoopKeepsThenFloorEscalates(t *testing.T) {
	reportPath := writeDojoRSIReport(t, dojoRSITestReport(t))
	journal := filepath.Join(t.TempDir(), "rsi-journal.jsonl")

	var out, errb bytes.Buffer
	code := runDojoRSI(&out, &errb, []string{
		"loop",
		"--report", reportPath,
		"--journal", journal,
		"--witness", `{"ok":true}`,
		"--ticks", "2",
		"--k", "1",
		"--now", "2026-06-29T00:00:00Z",
		"--dos-arbitrate=false",
	})
	if code != 3 {
		t.Fatalf("loop should exit 3 on floor ESCALATE, got %d\nstderr=%s\nstdout=%s", code, errb.String(), out.String())
	}
	rows := readDojoRSIJournal(journal)
	if len(rows) != 2 {
		t.Fatalf("journal rows = %d, want 2 (%s)", len(rows), out.String())
	}
	if rows[0].Kind != dojocal.RecalibrateKind || rows[0].Decision != "KEEP" || !rows[0].Kept {
		t.Fatalf("first tick should be mechanical KEEP, got %+v", rows[0])
	}
	if rows[1].Kind != dojocal.RouteFloor || rows[1].Decision != "ESCALATE" || !rows[1].AgentArm {
		t.Fatalf("second tick should be floor ESCALATE route, got %+v", rows[1])
	}
	if !strings.Contains(out.String(), "score=dojo_calibration") || !strings.Contains(out.String(), "grade=kept") {
		t.Fatalf("rendered loop should carry structured score summary, got:\n%s", out.String())
	}
	if !strings.Contains(out.String(), "floor-escalate=1") {
		t.Fatalf("rendered trend should report floor escalate, got:\n%s", out.String())
	}
}

func TestRunDojoRSITrendReadsCommittedJournal(t *testing.T) {
	journal := filepath.Join(t.TempDir(), "rsi-journal.jsonl")
	rows := []dojocal.JournalRow{
		{Schema: dojocal.JournalSchema, Tick: 1, Lever: "resume-posture", Metric: "cold_write_share", Kind: dojocal.RecalibrateKind, Decision: "KEEP", Kept: true},
		{Schema: dojocal.JournalSchema, Tick: 2, Lever: "compaction", Metric: "token_shed_ratio", Kind: dojocal.HarvestKind, Decision: "REVERT", AgentArm: true},
	}
	for _, r := range rows {
		if err := appendDojoRSIJournalRow(journal, r); err != nil {
			t.Fatal(err)
		}
	}

	var out, errb bytes.Buffer
	code := runDojoRSI(&out, &errb, []string{"trend", "--journal", journal, "--now", "2026-06-29T00:00:00Z"})
	if code != 0 {
		t.Fatalf("trend should exit 0, got %d (%s)", code, errb.String())
	}
	got := out.String()
	for _, want := range []string{"KEEP 1", "REVERT 1", "harvest=1"} {
		if !strings.Contains(got, want) {
			t.Fatalf("trend output missing %q:\n%s", want, got)
		}
	}
}

func TestDojoRSIDOSImproveArgsLowerBetter(t *testing.T) {
	row := rsiloop.Row{
		Cycle: 1, Candidate: "resume-posture/cold_write_share/RECALIBRATE",
		MetricName: dojoRSIMetricName, Baseline: 0.50, Candidate_: 0.25,
		LowerBetter: true, Improved: true, SuiteGreen: true, TruthClean: true,
		Decision: "KEEP",
	}
	args := dojoRSIDOSImproveArgs("repo", 3, row)
	joined := strings.Join(args, " ")
	if !strings.Contains(joined, "--lane dojocal") || !strings.Contains(joined, "--suite-passed") || !strings.Contains(joined, "--truth-clean") {
		t.Fatalf("dos improve args missing lane/witness bits: %v", args)
	}
	// Lower-better metrics are recorded as max-minus-value so the improved
	// candidate has strictly more work than the baseline for dos improve.
	work := argAfter(args, "--work")
	base := argAfter(args, "--baseline-work")
	if work == "" || base == "" || work <= base {
		t.Fatalf("lower-better work should be > baseline-work, got work=%q baseline=%q args=%v", work, base, args)
	}
}

func TestDojoRSIScorecardObserverAndNarration(t *testing.T) {
	report := dojoRSITestReport(t)
	now := time.Date(2026, 6, 29, 0, 0, 0, 0, time.UTC)
	payload := dojocal.ProposeRecals(report)
	ranked := dojocal.RankCandidates(payload.Candidates, nil, dojocal.SelectOptions{Now: now})
	scored, ok := dojocal.NextCandidate(ranked)
	if !ok {
		t.Fatal("fixture produced no dojo-RSI candidate")
	}
	it := dojocal.RunIteration(report, scored.Candidate, dojocal.DefaultMinSample, map[string]any{"ok": true})
	row := dojocal.NewJournalRow(1, it, "KEEP", 0, now, "abc123", dojocal.Wakeup{})

	obs := dojoRSIRowToObserverRow(row, it, scored, true)
	if obs.Score == nil {
		t.Fatal("observer row missing scorecard")
	}
	if obs.Score.Name != "dojo_calibration" || obs.Score.Grade != "kept" {
		t.Fatalf("scorecard identity = %+v, want dojo_calibration/kept", obs.Score)
	}
	for name, want := range map[string]float64{
		"measured_delta":   it.MeasuredDelta,
		"selector_score":   scored.Score,
		"candidate_sample": float64(scored.Candidate.Sample),
		"witnessed":        1,
		"kept":             1,
	} {
		if got := dojoRSIScoreComponentValue(obs.Score, name); got != want {
			t.Fatalf("score component %s = %.4g, want %.4g (score=%+v)", name, got, want, obs.Score)
		}
	}

	args := dojoRSIDOSImproveArgs("repo", 3, obs)
	narrated := argAfter(args, "--narrated")
	for _, want := range []string{"score=dojo_calibration", "grade=kept", "measured_delta=", "selector_score="} {
		if !strings.Contains(narrated, want) {
			t.Fatalf("narration missing %q:\n%s\nargs=%v", want, narrated, args)
		}
	}
}

func dojoRSITestReport(t *testing.T) dojo.Report {
	t.Helper()
	eps := []dojo.Episode{
		dojoRSITestEpisode(t, "resume-posture", "cold_write_share", 0.85, 0.40, 5, false, false),
		dojoRSITestEpisode(t, "resume-posture", "cold_write_share", 0.85, 0.50, 5, false, false),
		dojoRSITestEpisode(t, "resume-posture", "cold_write_share", 0.85, 0.60, 5, false, false),
		dojoRSITestEpisode(t, "vcache-warmth", "false_warm_rate", 0.0, 0.30, 5, true, true),
	}
	return dojo.Fold(eps, dojo.FoldOpts{
		Workspace:   ".",
		Commit:      "abc123",
		GeneratedAt: "2026-06-29T00:00:00Z",
		Date:        "2026-06-29",
	})
}

func dojoRSITestEpisode(t *testing.T, lever, metric string, claimed, realized float64, sample int, floor, lowerIsBetter bool) dojo.Episode {
	t.Helper()
	p := dojo.Prediction{
		Lever: lever, Metric: metric, Claimed: claimed, Unit: "fraction", Basis: "test",
		IntentionalFloor: floor, LowerIsBetter: lowerIsBetter,
	}
	o := dojo.Outcome{Realized: realized, Provenance: dojo.Witnessed, Source: "test", Measured: true, Sample: sample}
	return dojo.Score(lever, p, o, dojo.DefaultCalibBand())
}

func writeDojoRSIReport(t *testing.T, r dojo.Report) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "dojo-report.json")
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, append(b, '\n'), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}

func argAfter(args []string, flag string) string {
	for i := 0; i+1 < len(args); i++ {
		if args[i] == flag {
			return args[i+1]
		}
	}
	return ""
}

func dojoRSIScoreComponentValue(score *rsiloop.Scorecard, name string) float64 {
	if score == nil {
		return math.NaN()
	}
	for _, c := range score.Components {
		if c.Name == name {
			return c.Value
		}
	}
	return math.NaN()
}

func TestParseDojoRSITime(t *testing.T) {
	var errb bytes.Buffer
	tm, ok := parseDojoRSITime(&errb, "2026-06-29")
	if !ok || tm.Format(time.RFC3339) != "2026-06-29T00:00:00Z" {
		t.Fatalf("date parse = %s ok=%v err=%s", tm.Format(time.RFC3339), ok, errb.String())
	}
}
