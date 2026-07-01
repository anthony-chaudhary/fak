package frontierswe

import (
	"encoding/json"
	"math"
	"testing"
)

// tinyGeometry is a hand-checkable geometry so the projection assertions are NOT
// tautological (they pin a literal number, not a re-call of TTSRatio):
//
//	P=10, T=2, D=1, R=1
//	A = Σ_{t=0..1}(10 + t·2) = 10 + 12 = 22
//	B = P + (T-1)·R = 10 + 1 = 11
//	at r=0.5: C = A − r·(A−B) = 22 − 0.5·11 = 16.5; TTSRatio = C/A = 16.5/22 = 0.75
func tinyGeometry() TaskGeometry {
	return TaskGeometry{Name: "tiny", Prefix: 10, Turns: 2, Decode: 1, Result: 1}
}

const floatTol = 1e-9

// TestRecordTTS_CrossArmMeasuredVsProjection is the acceptance test: two arms (fak
// vs baseline) both reach correctness==1.0; the recorder emits the measured
// wall-clock-to-milestone and turns-to-first-correct per arm, the cross-arm measured
// TTS ratio, and the fak arm's C4 projection at its REALIZED reuse rate beside it so
// over-claim is a single visible number.
func TestRecordTTS_CrossArmMeasuredVsProjection(t *testing.T) {
	cfg := TTSConfig{
		Task: "tiny",
		Arms: []TTSArmTrace{
			{
				Arm:  "fak-agent",
				Role: "fak",
				Turns: []TTSTurn{
					{Turn: 1, WallClockSec: 10, Correctness: 0.5},
					{Turn: 3, WallClockSec: 30, Correctness: 1.0}, // first to hit 1.0
					{Turn: 4, WallClockSec: 40, Correctness: 1.0},
				},
				// Realized reuse 0.5 → projected TTSRatio 0.75 on tinyGeometry.
				Reuse:    &CacheWitnessSeries{RealizedReuseRate: 0.5},
				Geometry: geoPtr(tinyGeometry()),
			},
			{
				Arm:  "raw-harness",
				Role: "baseline",
				Turns: []TTSTurn{
					{Turn: 2, WallClockSec: 40, Correctness: 0.5},
					{Turn: 5, WallClockSec: 100, Correctness: 1.0}, // first to hit 1.0
				},
			},
		},
	}

	got := RecordTTS(cfg)

	if got.Schema != TTSMetricSchema {
		t.Fatalf("schema = %q, want %q", got.Schema, TTSMetricSchema)
	}
	if got.Milestone != 1.0 {
		t.Fatalf("milestone defaulted to %v, want 1.0", got.Milestone)
	}
	if len(got.Arms) != 2 {
		t.Fatalf("arms = %d, want 2", len(got.Arms))
	}

	fak := got.Arms[0]
	if !fak.Reached || fak.UsedFallback {
		t.Fatalf("fak reached=%v fallback=%v, want reached & no fallback", fak.Reached, fak.UsedFallback)
	}
	if fak.WallClockToMilestone != 30 {
		t.Errorf("fak wall-clock-to-milestone = %v, want 30", fak.WallClockToMilestone)
	}
	if fak.TurnsToFirstCorrect != 3 {
		t.Errorf("fak turns-to-first-correct = %v, want turn 3", fak.TurnsToFirstCorrect)
	}
	if fak.FinalCorrectness != 1.0 || fak.TotalTurns != 3 || fak.FinalWallClockSec != 40 {
		t.Errorf("fak end-state final=%v turns=%v wall=%v, want 1.0/3/40", fak.FinalCorrectness, fak.TotalTurns, fak.FinalWallClockSec)
	}
	if fak.RealizedReuseRate != 0.5 {
		t.Errorf("fak realized reuse = %v, want 0.5", fak.RealizedReuseRate)
	}
	if math.Abs(fak.ProjectedTTSRatio-0.75) > floatTol {
		t.Errorf("fak projected TTS ratio = %v, want 0.75 (hand-computed)", fak.ProjectedTTSRatio)
	}

	base := got.Arms[1]
	if !base.Reached || base.WallClockToMilestone != 100 || base.TurnsToFirstCorrect != 5 {
		t.Errorf("baseline reached=%v wall=%v turns=%v, want true/100/5", base.Reached, base.WallClockToMilestone, base.TurnsToFirstCorrect)
	}

	// Measured cross-arm ratio: 30/100 = 0.30, beside the projection 0.75.
	if !got.Comparable {
		t.Fatalf("expected comparable (both reached same milestone)")
	}
	if math.Abs(got.MeasuredTTSRatio-0.30) > floatTol {
		t.Errorf("measured TTS ratio = %v, want 0.30", got.MeasuredTTSRatio)
	}
	if math.Abs(got.ProjectedTTSRatio-0.75) > floatTol {
		t.Errorf("metric projected TTS ratio = %v, want 0.75", got.ProjectedTTSRatio)
	}
	// Over-claim = measured - projected = 0.30 - 0.75 = -0.45 (the run BEAT the floor).
	if math.Abs(got.OverClaim-(-0.45)) > floatTol {
		t.Errorf("over-claim = %v, want -0.45", got.OverClaim)
	}

	// The record must round-trip through JSON under the v1 schema id.
	b, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var back TTSMetric
	if err := json.Unmarshal(b, &back); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if back.Schema != TTSMetricSchema || len(back.Arms) != 2 {
		t.Fatalf("round-trip lost fields: %+v", back)
	}
}

// TestRecordTTS_FallbackMilestone covers the configurable fixed-quality fallback for
// a task that never reaches correctness==1.0: the arm is measured against the
// fallback and flagged UsedFallback so the milestone the number was taken at is
// never ambiguous.
func TestRecordTTS_FallbackMilestone(t *testing.T) {
	cfg := TTSConfig{
		Task:              "hard-task",
		FallbackMilestone: 0.5,
		Arms: []TTSArmTrace{{
			Arm:  "fak-agent",
			Role: "fak",
			Turns: []TTSTurn{
				{Turn: 1, WallClockSec: 20, Correctness: 0.3},
				{Turn: 4, WallClockSec: 80, Correctness: 0.6}, // crosses the 0.5 fallback
				{Turn: 7, WallClockSec: 140, Correctness: 0.7},
			},
		}},
	}

	got := RecordTTS(cfg)
	m := got.Arms[0]
	if !m.Reached || !m.UsedFallback {
		t.Fatalf("reached=%v fallback=%v, want reached via fallback", m.Reached, m.UsedFallback)
	}
	if m.MilestoneUsed != 0.5 {
		t.Errorf("milestone used = %v, want 0.5", m.MilestoneUsed)
	}
	if m.WallClockToMilestone != 80 || m.TurnsToFirstCorrect != 4 {
		t.Errorf("fallback wall=%v turns=%v, want 80/4", m.WallClockToMilestone, m.TurnsToFirstCorrect)
	}
	// A single fak arm with no baseline is not cross-comparable.
	if got.Comparable {
		t.Errorf("comparable=true with no baseline arm")
	}
}

// TestRecordTTS_NeverReached covers a run that never reaches the milestone and has no
// fallback: it is reported honestly as not reached with zeroed timing, never a zero
// masquerading as an instant solve.
func TestRecordTTS_NeverReached(t *testing.T) {
	cfg := TTSConfig{
		Task: "unsolved",
		Arms: []TTSArmTrace{{
			Arm:  "fak-agent",
			Role: "fak",
			Turns: []TTSTurn{
				{Turn: 1, WallClockSec: 10, Correctness: 0.2},
				{Turn: 2, WallClockSec: 20, Correctness: 0.4},
			},
		}},
	}

	m := RecordTTS(cfg).Arms[0]
	if m.Reached {
		t.Fatalf("reached=true, want false")
	}
	if m.WallClockToMilestone != 0 || m.TurnsToFirstCorrect != 0 {
		t.Errorf("unreached timing wall=%v turns=%v, want 0/0", m.WallClockToMilestone, m.TurnsToFirstCorrect)
	}
	if m.FinalCorrectness != 0.4 {
		t.Errorf("final correctness = %v, want 0.4", m.FinalCorrectness)
	}
}

// TestRecordTTS_DifferentMilestonesNotComparable guards the cross-arm ratio: when one
// arm reaches 1.0 and the other only the fallback, the two wall-clock numbers were
// measured against DIFFERENT milestones, so the ratio must not be reported.
func TestRecordTTS_DifferentMilestonesNotComparable(t *testing.T) {
	cfg := TTSConfig{
		Task:              "mixed",
		FallbackMilestone: 0.5,
		Arms: []TTSArmTrace{
			{
				Arm:   "fak-agent",
				Role:  "fak",
				Turns: []TTSTurn{{Turn: 2, WallClockSec: 20, Correctness: 1.0}},
			},
			{
				Arm:   "raw-harness",
				Role:  "baseline",
				Turns: []TTSTurn{{Turn: 3, WallClockSec: 90, Correctness: 0.6}}, // only the fallback
			},
		},
	}

	got := RecordTTS(cfg)
	if got.Comparable || got.MeasuredTTSRatio != 0 {
		t.Fatalf("comparable=%v ratio=%v, want not-comparable across milestones", got.Comparable, got.MeasuredTTSRatio)
	}
	if !got.Arms[1].UsedFallback {
		t.Errorf("baseline should have used the fallback milestone")
	}
}

func geoPtr(g TaskGeometry) *TaskGeometry { return &g }
