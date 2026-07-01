package frontierswe

import "testing"

func TestScoreParityPassesOnMatchAndBeat(t *testing.T) {
	sp := func(f float64) *float64 { return &f }
	raw := []TrialScore{
		{ID: "r1", Task: "ffmpeg-swscale-rewrite", Correctness: 0.6, Score: 0.3},
		{ID: "r2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.2), Score: 1.1},
	}
	match := []TrialScore{
		{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 0.6, Score: 0.3},
		{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.2), Score: 1.1},
	}
	beat := []TrialScore{
		{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.3), Score: 1.15},
		{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.4), Score: 1.2},
	}
	if rep := ScoreParity(raw, match, 0); !rep.Passed || rep.Reason != "" {
		t.Fatalf("match should pass: %+v", rep)
	}
	if rep := ScoreParity(raw, beat, 0); !rep.Passed || rep.Fak.CorrectCount != 2 {
		t.Fatalf("beat should pass: %+v", rep)
	}
}

func TestScoreParityFailsRegressions(t *testing.T) {
	sp := func(f float64) *float64 { return &f }
	raw := []TrialScore{
		{ID: "r1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.2), Score: 1.1},
		{ID: "r2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.0), Score: 1.0},
	}
	cases := map[string][]TrialScore{
		"avg_score": {
			{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.2), Score: 1.1},
			{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 0.0, Score: 0.0},
		},
		"best_score": {
			{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(0.9), Score: 0.95},
			{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(0.9), Score: 0.95},
		},
		"correct_count": {
			{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.3), Score: 1.15},
			{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 0.8, Score: 0.4},
		},
		"speedup": {
			{ID: "f1", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(1.1), Score: 1.05},
			{ID: "f2", Task: "ffmpeg-swscale-rewrite", Correctness: 1.0, Speedup: sp(0.8), Score: 0.9},
		},
	}
	for name, fak := range cases {
		t.Run(name, func(t *testing.T) {
			rep := ScoreParity(raw, fak, 0)
			if rep.Passed {
				t.Fatalf("regression %s should fail: %+v", name, rep)
			}
			if rep.Reason != ScoreParityFailedReason {
				t.Fatalf("reason = %q, want %q", rep.Reason, ScoreParityFailedReason)
			}
			if len(rep.Failures) == 0 {
				t.Fatalf("failure report empty: %+v", rep)
			}
		})
	}
}

func TestScoreParityEmptyArmsFailClosed(t *testing.T) {
	rep := ScoreParity(nil, nil, 0)
	if rep.Passed || rep.Reason != ScoreParityFailedReason {
		t.Fatalf("empty arms should fail closed: %+v", rep)
	}
}

func TestScoreRewardTrialFeedsParity(t *testing.T) {
	raw := []TrialScore{ScoreRewardTrial("raw-1", "ffmpeg-swscale-rewrite", loadReward(t, "ffmpeg_full_speedup"), false)}
	fak := []TrialScore{ScoreRewardTrial("fak-1", "ffmpeg-swscale-rewrite", loadReward(t, "ffmpeg_full_speedup"), false)}
	rep := ScoreParity(raw, fak, 0)
	if !rep.Passed {
		t.Fatalf("same reward fixture should pass parity: %+v", rep)
	}
	flagged := []TrialScore{ScoreRewardTrial("fak-flagged", "ffmpeg-swscale-rewrite", loadReward(t, "ffmpeg_full_speedup"), true)}
	if rep := ScoreParity(raw, flagged, 0); rep.Passed || rep.Reason != ScoreParityFailedReason {
		t.Fatalf("anti-cheat zeroed fak trial should fail parity: %+v", rep)
	}
}
