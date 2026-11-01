package agent

import (
	"context"
	"testing"
)

// fak-test:runtime fast est=100ms
// A no-tool-call turn that ended on a budget cut must not be recorded as a clean
// final answer: the arm metrics carry the finish reason and the truncated flag,
// while a stop-finished turn stays clean. Pinned for #13562 box 4.
func TestArmLengthFinishIsNotCleanFinalAnswer(t *testing.T) {
	const answer = "Fixture final answer."
	for _, tc := range []struct {
		finish string
		trunc  bool
	}{
		{"length", true},
		{"stop", false},
	} {
		t.Run(tc.finish, func(t *testing.T) {
			cfg := runConfig{}
			var metrics ArmMetrics
			r := &armRunner{
				cfg:      &cfg,
				metrics:  &metrics,
				messages: []Message{{Role: RoleUser, Content: "task"}},
				complete: func(context.Context, []Message, []ToolDef, StreamSink, ...SampleOpt) (*Completion, error) {
					return &Completion{Message: Message{Content: answer}, FinishReason: tc.finish}, nil
				},
				stopTerminated: func() bool { return false },
			}
			if err := r.run(context.Background(), 2); err != nil {
				t.Fatalf("run: %v", err)
			}
			if metrics.FinalAnswer != answer {
				t.Fatalf("final answer = %q, want %q", metrics.FinalAnswer, answer)
			}
			if metrics.FinalFinishReason != tc.finish {
				t.Fatalf("finish reason = %q, want %q", metrics.FinalFinishReason, tc.finish)
			}
			if metrics.FinalAnswerTruncated != tc.trunc {
				t.Fatalf("truncated flag = %t, want %t for a %q-finished no-tool-call turn",
					metrics.FinalAnswerTruncated, tc.trunc, tc.finish)
			}
			if metrics.Turns != 1 || metrics.HitTurnCap || metrics.SynthesizedFinalTurn {
				t.Fatalf("unexpected loop shape: turns=%d cap=%t synth=%t",
					metrics.Turns, metrics.HitTurnCap, metrics.SynthesizedFinalTurn)
			}
		})
	}
}
