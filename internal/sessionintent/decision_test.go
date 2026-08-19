package sessionintent

import (
	"testing"
	"time"
)

func TestDecideStopSeparatesMinimumTargetMaximumAndCompletion(t *testing.T) {
	i := baseIntent()
	i.Effort = []EffortBound{{Kind: BoundMinimum, Clock: ClockActive, Duration: 2 * time.Hour}, {Kind: BoundTarget, Clock: ClockActive, Duration: 4 * time.Hour}, {Kind: BoundMaximum, Clock: ClockElapsed, Duration: 10 * time.Hour}}
	cases := []struct {
		name string
		p    Progress
		want StopState
	}{
		{"too early", Progress{Active: time.Hour, Elapsed: 3 * time.Hour}, StopContinue},
		{"minimum met before target", Progress{Active: 2 * time.Hour, Elapsed: 3 * time.Hour}, StopEligible},
		{"target does not force stop", Progress{Active: 5 * time.Hour, Elapsed: 6 * time.Hour}, StopEligible},
		{"completion waits for minimum", Progress{Active: time.Hour, Elapsed: 90 * time.Minute, Completed: true}, StopContinue},
		{"completion after minimum", Progress{Active: 2 * time.Hour, Elapsed: 3 * time.Hour, Completed: true}, StopComplete},
		{"maximum forces timeout", Progress{Active: 6 * time.Hour, Elapsed: 10 * time.Hour, Completed: true}, StopTimeout},
		{"failure does not burn minimum", Progress{Active: time.Minute, Elapsed: time.Minute, Failed: true}, StopFailed},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := i.DecideStop(tc.p); got.State != tc.want {
				t.Fatalf("got %+v want %s", got, tc.want)
			}
		})
	}
}
