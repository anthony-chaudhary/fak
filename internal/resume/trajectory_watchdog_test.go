package resume

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajctl"
)

func TestDecideTrajectoryWatchdogShapes(t *testing.T) {
	tests := []struct {
		name string
		in   TrajectoryWatchdogInput
		want TrajectoryWatchdogAction
	}{
		{"healthy slow alive leaves alone", TrajectoryWatchdogInput{Alive: true, Signal: trajctl.SignalHealthy}, TrajectoryLeaveAlone},
		{"stalled alive nudges first", TrajectoryWatchdogInput{Alive: true, Signal: trajctl.SignalStall}, TrajectoryNudge},
		{"stalled after nudge revives anchored", TrajectoryWatchdogInput{Alive: true, Signal: trajctl.SignalStall, NudgeAttempted: true}, TrajectoryReviveAnchor},
		{"dead revives anchored", TrajectoryWatchdogInput{Alive: false, Signal: trajctl.SignalHealthy}, TrajectoryReviveAnchor},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := DecideTrajectoryWatchdog(tt.in)
			if got.Action != tt.want || got.Reason == "" {
				t.Fatalf("decision=%+v want action=%s", got, tt.want)
			}
		})
	}
}
