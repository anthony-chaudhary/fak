package main

import (
	"testing"
	"time"
)

func TestGuardChildTerminalRestorePulseIncludesClaude(t *testing.T) {
	original := startGuardChildTerminalRestorePulse
	t.Cleanup(func() { startGuardChildTerminalRestorePulse = original })

	type pulse struct {
		duration time.Duration
		interval time.Duration
	}
	for _, tc := range []struct {
		name    string
		command []string
		want    int
	}{
		{name: "empty", command: nil, want: 0},
		{name: "other harness", command: []string{"other-agent"}, want: 0},
		{name: "Claude", command: []string{"claude"}, want: 1},
		{name: "Claude Windows launcher", command: []string{`C:\tools\claude.exe`}, want: 1},
		{name: "Codex unchanged", command: []string{"codex", "exec"}, want: 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var pulses []pulse
			startGuardChildTerminalRestorePulse = func(duration, interval time.Duration) {
				pulses = append(pulses, pulse{duration: duration, interval: interval})
			}

			maybeStartGuardChildHarnessTerminalRestorePulse(tc.command)

			if got := len(pulses); got != tc.want {
				t.Fatalf("restore pulse count = %d, want %d", got, tc.want)
			}
			for i, got := range pulses {
				if got.duration != guardCodexTerminalRestorePulseDuration || got.interval != guardCodexTerminalRestorePulseInterval {
					t.Errorf("pulse %d = (%s, %s), want (%s, %s)", i, got.duration, got.interval, guardCodexTerminalRestorePulseDuration, guardCodexTerminalRestorePulseInterval)
				}
			}
		})
	}
}
