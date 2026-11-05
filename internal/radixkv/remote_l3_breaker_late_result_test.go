package radixkv

import (
	"errors"
	"testing"
	"time"
)

// fak-test:runtime fast est=1ms lane=default
func TestRemoteL3BreakerLateOrdinaryResultPreservesProbe(t *testing.T) {
	backendErr := errors.New("remote store unavailable")
	for _, tc := range []struct {
		name     string
		halfOpen bool
		lateErr  error
	}{
		{name: "success during cooldown"},
		{name: "success during probe", halfOpen: true},
		{name: "fault during probe", halfOpen: true, lateErr: backendErr},
	} {
		t.Run(tc.name, func(t *testing.T) {
			now := time.Date(2026, 11, 5, 0, 0, 0, 0, time.UTC)
			const cooldown = 10 * time.Second
			b := NewRemoteL3Breaker(BreakerConfig{
				FaultThreshold: 1,
				Cooldown:       cooldown,
				Now:            func() time.Time { return now },
			})

			// Two ordinary reads start while closed. The second returns first
			// and opens the breaker; the first remains in flight.
			allowed, lateProbe := b.Allow()
			if !allowed || lateProbe {
				t.Fatal("first read was not admitted as ordinary")
			}
			allowed, faultProbe := b.Allow()
			if !allowed || faultProbe {
				t.Fatal("second read was not admitted as ordinary")
			}
			b.RecordResult(backendErr, faultProbe)
			if b.State() != BreakerOpen {
				t.Fatal("fault did not open breaker")
			}

			var recoveryProbe bool
			if tc.halfOpen {
				now = now.Add(cooldown)
				allowed, recoveryProbe = b.Allow()
				if !allowed || !recoveryProbe || b.State() != BreakerHalfOpen {
					t.Fatal("cooldown did not admit the exclusive recovery probe")
				}
			}
			want := b.Stats()
			if tc.lateErr != nil {
				want.TotalFaults++
				want.ConsecutiveFaults++
			}
			b.RecordResult(tc.lateErr, lateProbe)
			if got := b.Stats(); got != want {
				t.Fatalf("late ordinary result changed recovery state: got %+v, want %+v", got, want)
			}
			if allowed, probe := b.Allow(); allowed || probe {
				t.Fatal("late ordinary result admitted a read before recovery")
			}

			// Only the real probe may recover the breaker. In the cooldown
			// case it has not started yet, so advance the injected clock.
			if !tc.halfOpen {
				now = now.Add(cooldown)
				allowed, recoveryProbe = b.Allow()
				if !allowed || !recoveryProbe {
					t.Fatal("preserved cooldown did not admit recovery probe")
				}
			}
			b.RecordResult(nil, recoveryProbe)
			got := b.Stats()
			if got.State != BreakerClosed || got.ConsecutiveFaults != 0 || got.TotalFaults != want.TotalFaults ||
				got.ProbesAttempted != 1 || got.ProbeRecoveries != 1 || got.ProbeFailures != 0 || !got.OpenedAt.IsZero() {
				t.Fatalf("real probe did not recover exactly once: %+v", got)
			}
			if allowed, probe := b.Allow(); !allowed || probe {
				t.Fatal("recovered breaker did not admit an ordinary read")
			}
		})
	}
}
