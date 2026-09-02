package workspin

import (
	"testing"
	"time"
)

func obs(at time.Time, summary string, witnessed bool) Observation {
	return Observation{At: at, Kind: Commit, Summary: summary, Changed: 2, Witnessed: witnessed}
}

func TestOccasionalCleanupAmidDeliveryIsHealthy(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	in := []Observation{
		obs(now.Add(-20*24*time.Hour), "chore: tidy", false),
		obs(now.Add(-19*24*time.Hour), "fix: repair gate", true),
		obs(now.Add(-12*24*time.Hour), "docs: clarify", false),
		obs(now.Add(-11*24*time.Hour), "feat: expose receipt", true),
	}
	if got := Audit(in, DefaultConfig(now)); got.Spinning {
		t.Fatalf("unexpected spinning: %+v", got)
	}
}

func TestRepeatedMotionWithoutWitnessedDeliverySpins(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	var in []Observation
	for _, days := range []int{13, 12, 11, 6, 5, 4} {
		in = append(in, obs(now.Add(-time.Duration(days)*24*time.Hour), "chore: small cleanup", false))
	}
	got := Audit(in, DefaultConfig(now))
	if !got.Spinning || got.Verdict != "spinning" {
		t.Fatalf("want spinning: %+v", got)
	}
}

func TestLowActivityIsNotSpinning(t *testing.T) {
	now := time.Date(2026, 9, 1, 0, 0, 0, 0, time.UTC)
	got := Audit([]Observation{obs(now.Add(-6*24*time.Hour), "chore: one cleanup", false)}, DefaultConfig(now))
	if got.Spinning {
		t.Fatalf("low activity flagged: %+v", got)
	}
}
