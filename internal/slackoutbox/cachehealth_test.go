package slackoutbox

import (
	"strings"
	"testing"
	"time"
)

func TestHealthWindowAlertLoopPagesOnlySustainedBadWindow(t *testing.T) {
	start := time.Unix(1700000000, 0).UTC()
	cfg := HealthWindowConfig{Window: 3, MinUpgradeAttempts: 10, MinPrefixTurns: 10, MaxUpgradeRefusalFraction: .25, MaxColdPrefixFraction: .30, Destination: "#cache-alerts"}
	bad := []HealthWindowSample{
		{At: start, UpgradeAttempts: 100, UpgradeRefusals: 40, PrefixTurns: 100, ColdPrefixTurns: 20},
		{At: start.Add(time.Minute), UpgradeAttempts: 100, UpgradeRefusals: 35, PrefixTurns: 100, ColdPrefixTurns: 20},
		{At: start.Add(2 * time.Minute), UpgradeAttempts: 100, UpgradeRefusals: 50, PrefixTurns: 100, ColdPrefixTurns: 20},
	}
	out, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	posted, err := EnqueueHealthWindowAlert(out, bad, cfg)
	if err != nil || !posted {
		t.Fatalf("posted=%v err=%v, want seeded sustained breach posted", posted, err)
	}
	snapshot, err := out.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Rows) != 1 {
		t.Fatalf("pending=%d, want one alert", len(snapshot.Rows))
	}
	if got := snapshot.Rows[0].Text; got == "" || !containsAll(got, `"upgrade_refusal_breached":true`, `"cold_prefix_breached":false`) {
		t.Fatalf("payload=%s", got)
	}
}

func TestHealthWindowAlertLoopDoesNotPostHealthyOrTransientWindow(t *testing.T) {
	start := time.Unix(1700000000, 0).UTC()
	cfg := HealthWindowConfig{Window: 3, MinUpgradeAttempts: 10, MinPrefixTurns: 10, MaxUpgradeRefusalFraction: .25, MaxColdPrefixFraction: .30, Destination: "#cache-alerts"}
	samples := []HealthWindowSample{
		{At: start, UpgradeAttempts: 100, UpgradeRefusals: 10, PrefixTurns: 100, ColdPrefixTurns: 10},
		{At: start.Add(time.Minute), UpgradeAttempts: 100, UpgradeRefusals: 90, PrefixTurns: 100, ColdPrefixTurns: 90},
		{At: start.Add(2 * time.Minute), UpgradeAttempts: 100, UpgradeRefusals: 10, PrefixTurns: 100, ColdPrefixTurns: 10},
	}
	out, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	posted, err := EnqueueHealthWindowAlert(out, samples, cfg)
	if err != nil || posted {
		t.Fatalf("posted=%v err=%v, want transient spike suppressed", posted, err)
	}
	snapshot, err := out.Load()
	if err != nil {
		t.Fatal(err)
	}
	if len(snapshot.Rows) != 0 {
		t.Fatalf("pending=%d, want healthy window silent", len(snapshot.Rows))
	}
}

func containsAll(s string, wants ...string) bool {
	for _, w := range wants {
		if !strings.Contains(s, w) {
			return false
		}
	}
	return true
}
