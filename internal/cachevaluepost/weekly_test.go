package cachevaluepost

import (
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

func weeklyDigestFixture(t *testing.T) cachevaluereport.WeeklyDigest {
	t.Helper()
	now := time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)
	usage := []gatewayusageledger.Row{
		{Schema: gatewayusageledger.Schema, Kind: "exit", UnixMillis: now.Add(-24 * time.Hour).UnixMilli(),
			Counters: gatewayusageledger.Counters{
				CacheTTLUpgradesUpgraded: 3,
				CacheTTLUpgradeReasons:   map[string]uint64{"head_too_young": 1},
				CompactionFired:          2, CompactionBailed: 1,
				CompactionShedTokens: 40000,
			}},
		{Schema: gatewayusageledger.Schema, Kind: "exit", UnixMillis: now.Add(-48 * time.Hour).UnixMilli()},
		{Schema: gatewayusageledger.Schema, Kind: "exit", UnixMillis: now.Add(-8 * 24 * time.Hour).UnixMilli(),
			Counters: gatewayusageledger.Counters{CacheTTLUpgradeReasons: map[string]uint64{"head_too_young": 2}}},
	}
	track1 := []cachevalueledger.Row{
		{Schema: cachevalueledger.Schema, Date: "2026-07-09", SessionType: "serve",
			UnixMillis: now.Add(-24 * time.Hour).UnixMilli(), Turns: 10, PromptTokens: 1000, ReusedTokens: 750},
		{Schema: cachevalueledger.Schema, Date: "2026-07-01", SessionType: "serve",
			UnixMillis: now.Add(-8 * 24 * time.Hour).UnixMilli(), Turns: 10, PromptTokens: 1000, ReusedTokens: 500},
	}
	return cachevaluereport.FoldWeeklyDigest(track1, usage, now)
}

func TestWeeklyCardTextCarriesAllFourSignals(t *testing.T) {
	card := FoldWeekly(weeklyDigestFixture(t))
	card.Source = "test"
	text := card.Text()

	for _, want := range []string{
		"weekly fleet digest",
		"MEASURED",
		"posture adoption: 1/2 sessions (50%)",
		"reuse: 75.0% over 10 m-turns, prior 50.0%",
		"shed: 40000 tok over 2 fire(s)",
		"refused upgrades: 1 of 4 head(s) (25%)",
		"fence: " + cachevaluereport.PublishableValueFamily,
		"posted by test",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("weekly card text missing %q:\n%s", want, text)
		}
	}
}

func TestWeeklyCardBlocksMirrorText(t *testing.T) {
	card := FoldWeekly(weeklyDigestFixture(t))
	blocks := card.Blocks()
	if len(blocks) < 3 {
		t.Fatalf("weekly card blocks = %d, want headline + body + context at least", len(blocks))
	}
}

func TestWeeklyCardInsufficientRendersHonestly(t *testing.T) {
	d := cachevaluereport.FoldWeeklyDigest(nil, nil, time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC))
	text := FoldWeekly(d).Text()
	if !strings.Contains(text, "INSUFFICIENT") {
		t.Fatalf("empty digest card must read INSUFFICIENT:\n%s", text)
	}
	if !strings.Contains(text, "posture adoption: 0/0 sessions (n/a)") {
		t.Fatalf("empty digest must render n/a, never a fabricated 0%%:\n%s", text)
	}
}
