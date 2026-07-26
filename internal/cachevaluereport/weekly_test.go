package cachevaluereport

import (
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// digestNow is the fixed fold clock every weekly-digest test uses: windows are
// this-week (digestNow-7d, digestNow] and prior-week (digestNow-14d, digestNow-7d].
var digestNow = time.Date(2026, 7, 10, 12, 0, 0, 0, time.UTC)

// TestObservedProviderReuseRatio pins the OBSERVED provider cache-read share: cache_read over
// all prompt tokens processed (read + fresh input + cache_creation), folded from the usage
// ledger and kept distinct from the WITNESSED realized-reuse. nil when no prompt tokens moved.
func TestObservedProviderReuseRatio(t *testing.T) {
	var w WeeklyWindow
	// The measured subscription-seat shape: heavy cache_read, small fresh input + write.
	w.accumulateUsage(gatewayusageledger.Row{Kind: "exit", Counters: gatewayusageledger.Counters{
		CachedPromptTokens: 744, InputTokens: 200, CacheCreationTokens: 56,
	}})
	w.accumulateUsage(gatewayusageledger.Row{Kind: "exit", Counters: gatewayusageledger.Counters{}})
	w.finalize()
	if w.ObservedProviderReuse == nil {
		t.Fatal("ObservedProviderReuse nil, want ~0.744")
	}
	if got := *w.ObservedProviderReuse; got < 0.743 || got > 0.745 {
		t.Fatalf("ObservedProviderReuse = %.4f, want 0.744 (744/(744+200+56))", got)
	}

	// No prompt tokens at all -> nil (no evidence), never a manufactured 0.0.
	var empty WeeklyWindow
	empty.accumulateUsage(gatewayusageledger.Row{Kind: "exit", Counters: gatewayusageledger.Counters{}})
	empty.finalize()
	if empty.ObservedProviderReuse != nil {
		t.Fatalf("ObservedProviderReuse = %v, want nil (no prompt tokens)", empty.ObservedProviderReuse)
	}
}

func usageExitRow(age time.Duration, c gatewayusageledger.Counters) gatewayusageledger.Row {
	return gatewayusageledger.Row{
		Schema:     gatewayusageledger.Schema,
		Kind:       "exit",
		UnixMillis: digestNow.Add(-age).UnixMilli(),
		Counters:   c,
	}
}

func track1Row(age time.Duration, turns, prompt, reused uint64) cachevalueledger.Row {
	return cachevalueledger.Row{
		Schema:       cachevalueledger.Schema,
		Date:         digestNow.Add(-age).UTC().Format("2006-01-02"),
		SessionType:  "serve",
		UnixMillis:   digestNow.Add(-age).UnixMilli(),
		Turns:        turns,
		PromptTokens: prompt,
		ReusedTokens: reused,
	}
}

const day = 24 * time.Hour

func TestFoldWeeklyDigestMeasuresAllFourSignals(t *testing.T) {
	usage := []gatewayusageledger.Row{
		// This week: 3 exit sessions, 2 posture-active (one via upgrades, one via
		// refusal reasons only — the ON-but-ineligible shape), shed activity, and
		// TTL outcomes 3 upgraded / 1 refused.
		usageExitRow(1*day, gatewayusageledger.Counters{
			CacheTTLUpgradesUpgraded: 3,
			CompactionFired:          2, CompactionBailed: 1,
			CompactionShedTokens: 40000, CompactionDroppedTurns: 12,
		}),
		usageExitRow(2*day, gatewayusageledger.Counters{
			CacheTTLUpgradeReasons: map[string]uint64{"head_too_young": 1},
		}),
		usageExitRow(3*day, gatewayusageledger.Counters{}),
		// Prior week: 2 exit sessions, 1 posture-active, all heads refused.
		usageExitRow(8*day, gatewayusageledger.Counters{
			CacheTTLUpgradeReasons: map[string]uint64{"head_too_young": 2},
			CompactionFired:        1, CompactionShedTokens: 10000,
		}),
		usageExitRow(9*day, gatewayusageledger.Counters{}),
		// Noise that must be excluded: a periodic prefix snapshot, a carryforward
		// sum, and a row older than both windows.
		{Schema: gatewayusageledger.Schema, Kind: "periodic", UnixMillis: digestNow.Add(-1 * day).UnixMilli(),
			Counters: gatewayusageledger.Counters{CacheTTLUpgradesUpgraded: 99}},
		{Schema: gatewayusageledger.Schema, Kind: gatewayusageledger.KindCarryforward, UnixMillis: digestNow.Add(-2 * day).UnixMilli(),
			Counters: gatewayusageledger.Counters{CompactionShedTokens: 999999}},
		usageExitRow(20*day, gatewayusageledger.Counters{CacheTTLUpgradesUpgraded: 7}),
	}
	track1 := []cachevalueledger.Row{
		track1Row(1*day, 10, 1000, 800),  // this week
		track1Row(2*day, 10, 1000, 700),  // this week
		track1Row(8*day, 10, 1000, 500),  // prior week
		track1Row(3*day, 1, 500, 0),      // single-turn: excluded from the reuse gate
		track1Row(30*day, 10, 1000, 900), // too old: outside both windows
	}

	d := FoldWeeklyDigest(track1, usage, digestNow)

	if d.Verdict != "MEASURED" || !d.OK {
		t.Fatalf("verdict = %s ok=%v, want MEASURED/true; finding: %s", d.Verdict, d.OK, d.Finding)
	}
	tw := d.ThisWeek
	if tw.ExitSessions != 3 || tw.PostureActiveSessions != 2 {
		t.Fatalf("this week posture = %d/%d sessions, want 2/3", tw.PostureActiveSessions, tw.ExitSessions)
	}
	if tw.PostureAdoptionPct == nil || *tw.PostureAdoptionPct < 66 || *tw.PostureAdoptionPct > 67 {
		t.Fatalf("this week adoption pct = %v, want ~66.7", tw.PostureAdoptionPct)
	}
	if tw.ReuseRatio == nil || *tw.ReuseRatio != 0.75 {
		t.Fatalf("this week reuse = %v, want 0.75 (1500/2000)", tw.ReuseRatio)
	}
	if tw.ReuseThin {
		t.Fatalf("20 multi-turn turns must not read thin (floor %d)", MinBucketTurns)
	}
	if tw.ShedTokens != 40000 || tw.CompactionFired != 2 || tw.CompactionBailed != 1 || tw.DroppedTurns != 12 {
		t.Fatalf("this week shed = %d tok / %d fired / %d bailed / %d dropped, want 40000/2/1/12",
			tw.ShedTokens, tw.CompactionFired, tw.CompactionBailed, tw.DroppedTurns)
	}
	if tw.ShedTokensPerFire == nil || *tw.ShedTokensPerFire != 20000 {
		t.Fatalf("shed tokens/fire = %v, want 20000", tw.ShedTokensPerFire)
	}
	if tw.TTLUpgrades != 3 || tw.TTLRefusals != 1 {
		t.Fatalf("this week TTL = %d up / %d refused, want 3/1", tw.TTLUpgrades, tw.TTLRefusals)
	}
	if tw.RefusedUpgradePct == nil || *tw.RefusedUpgradePct != 25 {
		t.Fatalf("refused-upgrade pct = %v, want 25", tw.RefusedUpgradePct)
	}

	pw := d.PriorWeek
	if pw.ExitSessions != 2 || pw.PostureActiveSessions != 1 {
		t.Fatalf("prior week posture = %d/%d, want 1/2", pw.PostureActiveSessions, pw.ExitSessions)
	}
	if pw.ReuseRatio == nil || *pw.ReuseRatio != 0.5 {
		t.Fatalf("prior week reuse = %v, want 0.5", pw.ReuseRatio)
	}
	if pw.RefusedUpgradePct == nil || *pw.RefusedUpgradePct != 100 {
		t.Fatalf("prior week refused pct = %v, want 100", pw.RefusedUpgradePct)
	}

	// Week-over-week: adoption 50% -> 66.7% improved; reuse 0.50 -> 0.75 improved.
	if d.AdoptionTrend != TrendImproved {
		t.Fatalf("adoption trend = %s, want improved (delta %.2f)", d.AdoptionTrend, d.DeltaAdoptionPct)
	}
	if d.ReuseTrend != TrendImproved || d.DeltaReuseRatio != 0.25 {
		t.Fatalf("reuse trend = %s delta %.3f, want improved / 0.25", d.ReuseTrend, d.DeltaReuseRatio)
	}
	if d.PublishableValueFamily != PublishableValueFamily || !d.VsNaiveMultipleExcluded {
		t.Fatalf("digest must carry the #1066 fence self-labels verbatim")
	}
}

func TestFoldWeeklyDigestEmptyLedgersFallOpenInsufficient(t *testing.T) {
	d := FoldWeeklyDigest(nil, nil, digestNow)
	if d.Verdict != "INSUFFICIENT" || !d.OK {
		t.Fatalf("empty fold: verdict=%s ok=%v, want INSUFFICIENT/true", d.Verdict, d.OK)
	}
	if d.ThisWeek.PostureAdoptionPct != nil || d.ThisWeek.ReuseRatio != nil || d.ThisWeek.RefusedUpgradePct != nil {
		t.Fatalf("empty windows must report nil ratios, never a measured zero")
	}
	if d.AdoptionTrend != TrendNew || d.ReuseTrend != TrendNew {
		t.Fatalf("no prior evidence must read new, got adoption=%s reuse=%s", d.AdoptionTrend, d.ReuseTrend)
	}
}

func TestFoldWeeklyDigestThinReuseIsFlagged(t *testing.T) {
	track1 := []cachevalueledger.Row{track1Row(1*day, 2, 100, 50)} // 2 turns < MinBucketTurns
	d := FoldWeeklyDigest(track1, nil, digestNow)
	if !d.ThisWeek.ReuseThin {
		t.Fatalf("2 multi-turn turns must be flagged thin (floor %d)", MinBucketTurns)
	}
	if d.Verdict != "MEASURED" {
		t.Fatalf("a thin week is still MEASURED (flagged), got %s", d.Verdict)
	}
}
