package compactcohere

import "testing"

// TestLiveCompactAuditReplayActuatesReset replays the resident-token values captured by
// `fak session compact-audit --json --aggregate-only --limit 20 --since 2026-08-23` while
// #3187 was dogfooded (median pre=162356, median post=18667). The witness explicitly uses
// enforce mode: shadow mode would observe the posture but could not actuate either yield or reset.
func TestLiveCompactAuditReplayActuatesReset(t *testing.T) {
	const (
		preCompactResident  = int64(162356)
		postCompactResident = int64(18667)
		preCompactMode      = "enforce"
	)
	if preCompactMode != "enforce" {
		t.Fatal("live replay must exercise the actuating PreCompact mode")
	}

	coordinator := NewConfig(Config{
		ResidentCeiling:        20000,
		CeilingStreakToYield:   2,
		NonHoldRewritesToReset: 2,
	})
	type counters struct {
		overCeiling, rewriteNoDrop, recommendReset, resetFired int
		residentTokens                                         int64
	}
	var metrics counters
	resetOnBudget := func() { metrics.resetFired++ }
	observe := func(digest string, resident int64) Decision {
		d := coordinator.Observe(TurnObservation{InboundPrefixDigest: digest, CacheReadTokens: resident})
		metrics.residentTokens = d.ResidentTokens
		if d.OverCeiling {
			metrics.overCeiling++
		}
		if d.RewriteNoDrop {
			metrics.rewriteNoDrop++
		}
		if d.EscalateReset {
			metrics.recommendReset++
			resetOnBudget()
		}
		return d
	}

	observe("prefix-a", preCompactResident)
	if d := observe("prefix-a", preCompactResident); d.HarnessPosture != PostureAllow {
		t.Fatalf("resident ceiling streak must yield the harness net: posture=%q, want %q", d.HarnessPosture, PostureAllow)
	}
	for _, digest := range []string{"prefix-b", "prefix-c"} {
		if d := observe(digest, postCompactResident); d.Event != EventHarnessRewrite {
			t.Fatalf("captured compaction must register a harness rewrite: event=%q", d.Event)
		}
		observe(digest, preCompactResident)
	}

	if metrics.residentTokens != preCompactResident {
		t.Fatalf("resident gauge=%d, want captured pre-compact value %d", metrics.residentTokens, preCompactResident)
	}
	if metrics.overCeiling != 4 || metrics.rewriteNoDrop != 2 || metrics.recommendReset != 1 || metrics.resetFired != 1 {
		t.Fatalf("metric progression=%+v, want over=4 no_drop=2 recommend=1 fired=1", metrics)
	}
}
