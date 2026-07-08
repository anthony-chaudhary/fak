package compactcohere

import (
	"reflect"
	"strings"
	"testing"
	"time"
)

// TestClassify pins the attribution order: each event is reachable and the priority
// (world-break > harness-rewrite > cold-TTL > fak-cut > stable) holds.
func TestClassify(t *testing.T) {
	const ttl = 5 * time.Minute
	prev := TurnObservation{InboundPrefixDigest: "A"}

	cases := []struct {
		name string
		prev TurnObservation
		cur  TurnObservation
		want PrefixEvent
	}{
		{
			name: "first turn is stable (no prev to compare)",
			prev: TurnObservation{}, // zero: no previous digest
			cur:  TurnObservation{InboundPrefixDigest: "A"},
			want: EventStable,
		},
		{
			name: "unchanged prefix, fresh, no fak action -> stable",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "A", IdleSinceLastTurn: time.Minute},
			want: EventStable,
		},
		{
			name: "unchanged prefix + fak cut fired -> fak_cut",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "A", FakCompactFired: true},
			want: EventFakCut,
		},
		{
			name: "fak world-break dominates everything",
			prev: prev,
			cur: TurnObservation{
				InboundPrefixDigest: "B",  // even with a changed prefix...
				FakWorldBreak:       true, // ...fak's own break wins attribution
				FakCompactFired:     true,
			},
			want: EventFakWorldBreak,
		},
		{
			name: "inbound prefix digest changed -> harness rewrite",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "B"},
			want: EventHarnessRewrite,
		},
		{
			name: "harness rewrite beats a fak cut on the same turn",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "B", FakCompactFired: true},
			want: EventHarnessRewrite,
		},
		{
			name: "unchanged prefix but idle past TTL -> cold_ttl",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "A", IdleSinceLastTurn: 6 * time.Minute},
			want: EventColdTTL,
		},
		{
			name: "unchanged prefix, observed cache_creation with no read -> cold_ttl",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "A", CacheCreationTokens: 1200},
			want: EventColdTTL,
		},
		{
			name: "unchanged prefix with a healthy cache read is NOT cold",
			prev: prev,
			cur:  TurnObservation{InboundPrefixDigest: "A", CacheReadTokens: 9000, CacheCreationTokens: 0},
			want: EventStable,
		},
		{
			name: "cold-TTL is not flagged on the first turn (no prev)",
			prev: TurnObservation{},
			cur:  TurnObservation{InboundPrefixDigest: "A", CacheCreationTokens: 1200},
			want: EventStable,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := Classify(tc.prev, tc.cur, ttl); got != tc.want {
				t.Fatalf("Classify = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestClassifyTTLFallback proves a non-positive ttl falls back to the default (5 min):
// a 4-minute idle is warm, a 6-minute idle is cold.
func TestClassifyTTLFallback(t *testing.T) {
	prev := TurnObservation{InboundPrefixDigest: "A"}
	warm := TurnObservation{InboundPrefixDigest: "A", IdleSinceLastTurn: 4 * time.Minute}
	cold := TurnObservation{InboundPrefixDigest: "A", IdleSinceLastTurn: 6 * time.Minute}
	if got := Classify(prev, warm, 0); got != EventStable {
		t.Fatalf("warm under default TTL: got %q, want stable", got)
	}
	if got := Classify(prev, cold, 0); got != EventColdTTL {
		t.Fatalf("cold past default TTL: got %q, want cold_ttl", got)
	}
}

// TestCoordinatorDefaultPostureIsBlock: with fak coping, the standing posture blocks the
// harness's auto-compaction and a stable turn just proceeds.
func TestCoordinatorDefaultPostureIsBlock(t *testing.T) {
	c := New(0)
	if c.Posture() != PostureBlock {
		t.Fatalf("initial posture = %q, want block", c.Posture())
	}
	d := c.Observe(TurnObservation{InboundPrefixDigest: "A"})
	if d.Event != EventStable || d.Action != ActionProceed {
		t.Fatalf("stable turn: event=%q action=%q, want stable/proceed", d.Event, d.Action)
	}
	if d.HarnessPosture != PostureBlock {
		t.Fatalf("posture after stable turn = %q, want block", d.HarnessPosture)
	}
}

// TestCoordinatorYieldsAfterBailStreak: once fak's compaction bails for the streak length,
// the posture flips to allow (hand the net back), and a single clean fire resets it.
func TestCoordinatorYieldsAfterBailStreak(t *testing.T) {
	c := NewWith(0, 3)
	bail := func(i int) TurnObservation {
		return TurnObservation{
			InboundPrefixDigest: digestN(i),
			FakBailReason:       "window_no_drop",
		}
	}
	// Two bails: still blocking (streak 2 < 3).
	c.Observe(bail(1))
	c.Observe(bail(2))
	if c.Posture() != PostureBlock {
		t.Fatalf("posture after 2 bails = %q, want block", c.Posture())
	}
	// Third consecutive bail: yield.
	d := c.Observe(bail(3))
	if c.Posture() != PostureAllow {
		t.Fatalf("posture after 3 bails = %q, want allow", c.Posture())
	}
	// The 3rd bail's digest changed (harness rewrite) AND we are in yield, so the action is
	// to allow the harness net.
	if d.Action != ActionAllowHarnessCompact {
		t.Fatalf("yielded harness-rewrite action = %q, want allow_harness_compact", d.Action)
	}
	// A clean fire resets the streak and restores the block posture.
	c.Observe(TurnObservation{InboundPrefixDigest: digestN(3), FakCompactFired: true})
	if c.Posture() != PostureBlock {
		t.Fatalf("posture after a clean fire = %q, want block (streak reset)", c.Posture())
	}
}

// TestHealthyUnderBudgetDoesNotCountAsBail: a no-op "under_budget" turn (modeled as an
// empty FakBailReason) must NOT advance the bail streak — only a real bail does.
func TestHealthyUnderBudgetDoesNotCountAsBail(t *testing.T) {
	c := NewWith(0, 2)
	c.Observe(TurnObservation{InboundPrefixDigest: "A", FakBailReason: "splice_failed"}) // streak 1
	c.Observe(TurnObservation{InboundPrefixDigest: "A"})                                 // healthy no-op resets to 0
	c.Observe(TurnObservation{InboundPrefixDigest: "A", FakBailReason: "splice_failed"}) // streak 1 again
	if c.Posture() != PostureBlock {
		t.Fatalf("posture = %q, want block (streak should be 1, not 2)", c.Posture())
	}
}

// TestQuarantineAtRisk: a seal that precedes a harness rewrite raises the trust flag; a
// rewrite with no prior seal does not; and the flag clears after it fires.
func TestQuarantineAtRisk(t *testing.T) {
	c := New(0)
	// Turn 1: fak seals a poisoned span (prefix stable).
	d1 := c.Observe(TurnObservation{InboundPrefixDigest: "A", SealedSpanPresent: true})
	if d1.QuarantineAtRisk {
		t.Fatalf("no rewrite yet — QuarantineAtRisk should be false")
	}
	// Turn 2: the harness rewrites its history; the earlier seal may be in the summary.
	d2 := c.Observe(TurnObservation{InboundPrefixDigest: "B"})
	if d2.Event != EventHarnessRewrite || !d2.QuarantineAtRisk {
		t.Fatalf("seal-then-rewrite: event=%q atRisk=%v, want harness_rewrite/true", d2.Event, d2.QuarantineAtRisk)
	}
	// Turn 3: another rewrite with no NEW seal since — not at risk again.
	d3 := c.Observe(TurnObservation{InboundPrefixDigest: "C"})
	if !d3.BurstObserved || d3.QuarantineAtRisk {
		t.Fatalf("rewrite with no new seal: burst=%v atRisk=%v, want burst/true atRisk/false", d3.BurstObserved, d3.QuarantineAtRisk)
	}
}

// TestColdTTLRecommendsReset: a cold-cache turn routes to the reset path, never to a
// harness block/allow.
func TestColdTTLRecommendsReset(t *testing.T) {
	c := New(0)
	c.Observe(TurnObservation{InboundPrefixDigest: "A"})
	d := c.Observe(TurnObservation{InboundPrefixDigest: "A", IdleSinceLastTurn: 10 * time.Minute})
	if d.Event != EventColdTTL || d.Action != ActionRecommendReset {
		t.Fatalf("cold turn: event=%q action=%q, want cold_ttl/recommend_reset", d.Event, d.Action)
	}
	if !d.BurstObserved {
		t.Fatalf("cold turn should mark BurstObserved")
	}
}

// TestPreCompactExitCode pins the actuator mapping: block -> 2 (the hook blocks the
// harness auto-compaction), allow -> 0.
func TestPreCompactExitCode(t *testing.T) {
	if got := PreCompactExitCode(PostureBlock); got != 2 {
		t.Fatalf("block exit code = %d, want 2", got)
	}
	if got := PreCompactExitCode(PostureAllow); got != 0 {
		t.Fatalf("allow exit code = %d, want 0", got)
	}
}

// TestObserveIsContentFree is a guard-rail: the Decision a shadow log would emit carries no
// prompt content, only the enum/flag fields. (Compile-time shape check — if a Content field
// is ever added this test is where the reviewer re-justifies it.)
func TestObserveIsContentFree(t *testing.T) {
	const sentinel = "PROMPT-CONTENT-SENTINEL-9173"
	c := New(0)
	d := c.Observe(TurnObservation{InboundPrefixDigest: sentinel})
	// A Decision is shadow-loggable only because it carries enum/flag fields and no prompt
	// content. Reflect over every field and fail if any string echoes the inbound digest —
	// the regression this guard-rail exists to catch (a future Content field, or Observe
	// copying the digest into Reason/Event).
	v := reflect.ValueOf(d)
	for i := 0; i < v.NumField(); i++ {
		if f := v.Field(i); f.Kind() == reflect.String && strings.Contains(f.String(), sentinel) {
			t.Fatalf("Decision.%s leaked inbound content: %q", v.Type().Field(i).Name, f.String())
		}
	}
	// Event is always an attributed enum constant, never empty — the shape a shadow log reads.
	if d.Event == "" {
		t.Fatal("Observe returned an empty Event for a digest-only observation")
	}
}

func digestN(i int) string { return "digest-" + time.Duration(i).String() }

// TestNewConfigAppliesDefaults proves NewConfig fills every non-positive field with its package
// default and that New/NewWith delegate (so existing callers get the resident-token terms for free).
func TestNewConfigAppliesDefaults(t *testing.T) {
	c := NewConfig(Config{})
	if c.ttl != DefaultProviderCacheTTL || c.bailStreakToYield != DefaultBailStreakToYield {
		t.Fatalf("ttl/bail defaults not applied: ttl=%v bail=%d", c.ttl, c.bailStreakToYield)
	}
	if c.residentCeiling != DefaultResidentCeiling || c.ceilingStreakToYield != DefaultCeilingStreakToYield || c.nonHoldRewritesToReset != DefaultNonHoldRewritesToReset {
		t.Fatalf("resident-token defaults not applied: ceiling=%d ceilStreak=%d nonHold=%d",
			c.residentCeiling, c.ceilingStreakToYield, c.nonHoldRewritesToReset)
	}
	if c.posture != PostureBlock {
		t.Fatalf("default posture = %q, want block", c.posture)
	}
	// New and NewWith delegate through NewConfig with the resident-token defaults intact.
	if New(0).residentCeiling != DefaultResidentCeiling {
		t.Fatal("New must carry the default resident ceiling")
	}
	if nw := NewWith(0, 5); nw.bailStreakToYield != 5 || nw.residentCeiling != DefaultResidentCeiling {
		t.Fatalf("NewWith must honor the bail streak and default the ceiling: bail=%d ceiling=%d", nw.bailStreakToYield, nw.residentCeiling)
	}
}

// TestResidentCeilingForcesAllow proves the #3158 ceiling term: a resident window that sits over the
// ceiling for a streak forces PostureAllow regardless of the bail streak, and the Decision carries
// the content-free size signals.
func TestResidentCeilingForcesAllow(t *testing.T) {
	c := NewConfig(Config{ResidentCeiling: 1000, CeilingStreakToYield: 2})
	// Under the ceiling: block, not over.
	d := c.Observe(TurnObservation{InboundPrefixDigest: "A", CacheReadTokens: 500})
	if d.OverCeiling || d.HarnessPosture != PostureBlock {
		t.Fatalf("under ceiling: over=%v posture=%q, want false/block", d.OverCeiling, d.HarnessPosture)
	}
	// Two consecutive over-ceiling turns (no bail at all) -> yield.
	c.Observe(TurnObservation{InboundPrefixDigest: "A", CacheReadTokens: 1500}) // over, streak 1
	d = c.Observe(TurnObservation{InboundPrefixDigest: "A", CacheReadTokens: 1500})
	if !d.OverCeiling || d.ResidentTokens != 1500 {
		t.Fatalf("over-ceiling turn: over=%v resident=%d, want true/1500", d.OverCeiling, d.ResidentTokens)
	}
	if d.HarnessPosture != PostureAllow {
		t.Fatalf("posture after an over-ceiling streak = %q, want allow (no bail involved)", d.HarnessPosture)
	}
}

// TestCleanFireOverCeilingHoldsBailStreak proves the #3158 refinement: a clean fire (no bail reason)
// that leaves the window OVER the ceiling does not reset the bail streak — fak is not coping, so the
// streak is held rather than credited with a reset.
func TestCleanFireOverCeilingHoldsBailStreak(t *testing.T) {
	// Ceiling-yield effectively off (streak 100) so the allow below can only come from the bail streak.
	c := NewConfig(Config{ResidentCeiling: 1000, BailStreakToYield: 2, CeilingStreakToYield: 100})
	c.Observe(TurnObservation{InboundPrefixDigest: "A", FakBailReason: "window_no_drop", CacheReadTokens: 500}) // streak 1
	// A clean fire, but the window is still over the ceiling: must NOT reset the streak.
	c.Observe(TurnObservation{InboundPrefixDigest: "A", FakCompactFired: true, CacheReadTokens: 1500})
	// Another real bail: if the clean fire had reset, this would be streak 1 (block); held ⇒ streak 2 (allow).
	d := c.Observe(TurnObservation{InboundPrefixDigest: "A", FakBailReason: "window_no_drop", CacheReadTokens: 500})
	if d.HarnessPosture != PostureAllow {
		t.Fatalf("posture = %q, want allow — the over-ceiling clean fire must not have reset the bail streak", d.HarnessPosture)
	}
}

// TestNonHoldingRewriteEscalatesToReset proves the #3159 detection + escalation: a harness rewrite
// after which the window climbs back over the ceiling is flagged RewriteNoDrop, and once the
// non-holding streak reaches the threshold the coordinator raises EscalateReset + ActionRecommendReset.
func TestNonHoldingRewriteEscalatesToReset(t *testing.T) {
	c := NewConfig(Config{ResidentCeiling: 1000, CeilingStreakToYield: 100, NonHoldRewritesToReset: 2})
	feed := func(digest string, resident int64) Decision {
		return c.Observe(TurnObservation{InboundPrefixDigest: digest, CacheReadTokens: resident})
	}
	feed("A", 500) // establish the prefix, under ceiling

	// Rewrite 1 drops under the ceiling, then the window re-inflates over it -> non-holding #1.
	if d := feed("B", 500); d.Event != EventHarnessRewrite {
		t.Fatalf("digest change must be a harness rewrite, got %q", d.Event)
	}
	d := feed("B", 1500) // same prefix, over ceiling -> non-holding rewrite
	if !d.RewriteNoDrop {
		t.Fatal("window back over the ceiling after a rewrite must flag RewriteNoDrop")
	}
	if d.EscalateReset {
		t.Fatal("one non-holding rewrite must not escalate yet (threshold 2)")
	}

	// Rewrite 2 drops under, re-inflates over -> non-holding #2 -> escalate.
	if d := feed("C", 500); d.Event != EventHarnessRewrite {
		t.Fatalf("second digest change must be a harness rewrite, got %q", d.Event)
	}
	d = feed("C", 1500)
	if !d.RewriteNoDrop {
		t.Fatal("second re-inflation must flag RewriteNoDrop")
	}
	if !d.EscalateReset || d.Action != ActionRecommendReset {
		t.Fatalf("second non-holding rewrite must escalate: escalate=%v action=%q, want true/recommend_reset", d.EscalateReset, d.Action)
	}
}

// TestHoldingRewriteDoesNotFlagNoDrop proves the converse: a rewrite after which the window stays
// under the ceiling HELD — it must not be flagged non-holding and must never escalate.
func TestHoldingRewriteDoesNotFlagNoDrop(t *testing.T) {
	c := NewConfig(Config{ResidentCeiling: 1000, CeilingStreakToYield: 100, NonHoldRewritesToReset: 2})
	c.Observe(TurnObservation{InboundPrefixDigest: "A", CacheReadTokens: 500})
	if d := c.Observe(TurnObservation{InboundPrefixDigest: "B", CacheReadTokens: 500}); d.Event != EventHarnessRewrite {
		t.Fatalf("digest change must be a harness rewrite, got %q", d.Event)
	}
	for i := 0; i < 3; i++ { // window stays well under the ceiling for several turns
		d := c.Observe(TurnObservation{InboundPrefixDigest: "B", CacheReadTokens: 500})
		if d.RewriteNoDrop || d.EscalateReset {
			t.Fatalf("held rewrite (under ceiling) must not flag no-drop/escalate: %+v", d)
		}
	}
}
