package ablate

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevaluereport"
	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// opusTrack2Pricing is the trusted guarded-Claude base price the managed-cache A/B is folded at
// (the Track-2 SavingsPricing twin of compaction_ab_test.go's opusPricing).
var opusTrack2Pricing = cachevaluereport.SavingsPricing{
	InputPerMTokUSD:  gateway.ClaudeOpus48InputPerMTokUSD,
	OutputPerMTokUSD: gateway.ClaudeOpus48OutputPerMTokUSD,
	Source:           "test:opus",
}

func managedCacheNow(t *testing.T) time.Time {
	t.Helper()
	return time.Date(2026, 7, 12, 0, 0, 0, 0, time.UTC)
}

// A workload where the full managed-cache posture pays off: ACTIVE holds a large WARM resident
// prefix (mostly cache_read) with a modest 1h-upgraded write, while PASSIVE re-primes the same
// work COLD (large 5m cache_creation, little reuse). The whole-posture net-$ delta must be
// positive — ACTIVE saves net $ over PASSIVE end to end.
func TestManagedCacheAB_ActivePostureSavesNet(t *testing.T) {
	replay := ManagedCacheReplay{
		WorkloadHash: "workloadA",
		Active: gateway.CacheUsage{
			InputTokens: 1_000, CacheReadTokens: 50_000, CacheCreationTokens: 2_000, OutputTokens: 1_000,
		},
		ActiveUpgradedCreationTokens: 2_000, // the whole write rode the 1h upgrade
		Passive: gateway.CacheUsage{
			InputTokens: 1_000, CacheReadTokens: 5_000, CacheCreationTokens: 20_000, OutputTokens: 1_000,
		},
	}

	rep, err := ManagedCacheABSweep(replay, opusTrack2Pricing, managedCacheNow(t))
	if err != nil {
		t.Fatalf("ManagedCacheABSweep: %v", err)
	}
	if rep.ArmID != ManagedCacheABArmID {
		t.Errorf("ArmID = %q, want %q", rep.ArmID, ManagedCacheABArmID)
	}
	if rep.Active.ArmID != ManagedCacheArmActive || rep.Passive.ArmID != ManagedCacheArmPassive {
		t.Errorf("arm ids = %q/%q, want %q/%q", rep.Active.ArmID, rep.Passive.ArmID, ManagedCacheArmActive, ManagedCacheArmPassive)
	}
	if !rep.ActiveSavesNet() || rep.NetDeltaUSD <= 0 {
		t.Fatalf("expected ACTIVE to save net $ over PASSIVE, got Δ=%v (active=%v passive=%v)",
			rep.NetDeltaUSD, rep.Active.NetUSD, rep.Passive.NetUSD)
	}
	if v := rep.Verdict(); !strings.Contains(v, "IS net-beneficial vs PASSIVE") {
		t.Errorf("Verdict() = %q, want net-beneficial", v)
	}
	// The one-liner names the arm, both postures, and the delta.
	if line := rep.SweepRow(); !strings.Contains(line, ManagedCacheABArmID) || !strings.Contains(line, "ACTIVE net") || !strings.Contains(line, "PASSIVE net") {
		t.Errorf("SweepRow() = %q, missing arm/posture labels", line)
	}
}

// The honesty half of the done condition: a workload where ACTIVE is the WRONG posture — it
// pays a big 2.0x 1h write premium re-priming a prefix it barely reads, while PASSIVE stays
// warm — must read NET-NEGATIVE. The arm cannot launder a losing posture into a win.
func TestManagedCacheAB_NegativeDeltaSurfacedHonestly(t *testing.T) {
	replay := ManagedCacheReplay{
		WorkloadHash: "workloadB",
		Active: gateway.CacheUsage{
			InputTokens: 1_000, CacheReadTokens: 2_000, CacheCreationTokens: 40_000, OutputTokens: 500,
		},
		ActiveUpgradedCreationTokens: 40_000, // all cold, all at the 2.0x 1h tier — the pure loss
		Passive: gateway.CacheUsage{
			InputTokens: 1_000, CacheReadTokens: 40_000, CacheCreationTokens: 2_000, OutputTokens: 500,
		},
	}

	rep, err := ManagedCacheABSweep(replay, opusTrack2Pricing, managedCacheNow(t))
	if err != nil {
		t.Fatalf("ManagedCacheABSweep: %v", err)
	}
	if rep.ActiveSavesNet() || rep.NetDeltaUSD >= 0 {
		t.Fatalf("expected ACTIVE to read NET-NEGATIVE vs PASSIVE, got Δ=%v (active=%v passive=%v)",
			rep.NetDeltaUSD, rep.Active.NetUSD, rep.Passive.NetUSD)
	}
	v := rep.Verdict()
	if !strings.Contains(v, "NET-NEGATIVE vs PASSIVE") {
		t.Errorf("Verdict() = %q, want NET-NEGATIVE surfaced", v)
	}
	// The signed delta is on the artifact, not floored away.
	if !strings.Contains(string(rep.JSON()), "\"net_delta_usd\"") {
		t.Errorf("JSON artifact must carry the signed net_delta_usd")
	}
}

// Reconciliation with the Track-2 audit fold (the #3631 witness): the arm's dollars ARE the
// ledger's dollars. For each posture arm the audit's two identities must hold on the reported
// numbers — NetUSD == rebate − write-premium − spend (no compaction rows), and rebate −
// write-premium == counterfactual − spend — and the whole-posture delta is exactly the two
// arms' NET difference. A direct re-fold of the ACTIVE arm through FoldAudit reproduces its
// NetUSD, proving the arm reports net-$ VIA the Track-2 audit fold, not a private re-derivation.
func TestManagedCacheAB_ReconcilesWithTrack2AuditFold(t *testing.T) {
	now := managedCacheNow(t)
	replay := ManagedCacheReplay{
		WorkloadHash:                 "workloadC",
		Active:                       gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 30_000, CacheCreationTokens: 4_000, OutputTokens: 800},
		ActiveUpgradedCreationTokens: 3_000,
		Passive:                      gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 8_000, CacheCreationTokens: 18_000, OutputTokens: 800},
	}
	rep, err := ManagedCacheABSweep(replay, opusTrack2Pricing, now)
	if err != nil {
		t.Fatalf("ManagedCacheABSweep: %v", err)
	}

	for _, arm := range []ManagedCacheArm{rep.Active, rep.Passive} {
		// NetUSD identity (the fold's own drift tripwire, at the arm level): a compaction-free
		// provider arm's NET is rebate − write-premium − spend.
		if got, want := arm.NetUSD, arm.RebateUSD-arm.WritePremiumUSD-arm.SpendUSD; math.Abs(got-want) > 1e-12 {
			t.Errorf("%s NetUSD=%v does not reconcile with rebate−writeprem−spend=%v", arm.ArmID, got, want)
		}
		// The audit's core identity: rebate − write-premium == counterfactual − actual spend.
		if lhs, rhs := arm.RebateUSD-arm.WritePremiumUSD, arm.CounterfactualUSD-arm.SpendUSD; math.Abs(lhs-rhs) > 1e-12 {
			t.Errorf("%s (rebate−writeprem)=%v != (counterfactual−spend)=%v", arm.ArmID, lhs, rhs)
		}
	}
	if got, want := rep.NetDeltaUSD, rep.Active.NetUSD-rep.Passive.NetUSD; math.Abs(got-want) > 1e-12 {
		t.Errorf("NetDeltaUSD=%v != Active.NetUSD−Passive.NetUSD=%v", got, want)
	}

	// Re-fold the ACTIVE arm independently through the Track-2 audit and confirm the arm read
	// its NetUSD straight off FoldAudit's cumulative account (drift ~0, NetUSD reproduced).
	activeObs := cachevaluereport.SavingsObservation{
		SessionType: "ablate", Provider: "anthropic", Context: "test:opus",
		InputTokens: 2_000, CacheReadTokens: 30_000, CacheCreationTokens: 4_000, OutputTokens: 800,
		CacheCreationTokensUpgraded: 3_000, Pricing: opusTrack2Pricing,
	}
	audit := cachevaluereport.FoldAudit(cachevaluereport.NewSavingsRows(activeObs, now), now)
	if math.Abs(audit.NetReconciliationDriftUSD) > 1e-9 {
		t.Errorf("Track-2 audit NET reconciliation drift=%v, want ~0", audit.NetReconciliationDriftUSD)
	}
	if math.Abs(audit.Cumulative.NetUSD-rep.Active.NetUSD) > 1e-12 {
		t.Errorf("ACTIVE arm NetUSD=%v != FoldAudit cumulative NetUSD=%v (arm did not report via the audit fold)",
			rep.Active.NetUSD, audit.Cumulative.NetUSD)
	}
}

// One WorkloadHash binds both arms, and the sweep fails closed on the states that would make a
// whole-posture delta dishonest: no hash, a dollar-blind price, both arms empty, or an arm with
// no cache activity.
func TestManagedCacheAB_WorkloadHashBindsAndFailsClosed(t *testing.T) {
	now := managedCacheNow(t)
	ok := ManagedCacheReplay{
		WorkloadHash: "bindme",
		Active:       gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 10_000, CacheCreationTokens: 1_000, OutputTokens: 500},
		Passive:      gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 1_000, CacheCreationTokens: 8_000, OutputTokens: 500},
	}
	rep, err := ManagedCacheABSweep(ok, opusTrack2Pricing, now)
	if err != nil {
		t.Fatalf("valid replay errored: %v", err)
	}
	if rep.WorkloadHash != "bindme" {
		t.Errorf("report WorkloadHash = %q, want %q", rep.WorkloadHash, "bindme")
	}

	// No WorkloadHash → refuse (the two arms are not pinned to one workload).
	noHash := ok
	noHash.WorkloadHash = "   "
	if _, err := ManagedCacheABSweep(noHash, opusTrack2Pricing, now); err == nil {
		t.Errorf("missing WorkloadHash should error, got nil")
	}

	// Dollar-blind price → refuse (no trusted $/MTok, so no honest net-$).
	if _, err := ManagedCacheABSweep(ok, cachevaluereport.SavingsPricing{DollarBlind: true, Source: "unpriced"}, now); err == nil {
		t.Errorf("dollar-blind pricing should error, got nil")
	}

	// Both arms empty → refuse.
	empty := ManagedCacheReplay{WorkloadHash: "void"}
	if _, err := ManagedCacheABSweep(empty, opusTrack2Pricing, now); err == nil {
		t.Errorf("both arms empty should error, got nil")
	}

	// An arm with input/output only (no cache activity) → refuse: not a cache posture measurement.
	noCache := ManagedCacheReplay{
		WorkloadHash: "nocache",
		Active:       gateway.CacheUsage{InputTokens: 5_000, OutputTokens: 1_000},
		Passive:      gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 1_000, CacheCreationTokens: 8_000, OutputTokens: 500},
	}
	if _, err := ManagedCacheABSweep(noCache, opusTrack2Pricing, now); err == nil {
		t.Errorf("an arm with no cache activity should error, got nil")
	}
}

// The JSON artifact round-trips: the two posture arms and the signed delta survive a
// marshal/unmarshal, so a saved sweep can be re-read.
func TestManagedCacheAB_JSONArtifactRoundTrips(t *testing.T) {
	replay := ManagedCacheReplay{
		WorkloadHash:                 "roundtrip",
		Active:                       gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 20_000, CacheCreationTokens: 2_000, OutputTokens: 600},
		ActiveUpgradedCreationTokens: 2_000,
		Passive:                      gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 4_000, CacheCreationTokens: 12_000, OutputTokens: 600},
	}
	rep, err := ManagedCacheABSweep(replay, opusTrack2Pricing, managedCacheNow(t))
	if err != nil {
		t.Fatalf("ManagedCacheABSweep: %v", err)
	}
	var back ManagedCacheABReport
	if err := json.Unmarshal(rep.JSON(), &back); err != nil {
		t.Fatalf("JSON round-trip: %v", err)
	}
	if back.ArmID != rep.ArmID || back.WorkloadHash != rep.WorkloadHash {
		t.Errorf("round-trip lost identity: %+v vs %+v", back, rep)
	}
	if math.Abs(back.NetDeltaUSD-rep.NetDeltaUSD) > 1e-12 {
		t.Errorf("round-trip changed NetDeltaUSD: %v vs %v", back.NetDeltaUSD, rep.NetDeltaUSD)
	}
	if back.Active.ArmID != ManagedCacheArmActive || back.Passive.ArmID != ManagedCacheArmPassive {
		t.Errorf("round-trip lost arm ids: %q/%q", back.Active.ArmID, back.Passive.ArmID)
	}
}
