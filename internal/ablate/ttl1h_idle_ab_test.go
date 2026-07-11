package ablate

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// The acceptance (#3629): a two-arm report showing the post-idle cache-read fraction is MATERIALLY
// higher with the 1h upgrade ON. A realistic re-entry — a 40k surviving prefix over a 2k fresh user
// turn after a 20m idle gap — lands the ON arm's prefix WARM (all cache_read) and the OFF arm's prefix
// COLD (all cache_creation, the 5m tier having expired). Same prompt tokens both arms; only the split
// differs.
func TestTTL1HIdleABSweep_UpgradeRealizesPostIdleRead(t *testing.T) {
	const prefix, fresh, out = 40_000, 2_000, 500
	// ON (1h): the >5m idle survived, so the re-entered prefix is served warm as cache_read.
	on := gateway.CacheUsage{InputTokens: fresh, CacheReadTokens: prefix, OutputTokens: out, WriteTTL: gateway.CacheTTL1h}
	// OFF (5m): the tier expired across the idle, so the SAME prefix re-primes cold as cache_creation.
	off := gateway.CacheUsage{InputTokens: fresh, CacheCreationTokens: prefix, OutputTokens: out, WriteTTL: gateway.CacheTTL5m}

	rep, err := TTL1HIdleABSweep(20*time.Minute, on, off)
	if err != nil {
		t.Fatalf("TTL1HIdleABSweep: %v", err)
	}
	if rep.ArmID != TTL1HIdleABArmID {
		t.Errorf("ArmID = %q, want %q", rep.ArmID, TTL1HIdleABArmID)
	}
	if rep.On.ArmID != TTL1HIdleArmOn || rep.On.TTL != string(gateway.CacheTTL1h) {
		t.Errorf("on arm mislabeled: id=%q ttl=%q", rep.On.ArmID, rep.On.TTL)
	}
	if rep.Off.ArmID != TTL1HIdleArmOff || rep.Off.TTL != string(gateway.CacheTTL5m) {
		t.Errorf("off arm mislabeled: id=%q ttl=%q", rep.Off.ArmID, rep.Off.TTL)
	}

	// Observed fractions: ON = prefix/(prefix+fresh), OFF = 0 (the prefix landed as a cold write).
	wantOn := float64(prefix) / float64(prefix+fresh)
	if math.Abs(rep.On.CacheReadFraction-wantOn) > 1e-12 {
		t.Errorf("On.CacheReadFraction = %v, want %v", rep.On.CacheReadFraction, wantOn)
	}
	if rep.Off.CacheReadFraction != 0 {
		t.Errorf("Off.CacheReadFraction = %v, want 0 (expired tier re-primes cold)", rep.Off.CacheReadFraction)
	}
	if math.Abs(rep.ReadFractionDelta-wantOn) > 1e-12 {
		t.Errorf("ReadFractionDelta = %v, want %v", rep.ReadFractionDelta, wantOn)
	}
	// The identical-workload guard passed ⇒ both arms report the same post-idle prompt tokens.
	if rep.PromptTokens != prefix+fresh || rep.On.PromptTokens != rep.Off.PromptTokens {
		t.Errorf("prompt tokens not identical across arms: report=%d on=%d off=%d", rep.PromptTokens, rep.On.PromptTokens, rep.Off.PromptTokens)
	}

	// The #3629 done condition: materially higher with the upgrade on.
	if !rep.UpgradeRealizesRead() {
		t.Errorf("expected the upgrade to realize the post-idle read (Δ=%v ≥ %v)", rep.ReadFractionDelta, MaterialReadFractionGap)
	}
	line := rep.SweepRow()
	for _, want := range []string{TTL1HIdleABArmID, "materially higher", "idle gap"} {
		if !strings.Contains(line, want) {
			t.Errorf("SweepRow() = %q, missing %q", line, want)
		}
	}
	// The artifact renders as valid JSON carrying both arms' observed fractions.
	var round map[string]any
	if err := json.Unmarshal(rep.JSON(), &round); err != nil {
		t.Fatalf("JSON() is not valid JSON: %v", err)
	}
	if _, ok := round["read_fraction_delta"]; !ok {
		t.Errorf("JSON artifact missing read_fraction_delta: %s", rep.JSON())
	}
}

// The identical-workload guard: two arms that re-entered DIFFERENT-sized prefixes are not
// apples-to-apples and must fail closed, never produce a bogus delta.
func TestTTL1HIdleABSweep_IdenticalWorkloadGuard(t *testing.T) {
	on := gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 40_000, WriteTTL: gateway.CacheTTL1h}
	off := gateway.CacheUsage{InputTokens: 2_000, CacheCreationTokens: 30_000, WriteTTL: gateway.CacheTTL5m} // 32k ≠ 42k prompt tokens
	if _, err := TTL1HIdleABSweep(20*time.Minute, on, off); err == nil {
		t.Fatalf("expected the identical-workload guard to reject mismatched post-idle prompt tokens, got nil")
	}
}

// The premise is a gap that EXPIRES the default 5m tier. A gap that is not strictly >5m survives on
// both arms — nothing to A/B — so the builder rejects it (boundary: exactly 5m is rejected).
func TestTTL1HIdleABSweep_RejectsNonExpiringGap(t *testing.T) {
	on := gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 40_000, WriteTTL: gateway.CacheTTL1h}
	off := gateway.CacheUsage{InputTokens: 2_000, CacheCreationTokens: 40_000, WriteTTL: gateway.CacheTTL5m}
	for _, gap := range []time.Duration{3 * time.Minute, 5 * time.Minute} {
		if _, err := TTL1HIdleABSweep(gap, on, off); err == nil {
			t.Errorf("gap %s is not >5m and should be rejected, got nil", gap)
		}
	}
}

// Fail closed on a post-idle turn with no billable prompt tokens on either arm — that is not a
// re-entry, and a fabricated zero-token fraction would read as a measured no-op.
func TestTTL1HIdleABSweep_FailsClosedOnEmptyTurn(t *testing.T) {
	valid := gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 40_000, WriteTTL: gateway.CacheTTL1h}
	if _, err := TTL1HIdleABSweep(20*time.Minute, gateway.CacheUsage{}, valid); err == nil {
		t.Errorf("empty ON turn should fail closed, got nil")
	}
	if _, err := TTL1HIdleABSweep(20*time.Minute, valid, gateway.CacheUsage{}); err == nil {
		t.Errorf("empty OFF turn should fail closed, got nil")
	}
}

// Honesty the other direction: if the ON arm did NOT realize a surviving read (e.g. the injected gap
// exceeded the 1h tier's OWN retention, so both arms re-primed cold), the report reads false and the
// caveat names the 1h limit — the upgrade cannot launder a paid write premium into a realized read.
func TestTTL1HIdleABSweep_NoRealizedReadReadsFalseWithCaveat(t *testing.T) {
	const prefix, fresh = 40_000, 2_000
	on := gateway.CacheUsage{InputTokens: fresh, CacheCreationTokens: prefix, WriteTTL: gateway.CacheTTL1h}  // expired too
	off := gateway.CacheUsage{InputTokens: fresh, CacheCreationTokens: prefix, WriteTTL: gateway.CacheTTL5m} // expired
	rep, err := TTL1HIdleABSweep(90*time.Minute, on, off)
	if err != nil {
		t.Fatalf("TTL1HIdleABSweep: %v", err)
	}
	if rep.UpgradeRealizesRead() {
		t.Errorf("both-cold arms should NOT read as a realized-read win (Δ=%v)", rep.ReadFractionDelta)
	}
	if !strings.Contains(rep.Caveat, "1h tier") {
		t.Errorf("a gap ≥1h should carry the tier-retention caveat, got %q", rep.Caveat)
	}
	if !strings.Contains(rep.SweepRow(), "NOT materially higher") {
		t.Errorf("SweepRow should report the non-win: %q", rep.SweepRow())
	}
}
