package ablate

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// opusPricing is the default guarded-Claude base price this arm is exercised at.
var opusPricing = gateway.CachePricing{
	InputPerMTokUSD:  gateway.ClaudeOpus48InputPerMTokUSD,
	OutputPerMTokUSD: gateway.ClaudeOpus48OutputPerMTokUSD,
}

// A session whose compaction-OFF arm carries a large warm resident prefix and whose
// compaction-ON arm sheds it but pays a cache_creation burst: the net-of-burst delta must
// equal CostUSD(Off) − CostUSD(On), with the burst already priced into On (not gross shed).
func TestNetOfBurstDelta_PricesBurstIntoOnArm(t *testing.T) {
	s := MatchedCompactionSession{
		SessionID: "sess-1",
		// Compaction ON: small uncached input, a cache_creation BURST re-priming the shed prefix.
		On: gateway.CacheUsage{InputTokens: 2_000, CacheCreationTokens: 8_000, OutputTokens: 500, WriteTTL: gateway.CacheTTL5m},
		// Compaction OFF: the full resident prefix carried forward as warm cache_read, no burst.
		Off: gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 40_000, OutputTokens: 500},
	}

	got := s.NetOfBurstDeltaUSD(opusPricing)
	want := opusPricing.CostUSD(s.Off) - opusPricing.CostUSD(s.On)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("NetOfBurstDeltaUSD = %v, want CostUSD(Off)-CostUSD(On) = %v", got, want)
	}

	// The burst must be inside the On arm: an On arm with the SAME shed but NO burst is
	// strictly cheaper, so its delta is strictly larger — proving the burst is netted, not
	// dropped as gross shed.
	noBurst := s
	noBurst.On = gateway.CacheUsage{InputTokens: 2_000, OutputTokens: 500}
	if noBurst.NetOfBurstDeltaUSD(opusPricing) <= got {
		t.Fatalf("removing the burst did not raise the net delta: with-burst=%v no-burst=%v", got, noBurst.NetOfBurstDeltaUSD(opusPricing))
	}
}

// The acceptance: a sweep row over matched sessions shows the net delta with a confidence
// interval. Deltas {2,4,6,8} priced in dollars via a trivial input-only usage → mean 5 units,
// with a CI that is symmetric about the mean and clears zero.
func TestCompactionABSweep_SweepRowWithConfidenceInterval(t *testing.T) {
	// Build sessions whose Off−On net delta is exactly k input-tokens for k in {2,4,6,8},
	// by giving Off k extra uncached input tokens over On (priced at full input).
	mk := func(id string, extra int) MatchedCompactionSession {
		return MatchedCompactionSession{
			SessionID: id,
			On:        gateway.CacheUsage{InputTokens: 1_000},
			Off:       gateway.CacheUsage{InputTokens: 1_000 + extra},
		}
	}
	sessions := []MatchedCompactionSession{mk("a", 2), mk("b", 4), mk("c", 6), mk("d", 8)}

	row, err := CompactionABSweep(sessions, opusPricing, "test:opus")
	if err != nil {
		t.Fatalf("CompactionABSweep: %v", err)
	}
	if row.N != 4 || len(row.Sessions) != 4 {
		t.Fatalf("row N=%d len(Sessions)=%d, want 4/4", row.N, len(row.Sessions))
	}
	if row.ArmID != CompactionABArmID {
		t.Errorf("ArmID = %q, want %q", row.ArmID, CompactionABArmID)
	}

	// Mean of {2,4,6,8} input-token deltas priced at full input.
	perTok := gateway.ClaudeOpus48InputPerMTokUSD / 1_000_000
	wantMean := 5 * perTok
	if math.Abs(row.MeanNetUSD-wantMean) > 1e-15 {
		t.Errorf("MeanNetUSD = %v, want %v", row.MeanNetUSD, wantMean)
	}

	// The interval is real (nonzero width for N>1) and symmetric about the mean.
	if row.CI95HighUSD <= row.CI95LowUSD {
		t.Errorf("CI has non-positive width: [%v, %v]", row.CI95LowUSD, row.CI95HighUSD)
	}
	mid := (row.CI95LowUSD + row.CI95HighUSD) / 2
	if math.Abs(mid-row.MeanNetUSD) > 1e-15 {
		t.Errorf("CI not centered on mean: mid=%v mean=%v", mid, row.MeanNetUSD)
	}
	// Sample sd of {2,4,6,8} = sqrt(20/3); SE = sd/2; half = z*SE. Verify the reported width.
	sd := math.Sqrt(((math.Pow(2-5, 2) + math.Pow(4-5, 2) + math.Pow(6-5, 2) + math.Pow(8-5, 2)) / 3))
	wantHalf := zScore95 * (sd * perTok) / 2
	if math.Abs((row.CI95HighUSD-row.MeanNetUSD)-wantHalf) > 1e-15 {
		t.Errorf("CI half-width = %v, want %v", row.CI95HighUSD-row.MeanNetUSD, wantHalf)
	}

	// All four deltas are positive → the interval clears zero, and the sweep-row string
	// names the delta, the interval, and the count.
	if !row.ClearsZero() {
		t.Errorf("expected all-positive deltas to clear zero: [%v, %v]", row.CI95LowUSD, row.CI95HighUSD)
	}
	line := row.SweepRow()
	for _, want := range []string{CompactionABArmID, "95% CI", "N=4"} {
		if !strings.Contains(line, want) {
			t.Errorf("SweepRow() = %q, missing %q", line, want)
		}
	}
}

// A burst that outweighs the shed must read NEGATIVE — the arm cannot launder a cost into a
// win. Compaction ON drops warm reads (cheap, 0.1x) but writes a fresh prefix (1.25x); net < 0.
func TestCompactionABSweep_BurstOutweighingShedReadsNegative(t *testing.T) {
	s := MatchedCompactionSession{
		SessionID: "cold-burst",
		On:        gateway.CacheUsage{InputTokens: 1_000, CacheCreationTokens: 20_000, WriteTTL: gateway.CacheTTL5m},
		Off:       gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 20_000},
	}
	row, err := CompactionABSweep([]MatchedCompactionSession{s}, opusPricing, "")
	if err != nil {
		t.Fatalf("CompactionABSweep: %v", err)
	}
	if row.MeanNetUSD >= 0 {
		t.Errorf("burst-heavy session should read negative net, got %v", row.MeanNetUSD)
	}
	// Single session → degenerate interval + the naming caveat.
	if row.StdErrUSD != 0 || row.CI95LowUSD != row.MeanNetUSD || row.CI95HighUSD != row.MeanNetUSD {
		t.Errorf("single-session interval should be degenerate, got SE=%v CI=[%v,%v]", row.StdErrUSD, row.CI95LowUSD, row.CI95HighUSD)
	}
	if row.Caveat == "" {
		t.Errorf("single-session row should carry the degenerate-interval caveat")
	}
}

// Fail closed: no matched sessions is an error (never a fabricated $0 no-op row); a session
// with no token activity on either arm cannot be a matched real session.
func TestCompactionABSweep_FailsClosed(t *testing.T) {
	if _, err := CompactionABSweep(nil, opusPricing, ""); err == nil {
		t.Errorf("empty session set should error, got nil")
	}
	empty := []MatchedCompactionSession{{SessionID: "void"}}
	if _, err := CompactionABSweep(empty, opusPricing, ""); err == nil {
		t.Errorf("session with no token activity should error, got nil")
	}
}
