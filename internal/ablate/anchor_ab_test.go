package ablate

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// opusPricing is shared with compaction_ab_test.go (same package): the default guarded-Claude
// base price both live A/B arms are exercised at.

// A session whose FirstBP arm carries a large warm resident prefix (idle lever) and whose Head
// arm sheds it but pays a cache_creation burst: the net-of-burst delta must equal
// CostUSD(FirstBP) − CostUSD(Head), with the burst already priced into Head (not gross shed).
func TestNetOfBurstDelta_PricesBurstIntoHeadArm(t *testing.T) {
	s := MatchedAnchorSession{
		SessionID: "sess-1",
		// FirstBP (idle default): the full resident prefix carried forward as warm cache_read, no burst.
		FirstBP: gateway.CacheUsage{InputTokens: 2_000, CacheReadTokens: 40_000, OutputTokens: 500},
		// Head (fires): small uncached input, a cache_creation BURST re-priming the shed prefix.
		Head: gateway.CacheUsage{InputTokens: 2_000, CacheCreationTokens: 8_000, OutputTokens: 500, WriteTTL: gateway.CacheTTL5m},
	}

	got := s.NetOfBurstDeltaUSD(opusPricing)
	want := opusPricing.CostUSD(s.FirstBP) - opusPricing.CostUSD(s.Head)
	if math.Abs(got-want) > 1e-12 {
		t.Fatalf("NetOfBurstDeltaUSD = %v, want CostUSD(FirstBP)-CostUSD(Head) = %v", got, want)
	}

	// The burst must be inside the Head arm: a Head arm with the SAME shed but NO burst is
	// strictly cheaper, so its delta is strictly larger — proving the burst is netted, not
	// dropped as gross shed.
	noBurst := s
	noBurst.Head = gateway.CacheUsage{InputTokens: 2_000, OutputTokens: 500}
	if noBurst.NetOfBurstDeltaUSD(opusPricing) <= got {
		t.Fatalf("removing the burst did not raise the net delta: with-burst=%v no-burst=%v", got, noBurst.NetOfBurstDeltaUSD(opusPricing))
	}
}

// The acceptance: a sweep row over matched sessions shows the net delta with a confidence
// interval. Deltas {2,4,6,8} priced in dollars via a trivial input-only usage → mean 5 units,
// with a CI that is symmetric about the mean and clears zero.
func TestAnchorABSweep_SweepRowWithConfidenceInterval(t *testing.T) {
	// Build sessions whose FirstBP−Head net delta is exactly k input-tokens for k in {2,4,6,8},
	// by giving FirstBP (the idle, more expensive arm) k extra uncached input tokens over Head.
	mk := func(id string, extra int) MatchedAnchorSession {
		return MatchedAnchorSession{
			SessionID: id,
			FirstBP:   gateway.CacheUsage{InputTokens: 1_000 + extra},
			Head:      gateway.CacheUsage{InputTokens: 1_000},
		}
	}
	sessions := []MatchedAnchorSession{mk("a", 2), mk("b", 4), mk("c", 6), mk("d", 8)}

	row, err := AnchorABSweep(sessions, opusPricing, "test:opus")
	if err != nil {
		t.Fatalf("AnchorABSweep: %v", err)
	}
	if row.N != 4 || len(row.Sessions) != 4 {
		t.Fatalf("row N=%d len(Sessions)=%d, want 4/4", row.N, len(row.Sessions))
	}
	if row.ArmID != AnchorABArmID {
		t.Errorf("ArmID = %q, want %q", row.ArmID, AnchorABArmID)
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
	for _, want := range []string{AnchorABArmID, "95% CI", "N=4"} {
		if !strings.Contains(line, want) {
			t.Errorf("SweepRow() = %q, missing %q", line, want)
		}
	}
}

// A Head burst that outweighs the shed must read NEGATIVE — head-anchoring cannot launder a
// cost into a win. Head drops warm reads (cheap, 0.1x) but writes a fresh prefix (1.25x); the
// net-of-burst delta FirstBP−Head is < 0, the exact case #1408's CacheBurstPaysBack refuses.
func TestAnchorABSweep_BurstOutweighingShedReadsNegative(t *testing.T) {
	s := MatchedAnchorSession{
		SessionID: "cold-burst",
		// FirstBP idle: the warm resident prefix, cheap cache_read.
		FirstBP: gateway.CacheUsage{InputTokens: 1_000, CacheReadTokens: 20_000},
		// Head fires: sheds that prefix but pays a fresh cache_creation burst re-priming it.
		Head: gateway.CacheUsage{InputTokens: 1_000, CacheCreationTokens: 20_000, WriteTTL: gateway.CacheTTL5m},
	}
	row, err := AnchorABSweep([]MatchedAnchorSession{s}, opusPricing, "")
	if err != nil {
		t.Fatalf("AnchorABSweep: %v", err)
	}
	if row.MeanNetUSD >= 0 {
		t.Errorf("burst-heavy Head session should read negative net, got %v", row.MeanNetUSD)
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
func TestAnchorABSweep_FailsClosed(t *testing.T) {
	if _, err := AnchorABSweep(nil, opusPricing, ""); err == nil {
		t.Errorf("empty session set should error, got nil")
	}
	empty := []MatchedAnchorSession{{SessionID: "void"}}
	if _, err := AnchorABSweep(empty, opusPricing, ""); err == nil {
		t.Errorf("session with no token activity should error, got nil")
	}
}

// The issue's acceptance stated IN WORDS: Verdict() renders the three-way net-dollar read (#2809).
// A positive delta whose CI clears zero → net-beneficial; a negative delta whose CI clears zero →
// net-negative; a CI straddling zero → not yet distinguishable. Each verdict must state the net
// dollars, the interval, and the matched-session count, and must agree with ClearsZero().
func TestAnchorABRow_VerdictThreeWay(t *testing.T) {
	cases := []struct {
		name string
		row  AnchorABRow
		want string // the distinguishing phrase the verdict must carry
	}{
		{"beneficial", AnchorABRow{MeanNetUSD: 0.50, CI95LowUSD: 0.10, CI95HighUSD: 0.90, N: 4}, "IS net-beneficial"},
		{"negative", AnchorABRow{MeanNetUSD: -0.50, CI95LowUSD: -0.90, CI95HighUSD: -0.10, N: 4}, "NET-NEGATIVE"},
		{"undetermined", AnchorABRow{MeanNetUSD: 0.10, CI95LowUSD: -0.20, CI95HighUSD: 0.40, N: 4}, "NOT YET DISTINGUISHABLE"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.row.Verdict()
			if !strings.Contains(got, tc.want) {
				t.Errorf("Verdict() = %q, want phrase %q", got, tc.want)
			}
			// A directional verdict is claimed iff the interval clears zero — the worded read may
			// not out-run the numbers.
			directional := strings.Contains(got, "net-beneficial") || strings.Contains(got, "NET-NEGATIVE")
			if directional != tc.row.ClearsZero() {
				t.Errorf("Verdict directional=%v but ClearsZero()=%v for %q", directional, tc.row.ClearsZero(), got)
			}
			// Every verdict states the net dollars, the 95% interval, and the matched count.
			for _, s := range []string{"net $", "95% CI", "N=4"} {
				if !strings.Contains(got, s) {
					t.Errorf("Verdict() = %q, missing %q", got, s)
				}
			}
		})
	}
}
