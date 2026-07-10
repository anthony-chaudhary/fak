package ablate

// Live compaction A/B arm (#2805). The frozen-trace sweep in ablate.go isolates a kernel
// knob on ONE workload hash; this file adds the LIVE rung the package doc reserves: a
// controlled compaction ON vs OFF split over MATCHED real guard sessions, reported in NET
// dollars — the shed saving MINUS the one-time cache-creation BURST compaction causes when
// it rewrites history — never gross shed. The only honest proof of compaction value is a
// causal ON/OFF delta on real traffic, netted for the burst; this is the wire that turns
// that split into a measured dollar delta with a confidence interval.
//
// NET OF BURST, not gross shed. Compaction OFF keeps the full resident prefix, most of it
// served WARM as provider cache_read (0.1x). Compaction ON sheds that prefix but pays a
// cache_creation burst to re-prime (1.25x/2.0x). So compaction can COST money when the shed
// tokens were already cheap warm reads and the burst writes fresh — exactly the tension a
// gross-shed number hides. This arm prices BOTH sessions with gateway.CachePricing.CostUSD,
// which already books the cache_creation burst at its write multiplier: the SAME
// burst-premium netting the report side uses (#2795), reused by construction rather than
// reinvented here, so the two program numbers cannot disagree.
//
// Generation posture (gen/next, #2783): this is a near-term foundation seam. It is a
// deterministic ($0, no model, no GPU) valuation over caller-supplied matched-session token
// axes — it does not itself run the split. Promotion evidence = a live guard capture feeding
// real matched ON/OFF sessions whose sweep-row CI clears zero. Invalidating assumptions are
// named on CompactionABRow (normal-approximation CI; caller-guaranteed session matching).

import (
	"errors"
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// zScore95 is the two-sided 95% critical value of the standard normal distribution — the
// same constant internal/agentdojo prices its 95% interval on. The mean net-of-burst delta's
// confidence interval uses the NORMAL approximation (mean ± z*SE); for a small matched-session
// count a Student-t interval would be wider, so a near-threshold row should be read as a
// lower bound on the interval width, not an exact one. This is the row's named invalidating
// assumption (see CompactionABRow.Caveat).
const zScore95 = 1.959963984540054

// CompactionArm names the two arms of the live compaction split, for labeling.
const (
	CompactionArmOn  = "compaction_on"
	CompactionArmOff = "compaction_off"
	// CompactionABArmID is the sweep-row arm id for the ON-vs-OFF net-of-burst delta.
	CompactionABArmID = "compaction_on_vs_off"
)

// MatchedCompactionSession is ONE real guard session measured under BOTH compaction settings
// on the same underlying traffic, so their dollar difference isolates compaction:
//
//   - On  is the token accounting with compaction ENABLED: a smaller resident prefix, but a
//     one-time cache_creation BURST (On.CacheCreationTokens) when compaction rewrites history.
//   - Off is the SAME session with compaction DISABLED: the full resident prefix carried
//     forward (mostly warm CacheReadTokens), no burst.
//
// Each side is a session-cumulative gateway.CacheUsage aggregate; the cache pricing model is
// linear, so one aggregate row reproduces the sum of the session's per-turn rows (the same
// convention AdjudicationSummary.ProviderCacheNetSavings relies on). Matching the two sides
// to comparable traffic is the CALLER's contract — a mismatch invalidates the delta, not this
// arithmetic (see CompactionABRow.Caveat).
type MatchedCompactionSession struct {
	SessionID string
	On        gateway.CacheUsage
	Off       gateway.CacheUsage
}

// NetOfBurstDeltaUSD is the dollars compaction ON saved on THIS session, NET of the burst it
// caused: CostUSD(Off) − CostUSD(On). CostUSD(On) already prices the cache_creation burst at
// its TTL write multiplier (the #2795 burst premium), so the result is net-of-burst by
// construction, never gross shed. Positive = compaction paid for its burst and then some;
// negative = the burst outweighed the shed (dropping already-warm reads to write a fresh
// prefix cost more than it saved).
func (m MatchedCompactionSession) NetOfBurstDeltaUSD(p gateway.CachePricing) float64 {
	return p.CostUSD(m.Off) - p.CostUSD(m.On)
}

// SessionDelta is one matched session's net-of-burst dollar delta, retained on the row so a
// reader can see the per-session spread the confidence interval summarizes, not just the mean.
type SessionDelta struct {
	SessionID  string  `json:"session_id"`
	NetUSD     float64 `json:"net_usd"`
	OnCostUSD  float64 `json:"on_cost_usd"`
	OffCostUSD float64 `json:"off_cost_usd"`
}

// CompactionABRow is the ONE sweep row the live compaction ON/OFF arm reports: the mean
// per-session net-of-burst dollar delta across N matched sessions, with a 95% confidence
// interval so a reader sees whether the delta clears zero, plus the per-session breakdown.
// Positive MeanNetUSD = compaction saved money net of burst on average; a CI straddling zero
// means the evidence does not yet distinguish the arm from no-op on this traffic.
type CompactionABRow struct {
	ArmID         string         `json:"arm_id"`
	N             int            `json:"n"`
	MeanNetUSD    float64        `json:"mean_net_usd"`
	StdErrUSD     float64        `json:"std_err_usd"`
	CI95LowUSD    float64        `json:"ci95_low_usd"`
	CI95HighUSD   float64        `json:"ci95_high_usd"`
	Sessions      []SessionDelta `json:"sessions"`
	PricingSource string         `json:"pricing_source,omitempty"`
	Caveat        string         `json:"caveat,omitempty"`
}

// SweepRow renders the human one-liner the acceptance calls for: the net delta with its
// confidence interval and the matched-session count. This is the "sweep row [that] shows the
// net delta with a confidence interval" (#2805).
func (r CompactionABRow) SweepRow() string {
	return fmt.Sprintf("%s: net $%+.4f/session [95%% CI $%+.4f, $%+.4f] over N=%d matched sessions",
		r.ArmID, r.MeanNetUSD, r.CI95LowUSD, r.CI95HighUSD, r.N)
}

// ClearsZero reports whether the whole 95% interval sits on one side of zero — the honest
// read of "is this delta real": true only when the CI does NOT straddle $0. A row whose
// interval brackets zero has not yet earned a directional claim.
func (r CompactionABRow) ClearsZero() bool {
	return r.CI95LowUSD > 0 || r.CI95HighUSD < 0
}

// CompactionABSweep folds MATCHED compaction ON/OFF sessions into the one sweep row: it prices
// each session's net-of-burst delta with CachePricing (the shared burst-premium engine),
// takes the mean over sessions, and brackets it with a 95% confidence interval. It fails
// closed on an empty session set — there is no row to report and a fabricated zero would read
// as a measured no-op — and on a session that is degenerate on both arms (no billable tokens
// either way), which cannot be a matched real session.
//
// The interval uses the sample standard deviation (Bessel-corrected, n−1) and the normal
// approximation mean ± z*SE. With a single matched session the interval is degenerate
// (zero width); the row carries a caveat naming that and the small-N t-vs-z gap so a reader
// never mistakes a thin interval for a strong one.
func CompactionABSweep(sessions []MatchedCompactionSession, p gateway.CachePricing, pricingSource string) (CompactionABRow, error) {
	if len(sessions) == 0 {
		return CompactionABRow{}, errors.New("ablate: compaction A/B needs at least one matched session")
	}
	deltas := make([]SessionDelta, 0, len(sessions))
	raw := make([]float64, 0, len(sessions))
	for i, s := range sessions {
		if s.On == (gateway.CacheUsage{}) && s.Off == (gateway.CacheUsage{}) {
			return CompactionABRow{}, fmt.Errorf("ablate: matched session %d (%q) has no token activity on either arm", i, s.SessionID)
		}
		on := p.CostUSD(s.On)
		off := p.CostUSD(s.Off)
		deltas = append(deltas, SessionDelta{
			SessionID:  s.SessionID,
			NetUSD:     off - on,
			OnCostUSD:  on,
			OffCostUSD: off,
		})
		raw = append(raw, off-on)
	}

	mean, stdErr := meanStdErr(raw)
	half := zScore95 * stdErr
	row := CompactionABRow{
		ArmID:         CompactionABArmID,
		N:             len(sessions),
		MeanNetUSD:    mean,
		StdErrUSD:     stdErr,
		CI95LowUSD:    mean - half,
		CI95HighUSD:   mean + half,
		Sessions:      deltas,
		PricingSource: pricingSource,
		Caveat:        compactionABCaveat(len(sessions)),
	}
	return row, nil
}

// meanStdErr returns the sample mean and the standard error of the mean (s/√n, with s the
// Bessel-corrected sample standard deviation). n<2 has no spread to estimate, so the standard
// error is 0 — the caller's caveat names the resulting degenerate interval.
func meanStdErr(xs []float64) (mean, stdErr float64) {
	n := float64(len(xs))
	if n == 0 {
		return 0, 0
	}
	var sum float64
	for _, x := range xs {
		sum += x
	}
	mean = sum / n
	if len(xs) < 2 {
		return mean, 0
	}
	var ss float64
	for _, x := range xs {
		d := x - mean
		ss += d * d
	}
	variance := ss / (n - 1)
	return mean, math.Sqrt(variance / n)
}

func compactionABCaveat(n int) string {
	if n < 2 {
		return "single matched session: the 95% interval is degenerate (zero width); it reports the point estimate only, not measured spread."
	}
	return "95% CI uses the normal approximation (mean ± z*SE); for this matched-session count a Student-t interval is wider, so read a near-zero interval as a lower bound on its width."
}
