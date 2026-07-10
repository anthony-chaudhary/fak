package ablate

// Live anchor-strategy A/B arm (#2809). Sibling to compaction_ab.go: where that arm splits
// compaction ON vs OFF, this one holds compaction ON and splits the ANCHOR — where the
// protected (verbatim-copied) prefix ends — reported in the same NET dollars, never gross shed.
// The two anchor strategies are the agent.CompactAnchor choices (#1407):
//
//   - FirstBP (agent.CompactAnchorFirstBP) is the warm-cache-safe default: the protected prefix
//     runs THROUGH the first messages[] cache_control breakpoint, so only the middle after it is
//     compactible. On real Claude Code traffic the sole message breakpoint is RECENT, so this
//     anchors near the end and the lever stays IDLE — the #1407 dormancy. Idle keeps the full
//     resident prefix, most of it served WARM as provider cache_read (0.1x).
//   - Head (agent.CompactAnchorHead) re-anchors on the stable provider head, making the WHOLE
//     message array compactible — this is what lets compaction actually FIRE on real traffic. A
//     fire sheds the middle but BURSTS the recent message breakpoint's cached suffix as fresh
//     cache_creation (1.25x/2.0x), which is why the real firing path gates each fire on
//     agent.CacheBurstPaysBack economics (#1408).
//
// So head-anchoring can COST money: it drops already-warm reads to write a fresh prefix, exactly
// the tension a gross-shed number hides. This arm prices BOTH sides with gateway.CachePricing.CostUSD
// — the SAME engine, already booking the cache_creation burst at its write multiplier (#2795),
// that compaction_ab.go and the report side use — so its net-of-burst delta cannot disagree with
// the compaction arm's by construction. The only honest proof that head-anchoring earns its burst
// on real traffic is a causal FirstBP/Head delta, netted for the burst; this wire turns that split
// into a measured dollar delta with a confidence interval.
//
// Generation posture (gen/next, #2783): a near-term foundation seam, same as compaction_ab.go. It
// is a deterministic ($0, no model, no GPU) valuation over caller-supplied matched-session token
// axes — it does not itself run the anchor split or decide CacheBurstPaysBack. Promotion evidence =
// a live guard capture feeding real matched FirstBP/Head sessions whose sweep-row CI clears zero.
// Invalidating assumptions are named on AnchorABRow (normal-approximation CI; caller-guaranteed
// session matching). The CI math (zScore95, meanStdErr) and the interval caveat (compactionABCaveat)
// are shared with compaction_ab.go rather than re-derived here, so the two arms report identically.

import (
	"errors"
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/gateway"
)

// AnchorArm names the two arms of the live anchor split, for labeling.
const (
	AnchorArmFirstBP = "anchor_firstbp"
	AnchorArmHead    = "anchor_head"
	// AnchorABArmID is the sweep-row arm id for the Head-vs-FirstBP net-of-burst delta. Head is
	// named first — it is the treatment (the lever that fires on real traffic), FirstBP the
	// dormant control — so a positive delta reads as "head-anchoring saved this much vs the
	// idle default", mirroring compaction_on_vs_off (treatment named first, positive = it wins).
	AnchorABArmID = "anchor_head_vs_firstbp"
)

// MatchedAnchorSession is ONE real guard session measured under BOTH anchor strategies on the
// same underlying traffic, so their dollar difference isolates the anchor choice:
//
//   - FirstBP is the token accounting under agent.CompactAnchorFirstBP: the lever stays idle on
//     recent-breakpoint traffic, so the full resident prefix is carried forward (mostly warm
//     CacheReadTokens), no burst.
//   - Head is the SAME session under agent.CompactAnchorHead: compaction fires and sheds the
//     middle, but pays a one-time cache_creation BURST (Head.CacheCreationTokens) re-priming the
//     shed suffix.
//
// Each side is a session-cumulative gateway.CacheUsage aggregate; cache pricing is linear, so one
// aggregate row reproduces the sum of the session's per-turn rows (the same convention
// compaction_ab.go's MatchedCompactionSession relies on). Matching the two sides to comparable
// traffic is the CALLER's contract — a mismatch invalidates the delta, not this arithmetic (see
// AnchorABRow.Caveat).
type MatchedAnchorSession struct {
	SessionID string
	FirstBP   gateway.CacheUsage
	Head      gateway.CacheUsage
}

// NetOfBurstDeltaUSD is the dollars head-anchoring saved on THIS session, NET of the burst it
// caused: CostUSD(FirstBP) − CostUSD(Head). CostUSD(Head) already prices the cache_creation burst
// at its TTL write multiplier (the #2795 burst premium), so the result is net-of-burst by
// construction, never gross shed. Positive = head-anchoring paid for its burst and then some
// (the #1408 CacheBurstPaysBack call was right to fire); negative = the burst outweighed the shed
// (dropping already-warm reads to write a fresh prefix cost more than it saved — the case the gate
// exists to refuse).
func (m MatchedAnchorSession) NetOfBurstDeltaUSD(p gateway.CachePricing) float64 {
	return p.CostUSD(m.FirstBP) - p.CostUSD(m.Head)
}

// AnchorSessionDelta is one matched session's net-of-burst dollar delta, retained on the row so a
// reader can see the per-session spread the confidence interval summarizes, not just the mean. Its
// per-arm cost fields are named for the anchors (firstbp/head), NOT on/off, so the report stays
// legible about which lever each column priced.
type AnchorSessionDelta struct {
	SessionID      string  `json:"session_id"`
	NetUSD         float64 `json:"net_usd"`
	FirstBPCostUSD float64 `json:"firstbp_cost_usd"`
	HeadCostUSD    float64 `json:"head_cost_usd"`
}

// AnchorABRow is the ONE sweep row the live anchor Head/FirstBP arm reports: the mean per-session
// net-of-burst dollar delta across N matched sessions, with a 95% confidence interval so a reader
// sees whether the delta clears zero, plus the per-session breakdown. Positive MeanNetUSD =
// head-anchoring saved money net of burst on average; a CI straddling zero means the evidence does
// not yet distinguish head-anchoring from the idle first-breakpoint default on this traffic.
type AnchorABRow struct {
	ArmID         string               `json:"arm_id"`
	N             int                  `json:"n"`
	MeanNetUSD    float64              `json:"mean_net_usd"`
	StdErrUSD     float64              `json:"std_err_usd"`
	CI95LowUSD    float64              `json:"ci95_low_usd"`
	CI95HighUSD   float64              `json:"ci95_high_usd"`
	Sessions      []AnchorSessionDelta `json:"sessions"`
	PricingSource string               `json:"pricing_source,omitempty"`
	Caveat        string               `json:"caveat,omitempty"`
}

// SweepRow renders the human one-liner the acceptance calls for: the net delta with its confidence
// interval and the matched-session count. This is the "sweep row [that] shows the net delta with a
// confidence interval" (#2809).
func (r AnchorABRow) SweepRow() string {
	return fmt.Sprintf("%s: net $%+.4f/session [95%% CI $%+.4f, $%+.4f] over N=%d matched sessions",
		r.ArmID, r.MeanNetUSD, r.CI95LowUSD, r.CI95HighUSD, r.N)
}

// ClearsZero reports whether the whole 95% interval sits on one side of zero — the honest read of
// "is this delta real": true only when the CI does NOT straddle $0. A row whose interval brackets
// zero has not yet earned a directional claim.
func (r AnchorABRow) ClearsZero() bool {
	return r.CI95LowUSD > 0 || r.CI95HighUSD < 0
}

// AnchorABSweep folds MATCHED FirstBP/Head sessions into the one sweep row: it prices each
// session's net-of-burst delta with CachePricing (the shared burst-premium engine), takes the mean
// over sessions, and brackets it with a 95% confidence interval. It fails closed on an empty
// session set — there is no row to report and a fabricated zero would read as a measured no-op —
// and on a session that is degenerate on both arms (no billable tokens either way), which cannot be
// a matched real session.
//
// The interval uses the sample standard deviation (Bessel-corrected, n−1) and the normal
// approximation mean ± z*SE via the shared meanStdErr/zScore95. With a single matched session the
// interval is degenerate (zero width); the row carries the shared caveat naming that and the
// small-N t-vs-z gap so a reader never mistakes a thin interval for a strong one.
func AnchorABSweep(sessions []MatchedAnchorSession, p gateway.CachePricing, pricingSource string) (AnchorABRow, error) {
	if len(sessions) == 0 {
		return AnchorABRow{}, errors.New("ablate: anchor A/B needs at least one matched session")
	}
	deltas := make([]AnchorSessionDelta, 0, len(sessions))
	raw := make([]float64, 0, len(sessions))
	for i, s := range sessions {
		if s.FirstBP == (gateway.CacheUsage{}) && s.Head == (gateway.CacheUsage{}) {
			return AnchorABRow{}, fmt.Errorf("ablate: matched session %d (%q) has no token activity on either arm", i, s.SessionID)
		}
		firstBP := p.CostUSD(s.FirstBP)
		head := p.CostUSD(s.Head)
		deltas = append(deltas, AnchorSessionDelta{
			SessionID:      s.SessionID,
			NetUSD:         firstBP - head,
			FirstBPCostUSD: firstBP,
			HeadCostUSD:    head,
		})
		raw = append(raw, firstBP-head)
	}

	mean, stdErr := meanStdErr(raw)
	half := zScore95 * stdErr
	row := AnchorABRow{
		ArmID:         AnchorABArmID,
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
