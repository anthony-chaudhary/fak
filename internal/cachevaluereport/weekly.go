package cachevaluereport

import (
	"fmt"
	"time"

	"github.com/anthony-chaudhary/fak/internal/cachevalueledger"
	"github.com/anthony-chaudhary/fak/internal/gatewayusageledger"
)

// WeeklyDigestSchema versions the weekly cache-health digest envelope (#3646) so a
// downstream reader can pin it, mirroring Schema for the Track-1 roll-up.
const WeeklyDigestSchema = "fak-cache-health-weekly-digest/1"

// adoptionEpsilon is the dead-band below which a week-over-week posture-adoption
// change reads flat rather than improved/regressed — the adoption sibling of
// reuseEpsilon, wider because adoption is a ratio over a handful of sessions and a
// single session flipping posture on a thin fleet should not manufacture a trend.
const adoptionEpsilon = 0.02

// WeeklyWindow is one 7-day window's fold of the fleet cache-health signals the
// weekly digest (#3646) reports: posture adoption, realized reuse, shed
// effectiveness, and the refused-upgrade rate. Every number is folded from EXISTING
// durable counters (the Track-1 cache-value ledger and the gateway-usage ledger) —
// the digest authors no new metric. Ratio fields are pointers so "no evidence this
// window" (nil) is never conflated with a measured zero.
type WeeklyWindow struct {
	Start string `json:"start"` // window open (YYYY-MM-DD, UTC, inclusive)
	End   string `json:"end"`   // window close (YYYY-MM-DD, UTC, exclusive)

	// Posture adoption (WITNESSED): exit sessions whose managed-cache TTL-upgrade
	// lever left durable evidence — an actual 1h upgrade or a refusal-reason row.
	// Both counters are zero/absent while --managed-cache is off, so their presence
	// is the honest "posture was active" witness (see gatewayusageledger.Counters).
	ExitSessions          int      `json:"exit_sessions"`
	PostureActiveSessions int      `json:"posture_active_sessions"`
	PostureAdoptionPct    *float64 `json:"posture_adoption_pct,omitempty"` // 0..100; nil when no exit sessions

	// Realized reuse (WITNESSED, Track-1 ledger, multi-turn sessions only — a
	// single-turn cold run has no previous turn to reuse from, exactly as
	// ScoreLedger excludes it).
	MultiTurnSessions int      `json:"multi_turn_sessions"`
	MultiTurnTurns    uint64   `json:"multi_turn_turns"`
	GatePromptTokens  uint64   `json:"gate_prompt_tokens"`
	GateReusedTokens  uint64   `json:"gate_reused_tokens"`
	ReuseRatio        *float64 `json:"reuse_ratio,omitempty"` // 0..1; nil when no multi-turn prompt tokens
	ReuseThin         bool     `json:"reuse_thin"`            // MultiTurnTurns < MinBucketTurns

	// Shed effectiveness (WITNESSED compaction counters): how often the shed fired
	// vs bailed, and how many resident tokens it actually removed.
	CompactionFired   uint64   `json:"compaction_fired"`
	CompactionBailed  uint64   `json:"compaction_bailed"`
	ShedTokens        uint64   `json:"shed_tokens"`
	DroppedTurns      uint64   `json:"dropped_turns"`
	ShedTokensPerFire *float64 `json:"shed_tokens_per_fire,omitempty"` // nil when nothing fired

	// Refused-upgrade rate (WITNESSED TTL-upgrade outcomes): refusal-reason rows vs
	// actual upgrades, over heads the lever even considered.
	TTLUpgrades       uint64   `json:"ttl_upgrades"`
	TTLRefusals       uint64   `json:"ttl_refusals"`
	RefusedUpgradePct *float64 `json:"refused_upgrade_pct,omitempty"` // 0..100; nil when the lever saw no heads

	// Observed provider reuse (OBSERVED, provider-relayed cache_read — NOT a fak claim,
	// and deliberately kept APART from the WITNESSED realized-reuse above so the two are
	// never summed or conflated). The share of prompt tokens the provider served from ITS
	// OWN prefix cache across every exit session — the honest "is prompt caching working
	// end to end" signal even on a wire where fak's own managed levers are passive (e.g. a
	// subscription-OAuth seat, whose 1h upgrade the provider rejects: see
	// docs/cache-frontier/2026-07-18-subscription-oauth-400s-1h-ttl-upgrade-MEASURED.md).
	// It is CONTEXT, not a scored cache-health family — a low WITNESSED score with a high
	// observed reuse means "the provider cache is carrying the caching, fak's own levers are
	// passive", which reads very differently from cold caching across the board.
	ObsCacheReadTokens     uint64   `json:"obs_cache_read_tokens"`
	ObsInputTokens         uint64   `json:"obs_input_tokens"`
	ObsCacheCreationTokens uint64   `json:"obs_cache_creation_tokens"`
	ObservedProviderReuse  *float64 `json:"observed_provider_reuse,omitempty"` // 0..1; nil when no observed prompt tokens
}

// WeeklyDigest is the weekly fleet cache-health digest (#3646) — the OPERATIONAL
// complement to the daily Track-1/$ card: not "what did the cache save" but "is the
// cache machinery actually working across the fleet this week". ThisWeek covers the
// 7 days ending at the fold's `now`; PriorWeek the 7 days before that, so every
// headline carries a week-over-week direction.
type WeeklyDigest struct {
	Schema      string `json:"schema"`
	GeneratedAt string `json:"generated_at"`
	WindowDays  int    `json:"window_days"` // 7

	ThisWeek  WeeklyWindow `json:"this_week"`
	PriorWeek WeeklyWindow `json:"prior_week"`

	// Week-over-week directions for the two headline ratios. Shed and refusals are
	// reported as raw this-vs-prior numbers on the card instead of a Trend — a
	// higher shed volume is not unambiguously better or worse.
	AdoptionTrend    Trend   `json:"adoption_trend"`
	DeltaAdoptionPct float64 `json:"delta_adoption_pct"` // this - prior, percentage points
	ReuseTrend       Trend   `json:"reuse_trend"`
	DeltaReuseRatio  float64 `json:"delta_reuse_ratio"` // this - prior, 0..1

	// #1066 fence self-labels, carried verbatim so a reader can never mistake the
	// realized reuse for the forbidden vs-naive multiple.
	PublishableValueFamily  string `json:"publishable_value_family"`
	VsNaiveMultipleExcluded bool   `json:"vs_naive_multiple_excluded"`

	OK         bool   `json:"ok"`
	Verdict    string `json:"verdict"` // MEASURED | INSUFFICIENT
	Finding    string `json:"finding"`
	NextAction string `json:"next_action"`
}

// windowAccumulate folds one usage exit row into a window's posture/shed/refusal
// totals. Only "exit" rows count: a periodic row is a prefix snapshot of the same
// session and would double-count it, and a carryforward row is a synthetic sum of a
// folded pre-cut era, not a session.
func (w *WeeklyWindow) accumulateUsage(r gatewayusageledger.Row) {
	w.ExitSessions++
	c := r.Counters
	if c.CacheTTLUpgradesUpgraded > 0 || len(c.CacheTTLUpgradeReasons) > 0 {
		w.PostureActiveSessions++
	}
	w.CompactionFired += c.CompactionFired
	w.CompactionBailed += c.CompactionBailed
	w.ShedTokens += c.CompactionShedTokens
	w.DroppedTurns += c.CompactionDroppedTurns
	w.TTLUpgrades += c.CacheTTLUpgradesUpgraded
	for _, n := range c.CacheTTLUpgradeReasons {
		w.TTLRefusals += n
	}
	// OBSERVED provider cache-read (relayed by the provider, never a fak claim): fold the
	// per-session prompt-token split so finalize can derive the provider-reuse share.
	w.ObsCacheReadTokens += c.CachedPromptTokens
	w.ObsInputTokens += c.InputTokens
	w.ObsCacheCreationTokens += c.CacheCreationTokens
}

// accumulateTrack1 folds one multi-turn Track-1 ledger row into the window's
// realized-reuse totals, mirroring ScoreLedger's multi-turn gate.
func (w *WeeklyWindow) accumulateTrack1(r cachevalueledger.Row) {
	w.MultiTurnSessions++
	w.MultiTurnTurns += r.Turns
	w.GatePromptTokens += r.PromptTokens
	w.GateReusedTokens += r.ReusedTokens
}

// finalize computes the window's derived ratios from its totals.
func (w *WeeklyWindow) finalize() {
	if w.ExitSessions > 0 {
		pct := 100 * float64(w.PostureActiveSessions) / float64(w.ExitSessions)
		w.PostureAdoptionPct = &pct
	}
	if w.GatePromptTokens > 0 {
		ratio := float64(w.GateReusedTokens) / float64(w.GatePromptTokens)
		w.ReuseRatio = &ratio
	}
	w.ReuseThin = w.MultiTurnTurns < MinBucketTurns
	if w.CompactionFired > 0 {
		per := float64(w.ShedTokens) / float64(w.CompactionFired)
		w.ShedTokensPerFire = &per
	}
	if heads := w.TTLUpgrades + w.TTLRefusals; heads > 0 {
		pct := 100 * float64(w.TTLRefusals) / float64(heads)
		w.RefusedUpgradePct = &pct
	}
	// OBSERVED provider reuse = provider cache_read / all prompt tokens processed (read +
	// fresh input + cache_creation write). nil when the fleet moved no prompt tokens.
	if obs := w.ObsCacheReadTokens + w.ObsInputTokens + w.ObsCacheCreationTokens; obs > 0 {
		ratio := float64(w.ObsCacheReadTokens) / float64(obs)
		w.ObservedProviderReuse = &ratio
	}
}

// trendOf maps a this-minus-prior delta onto the Trend vocabulary with the given
// dead-band. A window with no prior evidence reads TrendNew, never a manufactured
// improvement over an empty week.
func trendOf(this, prior *float64, epsilon float64) (Trend, float64) {
	if this == nil || prior == nil {
		return TrendNew, 0
	}
	delta := *this - *prior
	switch {
	case delta > epsilon:
		return TrendImproved, delta
	case delta < -epsilon:
		return TrendRegressed, delta
	}
	return TrendFlat, delta
}

// FoldWeeklyDigest folds the Track-1 cache-value ledger and the gateway-usage ledger
// into the weekly fleet cache-health digest (#3646). It is PURE and deterministic:
// rows + a caller-supplied `now` in, a digest out — no clock, no I/O, no network —
// exactly like Fold. Window membership comes from each row's own UnixMillis: this
// week is the 7 days ending at `now` (exclusive of anything after `now`), the prior
// week the 7 days before that; older rows are ignored. Missing or empty ledgers fold
// to the honest INSUFFICIENT digest rather than failing.
func FoldWeeklyDigest(track1 []cachevalueledger.Row, usage []gatewayusageledger.Row, now time.Time) WeeklyDigest {
	nowUTC := now.UTC()
	thisStart := nowUTC.AddDate(0, 0, -7)
	priorStart := nowUTC.AddDate(0, 0, -14)

	d := WeeklyDigest{
		Schema:                  WeeklyDigestSchema,
		GeneratedAt:             nowUTC.Format(time.RFC3339),
		WindowDays:              7,
		PublishableValueFamily:  PublishableValueFamily,
		VsNaiveMultipleExcluded: true,
		OK:                      true,
		Verdict:                 "INSUFFICIENT",
	}
	d.ThisWeek.Start, d.ThisWeek.End = thisStart.Format("2006-01-02"), nowUTC.Format("2006-01-02")
	d.PriorWeek.Start, d.PriorWeek.End = priorStart.Format("2006-01-02"), thisStart.Format("2006-01-02")

	window := func(ms int64) *WeeklyWindow {
		t := time.UnixMilli(ms).UTC()
		switch {
		case t.Before(priorStart) || t.After(nowUTC):
			return nil
		case t.Before(thisStart):
			return &d.PriorWeek
		}
		return &d.ThisWeek
	}

	for _, r := range usage {
		// Exit rows only: a periodic row is a prefix snapshot of the same session
		// (double count), a carryforward row a synthetic pre-cut sum (not a session).
		if r.Kind != "exit" {
			continue
		}
		if w := window(r.UnixMillis); w != nil {
			w.accumulateUsage(r)
		}
	}
	for _, r := range track1 {
		if r.Turns < 2 {
			continue // single-turn cold run: no previous turn to reuse from
		}
		if w := window(r.UnixMillis); w != nil {
			w.accumulateTrack1(r)
		}
	}
	d.ThisWeek.finalize()
	d.PriorWeek.finalize()

	d.AdoptionTrend, d.DeltaAdoptionPct = trendOf(d.ThisWeek.PostureAdoptionPct, d.PriorWeek.PostureAdoptionPct, 100*adoptionEpsilon)
	d.ReuseTrend, d.DeltaReuseRatio = trendOf(d.ThisWeek.ReuseRatio, d.PriorWeek.ReuseRatio, reuseEpsilon)

	if d.ThisWeek.ExitSessions == 0 && d.ThisWeek.MultiTurnSessions == 0 {
		d.Finding = fmt.Sprintf("INSUFFICIENT — no exit sessions and no multi-turn Track-1 rows in the 7 days ending %s; nothing to digest", d.ThisWeek.End)
		d.NextAction = "keep accumulating: fleet sessions append durable rows on exit; re-check next week"
		return d
	}
	d.Verdict = "MEASURED"
	adoption := "adoption n/a"
	if d.ThisWeek.PostureAdoptionPct != nil {
		adoption = fmt.Sprintf("posture-active %d/%d sessions (%.0f%%)", d.ThisWeek.PostureActiveSessions, d.ThisWeek.ExitSessions, *d.ThisWeek.PostureAdoptionPct)
	}
	reuse := "reuse n/a"
	if d.ThisWeek.ReuseRatio != nil {
		reuse = fmt.Sprintf("reuse %.1f%%", 100**d.ThisWeek.ReuseRatio)
	}
	d.Finding = fmt.Sprintf("MEASURED — %s; %s; shed %d tok over %d fire(s); refused-upgrade %s",
		adoption, reuse, d.ThisWeek.ShedTokens, d.ThisWeek.CompactionFired, pctOrNA(d.ThisWeek.RefusedUpgradePct))
	d.NextAction = "read the card; a regressed adoption or reuse week is the cue to check managed-cache defaults and the shed gate"
	return d
}

// pctOrNA renders an optional percentage for the finding line.
func pctOrNA(p *float64) string {
	if p == nil {
		return "n/a"
	}
	return fmt.Sprintf("%.0f%%", *p)
}
