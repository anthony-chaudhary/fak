package sessionaudit

import (
	"encoding/json"
	"os"
	"time"
)

// feed.go makes a session-audit run DURABLE. Every other nightrun signal — cache
// savings, gateway usage, memory value — appends a scrubbed row per sample to a
// docs/nightrun/*.jsonl ledger, so its health is a time series you can trend and gate.
// session-audit was the exception: it only ever emitted a one-off report, so "were the
// last few hours of sessions healthy?" had no durable answer and no regression witness.
//
// FoldFeedRow folds the existing CompactReport (the same aggregate the summary/actions
// surfaces already render) into ONE scrubbed FeedRow, and AppendFeedRow appends it as a
// single JSON line — the same fold-window-then-append-one-row shape `fak cachevalue feed`
// uses. The row carries only aggregate counters and UUID/namespace identifiers already in
// the report (no prose, no transcript bytes), so the ledger is safe to keep in-tree.

// DefaultFeedLedgerRel is the durable session-audit ledger, a sibling of the other
// docs/nightrun/*.jsonl fleet signals. One line per FoldFeedRow snapshot.
const DefaultFeedLedgerRel = "docs/nightrun/session-audit.jsonl"

// FeedSchema versions the durable row shape so a reader can pin the contract.
const FeedSchema = "fak.sessionaudit.feed.v1"

// FeedRow is one durable, scrubbed session-audit snapshot over a discovery window: the
// scope, the token/cost totals, the highest-cost model tier, the worst long-context
// session, the recommendation severity counts, and the cross-session process/reasoning
// friction folds. It is a strict aggregate of CompactReport — no per-session prose.
type FeedRow struct {
	Schema             string  `json:"schema"`
	TS                 string  `json:"ts"`
	WindowDays         float64 `json:"window_days,omitempty"`
	NamespaceScope     string  `json:"ns_scope"` // "" = all non-excluded namespaces
	SessionsAudited    int64   `json:"sessions_audited"`
	SessionsDiscovered int     `json:"sessions_discovered"`
	Clipped            bool    `json:"clipped,omitempty"`

	OutputTokens       int64   `json:"output_tokens"`
	CacheReadTokens    int64   `json:"cache_read_tokens"`
	TotalContextTokens int64   `json:"total_context_tokens"`
	CacheReadShare     float64 `json:"cache_read_share"`
	IORatio            float64 `json:"io_ratio"`
	EstCostUSD         float64 `json:"est_cost_usd"`

	TopTier          string  `json:"top_tier,omitempty"`
	TopTierCostShare float64 `json:"top_tier_cost_share,omitempty"`

	LongContextMax     int64  `json:"long_context_max,omitempty"`
	LongContextSession string `json:"long_context_session,omitempty"`

	RecHigh   int `json:"rec_high"`
	RecMedium int `json:"rec_medium"`
	RecLow    int `json:"rec_low"`

	StuckSessions           int64 `json:"stuck_sessions,omitempty"`
	TimeoutKills            int64 `json:"timeout_kills,omitempty"`
	RecurringFailureClasses int   `json:"recurring_failure_classes,omitempty"`
	ConfusedSessions        int64 `json:"confused_sessions,omitempty"`
	SilentConfusedSessions  int64 `json:"silent_confused_sessions,omitempty"`
}

// FoldFeedRow folds a CompactReport into one durable FeedRow stamped at now. Pure — no
// I/O, no clock read — so a caller passes the clock and a test asserts the row exactly.
func FoldFeedRow(rep CompactReport, now time.Time) FeedRow {
	row := FeedRow{
		Schema:             FeedSchema,
		TS:                 now.UTC().Format("2006-01-02T15:04:05Z"),
		NamespaceScope:     rep.Scope.NamespaceFilter,
		SessionsAudited:    rep.Scope.Audited,
		SessionsDiscovered: rep.Scope.Discovered,
		Clipped:            rep.Scope.Clipped,
		OutputTokens:       rep.Totals.OutputTokens,
		CacheReadTokens:    rep.Totals.CacheReadTokens,
		TotalContextTokens: rep.Totals.TotalContextTokens,
		CacheReadShare:     rep.Totals.CacheReadShare,
		IORatio:            rep.Totals.IORatio,
		EstCostUSD:         rep.Totals.EstimatedCostUSD,
	}
	if rep.Scope.SinceDays != nil {
		row.WindowDays = *rep.Scope.SinceDays
	}
	// The highest-cost model tier — the one to challenge before launching more of it.
	for _, t := range rep.Tiers {
		if t.EstimatedCostUSD > 0 && t.CostShare >= row.TopTierCostShare {
			row.TopTier = t.Tier
			row.TopTierCostShare = t.CostShare
		}
	}
	// TopLongContext is sorted worst-first; row[0] is the long-context pressure headline.
	if len(rep.TopLongContext) > 0 {
		row.LongContextMax = rep.TopLongContext[0].TotalContextTokens
		row.LongContextSession = rep.TopLongContext[0].Session
	}
	for _, r := range rep.Recommendations {
		switch r.Severity {
		case "high":
			row.RecHigh++
		case "medium":
			row.RecMedium++
		default:
			row.RecLow++
		}
	}
	if b := rep.Behavior; b != nil {
		row.StuckSessions = b.StuckSessions
		row.TimeoutKills = b.TimeoutKills
		row.RecurringFailureClasses = len(b.RecurringFailures)
	}
	if c := rep.Confusion; c != nil {
		row.ConfusedSessions = c.ConfusedSessions
		row.SilentConfusedSessions = c.SilentConfusedSessions
	}
	return row
}

// AppendFeedRow appends row as one JSON line to the ledger at path (created if absent).
// json.Encoder.Encode writes the marshaled row plus its '\n' terminator in a single
// Write, so the O_APPEND write is atomic — the single-line-append convention the sibling
// docs/nightrun ledgers use, under which concurrent fleet writers never interleave a
// partial row. HTML escaping is off so a session id or namespace bearing '<'/'&' stays
// byte-faithful and the ledger stays human-diffable.
func AppendFeedRow(path string, row FeedRow) error {
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetEscapeHTML(false)
	return enc.Encode(row)
}
