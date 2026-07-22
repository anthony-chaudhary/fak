package slackoutbox

import (
	"fmt"
	"time"
)

// Limits is the outbox's effective retention + compaction envelope, plus where the live
// spool currently sits against it. It answers, without running a pass, the operator's
// question behind `fak slack outbox limits`: what does compaction keep, what does it drop,
// and is a pass due right now? The retention windows and thresholds are the package
// defaults (there is no persisted per-outbox override today); the occupancy fields are
// folded from the live snapshot at the observed clock. Durations are seconds, matching the
// `_s` age fields Status already reports.
type Limits struct {
	RetainSettledS      int64 `json:"retain_settled_s"`       // superseded/refused window
	RetainPostedS       int64 `json:"retain_posted_s"`        // posted window (PostedTS-probe safety)
	RetainDeadS         int64 `json:"retain_dead_s"`          // dead window (operator-retryable)
	RetainCardsS        int64 `json:"retain_cards_s"`         // finalized run-card window
	CompactRowThreshold int   `json:"compact_row_threshold"`  // terminal backlog that forces a pass
	CompactMinIntervalS int64 `json:"compact_min_interval_s"` // min gap between automatic passes

	// Live occupancy, folded at the observed clock.
	TerminalRows  int  `json:"terminal_rows"`  // folded rows in a terminal state
	DroppableRows int  `json:"droppable_rows"` // terminal rows already past their window
	CompactionDue bool `json:"compaction_due"` // an automatic pass would run on the next drain
}

// Limits reports the effective retention/compaction envelope and where the live spool sits
// against it. It reads only: it folds the snapshot and reuses the same keepRow/compactionDue
// predicates a real pass would, so its droppable count and pass-due bit cannot drift from
// what `compact` actually does.
func (o *Outbox) Limits() (*Limits, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	opts := CompactOpts{Now: o.now()}.norm(o)
	lim := &Limits{
		RetainSettledS:      int64(opts.RetainSettled / time.Second),
		RetainPostedS:       int64(opts.RetainPosted / time.Second),
		RetainDeadS:         int64(opts.RetainDead / time.Second),
		RetainCardsS:        int64(opts.RetainCards / time.Second),
		CompactRowThreshold: CompactRowThreshold,
		CompactMinIntervalS: int64(CompactMinInterval / time.Second),
	}
	for _, r := range snap.Rows {
		s := snap.state(r.Nonce)
		if !s.terminal() {
			continue
		}
		lim.TerminalRows++
		if keep, _ := keepRow(r, s, opts); !keep {
			lim.DroppableRows++
		}
	}
	lim.CompactionDue = o.compactionDue(snap, opts)
	return lim, nil
}

// LimitsReportLine renders the effective limits and live occupancy as one human line (the
// CLI's non-JSON output).
func LimitsReportLine(l *Limits) string {
	due := "no"
	if l.CompactionDue {
		due = "YES"
	}
	return fmt.Sprintf(
		"retain settled %s  posted %s  dead %s  cards %s  | compact when >%d terminal or every %s  | live: %d terminal, %d droppable, pass due %s",
		durS(l.RetainSettledS), durS(l.RetainPostedS), durS(l.RetainDeadS), durS(l.RetainCardsS),
		l.CompactRowThreshold, durS(l.CompactMinIntervalS),
		l.TerminalRows, l.DroppableRows, due)
}

// durS renders a seconds count as a compact duration ("48h0m0s", "336h0m0s").
func durS(s int64) string { return (time.Duration(s) * time.Second).String() }
