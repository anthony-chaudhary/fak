package slackoutbox

import (
	"fmt"
	"sort"
	"time"
)

// Call-volume accounting (Slack rate-limit observability): the durable state log already
// records, per row, whether it POSTED (a chat.postMessage / chat.update that reached the
// wire), was SUPERSEDED (an edit coalesced into a newer one — a call the outbox avoided), or
// was REFUSED (a body the leak fence blocked — also a call avoided). CallStats folds those
// transitions into a per-source picture of how much Slack API budget each surface spends and
// how much the outbox already saves — WITHOUT a live API read and without a second metrics
// store. It is the measuring stick behind `fak slack outbox calls`: run it, change a
// producer's cadence, run it again, and the reduction is a number, not a vibe.
//
// Caveat, stated rather than hidden: the counts cover only the RETAINED log. Compaction
// folds settled rows out on their retention windows (superseded rows age out fastest), so a
// long-lived spool undercounts historical saves. LastCompactAgeS reports that window floor.

// SourceCalls is one producing surface's Slack API-call footprint.
type SourceCalls struct {
	Source    string `json:"source"`
	Posts     int    `json:"posts"`     // chat.postMessage calls that reached the wire
	Updates   int    `json:"updates"`   // chat.update calls that reached the wire
	Coalesced int    `json:"coalesced"` // edits collapsed into a newer one (calls the outbox saved)
	Refused   int    `json:"refused"`   // bodies the leak fence blocked (calls the outbox saved)
	Dead      int    `json:"dead"`      // rows dead-lettered after exhausting retries
	Pending   int    `json:"pending"`   // rows still owed a delivery (no call spent yet)
}

// Sent is the API calls this surface actually spent against the rate limit.
func (s SourceCalls) Sent() int { return s.Posts + s.Updates }

// Saved is the API calls the outbox avoided for this surface (coalesced duplicate edits plus
// fence refusals) — the noise it already suppresses.
func (s SourceCalls) Saved() int { return s.Coalesced + s.Refused }

// CallStats is the whole spool's call-volume fold: per source, sorted loudest-first, plus
// totals and the retention-window floor the counts cover.
type CallStats struct {
	Sources    []SourceCalls `json:"sources"`
	TotalSent  int           `json:"total_sent"`  // posts + updates across all sources
	TotalSaved int           `json:"total_saved"` // coalesced + refused across all sources
	// LastCompactAgeS is how long ago the log was last compacted (-1 = never). Settled rows
	// older than their retention window are gone, so this is the floor of the window covered.
	LastCompactAgeS int64 `json:"last_compact_age_s"`
}

// CallStats derives the per-source Slack API-call footprint from the durable log at `now`.
func (o *Outbox) CallStats(now time.Time) (*CallStats, error) {
	snap, err := o.Load()
	if err != nil {
		return nil, err
	}
	bySource := map[string]*SourceCalls{}
	get := func(src string) *SourceCalls {
		if src == "" {
			src = "unknown"
		}
		sc := bySource[src]
		if sc == nil {
			sc = &SourceCalls{Source: src}
			bySource[src] = sc
		}
		return sc
	}
	for _, r := range snap.Rows {
		sc := get(r.Source)
		switch snap.state(r.Nonce).State {
		case statePosted:
			// A posted UPDATE row was a chat.update; a posted POST row was a chat.postMessage.
			if r.UpdateTS != "" {
				sc.Updates++
			} else {
				sc.Posts++
			}
		case stateSuperseded:
			sc.Coalesced++
		case stateRefused:
			sc.Refused++
		case stateDead:
			sc.Dead++
		default: // pending / sending / failed — owed, no call spent
			sc.Pending++
		}
	}
	cs := &CallStats{LastCompactAgeS: -1}
	for _, sc := range bySource {
		cs.Sources = append(cs.Sources, *sc)
		cs.TotalSent += sc.Sent()
		cs.TotalSaved += sc.Saved()
	}
	sort.Slice(cs.Sources, func(i, j int) bool {
		if a, b := cs.Sources[i].Sent(), cs.Sources[j].Sent(); a != b {
			return a > b
		}
		return cs.Sources[i].Source < cs.Sources[j].Source
	})
	if !snap.LastCompactAt.IsZero() {
		cs.LastCompactAgeS = int64(now.Sub(snap.LastCompactAt) / time.Second)
	}
	return cs, nil
}

// CallStatsReportLine renders the one-line summary (the CLI's non-JSON headline).
func CallStatsReportLine(cs *CallStats) string {
	window := "all retained"
	if cs.LastCompactAgeS >= 0 {
		window = "since last compaction " + (time.Duration(cs.LastCompactAgeS) * time.Second).String() + " ago"
	}
	return fmt.Sprintf("sent %d call(s) (chat.postMessage+chat.update), saved %d (coalesced+refused) across %d source(s) — %s",
		cs.TotalSent, cs.TotalSaved, len(cs.Sources), window)
}
