package gatewayusageledger

import "sort"

// RewriteEvent is an inspectable increase in cache-creation usage after a
// session's initial cache fill. Ledger counters are cumulative, so the event
// values are deltas from the preceding observation of the same session.
type RewriteEvent struct {
	RowKey         string  `json:"row_key"`
	UnixMillis     int64   `json:"unix_millis"`
	CreatedTokens  uint64  `json:"created_tokens"`
	WastedWriteUSD float64 `json:"wasted_write_usd"`
}

// RewriteTally aggregates avoidable cold rewrites for one session.
type RewriteTally struct {
	SessionID         string         `json:"session_id"`
	EventCount        int            `json:"event_count"`
	WastedWriteTokens uint64         `json:"wasted_write_tokens"`
	WastedWriteUSD    float64        `json:"wasted_write_usd"`
	Events            []RewriteEvent `json:"events"`
}

// RewriteReport ranks sessions worst-first by wasted cache-write dollars.
type RewriteReport struct {
	Schema   string         `json:"schema"`
	Sessions []RewriteTally `json:"sessions"`
}

// RankPrefixRewrites finds cache-creation increases after each session's initial
// cache fill. writeUSDPerMillion is the provider's cache-write price.
// Input may be in any order. Counter resets start a new baseline rather than
// fabricating a negative or oversized event.
func RankPrefixRewrites(rows []Row, writeUSDPerMillion float64) RewriteReport {
	bySession := make(map[string][]Row)
	for _, row := range rows {
		if row.SessionID == "" {
			continue
		}
		bySession[row.SessionID] = append(bySession[row.SessionID], row)
	}

	report := RewriteReport{Schema: "fak-cache-break-report/1"}
	for sessionID, sessionRows := range bySession {
		sort.SliceStable(sessionRows, func(i, j int) bool {
			if sessionRows[i].UnixMillis != sessionRows[j].UnixMillis {
				return sessionRows[i].UnixMillis < sessionRows[j].UnixMillis
			}
			return sessionRows[i].RowKey < sessionRows[j].RowKey
		})

		var previousTokens uint64
		haveBaseline := false
		ranked := RewriteTally{SessionID: sessionID}
		for _, row := range sessionRows {
			tokens := row.Counters.CacheCreationTokens
			if !haveBaseline || tokens < previousTokens {
				previousTokens, haveBaseline = tokens, true
				continue
			}
			deltaTokens := tokens - previousTokens
			previousTokens = tokens
			if deltaTokens == 0 {
				continue
			}
			deltaUSD := float64(deltaTokens) * writeUSDPerMillion / 1_000_000
			ranked.EventCount++
			ranked.WastedWriteTokens += deltaTokens
			ranked.WastedWriteUSD += deltaUSD
			ranked.Events = append(ranked.Events, RewriteEvent{
				RowKey: row.RowKey, UnixMillis: row.UnixMillis,
				CreatedTokens: deltaTokens, WastedWriteUSD: deltaUSD,
			})
		}
		if ranked.EventCount > 0 {
			report.Sessions = append(report.Sessions, ranked)
		}
	}

	sort.Slice(report.Sessions, func(i, j int) bool {
		if report.Sessions[i].WastedWriteUSD != report.Sessions[j].WastedWriteUSD {
			return report.Sessions[i].WastedWriteUSD > report.Sessions[j].WastedWriteUSD
		}
		if report.Sessions[i].WastedWriteTokens != report.Sessions[j].WastedWriteTokens {
			return report.Sessions[i].WastedWriteTokens > report.Sessions[j].WastedWriteTokens
		}
		return report.Sessions[i].SessionID < report.Sessions[j].SessionID
	})
	return report
}
