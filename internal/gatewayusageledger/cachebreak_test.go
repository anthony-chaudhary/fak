package gatewayusageledger

import "testing"

func TestRankPrefixRewritesRanksSyntheticSpikesWorstFirst(t *testing.T) {
	row := func(session, rowKey string, at int64, tokens uint64) Row {
		return Row{SessionID: session, RowKey: rowKey, UnixMillis: at, Counters: Counters{CacheCreationTokens: tokens}}
	}
	rows := []Row{
		row("small", "small-2", 20, 130), row("worst", "worst-3", 30, 900),
		row("worst", "worst-1", 10, 100), row("small", "small-1", 10, 100),
		row("worst", "worst-2", 20, 400), row("steady", "steady-1", 10, 200),
		row("steady", "steady-2", 20, 200),
	}

	got := RankPrefixRewrites(rows, 10_000) // $0.01/token keeps assertions readable.
	if got.Schema != "fak-cache-break-report/1" {
		t.Fatalf("schema = %q", got.Schema)
	}
	if len(got.Sessions) != 2 {
		t.Fatalf("sessions = %#v, want two sessions with post-fill writes", got.Sessions)
	}
	if got.Sessions[0].SessionID != "worst" || got.Sessions[0].WastedWriteTokens != 800 || got.Sessions[0].WastedWriteUSD != 8 || got.Sessions[0].EventCount != 2 {
		t.Fatalf("top session = %#v", got.Sessions[0])
	}
	if len(got.Sessions[0].Events) != 2 || got.Sessions[0].Events[0].RowKey != "worst-2" || got.Sessions[0].Events[1].RowKey != "worst-3" {
		t.Fatalf("top events not inspectable/ordered: %#v", got.Sessions[0].Events)
	}
	if got.Sessions[1].SessionID != "small" || got.Sessions[1].WastedWriteUSD != .30 {
		t.Fatalf("second session = %#v", got.Sessions[1])
	}
}

func TestRankPrefixRewritesTreatsCounterResetAsNewBaseline(t *testing.T) {
	var rows []Row
	for _, v := range []struct {
		id     string
		at     int64
		tokens uint64
	}{
		{"a", 10, 100}, {"b", 20, 150}, {"reset", 30, 10}, {"c", 40, 30},
	} {
		rows = append(rows, Row{SessionID: "session", RowKey: v.id, UnixMillis: v.at, Counters: Counters{CacheCreationTokens: v.tokens}})
	}
	got := RankPrefixRewrites(rows, 10_000)
	if len(got.Sessions) != 1 || got.Sessions[0].WastedWriteTokens != 70 || got.Sessions[0].WastedWriteUSD != .7 {
		t.Fatalf("reset report = %#v", got)
	}
}
