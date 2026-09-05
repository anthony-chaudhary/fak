package knownbad

import (
	"fmt"
	"strings"
	"testing"
)

var (
	benchRecordSink   Record
	benchRecordsSink  []Record
	benchStatsSink    CompactStats
	benchCoalesceSink CoalesceStats
	benchStringSink   string
	benchBoolSink     bool
	benchStateSink    string
)

func makeBenchRecords(count int, nowUnix int64) []Record {
	recs := make([]Record, 0, count)
	sigCount := count / 3
	if sigCount < 1 {
		sigCount = 1
	}
	for i := 0; i < sigCount; i++ {
		tree := fmt.Sprintf("internal/pkg%d/**", i)
		reason := "BUILD_FAILURE"
		if i%2 == 1 {
			reason = "TEST_TIMEOUT"
		}
		hash := ""
		if i%3 == 0 {
			hash = fmt.Sprintf("sha256:%064x", i)
		}
		discovered := nowUnix - int64(100*(sigCount-i))
		ttl := int64(3600)
		if i%5 == 0 {
			ttl = 50 // some expired relative to nowUnix
		}

		open := NewRecord(reason, []string{tree, fmt.Sprintf("internal/pkg%d/sub/**", i)}, "issue", fmt.Sprintf("agent-%d", i), hash, discovered, ttl)
		recs = append(recs, open)
		if len(recs) >= count {
			break
		}

		// Revision 2: claim
		claimed := open.WithClaim(fmt.Sprintf("fixer-%d", i), discovered+20)
		recs = append(recs, claimed)
		if len(recs) >= count {
			break
		}

		// Revision 3: resolve or revoke for some
		if i%4 == 0 {
			resolved := claimed.WithResolve(fmt.Sprintf("fixer-%d", i), discovered+40, "tests")
			recs = append(recs, resolved)
		} else if i%4 == 1 {
			revoked := claimed.WithRevoke("operator", discovered+40, "not a bug")
			recs = append(recs, revoked)
		}
		if len(recs) >= count {
			break
		}
	}
	return recs
}

func makeBenchJSONL(records []Record) []byte {
	var sb strings.Builder
	for _, r := range records {
		line, _ := MarshalLine(r)
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

func BenchmarkSignature(b *testing.B) {
	globs := []string{"internal/gateway/**", "internal/engine/*", "internal/adjudicator/policy.go"}
	reason := "LIVELOCK_DETECTED"
	hash := "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

	b.Run("SingleGlob", func(b *testing.B) {
		single := []string{"internal/gateway/**"}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Signature(reason, single, "")
		}
	})

	b.Run("MultipleGlobsWithHash", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Signature(reason, globs, hash)
		}
	})
}

func BenchmarkNormalizeTree(b *testing.B) {
	paths := []string{
		"internal/gateway/**",
		`internal\engine\sub\file.go`,
		"internal/adjudicator/policy/*/**",
		"internal/pkg/../pkg/sub",
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = NormalizeTree(paths[i%len(paths)])
	}
}

func BenchmarkTreesIntersect(b *testing.B) {
	a := []string{"internal/gateway/**", "internal/adjudicator/**"}
	bMatch := []string{"internal/gateway/server.go"}
	bDisjoint := []string{"cmd/fak/main.go"}

	b.Run("Match", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = TreesIntersect(a, bMatch)
		}
	})

	b.Run("Disjoint", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchBoolSink = TreesIntersect(a, bDisjoint)
		}
	})
}

func BenchmarkMatch(b *testing.B) {
	const now = 1_700_000_000
	query := Query{TreeGlobs: []string{"internal/pkg5/sub/file.go"}}

	for _, size := range []int{10, 100, 500} {
		records := makeBenchRecords(size, now)
		b.Run(fmt.Sprintf("LedgerSize_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchRecordsSink = Match(records, query, now)
			}
		})
	}
}

func BenchmarkLiveRecords(b *testing.B) {
	const now = 1_700_000_000
	for _, size := range []int{50, 200, 500} {
		records := makeBenchRecords(size, now)
		b.Run(fmt.Sprintf("LedgerSize_%d", size), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchRecordsSink = LiveRecords(records, now)
			}
		})
	}
}

func BenchmarkFindLatestLive(b *testing.B) {
	const now = 1_700_000_000
	records := makeBenchRecords(100, now)
	targetSig := records[len(records)-1].Signature

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec, ok := FindLatestLive(records, targetSig, now)
		benchRecordSink = rec
		benchBoolSink = ok
	}
}

func BenchmarkLatestState(b *testing.B) {
	const now = 1_700_000_000
	records := makeBenchRecords(100, now)
	targetSig := records[len(records)-1].Signature

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec, seen, state := LatestState(records, targetSig, now)
		benchRecordSink = rec
		benchBoolSink = seen
		benchStateSink = state
	}
}

func BenchmarkCompact(b *testing.B) {
	const now = 1_700_000_000
	records := makeBenchRecords(250, now)

	b.Run("KeepTerminalTail50", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kept, stats := Compact(records, now, 50)
			benchRecordsSink = kept
			benchStatsSink = stats
		}
	})

	b.Run("KeepLiveOnly", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			kept, stats := Compact(records, now, 0)
			benchRecordsSink = kept
			benchStatsSink = stats
		}
	})
}

func BenchmarkParseLedger(b *testing.B) {
	const now = 1_700_000_000
	records := makeBenchRecords(100, now)
	data := makeBenchJSONL(records)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchRecordsSink = ParseLedger(data)
	}
}

func BenchmarkMarshalLine(b *testing.B) {
	rec := NewRecord("SIGNAL_CRASH", []string{"internal/gateway/**", "internal/engine/**"}, "critical failure", "agent-1", "sha256:abcdef", 1_700_000_000, 3600)
	rec = rec.WithClaim("fixer-42", 1_700_000_100).WithOccurrences(15, 1_700_000_200)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		line, err := MarshalLine(rec)
		if err != nil {
			b.Fatal(err)
		}
		benchStringSink = line
	}
}

func BenchmarkCoalesceCrashes(b *testing.B) {
	const now = 1_700_000_000
	const ttl = 900

	events100 := crashEvents(100, "SIGNAL_CRASH", []string{"internal/pkg/**"}, "sha256:111", now)
	priorRecord := NewRecord("SIGNAL_CRASH", []string{"internal/pkg/**"}, "", "agent", "sha256:111", now-100, ttl).WithOccurrences(50, now-50)
	priorLedger := []Record{priorRecord}

	b.Run("OpenFreshWindow_100Events", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows, stats := CoalesceCrashes(nil, events100, now, ttl)
			benchRecordsSink = rows
			benchCoalesceSink = stats
		}
	})

	b.Run("RefreshLiveWindow_100Events", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows, stats := CoalesceCrashes(priorLedger, events100, now, ttl)
			benchRecordsSink = rows
			benchCoalesceSink = stats
		}
	})

	// Multi-cause storm: 5 causes, 100 events each = 500 events
	events500 := make([]CrashEvent, 0, 500)
	for cause := 0; cause < 5; cause++ {
		class := fmt.Sprintf("CRASH_CLASS_%d", cause)
		tree := []string{fmt.Sprintf("internal/module%d/**", cause)}
		hash := fmt.Sprintf("sha256:hash%d", cause)
		events500 = append(events500, crashEvents(100, class, tree, hash, now)...)
	}

	b.Run("MultiCauseStorm_500Events", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows, stats := CoalesceCrashes(nil, events500, now, ttl)
			benchRecordsSink = rows
			benchCoalesceSink = stats
		}
	})
}

func BenchmarkLeaseID(b *testing.B) {
	sig := "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = LeaseID(sig)
	}
}
