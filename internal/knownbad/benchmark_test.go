// Package knownbad benchmarks measure failure signature hashing, tree path
// normalization, intersection queries, ledger compaction, and crash storm
// coalescing for persistent failure tracking.
//
// Query matching executes during lane arbitration and worker preflight,
// requiring bounded allocation overhead and sub-millisecond query evaluation
// across hundreds of active failure records.
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

// makeBenchRecords synthesizes count failure records across mixed causes, file trees,
// and lifecycle states (open, claimed, resolved, revoked).
// Operating envelope: count synthetic failure records across 1 to count/3 distinct failure signatures.
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

// makeBenchJSONL serializes a slice of records into line-delimited JSON bytes for parser benchmarking.
// Operating envelope: slice of structured Record objects.
func makeBenchJSONL(records []Record) []byte {
	var sb strings.Builder
	for _, r := range records {
		line, _ := MarshalLine(r)
		sb.WriteString(line)
		sb.WriteByte('\n')
	}
	return []byte(sb.String())
}

// BenchmarkSignature measures signature calculation for failure classification and dedup.
// Operating envelope: single tree glob and multi-glob lists with optional SHA-256 commit hash.
// Allocation budget: <= 1 alloc/op and <= 64 B/op for single glob; <= 4 allocs/op and <= 256 B/op for multi-glob.
// Latency ceiling: P50 < 200ns, P99 < 1µs.
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

// BenchmarkNormalizeTree measures tree path sanitization and wildcard normalization.
// Operating envelope: cyclic evaluation over 4 varied path patterns (slashes, backslashes, relative dots).
// Allocation budget: <= 2 allocs/op and <= 64 B/op.
// Latency ceiling: P50 < 100ns, P99 < 500ns.
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

// BenchmarkTreesIntersect measures disjointness checking between two tree glob sets.
// Operating envelope: 2-pattern candidate tree evaluated against matching and disjoint target sets.
// Allocation budget: 0 allocs/op on the intersection check.
// Latency ceiling: P50 < 50ns, P99 < 250ns.
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

// BenchmarkMatch evaluates query matching across 10, 100, and 500 historical failure records.
// Operating envelope: 10 to 500 records evaluated against a specific file path query.
// Allocation budget: <= 10 allocs/op and <= 2 KB/op at 100 records.
// Latency ceiling: P50 < 5µs, P99 < 25µs at 100 records with O(N) scaling.
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

// BenchmarkLiveRecords measures filtering of active, unexpired, and unresolved records.
// Operating envelope: 50 to 500 records evaluated against current wall-clock timestamp.
// Allocation budget: <= 5 allocs/op and <= 4 KB/op at 200 records.
// Latency ceiling: P50 < 8µs, P99 < 40µs at 200 records.
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

// BenchmarkFindLatestLive measures reverse chronological search for the newest live record of a signature.
// Operating envelope: 100-record ledger searching for target signature.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 1µs, P99 < 5µs.
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

// BenchmarkLatestState measures signature state resolution (open, claimed, resolved, revoked).
// Operating envelope: 100-record ledger resolving state for target signature.
// Allocation budget: 0 allocs/op.
// Latency ceiling: P50 < 1µs, P99 < 5µs.
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

// BenchmarkCompact measures ledger pruning retaining active records and terminal tail history.
// Operating envelope: 250-record ledger evaluated with tail caps of 50 and 0 (live-only).
// Allocation budget: <= 10 allocs/op and <= 8 KB/op.
// Latency ceiling: P50 < 15µs, P99 < 75µs.
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

// BenchmarkParseLedger evaluates JSONL parsing of serialized failure records.
// Operating envelope: 100 JSONL records (~15 KB payload).
// Allocation budget: linear with record count (~6 allocs/row).
// Latency ceiling: P50 < 50µs, P99 < 250µs for 100 records.
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

// BenchmarkMarshalLine measures JSON serialization of a single failure record.
// Operating envelope: fully-populated Record with claim and occurrence history.
// Allocation budget: <= 5 allocs/op and <= 512 B/op.
// Latency ceiling: P50 < 500ns, P99 < 2.5µs.
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

// BenchmarkCoalesceCrashes measures crash event coalescing across single-window and multi-cause storm topologies.
// Operating envelope: 100 to 500 crash events evaluated against empty and existing live ledgers.
// Allocation budget: <= 20 allocs/op and <= 8 KB/op at 100 events.
// Latency ceiling: P50 < 30µs, P99 < 150µs at 100 events.
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

// BenchmarkLeaseID measures deterministic lease identifier generation from a failure signature.
// Operating envelope: SHA-256 signature string formatted into an authoritative lease ID.
// Allocation budget: <= 2 allocs/op and <= 64 B/op.
// Latency ceiling: P50 < 80ns, P99 < 400ns.
func BenchmarkLeaseID(b *testing.B) {
	sig := "sha256:9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = LeaseID(sig)
	}
}
