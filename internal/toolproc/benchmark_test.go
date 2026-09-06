package toolproc

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
	"time"
)

var (
	benchTableSink     Table
	benchEventsSink    []Event
	benchProcsSink     []Proc
	benchNormalSink    NormalCall
	benchReportSink    RepeatReport
	benchReceiptSink   Receipt
	benchProcessStatus ProcessStatus
)

// makeBenchEvents constructs a realistic, validated journal of procCount procs.
func makeBenchEvents(procCount int) []Event {
	const base int64 = 1_700_000_000_000
	events := make([]Event, 0, procCount*3)
	for i := 0; i < procCount; i++ {
		callID := fmt.Sprintf("call-%04d", i)
		sessID := fmt.Sprintf("sess-%02d", i%10)
		start := base + int64(i*50)

		events = append(events, Event{
			Kind:             EvSpawn,
			CallID:           callID,
			Session:          sessID,
			Tool:             "bench_tool",
			AtMS:             start,
			DeadlineMS:       60_000,
			HeartbeatEveryMS: 5_000,
		})
		events = append(events, Event{
			Kind:   EvPulse,
			CallID: callID,
			AtMS:   start + 2_000,
		})

		if i%4 == 0 {
			// Leave running (some will stall / go overdue)
		} else if i%10 == 9 {
			events = append(events, Event{
				Kind:   EvKill,
				CallID: callID,
				AtMS:   start + 4_000,
				Reason: ReasonToolDeadlineExceededName,
			})
		} else {
			events = append(events, Event{
				Kind:   EvExit,
				CallID: callID,
				AtMS:   start + 4_000,
				Status: "ok",
			})
		}
	}
	return events
}

// makeRepeatsBenchRecords constructs a mixed stream of 100 tool call observations.
func makeRepeatsBenchRecords() []CallRecord {
	const base int64 = 1_700_000_000_000
	records := make([]CallRecord, 0, 100)
	for i := 0; i < 30; i++ {
		records = append(records, CallRecord{
			Tool:        "shell",
			Raw:         "git status --short --branch",
			AtMS:        base + int64(i*5000),
			OutputBytes: 380,
		})
	}
	for i := 0; i < 20; i++ {
		records = append(records, CallRecord{
			Tool:        "shell",
			Raw:         "cat internal/toolproc/toolproc.go",
			AtMS:        base + int64(i*6000),
			OutputBytes: 24500,
			Digest:      "sha256:feedbeef00112233",
		})
	}
	for i := 0; i < 20; i++ {
		records = append(records, CallRecord{
			Tool:        "shell",
			Raw:         "Get-Content -Raw internal/toolproc/toolproc.go",
			AtMS:        base + int64(i*6000+100),
			OutputBytes: 24500,
			Digest:      "sha256:feedbeef00112233",
		})
	}
	for i := 0; i < 10; i++ {
		records = append(records, CallRecord{
			Tool:        "shell",
			Raw:         "git push origin main",
			AtMS:        base + int64(i*15000),
			OutputBytes: 120,
		})
	}
	for i := 0; i < 20; i++ {
		records = append(records, CallRecord{
			Tool:        "shell",
			Raw:         fmt.Sprintf("curl -H 'Authorization: Bearer sk-ant-secret-%03d' https://api.example.com/v1", i),
			AtMS:        base + int64(i*3000),
			OutputBytes: 1024,
		})
	}
	return records
}

// BenchmarkFold_Sample benchmarks folding the built-in sample journal
// exercising every lifecycle verdict class.
func BenchmarkFold_Sample(b *testing.B) {
	events, nowMS, cfg := Sample()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tab, err := Fold(events, nowMS, cfg)
		if err != nil {
			b.Fatalf("Fold: %v", err)
		}
		benchTableSink = tab
	}
}

// BenchmarkFold_Scaling benchmarks folding scaled journals of 100 and 1000 events.
func BenchmarkFold_Scaling(b *testing.B) {
	for _, count := range []int{100, 1000} {
		b.Run(fmt.Sprintf("%d_Procs", count), func(b *testing.B) {
			events := makeBenchEvents(count)
			nowMS := int64(1_700_000_000_000) + int64(count*50) + 10_000
			cfg := Config{DefaultDeadlineMS: 60_000, StallMultiplier: DefaultStallMultiplier}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				tab, err := Fold(events, nowMS, cfg)
				if err != nil {
					b.Fatalf("Fold: %v", err)
				}
				benchTableSink = tab
			}
		})
	}
}

// BenchmarkTable_Subtree benchmarks walking descendant hierarchy trees.
func BenchmarkTable_Subtree(b *testing.B) {
	var procs []Proc
	rootID := "proc-root"
	procs = append(procs, Proc{CallID: rootID, State: StateRunning, StartMS: 1000})

	idCounter := 1
	currentLevel := []string{rootID}
	for depth := 0; depth < 3; depth++ {
		var nextLevel []string
		for _, parent := range currentLevel {
			for branch := 0; branch < 3; branch++ {
				childID := fmt.Sprintf("proc-%d", idCounter)
				idCounter++
				procs = append(procs, Proc{
					CallID:       childID,
					ParentCallID: parent,
					State:        StateRunning,
					StartMS:      int64(1000 + idCounter),
				})
				nextLevel = append(nextLevel, childID)
			}
		}
		currentLevel = nextLevel
	}
	tab := Table{Procs: procs}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sub := tab.Subtree(rootID)
		if len(sub) != len(procs)-1 {
			b.Fatalf("got %d descendants, want %d", len(sub), len(procs)-1)
		}
		benchProcsSink = sub
	}
}

// BenchmarkParseEvents benchmarks JSONL event deserialization and boundary validation.
func BenchmarkParseEvents(b *testing.B) {
	events := makeBenchEvents(100)
	var buf bytes.Buffer
	for _, ev := range events {
		data, err := json.Marshal(ev)
		if err != nil {
			b.Fatalf("marshal: %v", err)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}
	raw := buf.Bytes()

	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(raw)
		parsed, err := ParseEvents(r)
		if err != nil {
			b.Fatalf("ParseEvents: %v", err)
		}
		benchEventsSink = parsed
	}
}

// BenchmarkParseTail benchmarks reading and record-boundary alignment of journal tails.
func BenchmarkParseTail(b *testing.B) {
	events := makeBenchEvents(500)
	var buf bytes.Buffer
	for _, ev := range events {
		data, err := json.Marshal(ev)
		if err != nil {
			b.Fatalf("marshal: %v", err)
		}
		buf.Write(data)
		buf.WriteByte('\n')
	}
	raw := buf.Bytes()
	const tailWindow = 4096

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := bytes.NewReader(raw)
		parsed, err := ParseTail(r, tailWindow)
		if err != nil {
			b.Fatalf("ParseTail: %v", err)
		}
		benchEventsSink = parsed
	}
}

// BenchmarkCompactJournal benchmarks journal compaction keeping live calls and recent tail window.
func BenchmarkCompactJournal(b *testing.B) {
	events := makeBenchEvents(500)
	const tailKeep = 50

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		compacted := CompactJournal(events, tailKeep)
		benchEventsSink = compacted
	}
}

// BenchmarkNormalize benchmarks redaction, path canonicalization, and mutability classification.
func BenchmarkNormalize(b *testing.B) {
	cfg := RepeatConfig{DefaultFreshnessMS: DefaultFreshnessWindowMS}

	cases := []struct {
		name string
		rec  CallRecord
	}{
		{
			name: "ImmutableRead",
			rec: CallRecord{
				Tool:        "shell",
				Raw:         "cat internal/toolproc/toolproc.go",
				AtMS:        1700000000000,
				OutputBytes: 24500,
			},
		},
		{
			name: "MutableQuery",
			rec: CallRecord{
				Tool:        "shell",
				Raw:         "git status --short --branch",
				AtMS:        1700000000000,
				OutputBytes: 420,
			},
		},
		{
			name: "IdempotentWrite",
			rec: CallRecord{
				Tool:        "shell",
				Raw:         "git push origin main",
				AtMS:        1700000000000,
				OutputBytes: 150,
			},
		},
		{
			name: "Redaction",
			rec: CallRecord{
				Tool:        "shell",
				Raw:         "curl -H 'Authorization: Bearer sk-ant-api03-abcdef123456789' https://api.anthropic.com/v1",
				AtMS:        1700000000000,
				OutputBytes: 1024,
			},
		},
		{
			name: "Unknown",
			rec: CallRecord{
				Tool:        "custom",
				Raw:         "custom_daemon --mode=aggressive --workers=4",
				AtMS:        1700000000000,
				OutputBytes: 80,
			},
		},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchNormalSink = Normalize(tc.rec, cfg)
			}
		})
	}
}

// BenchmarkClassifyRepeats benchmarks folding repeated call observations into a RepeatReport.
func BenchmarkClassifyRepeats(b *testing.B) {
	records := makeRepeatsBenchRecords()
	cfg := RepeatConfig{
		PollingMinRepeats:         DefaultPollingMinRepeats,
		PollingMaxMedianSpacingMS: DefaultPollingMaxMedianSpacingMS,
		DefaultFreshnessMS:        DefaultFreshnessWindowMS,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rep := ClassifyRepeats(records, cfg)
		benchReportSink = rep
	}
}

// BenchmarkReuseStore_Admit benchmarks cache hit decisions, freshness windows, and invalidation.
func BenchmarkReuseStore_Admit(b *testing.B) {
	cfg := RepeatConfig{DefaultFreshnessMS: DefaultFreshnessWindowMS}

	b.Run("ImmutableHit", func(b *testing.B) {
		store := NewReuseStore(cfg)
		rec := CallRecord{
			Tool:        "shell",
			Raw:         "cat internal/toolproc/toolproc.go",
			AtMS:        1700000000000,
			Digest:      "sha256:feedbeef00112233",
			OutputBytes: 4096,
		}
		// Seed the initial cache entry
		_ = store.Admit(rec)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rec.AtMS = 1700000000000 + int64(i)
			benchReceiptSink = store.Admit(rec)
		}
	})

	b.Run("FreshnessHit", func(b *testing.B) {
		store := NewReuseStore(cfg)
		rec := CallRecord{
			Tool:        "shell",
			Raw:         "git status --short --branch",
			AtMS:        1700000000000,
			OutputBytes: 500,
		}
		// Seed the query state
		_ = store.Admit(rec)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			// Query within freshness window (500ms < 2000ms)
			rec.AtMS = 1700000000500
			benchReceiptSink = store.Admit(rec)
		}
	})

	b.Run("MutationInvalidation", func(b *testing.B) {
		store := NewReuseStore(cfg)
		recA := CallRecord{
			Tool:        "shell",
			Raw:         "cat internal/toolproc/toolproc.go",
			AtMS:        1700000000000,
			Digest:      "sha256:1111111111111111",
			OutputBytes: 4096,
		}
		recB := CallRecord{
			Tool:        "shell",
			Raw:         "cat internal/toolproc/toolproc.go",
			AtMS:        1700000000000,
			Digest:      "sha256:2222222222222222",
			OutputBytes: 4096,
		}

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if i%2 == 0 {
				recA.AtMS = 1700000000000 + int64(i)
				benchReceiptSink = store.Admit(recA)
			} else {
				recB.AtMS = 1700000000000 + int64(i)
				benchReceiptSink = store.Admit(recB)
			}
		}
	})
}

// BenchmarkRenderRepeatReport benchmarks formatting repeat analysis for operator presentation.
func BenchmarkRenderRepeatReport(b *testing.B) {
	records := makeRepeatsBenchRecords()
	rep := ClassifyRepeats(records, RepeatConfig{})

	var buf bytes.Buffer
	buf.Grow(4096)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		RenderRepeatReport(&buf, rep, 10)
	}
}

// BenchmarkProcessSupervisor benchmarks process tracking, status polling, and tombstone lookups.
func BenchmarkProcessSupervisor(b *testing.B) {
	b.Run("PollActive", func(b *testing.B) {
		sup := NewProcessSupervisor()
		_ = sup.RegisterProcess(42, "worker --sync")

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st, err := sup.PollProcess(42)
			if err != nil {
				b.Fatalf("PollProcess: %v", err)
			}
			benchProcessStatus = st
		}
	})

	b.Run("PollTombstoned", func(b *testing.B) {
		sup := NewProcessSupervisor(WithPollLivelockThreshold(1_000_000_000))
		sup.RegisterProcess(99, "worker --sync")
		sup.RecordExit(99, 0, "ok", "", 500*time.Millisecond)

		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			st, _ := sup.PollProcess(99)
			benchProcessStatus = st
		}
	})
}
