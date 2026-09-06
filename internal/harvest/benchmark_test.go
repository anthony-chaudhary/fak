package harvest

import (
	"fmt"
	"strconv"
	"sync/atomic"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// makeBenchCorpus returns a populated corpus with n rows matching a realistic
// distribution of allows, denies, hard negatives, and quarantines.
func makeBenchCorpus(n int) *Corpus {
	c := NewCorpus()
	c.SetMaxRows(n)
	reasons := []abi.ReasonCode{
		abi.ReasonPolicyBlock,
		abi.ReasonTrustViolation,
		abi.ReasonMalformed,
		abi.ReasonUnwitnessed,
	}
	for i := 0; i < n; i++ {
		hash := "tool@" + strconv.Itoa(i)
		switch {
		case i%20 == 0:
			c.add(abi.LabelRow{
				CallHash:   hash,
				RungPassed: 1,
				RungFailed: 3,
				Verdict:    abi.VerdictDeny,
				Reason:     reasons[i%len(reasons)],
			})
		case i%20 == 1:
			c.add(abi.LabelRow{
				CallHash:   hash,
				RungPassed: -1,
				RungFailed: -1,
				Verdict:    abi.VerdictQuarantine,
				Reason:     abi.ReasonTrustViolation,
			})
		case i%5 == 0:
			c.add(abi.LabelRow{
				CallHash:   hash,
				RungPassed: -1,
				RungFailed: -1,
				Verdict:    abi.VerdictDeny,
				Reason:     reasons[i%len(reasons)],
			})
		default:
			c.add(abi.LabelRow{
				CallHash:   hash,
				RungPassed: -1,
				RungFailed: -1,
				Verdict:    abi.VerdictAllow,
				Reason:     0,
			})
		}
	}
	return c
}

// BenchmarkHarvesterEmit measures event intake throughput and allocations across
// production event paths handled by Harvester.Emit.
func BenchmarkHarvesterEmit(b *testing.B) {
	b.Run("ExplicitLabel", func(b *testing.B) {
		h := New(NewCorpus())
		label := abi.LabelRow{
			CallHash:   "send_email@sha256:deadbeef",
			RungPassed: 1,
			RungFailed: 3,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonPolicyBlock,
		}
		ev := abi.Event{
			Kind:  abi.EvRungLabel,
			Label: &label,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("DerivedDeny_WithDigest", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "send_email",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:0123456789abcdef"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonTrustViolation,
			By:     "ifc-sink",
		}
		ev := abi.Event{
			Kind:    abi.EvDeny,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("DerivedDeny_InlineArgs", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "send_email",
			Args: abi.Ref{
				Kind:   abi.RefInline,
				Inline: []byte(`{"to":"victim@corp.internal","subject":"exfil","body":"secret_payload"}`),
			},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonPolicyBlock,
			By:     "policy-gate",
		}
		ev := abi.Event{
			Kind:    abi.EvDeny,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("DerivedAllow_EvDecide", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "search_kb",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:cafebabedeadbeef"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictAllow,
			Reason: 0,
			By:     "default-allow",
		}
		ev := abi.Event{
			Kind:    abi.EvDecide,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("RedundantEvent_Skipped", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "send_email",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:feedface"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonPolicyBlock,
			By:     "policy-gate",
		}
		ev := abi.Event{
			Kind:    abi.EvDecide,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("ResultDeny", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "fetch_webpage",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:beef1234"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonUnwitnessed,
			By:     "result-admit",
		}
		ev := abi.Event{
			Kind:    abi.EvResultDeny,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})

	b.Run("Quarantine", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "exec_bash",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:c001d00d"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictQuarantine,
			Reason: abi.ReasonTrustViolation,
			By:     "runtime-sentinel",
		}
		ev := abi.Event{
			Kind:    abi.EvQuarantine,
			Call:    call,
			Verdict: verdict,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			h.Emit(ev)
		}
	})
}

// BenchmarkHarvesterEmit_Parallel tests concurrent event ingestion across multiple
// goroutines, simulating parallel kernel adjudications folding into a shared corpus.
func BenchmarkHarvesterEmit_Parallel(b *testing.B) {
	b.Run("ExplicitLabel", func(b *testing.B) {
		h := New(NewCorpus())
		label := abi.LabelRow{
			CallHash:   "bench_tool@sha256:1234",
			RungPassed: 0,
			RungFailed: 2,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonPolicyBlock,
		}
		ev := abi.Event{Kind: abi.EvRungLabel, Label: &label}

		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				h.Emit(ev)
			}
		})
	})

	b.Run("DerivedDeny", func(b *testing.B) {
		h := New(NewCorpus())
		call := &abi.ToolCall{
			Tool: "curl",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:parallel-test"},
		}
		verdict := &abi.Verdict{
			Kind:   abi.VerdictDeny,
			Reason: abi.ReasonTrustViolation,
			By:     "parallel-sink",
		}
		ev := abi.Event{Kind: abi.EvDeny, Call: call, Verdict: verdict}

		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				h.Emit(ev)
			}
		})
	})

	b.Run("MixedStream", func(b *testing.B) {
		h := New(NewCorpus())
		events := []abi.Event{
			{
				Kind:    abi.EvDecide,
				Call:    &abi.ToolCall{Tool: "read", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:r1"}},
				Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"},
			},
			{
				Kind:    abi.EvDecide,
				Call:    &abi.ToolCall{Tool: "write", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:w1"}},
				Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
			},
			{
				Kind:    abi.EvDeny,
				Call:    &abi.ToolCall{Tool: "write", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:w1"}},
				Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
			},
			{
				Kind: abi.EvRungLabel,
				Label: &abi.LabelRow{
					CallHash:   "exec@sha:e1",
					RungPassed: 1,
					RungFailed: 2,
					Verdict:    abi.VerdictDeny,
					Reason:     abi.ReasonMalformed,
				},
			},
		}

		var idx uint64
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				i := atomic.AddUint64(&idx, 1)
				h.Emit(events[i%uint64(len(events))])
			}
		})
	})
}

// BenchmarkCorpusAdd measures row insertion and window eviction performance in Corpus.add.
func BenchmarkCorpusAdd(b *testing.B) {
	b.Run("BoundedDefault_1024", func(b *testing.B) {
		c := NewCorpus()
		for i := 0; i < defaultMaxCorpusRows; i++ {
			c.add(abi.LabelRow{CallHash: "prefill", Verdict: abi.VerdictAllow})
		}
		row := abi.LabelRow{
			CallHash:   "steady_state",
			RungPassed: -1,
			RungFailed: -1,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonPolicyBlock,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			c.add(row)
		}
	})

	b.Run("BoundedSmall_64", func(b *testing.B) {
		c := NewCorpus()
		c.SetMaxRows(64)
		for i := 0; i < 64; i++ {
			c.add(abi.LabelRow{CallHash: "prefill", Verdict: abi.VerdictAllow})
		}
		row := abi.LabelRow{
			CallHash:   "small_cap",
			RungPassed: 1,
			RungFailed: 3,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonTrustViolation,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			c.add(row)
		}
	})

	b.Run("Unbounded", func(b *testing.B) {
		c := NewCorpus()
		c.SetMaxRows(-1)
		row := abi.LabelRow{
			CallHash:   "unbounded",
			RungPassed: -1,
			RungFailed: -1,
			Verdict:    abi.VerdictAllow,
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			c.add(row)
		}
	})

	b.Run("Parallel_Bounded1024", func(b *testing.B) {
		c := NewCorpus()
		row := abi.LabelRow{
			CallHash:   "parallel_row",
			RungPassed: -1,
			RungFailed: -1,
			Verdict:    abi.VerdictDeny,
			Reason:     abi.ReasonPolicyBlock,
		}
		b.ReportAllocs()
		b.ResetTimer()
		b.RunParallel(func(pb *testing.PB) {
			for pb.Next() {
				c.add(row)
			}
		})
	})
}

// BenchmarkCorpusQueries measures query operations on a fully populated (1024-row) corpus.
func BenchmarkCorpusQueries(b *testing.B) {
	c := makeBenchCorpus(defaultMaxCorpusRows)

	b.Run("Rows_Snapshot", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			rows := c.Rows()
			if len(rows) != defaultMaxCorpusRows {
				b.Fatalf("expected %d rows, got %d", defaultMaxCorpusRows, len(rows))
			}
		}
	})

	b.Run("Positives_Filter", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			pos := c.Positives()
			if len(pos) == 0 {
				b.Fatal("expected positive rows")
			}
		}
	})

	b.Run("HardNegatives_Filter", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			hn := c.HardNegatives()
			if len(hn) == 0 {
				b.Fatal("expected hard negative rows")
			}
		}
	})

	b.Run("ByReason_Tally", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			m := c.ByReason()
			if len(m) == 0 {
				b.Fatal("expected reason entries")
			}
		}
	})

	b.Run("Len", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			n := c.Len()
			if n != defaultMaxCorpusRows {
				b.Fatalf("expected len %d, got %d", defaultMaxCorpusRows, n)
			}
		}
	})
}

// BenchmarkCorpusConcurrentReadWrite measures read/write contention when collectors
// append events while analysis consumers periodically query the corpus.
func BenchmarkCorpusConcurrentReadWrite(b *testing.B) {
	c := makeBenchCorpus(defaultMaxCorpusRows)
	var opCount uint64

	row := abi.LabelRow{
		CallHash:   "concurrent_rw",
		RungPassed: 1,
		RungFailed: 2,
		Verdict:    abi.VerdictDeny,
		Reason:     abi.ReasonPolicyBlock,
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			n := atomic.AddUint64(&opCount, 1)
			switch n % 10 {
			case 0:
				_ = c.Positives()
			case 1:
				_ = c.ByReason()
			case 2:
				_ = c.Len()
			default:
				c.add(row)
			}
		}
	})
}

// BenchmarkCallHash measures call identity computation across precomputed digest
// and varying sizes of inline argument payloads.
func BenchmarkCallHash(b *testing.B) {
	b.Run("PrecomputedDigest", func(b *testing.B) {
		call := &abi.ToolCall{
			Tool: "send_email",
			Args: abi.Ref{Kind: abi.RefInline, Digest: "sha256:abcdef0123456789"},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = callHash(call)
		}
	})

	b.Run("InlineArgs_Small", func(b *testing.B) {
		call := &abi.ToolCall{
			Tool: "search",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(`{"q":"test"}`)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = callHash(call)
		}
	})

	b.Run("InlineArgs_Medium512B", func(b *testing.B) {
		payload := fmt.Sprintf(`{"action":"update","context":"%0480d"}`, 42)
		call := &abi.ToolCall{
			Tool: "update_record",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(payload)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = callHash(call)
		}
	})

	b.Run("InlineArgs_Large4KB", func(b *testing.B) {
		payload := fmt.Sprintf(`{"document":"%04000d"}`, 99)
		call := &abi.ToolCall{
			Tool: "upload_document",
			Args: abi.Ref{Kind: abi.RefInline, Inline: []byte(payload)},
		}
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			_ = callHash(call)
		}
	})
}

// BenchmarkEndToEndStream simulates a defender-side supervisory loop consuming
// a continuous multi-verdict adjudication stream.
func BenchmarkEndToEndStream(b *testing.B) {
	h := New(NewCorpus())

	stream := []abi.Event{
		// 1. Benign allow
		{
			Kind:    abi.EvDecide,
			Call:    &abi.ToolCall{Tool: "read_file", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:f1"}},
			Verdict: &abi.Verdict{Kind: abi.VerdictAllow, By: "test"},
		},
		// 2. Redundant decide deny
		{
			Kind:    abi.EvDecide,
			Call:    &abi.ToolCall{Tool: "exfil", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:x1"}},
			Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
		},
		// 3. Concrete deny
		{
			Kind:    abi.EvDeny,
			Call:    &abi.ToolCall{Tool: "exfil", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:x1"}},
			Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonPolicyBlock, By: "test"},
		},
		// 4. Ladder hard negative
		{
			Kind: abi.EvRungLabel,
			Label: &abi.LabelRow{
				CallHash:   "probe@sha:p1",
				RungPassed: 0,
				RungFailed: 2,
				Verdict:    abi.VerdictDeny,
				Reason:     abi.ReasonTrustViolation,
			},
		},
		// 5. Result admission deny
		{
			Kind:    abi.EvResultDeny,
			Call:    &abi.ToolCall{Tool: "fetch", Args: abi.Ref{Kind: abi.RefInline, Digest: "sha:u1"}},
			Verdict: &abi.Verdict{Kind: abi.VerdictDeny, Reason: abi.ReasonUnwitnessed, By: "test"},
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ev := stream[i%len(stream)]
		h.Emit(ev)
		if i > 0 && i%1000 == 0 {
			_ = h.corpus.Positives()
			_ = h.corpus.ByReason()
		}
	}
}
