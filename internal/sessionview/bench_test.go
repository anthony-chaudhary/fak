package sessionview

import (
	"sync/atomic"
	"testing"
	"time"
)

func BenchmarkMaterializedView_Append_Sequential(b *testing.B) {
	v := New("bench-sess", 500)
	row := LlmCallRow{
		CallID:       "call-bench",
		SessionID:    "bench-sess",
		Model:        "qwen-2.5",
		PromptTokens: 100,
		OutputTokens: 50,
		TotalTokens:  150,
		Timestamp:    time.Now().UTC(),
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.Append(row)
	}
}

func BenchmarkMaterializedView_Append_Parallel(b *testing.B) {
	v := New("bench-sess", 1000)
	var idGen atomic.Int64

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = v.Append(ToolCallRow{
				ToolCallID: "tool-bench",
				SessionID:  "bench-sess",
				ToolName:   "bash",
				Duration:   10 * time.Millisecond,
			})
			idGen.Add(1)
		}
	})
}

func BenchmarkMaterializedView_Snapshot(b *testing.B) {
	const capacity = 500
	v := New("bench-sess", capacity)
	for i := 0; i < capacity; i++ {
		_ = v.Append(LlmCallRow{
			CallID:       "call-bench",
			SessionID:    "bench-sess",
			PromptTokens: 50,
			OutputTokens: 25,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.Snapshot()
	}
}

func BenchmarkMaterializedView_ConcurrentAppendAndSnapshot(b *testing.B) {
	v := New("bench-sess", 500)

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			if i%10 == 0 {
				_ = v.Snapshot()
			} else {
				_ = v.Append(AuditEventRow{
					EventID:   "audit-bench",
					SessionID: "bench-sess",
					Component: "guard",
				})
			}
			i++
		}
	})
}

func BenchmarkRingBuffer_PushAndEvict(b *testing.B) {
	rb := newRingBuffer(200)
	row := ToolCallRow{ToolCallID: "bench"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rb.push(row)
	}
}

func BenchmarkMaterializedView_Ingest_CappedRingEviction(b *testing.B) {
	const capacity = 10000
	v := NewMaterializedView(capacity)
	row := LlmCallRow{
		CallID:       "call-bench",
		SessionID:    "bench-sess",
		Model:        "qwen-2.5",
		PromptTokens: 100,
		OutputTokens: 50,
		TotalTokens:  150,
		CachedTokens: 20,
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = v.Ingest(row)
	}
}
