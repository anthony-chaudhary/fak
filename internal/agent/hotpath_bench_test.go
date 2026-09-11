package agent

import (
	"fmt"
	"strings"
	"testing"
)

// This file carries the micro-benchmarks and allocation floors for two hot paths
// the perf audit (issue #995) found unguarded in the public core:
//
//  1. The decode-loop emit closure (internal/agent/inkernel_planner.go, the closure
//     built at Complete: it runs per generated token and re-scans the accumulated
//     text on every token through checkStop). This is the O(N^2)-shaped work the
//     audit flagged; the closure itself needs a live InKernelPlanner + tokenizer, so
//     the benchmark targets the reusable per-token primitive it calls —
//     checkStop(text, stop) — with the builder accumulation shape it wraps.
//  2. Transcript elision / ctxview linear scan (agent.ElideStaleReadMessages),
//     exercised at agent scale (an 8k-message transcript).
//
// LIMITATION: the full emit closure at inkernel_planner.go cannot be invoked
// without a constructed InKernelPlanner backed by a real model/tokenizer fixture.
// Building a fake model only to reach the closure would be a brittle harness, so
// this file benches the callable primitive (checkStop + strings.Builder) that
// carries the per-token cost, and documents that boundary here.

// BenchmarkDecodeEmitCheckStop measures the per-token stop scan the decode emit
// closure performs: checkStop(sb.String(), stops) materializes the full accumulated
// text and HasSuffix-scans it against every stop sequence. b.SetBytes reports the
// transcript length so ns/op can be read as a scan rate.
func BenchmarkDecodeEmitCheckStop(b *testing.B) {
	stops := []string{"<|im_end|>", "\nuser:", "</tool_call>"}
	for _, tokens := range []int{128, 1024, 8192} {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			var sb strings.Builder
			for i := 0; i < tokens; i++ {
				sb.WriteString("token ")
			}
			text := sb.String()
			b.SetBytes(int64(len(text)))
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, hit := checkStop(text, stops); hit {
					b.Fatalf("unexpected stop hit")
				}
			}
		})
	}
}

// BenchmarkDecodeEmitAccumulate measures the emit closure's builder accumulation
// shape (WriteString per decoded token) at the same scales, so the two halves of
// per-token work — detokenized append + stop rescan — are both witnessed.
func BenchmarkDecodeEmitAccumulate(b *testing.B) {
	for _, tokens := range []int{128, 1024, 8192} {
		b.Run(fmt.Sprintf("tokens=%d", tokens), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				var sb strings.Builder
				for j := 0; j < tokens; j++ {
					sb.WriteString("token ")
				}
				if sb.Len() == 0 {
					b.Fatal("empty")
				}
			}
		})
	}
}

// TestDecodeEmitPerTokenAllocsFlat asserts the allocation floor for the emit
// primitive: checkStop must allocate nothing per call, and the builder's
// per-token accumulation must not allocate per token (only the amortized growth
// of its backing array). A reintroduced per-token full-buffer COPY (the regression
// the audit warns about) would push allocations up and fail here.
func TestDecodeEmitPerTokenAllocsFlat(t *testing.T) {
	stops := []string{"<|im_end|>", "\nuser:", "</tool_call>"}
	text := strings.Repeat("token ", 8192)

	stopAllocs := testing.AllocsPerRun(200, func() {
		_, _ = checkStop(text, stops)
	})
	if stopAllocs > 0 {
		t.Fatalf("checkStop must not allocate per token: got %v allocs/op, want 0", stopAllocs)
	}

	// A short (no-stop-hit) emit over 256 tokens must not scale allocations with
	// token count: WriteString into a Builder reuses its buffer, so the loop is
	// allocation-flat modulo the amortized backing-array growth.
	const n = 256
	builderAllocs := testing.AllocsPerRun(200, func() {
		var sb strings.Builder
		for i := 0; i < n; i++ {
			sb.WriteString("token ")
		}
		_, _ = checkStop(sb.String(), stops)
	})
	// strings.Builder growth is geometric: log2(256*6/64) ~= 5 growth allocs plus a
	// possible String() copy. A per-token copy would be >= n. Budget well under that.
	const budget = 16.0
	if builderAllocs > budget {
		t.Fatalf("emit accumulation allocates per token: got %v allocs for %d tokens (budget %v)", builderAllocs, n, budget)
	}
}

// BenchmarkElideStaleReadScan exercises the decoded-path transcript elision linear
// scan (agent.ElideStaleReadMessages) at agent scale: an 8k-message transcript where
// every read is superseded by a later edit. This is the ctxview/elision path the
// audit named.
func BenchmarkElideStaleReadScan(b *testing.B) {
	const messages = 8000
	transcript := benchStaleReadTranscript(messages)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out := ElideStaleReadMessages(transcript, nil)
		if len(out) != len(transcript) {
			b.Fatalf("elision changed length: got %d want %d", len(out), len(transcript))
		}
	}
}

// benchStaleReadTranscript builds an 8k-message transcript of regular read results
// interleaved with later edits to the same paths, so the linear scan has real work
// to do on every pass without bailing on the protected tail.
func benchStaleReadTranscript(n int) []Message {
	msgs := make([]Message, 0, n+8)
	body := strings.Repeat("file body line\n", 32)
	for i := 0; i < n; i++ {
		switch i % 3 {
		case 0:
			path := fmt.Sprintf("/src/file_%d.go", i)
			msgs = append(msgs, Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{{
					ID:       fmt.Sprintf("read_%d", i),
					Type:     "function",
					Function: Func{Name: "Read", Arguments: fmt.Sprintf(`{"file_path":%q}`, path)},
				}},
			})
		case 1:
			path := fmt.Sprintf("/src/file_%d.go", i-1)
			msgs = append(msgs, Message{Role: RoleTool, ToolCallID: fmt.Sprintf("read_%d", i-1), Content: body})
			_ = path
		case 2:
			path := fmt.Sprintf("/src/file_%d.go", i-2)
			msgs = append(msgs, Message{
				Role: RoleAssistant,
				ToolCalls: []ToolCall{{
					ID:       fmt.Sprintf("edit_%d", i),
					Type:     "function",
					Function: Func{Name: "Edit", Arguments: fmt.Sprintf(`{"file_path":%q}`, path)},
				}},
			})
		}
	}
	// Protected working-set tail: the scan never elides the last 6 messages.
	for i := 0; i < 6; i++ {
		msgs = append(msgs, Message{Role: RoleAssistant, Content: fmt.Sprintf("tail reply %d", i)})
	}
	return msgs
}
