// These benchmarks measure the package's two production hot paths: Check,
// the vLLM check_logprobs_close scoring core, and SegmentText, text-piece
// recovery for text-only candidates.
package logprobparity_test

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/logprobparity"
)

func BenchmarkCheckAgreeingPrefix(b *testing.B) {
	const n = 256
	actual := make([]logprobparity.Position, n)
	ref := make([]logprobparity.Position, n)
	for i := range actual {
		topk := make([]logprobparity.Candidate, 5)
		for j := range topk {
			id := i
			if j > 0 {
				id = i + 1000 + j
			}
			topk[j] = logprobparity.Candidate{ID: id, Logprob: -float64(j + 1)}
		}
		actual[i] = logprobparity.Position{ID: i, TopK: topk, ReportedTopK: 5}
		ref[i] = logprobparity.Position{ID: i, TopK: topk, ReportedTopK: 5}
	}
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		r := logprobparity.Check(actual, ref, 5, logprobparity.TokenIDs)
		if i == 0 && r.Verdict != logprobparity.Match {
			b.Fatalf("verdict %q, want Match", r.Verdict)
		}
		sink += r.Compared
	}
	_ = sink
}

func BenchmarkCheckDivergentWithTopK(b *testing.B) {
	actual := []logprobparity.Position{{
		ID:           3,
		TopK:         []logprobparity.Candidate{{ID: 3, Logprob: -0.2}, {ID: 2, Logprob: -1.4}},
		ReportedTopK: 2,
	}}
	ref := []logprobparity.Position{{
		ID:           2,
		TopK:         []logprobparity.Candidate{{ID: 2, Logprob: -0.3}, {ID: 3, Logprob: -1.1}},
		ReportedTopK: 2,
	}}
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		r := logprobparity.Check(actual, ref, 5, logprobparity.TokenIDs)
		if i == 0 && r.Verdict != logprobparity.Close {
			b.Fatalf("verdict %q, want Close", r.Verdict)
		}
		sink += r.Compared
	}
	_ = sink
}

func BenchmarkSegmentTextFullMatch(b *testing.B) {
	ref := []logprobparity.Position{
		{ID: 1, Piece: " Paris"},
		{ID: 2, Piece: "."},
		{ID: 3, Piece: " In"},
		{
			ID:    4,
			Piece: " 1923",
			TopK: []logprobparity.Candidate{
				{ID: 4, Piece: " 1923", Logprob: -0.4},
				{ID: 4, Piece: " 1923,", Logprob: -0.9},
			},
		},
	}
	text := " Paris. In 1923,"
	b.ResetTimer()
	var sink int
	for i := 0; i < b.N; i++ {
		got := logprobparity.SegmentText(text, ref)
		if i == 0 {
			if len(got) != len(ref) {
				b.Fatalf("segmented %d of %d positions", len(got), len(ref))
			}
			if r := logprobparity.Check(got, ref, 5, logprobparity.TextPieces); r.Verdict != logprobparity.Match {
				b.Fatalf("verdict %q, want Match", r.Verdict)
			}
		}
		sink += len(got)
	}
	_ = sink
}
