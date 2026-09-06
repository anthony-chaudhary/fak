package guardcorpus

import (
	"fmt"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

var (
	sinkRecord   SessionRecord
	sinkExamples []Example
	sinkVerdict  string
)

// BenchmarkFoldPlanted measures the execution throughput and allocations of Fold
// over the canonical planted test session covering all decision types, honesty holes,
// and exit outcomes.
func BenchmarkFoldPlanted(b *testing.B) {
	meta := SessionMeta{
		TraceID:       "trace-bench-planted",
		Agent:         "claude-code",
		HostClass:     "desktop",
		PolicyDigest:  "sha256:d8e8fca2dc0f896fd7cb4cb0031ba249",
		ChainVerified: true,
	}
	rows := planted()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRecord, sinkExamples = Fold(meta, rows)
	}
}

// BenchmarkFoldScaling measures Fold throughput and memory allocation behavior
// across varying session journal sizes using b.Run sub-benchmarks.
func BenchmarkFoldScaling(b *testing.B) {
	sizes := []int{10, 100, 1000}
	meta := SessionMeta{
		TraceID:       "trace-bench-scaling",
		Agent:         "claude-code",
		HostClass:     "desktop",
		PolicyDigest:  "sha256:scale-digest",
		ChainVerified: true,
	}

	for _, size := range sizes {
		b.Run(fmt.Sprintf("Rows_%d", size), func(b *testing.B) {
			rows := make([]journal.Row, size)
			for i := 0; i < size; i++ {
				verdict := "ALLOW"
				kind := "DECIDE"
				reason := ""
				if i%10 == 1 {
					verdict = "DENY"
					kind = "DENY"
					reason = "POLICY_BLOCK"
				} else if i%20 == 2 {
					verdict = "ADVISORY"
					kind = "TOOL_DEFINITION_PRUNED"
				} else if i%50 == 3 {
					verdict = "QUARANTINE"
					kind = "QUARANTINE"
					reason = "SECRET_DISCOVERED"
				}
				rows[i] = journal.Row{
					Seq:        uint64(i + 1),
					TSUnixNano: int64(1000 + i*10),
					Kind:       kind,
					Tool:       "Bash",
					Verdict:    verdict,
					Reason:     reason,
					By:         "floor",
					Witness:    "benchmark-witness-evidence",
					ArgsLabel:  "command=go",
				}
			}

			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				sinkRecord, sinkExamples = Fold(meta, rows)
			}
		})
	}
}

// BenchmarkFoldAllowsBounded verifies the performance profile of Fold when processing
// sessions dominated by ALLOW decisions that exceed maxAllowExamples capping limit.
func BenchmarkFoldAllowsBounded(b *testing.B) {
	const count = 500
	rows := make([]journal.Row, count)
	for i := 0; i < count; i++ {
		rows[i] = journal.Row{
			Seq:        uint64(i + 1),
			TSUnixNano: int64(100 + i),
			Kind:       "DECIDE",
			Tool:       "Read",
			Verdict:    "ALLOW",
			By:         "floor",
			ArgsLabel:  "path=main.go",
		}
	}
	meta := SessionMeta{TraceID: "trace-bench-allows", PolicyDigest: "sha256:allows"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkRecord, sinkExamples = Fold(meta, rows)
	}
}

// BenchmarkNormalizeVerdict measures verdict resolution and fallback canonicalization
// throughput across standard verdicts and kind-fallback cases.
func BenchmarkNormalizeVerdict(b *testing.B) {
	cases := []struct {
		verdict string
		kind    string
	}{
		{"ALLOW", "DECIDE"},
		{"DENY", "DENY"},
		{"", "RESULT_DENY"},
		{"", "QUARANTINE"},
		{"ADVISORY", "TOOL_DEFINITION_PRUNED"},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c := cases[i%len(cases)]
		sinkVerdict = normalizeVerdict(c.verdict, c.kind)
	}
}
