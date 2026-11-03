package coherence

import (
	"strconv"
	"strings"
	"testing"
)

// BenchmarkCheckCoherent measures the engine readiness-gate verdict over a
// full-size answer: Check(text, MaxCount) on a realistic model reply carrying
// a chat preamble, a closed <think>...</think> block, and the contiguous run
// 1..MaxCount.
func BenchmarkCheckCoherent(b *testing.B) {
	var sb strings.Builder
	sb.WriteString("Sure, here is the count you asked for.")
	sb.WriteString("\n")
	sb.WriteString("Listing the integers in order and stopping at sixty-four.")
	sb.WriteString("\n")
	for i := 1; i <= MaxCount; i++ {
		if i > 1 {
			sb.WriteByte(' ')
		}
		sb.WriteString(strconv.Itoa(i))
	}
	input := sb.String()
	if got := Check(input, MaxCount); got != Coherent {
		b.Fatalf("fixture Check = %q, want %q", got, Coherent)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = Check(input, MaxCount)
	}
}
