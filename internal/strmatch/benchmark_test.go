package strmatch

import "testing"

var (
	benchSinkBool   bool
	benchSinkString string
	benchSinkInt    int
	benchSinkSlice  []string
)

// BenchmarkContainsAny measures scanning string inputs for any matching pattern.
func BenchmarkContainsAny(b *testing.B) {
	haystack := "error: compilation failed due to merge conflict in rebase state"
	patterns := []string{"auth", "timeout", "conflict", "permission", "unreachable"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = ContainsAny(haystack, patterns...)
	}
}

// BenchmarkFirstContained measures witness-carrying first matching substring lookup.
func BenchmarkFirstContained(b *testing.B) {
	haystack := "operation refused: target lane lease expired before commit"
	needles := []string{"not found", "conflict", "lease expired", "permission denied"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, ok := FirstContained(haystack, needles)
		benchSinkString = s
		benchSinkBool = ok
	}
}

// BenchmarkCommonPrefixLen measures byte-level common prefix calculation across strings.
func BenchmarkCommonPrefixLen(b *testing.B) {
	a := "github.com/anthony-chaudhary/fak/internal/adjudicator/policy_matrix_evaluation.go"
	c := "github.com/anthony-chaudhary/fak/internal/adjudicator/policy_matrix_validation.go"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInt = CommonPrefixLen(a, c)
	}
}

// BenchmarkTail measures clamping trailing bytes of trimmed error or log outputs.
func BenchmarkTail(b *testing.B) {
	output := "   [kernel] warning: lease heartbeat delayed\n[kernel] error: command failed with exit status 1   \n"

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = Tail(output, 48)
	}
}

// BenchmarkFirstNonBlank measures evaluating fallback chains for non-whitespace text.
func BenchmarkFirstNonBlank(b *testing.B) {
	vals := []string{"", "   ", "\t\n", "fak-primary-worker", "fallback-worker"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = FirstNonBlank(vals...)
	}
}

// BenchmarkCommonSlicePrefixLen measures finding common prefix length in generic slices.
func BenchmarkCommonSlicePrefixLen(b *testing.B) {
	s1 := []string{"cmd", "fak", "internal", "strmatch", "benchmark_test.go"}
	s2 := []string{"cmd", "fak", "internal", "strmatch", "strmatch_test.go"}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkInt = CommonSlicePrefixLen(s1, s2)
	}
}

// BenchmarkStripUnquotedComment measures stripping trailing comments outside double quotes.
func BenchmarkStripUnquotedComment(b *testing.B) {
	line := `lane = "strmatch # not a comment" # real comment with details`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = StripUnquotedComment(line, '#')
	}
}

// BenchmarkSplitQuoted measures splitting separated tokens outside double quotes.
func BenchmarkSplitQuoted(b *testing.B) {
	entry := `"lane1,with,comma", "lane2", "lane3,with,comma", "lane4"`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSlice = SplitQuoted(entry, ',')
	}
}

// BenchmarkParseQuotedScalar measures parsing strict double-quoted scalars.
func BenchmarkParseQuotedScalar(b *testing.B) {
	raw := `"production-grade-lane"`

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s, err := ParseQuotedScalar(raw)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkString = s
	}
}
