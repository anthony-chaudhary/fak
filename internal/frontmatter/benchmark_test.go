package frontmatter

import (
	"testing"
)

var (
	benchSinkVal string
	benchSinkOK  bool
)

func BenchmarkDecodeScalarPlain(b *testing.B) {
	raw := "   production-scalar-value-without-quotes   "
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(raw)
	}
}

func BenchmarkDecodeScalarSingleQuoted(b *testing.B) {
	raw := `'Bob''s skill says ''hello world'' from the agent''s context'`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(raw)
	}
}

func BenchmarkDecodeScalarDoubleQuoted(b *testing.B) {
	raw := `"Use a colon: safely and say \"hello\" from C:\\tools.\nNext\tstep with escaped \"quotes\".\r\n"`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(raw)
	}
}

func BenchmarkDecodeScalarUnicode(b *testing.B) {
	raw := `"Unicode escape \u00a0 and hex \x20 with rune \U00000041 and newline \n"`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(raw)
	}
}

func BenchmarkDecodeScalarMalformed(b *testing.B) {
	raw := `"unterminated string with unknown \q escape and bad surrogate \uD800`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(raw)
	}
}

func BenchmarkDecodeScalarProductionCorpus(b *testing.B) {
	corpus := []string{
		"plain-skill-name",
		"1.2.0",
		"true",
		"  trimmed-identifier  ",
		`'Bob''s skill says ''hello''.'`,
		`'internal/frontmatter/**'`,
		`"Use a colon: safely and say \"hello\" from C:\\tools.\nNext\tstep."`,
		`"Commit finished work cleanly on the trunk \u2014 lint before staging.\nNext line."`,
		`"unterminated escape sequence \q`,
		"",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkVal, benchSinkOK = DecodeScalar(corpus[i%len(corpus)])
	}
}

func TestBenchmarkExecution(t *testing.T) {
	res := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			benchSinkVal, benchSinkOK = DecodeScalar(`"hello\nworld"`)
		}
	})
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
