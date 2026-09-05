package binstamp

import (
	"runtime/debug"
	"testing"
)

var (
	benchSinkStamp     Stamp
	benchSinkFreshness Freshness
	benchSinkCause     Cause
	benchSinkString    string
	benchSinkBool      bool
)

func BenchmarkBinStamp(b *testing.B) {
	head := "abcdef1234567890abcdef1234567890abcdef12"
	stampFresh := Stamp{Revision: head, HasVCS: true}
	stampStale := Stamp{Revision: "ffffffffffffffff", HasVCS: true}
	bi := &debug.BuildInfo{
		Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: head},
			{Key: "vcs.modified", Value: "false"},
		},
	}

	b.Run("CompareFresh", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkFreshness = Compare(stampFresh, head)
		}
	})

	b.Run("CompareStale", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkFreshness = Compare(stampStale, head)
		}
	})

	b.Run("Explain", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkFreshness, benchSinkCause = Explain(stampFresh, head)
		}
	})

	b.Run("FromBuildInfo", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStamp = FromBuildInfo(bi)
		}
	})

	b.Run("Self", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkStamp = Self()
		}
	})
}

func BenchmarkCompare(b *testing.B) {
	head := "abcdef1234567890abcdef1234567890abcdef12"
	stamp := Stamp{Revision: head, HasVCS: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFreshness = Compare(stamp, head)
	}
}

func BenchmarkExplain(b *testing.B) {
	head := "abcdef1234567890abcdef1234567890abcdef12"
	stamp := Stamp{Revision: head, HasVCS: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkFreshness, benchSinkCause = Explain(stamp, head)
	}
}

func BenchmarkFromBuildInfo(b *testing.B) {
	bi := &debug.BuildInfo{
		Settings: []debug.BuildSetting{
			{Key: "vcs.revision", Value: "abcdef1234567890abcdef1234567890abcdef12"},
			{Key: "vcs.modified", Value: "false"},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStamp = FromBuildInfo(bi)
	}
}

func BenchmarkSelf(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStamp = Self()
	}
}

func BenchmarkFreshnessString(b *testing.B) {
	f := Fresh
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = f.String()
	}
}

func BenchmarkCauseString(b *testing.B) {
	c := CauseMatched
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString = c.String()
	}
}

func BenchmarkRevisionsMatch(b *testing.B) {
	revA := "abcdef1234567"
	revB := "abcdef1234567890abcdef1234567890abcdef12"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = revisionsMatch(revA, revB)
	}
}

func TestBenchmarkExecution(t *testing.T) {
	res := testing.Benchmark(func(b *testing.B) {
		head := "abcdef1234567890abcdef1234567890abcdef12"
		stamp := Stamp{Revision: head, HasVCS: true}
		for i := 0; i < b.N; i++ {
			benchSinkFreshness, benchSinkCause = Explain(stamp, head)
		}
	})
	if res.N <= 0 {
		t.Fatalf("expected benchmark iterations > 0, got %d", res.N)
	}
}
