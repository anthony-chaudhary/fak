package generation

import (
	"testing"
)

var (
	benchStringSink string
	benchSliceSink  []string
	benchMapSink    map[string]int
)

var benchmarkInputsCanonical = []string{
	"now",
	"next",
	"second-next",
	"future",
}

var benchmarkInputsPrefixed = []string{
	"gen/now",
	"gen/next",
	"gen/second-next",
	"gen/future",
}

var benchmarkInputsFormatted = []string{
	"  GEN/NOW  ",
	"  Next\t",
	"\nsecond-next  ",
	"  GEN/FUTURE  ",
}

var benchmarkInputsUnclassified = []string{
	"custom",
	"bug",
	"feature",
	"priority/p0",
	"gen/unknown",
	"",
}

var benchmarkInputsMixed = []string{
	"gen/now",
	"custom-feature",
	"  SECOND-NEXT  ",
	"bug/regression",
	"next",
	"gen/future",
	"priority/p1",
	"  GEN/NEXT \n",
	"documentation",
	"unclassified",
	"future",
	"",
}

// BenchmarkOrder measures canonical horizon order slice construction and copying.
func BenchmarkOrder(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSliceSink = Order()
	}
}

// BenchmarkNormalize measures normalization across representative production inputs.
func BenchmarkNormalize(b *testing.B) {
	b.Run("Canonical", func(b *testing.B) {
		inputs := benchmarkInputsCanonical
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Normalize(inputs[i%n])
		}
	})

	b.Run("Prefixed", func(b *testing.B) {
		inputs := benchmarkInputsPrefixed
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Normalize(inputs[i%n])
		}
	})

	b.Run("Formatted", func(b *testing.B) {
		inputs := benchmarkInputsFormatted
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Normalize(inputs[i%n])
		}
	})

	b.Run("Unclassified", func(b *testing.B) {
		inputs := benchmarkInputsUnclassified
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Normalize(inputs[i%n])
		}
	})

	b.Run("Mixed", func(b *testing.B) {
		inputs := benchmarkInputsMixed
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Normalize(inputs[i%n])
		}
	})
}

// BenchmarkNormalizeCanonical measures direct canonical bare horizon lookup.
func BenchmarkNormalizeCanonical(b *testing.B) {
	inputs := benchmarkInputsCanonical
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Normalize(inputs[i%n])
	}
}

// BenchmarkNormalizePrefixed measures gen/ prefixed horizon lookup.
func BenchmarkNormalizePrefixed(b *testing.B) {
	inputs := benchmarkInputsPrefixed
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Normalize(inputs[i%n])
	}
}

// BenchmarkNormalizeFormatted measures whitespace-trimmed and case-folded inputs.
func BenchmarkNormalizeFormatted(b *testing.B) {
	inputs := benchmarkInputsFormatted
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Normalize(inputs[i%n])
	}
}

// BenchmarkNormalizeUnclassified measures unclassified label throughput.
func BenchmarkNormalizeUnclassified(b *testing.B) {
	inputs := benchmarkInputsUnclassified
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Normalize(inputs[i%n])
	}
}

// BenchmarkNormalizeMixed measures real-world mixed input stream normalization.
func BenchmarkNormalizeMixed(b *testing.B) {
	inputs := benchmarkInputsMixed
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Normalize(inputs[i%n])
	}
}

// BenchmarkLabel measures gen/ canonical label formatting across inputs.
func BenchmarkLabel(b *testing.B) {
	b.Run("Canonical", func(b *testing.B) {
		inputs := benchmarkInputsCanonical
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Label(inputs[i%n])
		}
	})

	b.Run("Prefixed", func(b *testing.B) {
		inputs := benchmarkInputsPrefixed
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Label(inputs[i%n])
		}
	})

	b.Run("Unclassified", func(b *testing.B) {
		inputs := benchmarkInputsUnclassified
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Label(inputs[i%n])
		}
	})

	b.Run("Mixed", func(b *testing.B) {
		inputs := benchmarkInputsMixed
		n := len(inputs)
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchStringSink = Label(inputs[i%n])
		}
	})
}

// BenchmarkLabelCanonical measures formatting canonical bare names to gen/ labels.
func BenchmarkLabelCanonical(b *testing.B) {
	inputs := benchmarkInputsCanonical
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Label(inputs[i%n])
	}
}

// BenchmarkLabelPrefixed measures formatting already-prefixed names.
func BenchmarkLabelPrefixed(b *testing.B) {
	inputs := benchmarkInputsPrefixed
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Label(inputs[i%n])
	}
}

// BenchmarkLabelMixed measures formatting on mixed production label streams.
func BenchmarkLabelMixed(b *testing.B) {
	inputs := benchmarkInputsMixed
	n := len(inputs)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStringSink = Label(inputs[i%n])
	}
}

// BenchmarkBatchClassify measures scanning and filtering issue label lists into generation horizons,
// simulating production loops in milestonereport and projectreport.
func BenchmarkBatchClassify(b *testing.B) {
	issueLabelSets := [][]string{
		{"kind/feature", "gen/now", "team/core"},
		{"area/serving", "gen/next"},
		{"bug", "priority/urgent"},
		{"gen/second-next", "size/m"},
		{"area/inference", "gen/future", "blocked"},
		{"doc", "gen/now"},
		{"untagged"},
	}
	setCount := len(issueLabelSets)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		labels := issueLabelSets[i%setCount]
		gen := "unclassified"
		for _, lbl := range labels {
			normalized := Normalize(lbl)
			if normalized != "unclassified" {
				gen = normalized
				break
			}
		}
		benchStringSink = gen
	}
}

// BenchmarkHorizonPartition measures aggregating issues by horizon bucket,
// modeling milestone reporting workload over 100 issues.
func BenchmarkHorizonPartition(b *testing.B) {
	issues := make([]string, 100)
	for i := range issues {
		switch i % 5 {
		case 0:
			issues[i] = "gen/now"
		case 1:
			issues[i] = "gen/next"
		case 2:
			issues[i] = "gen/second-next"
		case 3:
			issues[i] = "gen/future"
		default:
			issues[i] = "unclassified"
		}
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buckets := make(map[string]int, 5)
		for _, raw := range issues {
			h := Normalize(raw)
			buckets[h]++
		}
		benchMapSink = buckets
	}
}
