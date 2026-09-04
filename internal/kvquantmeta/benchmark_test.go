package kvquantmeta

import (
	"testing"
)

var benchSupport = Support{
	Schemes: map[string][]string{
		"kvq": {"1", "2"},
	},
	Precisions: []Precision{
		PrecisionFP16,
		PrecisionBF16,
		PrecisionFP8,
		PrecisionINT8,
		PrecisionINT4,
		PrecisionINT2,
	},
	Groupings: []Grouping{
		GroupingPerToken,
		GroupingPerChannel,
		GroupingPerTokenChannel,
	},
	Transforms: []string{
		"hadamard",
		"rope-aligned",
	},
	Transitions: map[string][]string{
		"hot":  {"warm"},
		"warm": {"cold"},
	},
}

// BenchmarkKVQuantMeta measures end-to-end descriptor and tier transition validation throughput.
func BenchmarkKVQuantMeta(b *testing.B) {
	dHot := Descriptor{
		ID:                   "kvq",
		Version:              "1",
		KeyPrecision:         PrecisionFP8,
		ValuePrecision:       PrecisionINT4,
		Grouping:             GroupingPerToken,
		ResidualWindowTokens: 128,
		Transform:            "hadamard",
		Tier:                 "hot",
		Recoverability:       RecoverableApproximate,
	}
	dWarm := Descriptor{
		ID:                   "kvq",
		Version:              "1",
		KeyPrecision:         PrecisionINT4,
		ValuePrecision:       PrecisionINT4,
		Grouping:             GroupingPerToken,
		ResidualWindowTokens: 64,
		Transform:            "hadamard",
		Tier:                 "warm",
		Recoverability:       RecoverableNone,
	}
	trans := Transition{From: dHot, To: dWarm}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res1 := Validate(dHot, benchSupport)
		if !res1.Supported {
			b.Fatalf("validate hot descriptor failed: %v", res1)
		}
		res2 := ValidateTransition(trans, benchSupport)
		if !res2.Supported {
			b.Fatalf("validate transition failed: %v", res2)
		}
	}
}

// BenchmarkValidateDescriptor measures validation performance for single cache precision descriptors.
func BenchmarkValidateDescriptor(b *testing.B) {
	desc := Descriptor{
		ID:                   "kvq",
		Version:              "1",
		KeyPrecision:         PrecisionINT8,
		ValuePrecision:       PrecisionINT4,
		Grouping:             GroupingPerChannel,
		GroupSize:            32,
		ResidualWindowTokens: 128,
		Transform:            "rope-aligned",
		Tier:                 "hot",
		Recoverability:       RecoverableExact,
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Validate(desc, benchSupport)
		if !res.Supported {
			b.Fatalf("validation failed: %v", res)
		}
	}
}

// BenchmarkValidateTransition measures validation performance for directed tier migrations.
func BenchmarkValidateTransition(b *testing.B) {
	from := Descriptor{
		ID:                   "kvq",
		Version:              "1",
		KeyPrecision:         PrecisionFP8,
		ValuePrecision:       PrecisionFP8,
		Grouping:             GroupingPerToken,
		ResidualWindowTokens: 128,
		Tier:                 "hot",
		Recoverability:       RecoverableApproximate,
	}
	to := Descriptor{
		ID:                   "kvq",
		Version:              "1",
		KeyPrecision:         PrecisionINT4,
		ValuePrecision:       PrecisionINT4,
		Grouping:             GroupingPerToken,
		ResidualWindowTokens: 64,
		Tier:                 "warm",
		Recoverability:       RecoverableNone,
	}
	trans := Transition{From: from, To: to}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := ValidateTransition(trans, benchSupport)
		if !res.Supported {
			b.Fatalf("transition validation failed: %v", res)
		}
	}
}

// BenchmarkValidateMatrix measures throughput across varied precision and grouping configurations.
func BenchmarkValidateMatrix(b *testing.B) {
	descriptors := []Descriptor{
		{
			ID:                   "kvq",
			Version:              "1",
			KeyPrecision:         PrecisionFP16,
			ValuePrecision:       PrecisionFP16,
			Grouping:             GroupingPerToken,
			ResidualWindowTokens: 256,
			Tier:                 "hot",
			Recoverability:       RecoverableExact,
		},
		{
			ID:                   "kvq",
			Version:              "1",
			KeyPrecision:         PrecisionBF16,
			ValuePrecision:       PrecisionBF16,
			Grouping:             GroupingPerToken,
			ResidualWindowTokens: 256,
			Tier:                 "hot",
			Recoverability:       RecoverableExact,
		},
		{
			ID:                   "kvq",
			Version:              "1",
			KeyPrecision:         PrecisionFP8,
			ValuePrecision:       PrecisionFP8,
			Grouping:             GroupingPerToken,
			ResidualWindowTokens: 128,
			Tier:                 "hot",
			Recoverability:       RecoverableApproximate,
		},
		{
			ID:                   "kvq",
			Version:              "1",
			KeyPrecision:         PrecisionINT8,
			ValuePrecision:       PrecisionINT4,
			Grouping:             GroupingPerChannel,
			GroupSize:            64,
			ResidualWindowTokens: 64,
			Tier:                 "warm",
			Recoverability:       RecoverableApproximate,
		},
		{
			ID:                   "kvq",
			Version:              "1",
			KeyPrecision:         PrecisionINT4,
			ValuePrecision:       PrecisionINT2,
			Grouping:             GroupingPerTokenChannel,
			GroupSize:            32,
			ResidualWindowTokens: 0,
			Tier:                 "cold",
			Recoverability:       RecoverableNone,
		},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for idx := range descriptors {
			res := Validate(descriptors[idx], benchSupport)
			if !res.Supported {
				b.Fatalf("matrix descriptor validation failed at index %d: %v", idx, res)
			}
		}
	}
}

// TestBenchmarkKVQuantMetaSanity ensures benchmark routines execute cleanly under test runners.
func TestBenchmarkKVQuantMetaSanity(t *testing.T) {
	res := testing.Benchmark(BenchmarkKVQuantMeta)
	if res.N <= 0 {
		t.Fatalf("expected positive benchmark iterations, got %d", res.N)
	}
}
