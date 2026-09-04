package archrank

import (
	"bytes"
	"encoding/json"
	"fmt"
	"testing"
)

// BenchmarkArchRankSort measures throughput and memory allocations when ranking
// and sorting architecture candidates by quality efficiency and tie-break rules.
func BenchmarkArchRankSort(b *testing.B) {
	candidates := make([]Candidate, 64)
	for i := 0; i < len(candidates); i++ {
		candidates[i] = Candidate{
			ID:                fmt.Sprintf("arch-%03d", (i*37)%len(candidates)),
			Architecture:      "transformer-moe",
			MigrationClass:    "clean-room",
			EnvelopeID:        "tier-7b-ctx4k",
			QualityMetric:     "mmlu-pro",
			QualitySourceKind: "measured-benchmark",
			MeasurementStatus: "accepted",
			Quality:           0.50 + float64(i%20)*0.02,
			ActiveWeightBytes: 7000000000 + uint64((i*17)%100)*10000000,
			StateBytes:        2000000,
			KVBytesAtEnvelope: 500000000,
			Provenance: Provenance{
				Kind:    "synthetic_control_measurement",
				Locator: "eval.log",
			},
		}
	}
	dataset := validDataset(candidates...)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, err := Rank(dataset)
		if err != nil {
			b.Fatalf("Rank failed: %v", err)
		}
		if len(res.Groups) != 1 || len(res.Groups[0].Rows) != len(candidates) {
			b.Fatalf("unexpected group or row count in ranking")
		}
	}
}

// BenchmarkArchRank measures throughput when ranking small candidate datasets.
func BenchmarkArchRank(b *testing.B) {
	c1 := measuredCandidate("bench-cand-1")
	c2 := measuredCandidate("bench-cand-2")
	c3 := measuredCandidate("bench-cand-3")
	c4 := measuredCandidate("bench-cand-4")
	dataset := validDataset(c1, c2, c3, c4)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res, err := Rank(dataset)
		if err != nil {
			b.Fatalf("Rank failed: %v", err)
		}
		if len(res.Groups) == 0 {
			b.Fatal("unexpected empty groups in benchmark")
		}
	}
}

// BenchmarkDatasetValidate measures the cost of schema, formula, and candidate invariant validation.
func BenchmarkDatasetValidate(b *testing.B) {
	c1 := measuredCandidate("bench-val-1")
	c2 := measuredCandidate("bench-val-2")
	dataset := validDataset(c1, c2)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := dataset.Validate(); err != nil {
			b.Fatalf("Validate failed: %v", err)
		}
	}
}

// BenchmarkActiveBytes measures active-byte accounting formula calculation and overflow bounds.
func BenchmarkActiveBytes(b *testing.B) {
	cand := measuredCandidate("bench-active-bytes")

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		bytes, err := cand.ActiveBytes()
		if err != nil || bytes == 0 {
			b.Fatalf("ActiveBytes failed: %v", err)
		}
	}
}

// BenchmarkLoadJSON measures JSON deserialization with strict unknown-field checking.
func BenchmarkLoadJSON(b *testing.B) {
	c1 := measuredCandidate("bench-load-1")
	c2 := measuredCandidate("bench-load-2")
	dataset := validDataset(c1, c2)
	data, err := json.Marshal(dataset)
	if err != nil {
		b.Fatalf("json.Marshal failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_, err := LoadJSON(bytes.NewReader(data))
		if err != nil {
			b.Fatalf("LoadJSON failed: %v", err)
		}
	}
}
