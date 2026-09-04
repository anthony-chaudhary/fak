package causalreceipt

import (
	"encoding/json"
	"testing"
)

func TestJSONRoundTrip(t *testing.T) {
	r := fixture("completed")
	data, err := json.Marshal(r)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var decoded Receipt
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	if err := Validate(decoded); err != nil {
		t.Fatalf("Validate(decoded): %v", err)
	}

	m1, err := DeriveMetrics(r)
	if err != nil {
		t.Fatalf("DeriveMetrics(r): %v", err)
	}
	m2, err := DeriveMetrics(decoded)
	if err != nil {
		t.Fatalf("DeriveMetrics(decoded): %v", err)
	}

	if m1.PhaseCount != m2.PhaseCount || m1.Tokens != m2.Tokens || m1.Bytes != m2.Bytes ||
		m1.CacheReuseBytes != m2.CacheReuseBytes || m1.OverheadNS != m2.OverheadNS ||
		m1.Outcomes["completed"] != m2.Outcomes["completed"] {
		t.Fatalf("metrics mismatch: %+v vs %+v", m1, m2)
	}
}

func BenchmarkValidate(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := Validate(r); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDeriveMetrics(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := DeriveMetrics(r); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMetricLabels(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = MetricLabels(r)
	}
}

func BenchmarkIncidentAnswers(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = IncidentAnswers(r)
	}
}

func BenchmarkJSONRoundTrip(b *testing.B) {
	r := fixture("completed")
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		data, err := json.Marshal(r)
		if err != nil {
			b.Fatal(err)
		}
		var decoded Receipt
		if err := json.Unmarshal(data, &decoded); err != nil {
			b.Fatal(err)
		}
	}
}
