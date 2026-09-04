package ultracodetokenizer

import (
	"encoding/json"
	"os"
	"testing"
)

func benchFixture(b *testing.B) (CanonicalInput, []Measurement) {
	b.Helper()
	var in CanonicalInput
	data, err := os.ReadFile("testdata/canonical.json")
	if err != nil {
		b.Fatal(err)
	}
	if err := json.Unmarshal(data, &in); err != nil {
		b.Fatal(err)
	}
	var measurements []Measurement
	data, err = os.ReadFile("testdata/measurements.json")
	if err != nil {
		b.Fatal(err)
	}
	if err := json.Unmarshal(data, &measurements); err != nil {
		b.Fatal(err)
	}
	digest := Digest(in)
	for i := range measurements {
		measurements[i].CanonicalDigest = digest
	}
	return in, measurements
}

func BenchmarkUltraCodeTokenizer(b *testing.B) {
	in, measurements := benchFixture(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		report, err := Evaluate(in, measurements)
		if err != nil {
			b.Fatal(err)
		}
		if len(report.Results) == 0 {
			b.Fatal("unexpected empty results")
		}
	}
}

func TestBenchmarkUltraCodeTokenizer(t *testing.T) {
	var in CanonicalInput
	data, err := os.ReadFile("testdata/canonical.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &in); err != nil {
		t.Fatal(err)
	}
	var measurements []Measurement
	data, err = os.ReadFile("testdata/measurements.json")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(data, &measurements); err != nil {
		t.Fatal(err)
	}
	digest := Digest(in)
	for i := range measurements {
		measurements[i].CanonicalDigest = digest
	}
	report, err := Evaluate(in, measurements)
	if err != nil {
		t.Fatal(err)
	}
	if len(report.Results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(report.Results))
	}
}
