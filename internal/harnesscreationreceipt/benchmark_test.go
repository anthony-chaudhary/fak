package harnesscreationreceipt

import (
	"encoding/json"
	"testing"
)

func BenchmarkHarnessCreationReceipt(b *testing.B) {
	r := validReceipt()
	raw, err := json.Marshal(r)
	if err != nil {
		b.Fatal(err)
	}
	study := []byte(`{"protocol":{"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"runs":[]}`)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		parsed, err := Parse(raw)
		if err != nil {
			b.Fatal(err)
		}
		res := Evaluate(parsed)
		if !res.Valid {
			b.Fatal("invalid result")
		}
		if err := CheckUnique(study, res.Row); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkParse(b *testing.B) {
	r := validReceipt()
	raw, err := json.Marshal(r)
	if err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, err := Parse(raw); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkEvaluate(b *testing.B) {
	r := validReceipt()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		_ = Evaluate(r)
	}
}

func BenchmarkCheckUnique(b *testing.B) {
	row := Evaluate(validReceipt()).Row
	study := []byte(`{"protocol":{"task_digest":"sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"},"runs":[]}`)

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := CheckUnique(study, row); err != nil {
			b.Fatal(err)
		}
	}
}
