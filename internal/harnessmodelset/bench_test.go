package harnessmodelset_test

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessmodelset"
)

func BenchmarkParseJSON(b *testing.B) {
	raw, err := os.ReadFile("testdata/two-role.json")
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		intent, err := harnessmodelset.ParseJSON(raw)
		if err != nil {
			b.Fatalf("ParseJSON: %v", err)
		}
		if len(intent.Roles) == 0 {
			b.Fatalf("no roles parsed")
		}
	}
}

func BenchmarkCanonicalJSON(b *testing.B) {
	raw, err := os.ReadFile("testdata/two-role.json")
	if err != nil {
		b.Fatal(err)
	}
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := harnessmodelset.CanonicalJSON(intent)
		if err != nil {
			b.Fatalf("CanonicalJSON: %v", err)
		}
		if len(out) == 0 {
			b.Fatalf("empty output")
		}
	}
}

func BenchmarkValidate(b *testing.B) {
	raw, err := os.ReadFile("testdata/two-role.json")
	if err != nil {
		b.Fatal(err)
	}
	intent, err := harnessmodelset.ParseJSON(raw)
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := intent.Validate(); err != nil {
			b.Fatalf("intent.Validate(): %v", err)
		}
	}
}
