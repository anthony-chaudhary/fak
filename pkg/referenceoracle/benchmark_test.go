package referenceoracle_test

// These benchmarks measure referenceoracle's production decode paths:
// Parse over in-memory bytes (legacy V4.1 fixture and a minimal native
// document) and ReadFile over the legacy fixture. They run only under
// `go test -bench` and exercise no network or hardware.

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/referenceoracle"
)

// benchSinkOracle keeps decoded results live so the compiler cannot
// optimize the decode calls away.
var benchSinkOracle referenceoracle.Oracle

// benchNativeDoc is a minimal valid fak.reference_oracle.v1 document.
const benchNativeDoc = `{
  "schema": "fak.reference_oracle.v1",
  "reference": {"repo": "example/ref-engine", "sha": "abc123def456"},
  "artifact": {"repo": "example/checkpoint"},
  "prompts": [
    {
      "text": "bench prompt",
      "input_token_ids": [0, 1],
      "greedy_ids": [2],
      "top5": [[{"id": 2, "logprob": -0.5}]]
    }
  ],
  "self_parity": {"checked": true, "identical": true}
}`

func loadBenchFixture(b *testing.B, path string) []byte {
	data, err := os.ReadFile(path)
	if err != nil {
		b.Fatal(err)
	}
	return data
}

func BenchmarkParseLegacyV41(b *testing.B) {
	fixture := loadBenchFixture(b, "testdata/v41-reference-oracle.json")
	if _, err := referenceoracle.Parse(fixture); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(fixture)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, _ := referenceoracle.Parse(fixture)
		benchSinkOracle = o
	}
}

func BenchmarkParseNative(b *testing.B) {
	doc := []byte(benchNativeDoc)
	if _, err := referenceoracle.Parse(doc); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(doc)))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, _ := referenceoracle.Parse(doc)
		benchSinkOracle = o
	}
}

func BenchmarkReadFileLegacyV41(b *testing.B) {
	info, err := os.Stat("testdata/v41-reference-oracle.json")
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(info.Size()))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		o, err := referenceoracle.ReadFile("testdata/v41-reference-oracle.json")
		if err != nil {
			b.Fatal(err)
		}
		benchSinkOracle = o
	}
}
