package stamp

import (
	"fmt"
	"testing"
)

// BenchmarkParseVersionJSON measures the smoke/provenance identity parse of a
// candidate's version --json document through ParseVersionJSON, including one
// unknown field that must be ignored.
func BenchmarkParseVersionJSON(b *testing.B) {
	const commit = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
	doc := []byte(fmt.Sprintf(
		`{"app_version":"0.55.0","commit":"%s","dirty":false,"stamped":true,"extra":{"x":1}}`,
		commit,
	))
	id, err := ParseVersionJSON(doc)
	if err != nil || id.Commit != commit {
		b.Fatalf("fixture ParseVersionJSON = (%+v, %v), want commit %q without error", id, err, commit)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = ParseVersionJSON(doc)
	}
}
