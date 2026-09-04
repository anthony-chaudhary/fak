package kimik3page

import (
	"strings"
	"testing"
)

func BenchmarkValidatePageContent(b *testing.B) {
	samplePage := strings.Join(DefaultRequiredTokens(), "\n") + "\n<p>Safe content without forbidden claims</p>\n"
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		report := ValidatePageContent(samplePage)
		if !report.Valid {
			b.Fatalf("unexpected invalid report: %+v", report)
		}
	}
}
