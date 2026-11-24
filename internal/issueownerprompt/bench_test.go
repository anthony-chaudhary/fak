package issueownerprompt

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testenv/repotest"
)

func BenchmarkIssueOwnerPrompt(b *testing.B) {
	dir := repotest.Path(b, ".claude/goal-prompts")

	if err := ValidateDir(dir); err != nil {
		b.Fatalf("ValidateDir failed: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if err := ValidateDir(dir); err != nil {
			b.Fatal(err)
		}
	}
}
