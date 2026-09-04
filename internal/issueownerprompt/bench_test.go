package issueownerprompt

import (
	"path/filepath"
	"runtime"
	"testing"
)

func BenchmarkIssueOwnerPrompt(b *testing.B) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		b.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	dir := filepath.Join(root, ".claude", "goal-prompts")

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
