package ultracodeborrow

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

func benchRepoRoot(tb testing.TB) string {
	tb.Helper()
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		tb.Fatal("runtime.Caller")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
}

func TestBenchmarkArtifactValidity(t *testing.T) {
	root := benchRepoRoot(t)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "notes", "CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json"))
	if err != nil {
		t.Fatal(err)
	}
	artifact, err := Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	if err := Validate(artifact); err != nil {
		t.Fatal(err)
	}
}

func BenchmarkUltraCodeBorrow(b *testing.B) {
	root := benchRepoRoot(b)
	raw, err := os.ReadFile(filepath.Join(root, "docs", "notes", "CONCEPT-STUDY-ULTRACODE-WORKFLOWS-2026-08-21.json"))
	if err != nil {
		b.Fatal(err)
	}
	artifact, err := Parse(raw)
	if err != nil {
		b.Fatal(err)
	}

	b.Run("FullPipeline", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			a, err := Parse(raw)
			if err != nil {
				b.Fatal(err)
			}
			if err := Validate(a); err != nil {
				b.Fatal(err)
			}
			if err := CheckPublicText("benchmark", raw); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("ValidateOnly", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := Validate(artifact); err != nil {
				b.Fatal(err)
			}
		}
	})

	b.Run("CheckPublicText", func(b *testing.B) {
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			if err := CheckPublicText("benchmark", raw); err != nil {
				b.Fatal(err)
			}
		}
	})
}
