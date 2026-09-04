package dependencyquarantine

import (
	"strings"
	"testing"
)

func BenchmarkDependencyQuarantine(b *testing.B) {
	root := repoRootBenchmark(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		violations, err := Check(root)
		if err != nil {
			b.Fatalf("Check failed: %v", err)
		}
		if len(violations) > 0 {
			b.Fatalf("unexpected repository violations: %v", violations)
		}
	}
}

func BenchmarkCheckRootRequirements(b *testing.B) {
	root := fixtureBenchmark(b, "module example.test/root\n\ngo 1.26\n\nrequire (\n golang.org/x/sys v0.46.0\n golang.org/x/term v0.44.0\n)\n", strings.Join(keys(allowedRootSum), "\n")+"\n")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		violations, err := Check(root)
		if err != nil {
			b.Fatalf("Check failed: %v", err)
		}
		if len(violations) > 0 {
			b.Fatalf("unexpected fixture violations: %v", violations)
		}
	}
}
