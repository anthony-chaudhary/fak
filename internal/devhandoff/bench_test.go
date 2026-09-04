package devhandoff

import "testing"

func BenchmarkDevHandoff(b *testing.B) {
	queries := []string{"amd-setup", "backend", "build", "ci-preflight", "unknown-command", "index"}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		query := queries[i%len(queries)]
		_ = IsCommand(query)
	}
}

func TestBenchmarkDevHandoff(t *testing.T) {
	if !IsCommand("backend") {
		t.Fatal("expected backend to be a known command")
	}
	if IsCommand("non-existent-dev-command-xyz") {
		t.Fatal("expected unknown command to return false")
	}
}
