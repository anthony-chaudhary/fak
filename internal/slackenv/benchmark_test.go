package slackenv

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

var (
	benchSinkResolved Resolved
	benchSinkString   string
	benchSinkBool     bool
)

const benchEnvBody = `# Production-representative .env.slack.local configuration for fak
export FAK_SCOREBOARD_TOKEN=xoxb-1234567890-abcdef1234567890
FAK_SCOREBOARD_CHANNEL=C0123456789
export FAK_BLOCKERS_CHANNEL=C0987654321
FAK_DISPATCH_CHANNEL=C1122334455
FAK_ALERT_WEBHOOK=https://hooks.slack.com/services/T00/B00/XXXX
FAK_CHATOPS_ADMINS=U123456,U789012
FAK_OTHER_VAR=some_value_with=equals_sign
`

// BenchmarkLookupEnv measures the hot-path lookup latency when the configuration
// key is already present in the process environment.
func BenchmarkLookupEnv(b *testing.B) {
	const testKey = "FAK_BENCH_TOKEN_ENV"
	b.Setenv(testKey, "xoxb-bench-env-token-value")
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchSinkResolved = Lookup(testKey)
	}
}

// BenchmarkLookupFile measures resolution when the key must be read and parsed
// from a local .env.slack.local file in the working directory.
func BenchmarkLookupFile(b *testing.B) {
	dir := b.TempDir()
	writeEnvFile(b, dir, benchEnvBody)
	b.Chdir(dir)

	const testKey = "FAK_BLOCKERS_CHANNEL"
	b.Setenv(testKey, "") // ensure environment does not short-circuit
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchSinkResolved = Lookup(testKey)
	}
}

// BenchmarkLookupUnset measures the full-miss resolution cost when a key is
// neither in the process environment nor in any .env.slack.local file.
func BenchmarkLookupUnset(b *testing.B) {
	dir := b.TempDir()
	b.Chdir(dir)

	const testKey = "FAK_UNSET_CHANNEL_KEY"
	b.Setenv(testKey, "")
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchSinkResolved = Lookup(testKey)
	}
}

// BenchmarkFileValueDirect measures direct file lookup when .env.slack.local
// is present in the starting directory without requiring an ancestor walk.
func BenchmarkFileValueDirect(b *testing.B) {
	dir := b.TempDir()
	writeEnvFile(b, dir, benchEnvBody)

	const testKey = "FAK_SCOREBOARD_CHANNEL"
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchSinkString = fileValueFrom(dir, testKey)
	}
}

// BenchmarkFileValueWalkUp measures ancestor directory traversal cost across
// varying nested directory depths (1, 3, and 5 levels deep).
func BenchmarkFileValueWalkUp(b *testing.B) {
	root := b.TempDir()
	writeEnvFile(b, root, benchEnvBody)

	for _, depth := range []int{1, 3, 5} {
		nested := root
		for d := 0; d < depth; d++ {
			nested = filepath.Join(nested, fmt.Sprintf("level%d", d+1))
		}
		if err := os.MkdirAll(nested, 0o755); err != nil {
			b.Fatalf("mkdir depth %d: %v", depth, err)
		}

		b.Run(fmt.Sprintf("Depth_%d", depth), func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkString = fileValueFrom(nested, "FAK_DISPATCH_CHANNEL")
			}
		})
	}
}

// BenchmarkFileValueNearerBlankOverride measures walk-up performance when a nearer
// directory defines KEY= to explicitly suppress an ancestor's value.
func BenchmarkFileValueNearerBlankOverride(b *testing.B) {
	root := b.TempDir()
	writeEnvFile(b, root, "FAK_OVERRIDE_KEY=ancestor-value\n")

	nested := filepath.Join(root, "sub", "pkg")
	if err := os.MkdirAll(nested, 0o755); err != nil {
		b.Fatalf("mkdir: %v", err)
	}
	writeEnvFile(b, filepath.Join(root, "sub"), "FAK_OVERRIDE_KEY=\n")

	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		benchSinkString = fileValueFrom(nested, "FAK_OVERRIDE_KEY")
	}
}

// BenchmarkScanFile measures file reading and line scanning throughput across
// first-line, middle-line, last-line, and absent key positions.
func BenchmarkScanFile(b *testing.B) {
	dir := b.TempDir()
	envPath := filepath.Join(dir, EnvFileName)
	if err := os.WriteFile(envPath, []byte(benchEnvBody), 0o600); err != nil {
		b.Fatalf("write env file: %v", err)
	}

	cases := []struct {
		name string
		key  string
	}{
		{"FirstLine", "FAK_SCOREBOARD_TOKEN"},
		{"MiddleLine", "FAK_DISPATCH_CHANNEL"},
		{"LastLine", "FAK_OTHER_VAR"},
		{"Absent", "FAK_NONEXISTENT_KEY"},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchSinkString, benchSinkBool = scanFile(envPath, tc.key)
			}
		})
	}
}

// BenchmarkResolve measures Resolution dispatch when hitting env, file, or falling back.
func BenchmarkResolve(b *testing.B) {
	dir := b.TempDir()
	writeEnvFile(b, dir, benchEnvBody)
	b.Chdir(dir)

	fallbackFn := func() string {
		return "fallback-channel-id"
	}

	b.Run("EnvHit", func(b *testing.B) {
		const key = "FAK_RESOLVE_ENV_KEY"
		b.Setenv(key, "env-channel-id")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = Resolve(key, fallbackFn)
		}
	})

	b.Run("FileHit", func(b *testing.B) {
		const key = "FAK_SCOREBOARD_CHANNEL"
		b.Setenv(key, "")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = Resolve(key, fallbackFn)
		}
	})

	b.Run("FallbackHit", func(b *testing.B) {
		const key = "FAK_RESOLVE_UNSET_KEY"
		b.Setenv(key, "")
		b.ReportAllocs()
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			benchSinkString = Resolve(key, fallbackFn)
		}
	})
}

// BenchmarkLookupParallel measures concurrent lookup throughput across goroutines.
func BenchmarkLookupParallel(b *testing.B) {
	const key = "FAK_PARALLEL_TOKEN"
	b.Setenv(key, "xoxb-parallel-token-12345")
	b.ReportAllocs()
	b.ResetTimer()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			res := Lookup(key)
			if !res.Set() {
				b.Fatal("expected key to be set")
			}
		}
	})
}

// writeEnvFile writes a .env.slack.local file in dir for benchmarks.
func writeEnvFile(b *testing.B, dir, body string) {
	b.Helper()
	if err := os.WriteFile(filepath.Join(dir, EnvFileName), []byte(body), 0o600); err != nil {
		b.Fatalf("write env file: %v", err)
	}
}

// TestBenchmarkOperationsSanity verifies that all benchmark scenarios execute
// and produce valid results under unit test runs.
func TestBenchmarkOperationsSanity(t *testing.T) {
	dir := t.TempDir()
	writeEnv(t, dir, benchEnvBody)
	t.Chdir(dir)

	// Env hit
	t.Setenv("FAK_SANITY_KEY", "sanity-val")
	if r := Lookup("FAK_SANITY_KEY"); r.Value != "sanity-val" || r.Source != SourceEnv {
		t.Fatalf("Lookup env: got %+v", r)
	}

	// File hit
	t.Setenv("FAK_SCOREBOARD_CHANNEL", "")
	if r := Lookup("FAK_SCOREBOARD_CHANNEL"); r.Value != "C0123456789" || r.Source != SourceFile {
		t.Fatalf("Lookup file: got %+v", r)
	}

	// Unset
	t.Setenv("FAK_NONEXISTENT", "")
	if r := Lookup("FAK_NONEXISTENT"); r.Set() || r.Source != SourceUnset {
		t.Fatalf("Lookup unset: got %+v", r)
	}

	// Resolve fallback
	if got := Resolve("FAK_NONEXISTENT", func() string { return "fallback" }); got != "fallback" {
		t.Fatalf("Resolve fallback: got %q, want fallback", got)
	}
}
