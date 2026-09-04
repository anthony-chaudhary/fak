package patchcommit

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// BenchmarkPatchCommit measures end-to-end patch commit transaction throughput.
func BenchmarkPatchCommit(b *testing.B) {
	repo := newBenchRepo(b)
	filePath := filepath.Join(repo, "bench.txt")
	if err := os.WriteFile(filePath, []byte("v0\n"), 0644); err != nil {
		b.Fatal(err)
	}
	gitBench(b, repo, "add", "bench.txt")
	gitBench(b, repo, "commit", "-m", "init")

	patchFile := filepath.Join(repo, "bench.patch")
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		nextContent := fmt.Sprintf("v%d\n", i+1)
		if err := os.WriteFile(filePath, []byte(nextContent), 0644); err != nil {
			b.Fatal(err)
		}
		patch := fmt.Sprintf("diff --git a/bench.txt b/bench.txt\n--- a/bench.txt\n+++ b/bench.txt\n@@ -1 +1 @@\n-v%d\n+v%d\n", i, i+1)
		if err := os.WriteFile(patchFile, []byte(patch), 0644); err != nil {
			b.Fatal(err)
		}

		res, err := Commit(ctx, Options{
			Dir:       repo,
			PatchFile: patchFile,
			Paths:     []string{"bench.txt"},
			Message:   fmt.Sprintf("feat(bench): iteration %d", i),
		})
		if err != nil || res.Reason != "" || res.SHA == "" {
			b.Fatalf("Commit iter %d failed: res=%+v, err=%v", i, res, err)
		}
	}
}

// BenchmarkNormalizePaths measures path sanitization and deduplication throughput.
func BenchmarkNormalizePaths(b *testing.B) {
	raw := []string{
		"a/b/c.go",
		"internal/pkg/foo.go",
		"./internal/pkg/foo.go",
		"cmd/fak/main.go",
		"a/b/c.go",
		"internal/safecommit/runner.go",
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		paths, ok := normalizePaths(raw)
		if !ok || len(paths) != 4 {
			b.Fatalf("unexpected normalizePaths result: ok=%v len=%d", ok, len(paths))
		}
	}
}

// BenchmarkForbiddenPatchForm measures validation throughput for forbidden patch patterns.
func BenchmarkForbiddenPatchForm(b *testing.B) {
	patch := []byte(`diff --git a/pkg/foo.go b/pkg/foo.go
--- a/pkg/foo.go
+++ b/pkg/foo.go
@@ -10,2 +10,3 @@
 func Run() {
+	println("ok")
 }
`)
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if forbiddenPatchForm(patch) {
			b.Fatal("unexpected forbidden patch result")
		}
	}
}

// TestBenchmarkPatchCommit verifies the benchmark workflow in unit test mode.
func TestBenchmarkPatchCommit(t *testing.T) {
	repo := newBenchRepo(t)
	filePath := filepath.Join(repo, "bench.txt")
	if err := os.WriteFile(filePath, []byte("v0\n"), 0644); err != nil {
		t.Fatal(err)
	}
	gitBench(t, repo, "add", "bench.txt")
	gitBench(t, repo, "commit", "-m", "init")

	patchFile := filepath.Join(repo, "bench.patch")
	ctx := context.Background()

	for i := 0; i < 3; i++ {
		nextContent := fmt.Sprintf("v%d\n", i+1)
		if err := os.WriteFile(filePath, []byte(nextContent), 0644); err != nil {
			t.Fatal(err)
		}
		patch := fmt.Sprintf("diff --git a/bench.txt b/bench.txt\n--- a/bench.txt\n+++ b/bench.txt\n@@ -1 +1 @@\n-v%d\n+v%d\n", i, i+1)
		if err := os.WriteFile(patchFile, []byte(patch), 0644); err != nil {
			t.Fatal(err)
		}

		res, err := Commit(ctx, Options{
			Dir:       repo,
			PatchFile: patchFile,
			Paths:     []string{"bench.txt"},
			Message:   fmt.Sprintf("feat(bench): iteration %d", i),
		})
		if err != nil || res.Reason != "" || res.SHA == "" {
			t.Fatalf("Commit iter %d failed: res=%+v, err=%v", i, res, err)
		}
	}
}

func newBenchRepo(tb testing.TB) string {
	tb.Helper()
	d := tb.TempDir()
	gitBench(tb, d, "init", "-b", "main")
	gitBench(tb, d, "config", "user.name", "bench")
	gitBench(tb, d, "config", "user.email", "bench@example.test")
	return d
}

func gitBench(tb testing.TB, dir string, args ...string) string {
	tb.Helper()
	c := exec.Command("git", args...)
	c.Dir = dir
	c.Env = append(os.Environ(), "GIT_CONFIG_NOSYSTEM=1")
	out, err := c.CombinedOutput()
	if err != nil {
		tb.Fatalf("git %s: %v\n%s", strings.Join(args, " "), err, out)
	}
	return string(out)
}
