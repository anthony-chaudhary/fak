package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codetools"
)

func mustWriteFixture(t testing.TB, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir: %v", err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func argsOf(t testing.TB, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	return b
}

func decodeResultMap(t testing.TB, b []byte) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatalf("decode: %v (%s)", err, string(b))
	}
	return m
}

// TestAgentZeroSubprocessSpawnsAcross1000Operations demonstrates zero OS process
// spawns across 1,000+ operations in the agent codetools suite.
func TestAgentZeroSubprocessSpawnsAcross1000Operations(t *testing.T) {
	root := t.TempDir()
	ts, err := codetools.New(codetools.Config{Root: root})
	if err != nil {
		t.Fatalf("codetools.New: %v", err)
	}

	mustWriteFixture(t, filepath.Join(root, "main.go"), "package main\n\nconst Version = 1\n")
	mustWriteFixture(t, filepath.Join(root, "pkg", "service.go"), "package pkg\n\nfunc Execute() int { return 100 }\n")
	mustWriteFixture(t, filepath.Join(root, "pkg", "worker.go"), "package pkg\n\nfunc Work() {}\n")

	codetools.ResetBufferPoolMetrics()
	ctx := context.Background()

	const iterations = 300 // 300 * 4 = 1,200 operations
	for i := 0; i < iterations; i++ {
		// 1. Read operation
		readPayload, bad := ts.Read(ctx, argsOf(t, codetools.ReadArgs{FilePath: "main.go"}))
		if bad {
			t.Fatalf("read failed at iter %d: %s", i, readPayload)
		}
		version := decodeResultMap(t, readPayload)["version"].(string)

		// 2. Grep operation
		grepPayload, bad := ts.Grep(ctx, argsOf(t, codetools.GrepArgs{Pattern: "Execute"}))
		if bad {
			t.Fatalf("grep failed at iter %d: %s", i, grepPayload)
		}

		// 3. Glob operation
		globPayload, bad := ts.Glob(ctx, argsOf(t, codetools.GlobArgs{Pattern: "**/*.go"}))
		if bad {
			t.Fatalf("glob failed at iter %d: %s", i, globPayload)
		}

		// 4. Edit operation
		oldStr := fmt.Sprintf("const Version = %d", i+1)
		newStr := fmt.Sprintf("const Version = %d", i+2)
		editPayload, bad := ts.Edit(ctx, argsOf(t, codetools.EditArgs{
			FilePath:        "main.go",
			ExpectedVersion: version,
			OldString:       oldStr,
			NewString:       newStr,
		}))
		if bad {
			t.Fatalf("edit failed at iter %d: %s", i, editPayload)
		}
	}

	m := codetools.GetBufferPoolMetrics()
	if m.SubprocessesAvoided < 1000 {
		t.Fatalf("SubprocessesAvoided = %d, want >= 1000", m.SubprocessesAvoided)
	}
	if m.HitRate() < 0.90 {
		t.Fatalf("HitRate = %.4f, want >= 0.90", m.HitRate())
	}
}

// TestAgentBufferPoolConcurrentContention tests bounded allocations and high hit
// rate under concurrent access from agent worker routines.
func TestAgentBufferPoolConcurrentContention(t *testing.T) {
	root := t.TempDir()
	ts, err := codetools.New(codetools.Config{Root: root})
	if err != nil {
		t.Fatalf("codetools.New: %v", err)
	}

	for i := 0; i < 10; i++ {
		mustWriteFixture(t, filepath.Join(root, fmt.Sprintf("file_%d.go", i)),
			fmt.Sprintf("package fixture\n\nconst ID = %d\n", i))
	}

	codetools.ResetBufferPoolMetrics()
	const workers = 20
	const opsPerWorker = 50

	var wg sync.WaitGroup
	wg.Add(workers)
	ctx := context.Background()

	for w := 0; w < workers; w++ {
		go func(workerID int) {
			defer wg.Done()
			for i := 0; i < opsPerWorker; i++ {
				file := fmt.Sprintf("file_%d.go", (workerID+i)%10)
				_, bad := ts.Read(ctx, argsOf(t, codetools.ReadArgs{FilePath: file}))
				if bad {
					t.Errorf("worker %d read failed", workerID)
				}
				_, bad = ts.Grep(ctx, argsOf(t, codetools.GrepArgs{Pattern: "fixture"}))
				if bad {
					t.Errorf("worker %d grep failed", workerID)
				}
			}
		}(w)
	}

	wg.Wait()

	m := codetools.GetBufferPoolMetrics()
	if m.Acquires < workers*opsPerWorker*2 {
		t.Fatalf("Acquires = %d, want >= %d", m.Acquires, workers*opsPerWorker*2)
	}
	// Bounded allocations under concurrent access
	if m.Allocations > workers*4 {
		t.Fatalf("Allocations = %d, want <= %d", m.Allocations, workers*4)
	}
	if m.HitRate() < 0.85 {
		t.Fatalf("HitRate = %.4f, want >= 0.85", m.HitRate())
	}
}

func BenchmarkAgentCodeToolsRead(b *testing.B) {
	root := b.TempDir()
	ts, err := codetools.New(codetools.Config{Root: root})
	if err != nil {
		b.Fatal(err)
	}
	p := filepath.Join(root, "target.go")
	mustWriteFixture(b, p, strings.Repeat("package bench\nconst Data = 42\n", 500))

	ctx := context.Background()
	args := argsOf(b, codetools.ReadArgs{FilePath: "target.go"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, bad := ts.Read(ctx, args)
		if bad {
			b.Fatal("read failed")
		}
	}
}

func BenchmarkAgentCodeToolsGrep(b *testing.B) {
	root := b.TempDir()
	ts, err := codetools.New(codetools.Config{Root: root})
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < 15; i++ {
		mustWriteFixture(b, filepath.Join(root, fmt.Sprintf("item_%d.go", i)),
			strings.Repeat("func BenchmarkTarget() string { return \"ok\" }\n", 100))
	}

	ctx := context.Background()
	args := argsOf(b, codetools.GrepArgs{Pattern: "BenchmarkTarget"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, bad := ts.Grep(ctx, args)
		if bad {
			b.Fatal("grep failed")
		}
	}
}

func BenchmarkAgentCodeToolsGlob(b *testing.B) {
	root := b.TempDir()
	ts, err := codetools.New(codetools.Config{Root: root})
	if err != nil {
		b.Fatal(err)
	}
	for i := 0; i < 30; i++ {
		mustWriteFixture(b, filepath.Join(root, fmt.Sprintf("dir_%d", i%5), fmt.Sprintf("file_%d.go", i)), "package bench\n")
	}

	ctx := context.Background()
	args := argsOf(b, codetools.GlobArgs{Pattern: "**/*.go"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, bad := ts.Glob(ctx, args)
		if bad {
			b.Fatal("glob failed")
		}
	}
}
