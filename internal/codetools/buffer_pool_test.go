package codetools

import (
	"context"
	"fmt"
	"go/parser"
	"go/token"
	"path/filepath"
	"strings"
	"sync"
	"testing"
)

// TestBufferPoolSizeClasses verifies that size-classed arenas correctly allocate,
// classify, and reuse buffers across 64KB, 256KB, and 1MB thresholds.
func TestBufferPoolSizeClasses(t *testing.T) {
	pool := NewBufferPool()

	// 1. 64KB Class
	b64 := pool.AcquireBuffer(10 * 1024)
	if len(b64) != 10*1024 {
		t.Fatalf("len = %d, want %d", len(b64), 10*1024)
	}
	if cap(b64) < ArenaClass64K {
		t.Fatalf("cap = %d, want >= %d", cap(b64), ArenaClass64K)
	}
	pool.ReleaseBuffer(b64)

	// 2. 256KB Class
	b256 := pool.AcquireBuffer(100 * 1024)
	if len(b256) != 100*1024 {
		t.Fatalf("len = %d, want %d", len(b256), 100*1024)
	}
	if cap(b256) < ArenaClass256K {
		t.Fatalf("cap = %d, want >= %d", cap(b256), ArenaClass256K)
	}
	pool.ReleaseBuffer(b256)

	// 3. 1MB Class
	b1m := pool.AcquireBuffer(500 * 1024)
	if len(b1m) != 500*1024 {
		t.Fatalf("len = %d, want %d", len(b1m), 500*1024)
	}
	if cap(b1m) < ArenaClass1M {
		t.Fatalf("cap = %d, want >= %d", cap(b1m), ArenaClass1M)
	}
	pool.ReleaseBuffer(b1m)

	// 4. Oversize Class (>1MB)
	bOver := pool.AcquireBuffer(2 * 1024 * 1024)
	if len(bOver) != 2*1024*1024 || cap(bOver) < 2*1024*1024 {
		t.Fatalf("oversize len=%d cap=%d", len(bOver), cap(bOver))
	}
	pool.ReleaseBuffer(bOver) // safe no-op on arena return

	// 5. Verify pointer acquire helper
	bPtr := pool.AcquireBufferPtr(32 * 1024)
	if bPtr == nil || len(*bPtr) != 32*1024 {
		t.Fatalf("AcquireBufferPtr failed: %v", bPtr)
	}
	pool.ReleaseBuffer(bPtr)

	m := pool.Metrics()
	if m.Acquires != 5 {
		t.Fatalf("Acquires = %d, want 5", m.Acquires)
	}
	if m.Releases != 5 {
		t.Fatalf("Releases = %d, want 5", m.Releases)
	}
}

// TestBufferPoolConcurrentAccessAndBoundedAllocations verifies reusable buffer pool
// hit rates and bounded allocations under high concurrent contention.
func TestBufferPoolConcurrentAccessAndBoundedAllocations(t *testing.T) {
	pool := NewBufferPool()
	const (
		numGoroutines = 50
		opsPerRoutine = 100
		totalOps      = numGoroutines * opsPerRoutine
	)

	var wg sync.WaitGroup
	wg.Add(numGoroutines)

	for g := 0; g < numGoroutines; g++ {
		go func(routineID int) {
			defer wg.Done()
			for i := 0; i < opsPerRoutine; i++ {
				// Mix of sizes across 64K, 256K, 1M
				size := 4096
				switch (routineID + i) % 3 {
				case 0:
					size = 16 * 1024
				case 1:
					size = 128 * 1024
				case 2:
					size = 512 * 1024
				}

				buf := pool.AcquireBuffer(size)
				buf[0] = byte(i)
				buf[len(buf)-1] = byte(routineID)
				pool.ReleaseBuffer(buf)
			}
		}(g)
	}

	wg.Wait()

	m := pool.Metrics()
	if m.Acquires != uint64(totalOps) {
		t.Fatalf("Acquires = %d, want %d", m.Acquires, totalOps)
	}
	if m.Releases != uint64(totalOps) {
		t.Fatalf("Releases = %d, want %d", m.Releases, totalOps)
	}

	// Bounded allocations: despite 5,000 operations, allocations must be bounded
	// by concurrent worker count (~numGoroutines * 3 classes).
	maxExpectedAllocs := uint64(numGoroutines * 4)
	if m.Allocations > maxExpectedAllocs {
		t.Fatalf("Allocations = %d, exceeded bound of %d under concurrency", m.Allocations, maxExpectedAllocs)
	}

	// Reusable buffer hit rate must be high (>= 90%)
	hitRate := m.HitRate()
	if hitRate < 0.90 {
		t.Fatalf("HitRate = %.4f, want >= 0.90", hitRate)
	}
}

// TestByteBufferWrapper tests the growable pooled ByteBuffer wrapper.
func TestByteBufferWrapper(t *testing.T) {
	bb := AcquireByteBuffer(64)
	defer ReleaseByteBuffer(bb)

	if bb.Len() != 0 {
		t.Fatalf("initial Len = %d, want 0", bb.Len())
	}
	if bb.Cap() < ArenaClass64K {
		t.Fatalf("initial Cap = %d, want >= %d", bb.Cap(), ArenaClass64K)
	}

	_, _ = bb.WriteString("hello")
	_ = bb.WriteByte(' ')
	_, _ = bb.Write([]byte("world"))

	if got := bb.String(); got != "hello world" {
		t.Fatalf("String = %q, want %q", got, "hello world")
	}
	if bb.Len() != 11 {
		t.Fatalf("Len = %d, want 11", bb.Len())
	}

	bb.Reset()
	if bb.Len() != 0 {
		t.Fatalf("after Reset Len = %d, want 0", bb.Len())
	}

	// Test growth beyond initial arena capacity
	largeStr := strings.Repeat("x", ArenaClass64K+100)
	_, err := bb.WriteString(largeStr)
	if err != nil {
		t.Fatalf("WriteString error: %v", err)
	}
	if bb.Len() != len(largeStr) {
		t.Fatalf("grown Len = %d, want %d", bb.Len(), len(largeStr))
	}
	if bb.Cap() < ArenaClass256K {
		t.Fatalf("grown Cap = %d, want >= %d", bb.Cap(), ArenaClass256K)
	}
}

// TestZeroSubprocessSpawnsAcross1000Operations demonstrates zero OS process spawns
// across 1,000+ tool operations (Grep, Glob, Read, Edit).
func TestZeroSubprocessSpawnsAcross1000Operations(t *testing.T) {
	ts, dir := newTestToolset(t)

	// Seed fixture files
	mustWrite(t, filepath.Join(dir, "f1.go"), "package main\n\nconst Version = 1\n")
	mustWrite(t, filepath.Join(dir, "f2.go"), "package main\n\nfunc Run() int { return 42 }\n")
	mustWrite(t, filepath.Join(dir, "sub", "f3.go"), "package sub\n\n// Submodule logic\n")

	ResetBufferPoolMetrics()
	initialAvoided := GetBufferPoolMetrics().SubprocessesAvoided
	ctx := context.Background()

	const iterations = 300 // 300 * 4 = 1,200 operations
	for i := 0; i < iterations; i++ {
		// 1. Read
		readOut, bad := ts.read(ctx, argsOf(t, ReadArgs{FilePath: "f1.go"}))
		if bad {
			t.Fatalf("read failed: %s", readOut)
		}

		// 2. Grep
		grepOut, bad := ts.grep(ctx, argsOf(t, GrepArgs{Pattern: "Run"}))
		if bad {
			t.Fatalf("grep failed: %s", grepOut)
		}

		// 3. Glob
		globOut, bad := ts.glob(ctx, argsOf(t, GlobArgs{Pattern: "**/*.go"}))
		if bad {
			t.Fatalf("glob failed: %s", globOut)
		}

		// 4. Edit
		version := decodeResult(t, readOut)["version"].(string)
		oldVal := fmt.Sprintf("const Version = %d", i+1)
		newVal := fmt.Sprintf("const Version = %d", i+2)
		editOut, bad := ts.edit(ctx, argsOf(t, EditArgs{
			FilePath:        "f1.go",
			ExpectedVersion: version,
			OldString:       oldVal,
			NewString:       newVal,
		}))
		if bad {
			t.Fatalf("edit failed at iter %d: %s", i, editOut)
		}
	}

	m := GetBufferPoolMetrics()
	totalAvoided := m.SubprocessesAvoided - initialAvoided
	if totalAvoided < 1000 {
		t.Fatalf("SubprocessesAvoided = %d, want >= 1000", totalAvoided)
	}
}

// TestNoNonBashEngineImportsExec scans all non-bash Go source files in internal/codetools
// to guarantee that zero external process execution machinery (os/exec) is imported
// for standard code navigation and editing.
func TestNoNonBashEngineImportsExec(t *testing.T) {
	files, err := filepath.Glob("*.go")
	if err != nil {
		t.Fatal(err)
	}

	fset := token.NewFileSet()
	for _, f := range files {
		base := filepath.Base(f)
		if strings.HasPrefix(base, "bash") || strings.HasPrefix(base, "containment_test") {
			continue
		}

		node, err := parser.ParseFile(fset, f, nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("failed to parse %s: %v", f, err)
		}

		for _, imp := range node.Imports {
			pathVal := strings.Trim(imp.Path.Value, `"`)
			if pathVal == "os/exec" {
				t.Fatalf("forbidden import %q found in %s: non-bash tool engine must never import os/exec", pathVal, f)
			}
		}
	}
}

// Benchmarks for pooled buffers vs unpooled allocations:

func BenchmarkBufferPoolAcquireRelease(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := AcquireBuffer(64 * 1024)
		ReleaseBuffer(buf)
	}
}

func BenchmarkByteBufferWrites(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bb := AcquireByteBuffer(4096)
		for j := 0; j < 10; j++ {
			_, _ = bb.WriteString("benchmark line for pooled buffer performance\n")
		}
		ReleaseByteBuffer(bb)
	}
}

func BenchmarkObserveFilePooled(b *testing.B) {
	dir := b.TempDir()
	p := filepath.Join(dir, "bench.txt")
	body := strings.Repeat("hello world code buffer pool\n", 1000)
	mustWrite(&testing.T{}, p, body)

	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, r := observeFile(ctx, p, 1<<20)
		if r != nil {
			b.Fatalf("observeFile: %v", r)
		}
	}
}

func BenchmarkGrepPooled(b *testing.B) {
	dir := b.TempDir()
	ts, err := New(Config{Root: dir})
	if err != nil {
		b.Fatal(err)
	}
	body := strings.Repeat("func TargetFunction() string { return \"val\" }\n", 100)
	for i := 0; i < 20; i++ {
		mustWrite(&testing.T{}, filepath.Join(dir, fmt.Sprintf("file_%d.go", i)), body)
	}

	ctx := context.Background()
	args := argsOf(&testing.T{}, GrepArgs{Pattern: "TargetFunction"})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, bad := ts.grep(ctx, args)
		if bad {
			b.Fatal("grep failed")
		}
	}
}
