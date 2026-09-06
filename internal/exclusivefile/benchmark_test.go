package exclusivefile

import (
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

// BenchmarkCreatePIDTime measures uncontended atomic creation and removal of
// the process marker file in a loop.
func BenchmarkCreatePIDTime(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "marker.lock")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := CreatePIDTime(path); err != nil {
			b.Fatalf("CreatePIDTime: %v", err)
		}
		if err := os.Remove(path); err != nil {
			b.Fatalf("Remove: %v", err)
		}
	}
}

// BenchmarkCreatePIDTimeUnique measures atomic creation of distinct process marker
// files across independent paths, isolating filesystem creation and serialization
// without immediate file-path deletion reuse.
func BenchmarkCreatePIDTimeUnique(b *testing.B) {
	dir := b.TempDir()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		p := filepath.Join(dir, "marker_"+strconv.Itoa(i)+".lock")
		if err := CreatePIDTime(p); err != nil {
			b.Fatalf("CreatePIDTime: %v", err)
		}
	}
}

// BenchmarkExclusiveFileLockContention measures rejection latency when competing
// processes attempt to create an already-held exclusive lock marker file.
func BenchmarkExclusiveFileLockContention(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "held.lock")
	if err := CreatePIDTime(path); err != nil {
		b.Fatalf("setup held lock: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := CreatePIDTime(path)
		if !os.IsExist(err) {
			b.Fatalf("expected existing file error, got: %v", err)
		}
	}
}

// BenchmarkExclusiveFileLockContentionParallel measures concurrent contention
// throughput across multiple parallel goroutines attempting to acquire an already-held
// exclusive lock file.
func BenchmarkExclusiveFileLockContentionParallel(b *testing.B) {
	dir := b.TempDir()
	path := filepath.Join(dir, "parallel_held.lock")
	if err := CreatePIDTime(path); err != nil {
		b.Fatalf("setup held lock: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			err := CreatePIDTime(path)
			if !os.IsExist(err) {
				b.Errorf("expected existing file error, got: %v", err)
				return
			}
		}
	})
}
