package flock

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// TestLockUnlockLifecycle exercises repeated lock acquisition and release on a
// single open handle to ensure the advisory lock is completely cleared between cycles.
func TestLockUnlockLifecycle(t *testing.T) {
	path := filepath.Join(t.TempDir(), "lifecycle.lock")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer f.Close()

	for i := 0; i < 20; i++ {
		if err := TryLock(f); err != nil {
			t.Fatalf("cycle %d: TryLock failed: %v", i, err)
		}
		if err := Unlock(f); err != nil {
			t.Fatalf("cycle %d: Unlock failed: %v", i, err)
		}
	}
}

// BenchmarkTryLockUnlock measures the baseline latency and allocation profile of
// taking and immediately releasing an uncontended advisory lock on an open handle.
func BenchmarkTryLockUnlock(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_uncontended.lock")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open lockfile: %v", err)
	}
	defer f.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TryLock(f); err != nil {
			b.Fatalf("TryLock: %v", err)
		}
		if err := Unlock(f); err != nil {
			b.Fatalf("Unlock: %v", err)
		}
	}
}

// BenchmarkContendedTryLock measures the fast-refusal overhead when another
// handle holds the advisory lock and TryLock returns ErrLockBusy.
func BenchmarkContendedTryLock(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_contended.lock")
	holder, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open holder: %v", err)
	}
	defer holder.Close()

	if err := TryLock(holder); err != nil {
		b.Fatalf("hold lock: %v", err)
	}
	defer func() { _ = Unlock(holder) }()

	contender, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open contender: %v", err)
	}
	defer contender.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		err := TryLock(contender)
		if !errors.Is(err, ErrLockBusy) {
			b.Fatalf("expected ErrLockBusy, got %v", err)
		}
	}
}

// BenchmarkAlternatingHandles measures the handoff latency of transferring
// lock ownership between two independent file handles on the same lockfile.
func BenchmarkAlternatingHandles(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_alt.lock")
	h1, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open h1: %v", err)
	}
	defer h1.Close()

	h2, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open h2: %v", err)
	}
	defer h2.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TryLock(h1); err != nil {
			b.Fatalf("h1 TryLock: %v", err)
		}
		if err := Unlock(h1); err != nil {
			b.Fatalf("h1 Unlock: %v", err)
		}
		if err := TryLock(h2); err != nil {
			b.Fatalf("h2 TryLock: %v", err)
		}
		if err := Unlock(h2); err != nil {
			b.Fatalf("h2 Unlock: %v", err)
		}
	}
}

// BenchmarkCriticalSection measures a production critical section cycle: acquiring
// an exclusive lock, updating metadata at offset 0 (e.g. PID or sequence), and releasing.
func BenchmarkCriticalSection(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_cs.lock")
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("open file: %v", err)
	}
	defer f.Close()

	payload := []byte("pid=12345 status=running\n")

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := TryLock(f); err != nil {
			b.Fatalf("TryLock: %v", err)
		}
		if _, err := f.WriteAt(payload, 0); err != nil {
			b.Fatalf("WriteAt: %v", err)
		}
		if err := Unlock(f); err != nil {
			b.Fatalf("Unlock: %v", err)
		}
	}
}

// BenchmarkConcurrentTryLockContention measures concurrent contending handles
// attempting non-blocking acquires in parallel with retry/backoff.
func BenchmarkConcurrentTryLockContention(b *testing.B) {
	path := filepath.Join(b.TempDir(), "bench_concurrent.lock")
	initF, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o644)
	if err != nil {
		b.Fatalf("init lockfile: %v", err)
	}
	initF.Close()

	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		f, err := os.OpenFile(path, os.O_RDWR, 0o644)
		if err != nil {
			b.Errorf("open lockfile: %v", err)
			return
		}
		defer f.Close()

		for pb.Next() {
			for {
				err := TryLock(f)
				if err == nil {
					_ = Unlock(f)
					break
				}
				if !errors.Is(err, ErrLockBusy) {
					b.Errorf("unexpected error: %v", err)
					return
				}
				runtime.Gosched()
			}
		}
	})
}
