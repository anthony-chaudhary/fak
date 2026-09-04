package procguard

import (
	"os"
	"runtime"
	"testing"
)

func TestYieldMemoryBasic(t *testing.T) {
	// Calling YieldMemory with no pids should run GC, FreeOSMemory, and yield current working set
	YieldMemory()
}

func TestYieldMemoryWithInvalidPIDs(t *testing.T) {
	// Negative and zero PIDs should be handled safely and ignored
	YieldMemory(0, -1, -9999)
}

func TestYieldMemoryWithCurrentPID(t *testing.T) {
	// Passing current PID should succeed without issue
	YieldMemory(os.Getpid())
}

func TestYieldMemoryMixedPIDs(t *testing.T) {
	// Mix of current PID, invalid PIDs, and high unlikely PIDs
	YieldMemory(os.Getpid(), 0, -5, 99999999)
}

func TestYieldMemoryHeapAllocation(t *testing.T) {
	// Allocate memory on the heap
	alloc := make([]byte, 20<<20) // 20 MB
	alloc[0] = 1
	alloc[len(alloc)-1] = 1

	var m1, m2 runtime.MemStats
	runtime.ReadMemStats(&m1)

	// Release reference to allocation and invoke YieldMemory
	alloc = nil
	_ = alloc
	YieldMemory()

	runtime.ReadMemStats(&m2)
	// Verify that GC has run by checking NumGC increased
	if m2.NumGC <= m1.NumGC {
		t.Fatalf("expected NumGC to increase after YieldMemory: before=%d after=%d", m1.NumGC, m2.NumGC)
	}
}
