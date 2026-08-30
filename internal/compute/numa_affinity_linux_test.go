//go:build linux && amd64

package compute

import (
	"runtime"
	"testing"
)

func TestPinCurrentThreadToCPUsRestoresPriorMask(t *testing.T) {
	runtime.LockOSThread()
	var processMask cpuAffinityMask
	if err := getCurrentThreadAffinity(&processMask); err != nil {
		runtime.UnlockOSThread()
		t.Fatal(err)
	}
	defer func() {
		if err := setCurrentThreadAffinity(&processMask); err != nil {
			t.Errorf("restore test process affinity: %v", err)
		}
		runtime.UnlockOSThread()
	}()

	cpus := affinityCPUs(processMask)
	if len(cpus) < 2 {
		t.Skip("affinity restoration test needs at least two permitted CPUs")
	}
	initial, err := affinityMaskForCPUs(cpus[:2])
	if err != nil {
		t.Fatal(err)
	}
	if err := setCurrentThreadAffinity(&initial); err != nil {
		t.Fatalf("set known initial affinity: %v", err)
	}

	unpin, err := PinCurrentThreadToCPUs(cpus[:1])
	if err != nil {
		t.Fatalf("pin current thread: %v", err)
	}
	wantPinned, _ := affinityMaskForCPUs(cpus[:1])
	if got := readAffinity(t); got != wantPinned {
		t.Fatalf("pinned affinity mismatch:\n got %x\nwant %x", got, wantPinned)
	}

	unpin()
	if got := readAffinity(t); got != initial {
		t.Fatalf("restored affinity mismatch:\n got %x\nwant %x", got, initial)
	}

	// A duplicate cleanup must neither change affinity nor perform a second unlock.
	unpin()
	if got := readAffinity(t); got != initial {
		t.Fatalf("affinity changed after duplicate unpin:\n got %x\nwant %x", got, initial)
	}
}

func TestPinCurrentThreadToCPUsApplyFailureRestoresAndUnlocks(t *testing.T) {
	runtime.LockOSThread()
	var original cpuAffinityMask
	if err := getCurrentThreadAffinity(&original); err != nil {
		runtime.UnlockOSThread()
		t.Fatal(err)
	}
	badCPU, ok := findRejectedCPU(t, original)
	if !ok {
		runtime.UnlockOSThread()
		t.Skip("kernel accepted every representable single-CPU affinity mask")
	}
	before := readAffinity(t)

	unpin, err := PinCurrentThreadToCPUs([]int{badCPU})
	if err == nil {
		unpin()
		t.Fatalf("pin unexpectedly accepted unavailable CPU %d", badCPU)
	}
	unpin() // the error path returns a safe no-op cleanup

	// The outer thread lock remains held across the helper's nested lock. Byte-identical readback
	// proves the failed apply was cleaned up on the exact thread the helper attempted to modify.
	defer runtime.UnlockOSThread()
	if got := readAffinity(t); got != before {
		t.Fatalf("apply failure changed affinity:\n got %x\nwant %x", got, before)
	}
}

func TestPinCurrentThreadToCPUsRejectsInvalidSets(t *testing.T) {
	for _, tc := range []struct {
		name string
		cpus []int
	}{
		{name: "empty"},
		{name: "negative", cpus: []int{-1}},
		{name: "too large", cpus: []int{cpuSetWords * 64}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			unpin, err := PinCurrentThreadToCPUs(tc.cpus)
			if err == nil {
				unpin()
				t.Fatal("expected invalid CPU set to fail")
			}
			unpin()
		})
	}
}

func readAffinity(t *testing.T) cpuAffinityMask {
	t.Helper()
	var mask cpuAffinityMask
	if err := getCurrentThreadAffinity(&mask); err != nil {
		t.Fatal(err)
	}
	return mask
}

func affinityCPUs(mask cpuAffinityMask) []int {
	var cpus []int
	for cpu := 0; cpu < cpuSetWords*64; cpu++ {
		if mask[cpu/64]&(uint64(1)<<(uint(cpu)%64)) != 0 {
			cpus = append(cpus, cpu)
		}
	}
	return cpus
}

func findRejectedCPU(t *testing.T, original cpuAffinityMask) (int, bool) {
	t.Helper()
	for cpu := cpuSetWords*64 - 1; cpu >= 0; cpu-- {
		probe, _ := affinityMaskForCPUs([]int{cpu})
		if err := setCurrentThreadAffinity(&probe); err != nil {
			if restoreErr := setCurrentThreadAffinity(&original); restoreErr != nil {
				t.Fatalf("restore affinity after rejected CPU %d probe: %v", cpu, restoreErr)
			}
			return cpu, true
		}
		if err := setCurrentThreadAffinity(&original); err != nil {
			t.Fatalf("restore affinity after CPU %d probe: %v", cpu, err)
		}
	}
	return 0, false
}
