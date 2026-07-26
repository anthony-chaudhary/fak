//go:build linux && amd64

package compute

import (
	"errors"
	"runtime"
	"syscall"
	"unsafe"
)

// numa_affinity_linux.go — the pinning half of the per-node replica read path. ScheduleDecodeNUMA
// says which CPUs a decode worker may run on; this binds the worker's OS thread to exactly that
// set, so the worker actually executes on the node whose replica it reads. Without the pin the Go
// scheduler is free to migrate the goroutine to any core, and a worker holding node k's replica
// could run on node j — reintroducing the fabric read the replica exists to remove.
//
// cpuSetWords covers the kernel's cpu_set_t for the affinity syscall: 1024 CPU ids in 64-bit
// words, the glibc CPU_SETSIZE default and comfortably above this box's 256 threads.
const cpuSetWords = 1024 / 64

// PinCurrentThreadToCPUs wires the CALLING goroutine to its OS thread and restricts that thread
// to cpus via sched_setaffinity(2). It must be called from inside the worker goroutine whose
// placement it enforces, and the caller keeps the thread locked for the worker's lifetime —
// hence it does NOT unlock: returning the thread to the pool would drop the affinity that the
// whole node-local read path depends on. The returned unpin releases the OS thread when the
// worker retires.
//
// A refusal is never fatal to decode: on error the caller keeps running unpinned, reads stay
// byte-correct, and only the locality claim is forfeited (the same fail-visible discipline the
// replica allocator uses for a failed bind).
func PinCurrentThreadToCPUs(cpus []int) (unpin func(), err error) {
	if len(cpus) == 0 {
		return func() {}, errors.New("compute: empty CPU set — nothing to pin to")
	}
	var mask [cpuSetWords]uint64
	set := 0
	for _, c := range cpus {
		if c < 0 || c >= cpuSetWords*64 {
			continue // a CPU id outside the mask cannot be expressed; skip rather than corrupt it
		}
		mask[c/64] |= 1 << (uint(c) % 64)
		set++
	}
	if set == 0 {
		return func() {}, errors.New("compute: no representable CPU ids in set")
	}

	runtime.LockOSThread()
	// tid 0 means "the calling thread", which is exactly the thread LockOSThread just pinned us to.
	_, _, errno := syscall.RawSyscall(
		syscall.SYS_SCHED_SETAFFINITY,
		0,
		uintptr(len(mask)*8),
		uintptr(unsafe.Pointer(&mask[0])),
	)
	if errno != 0 {
		runtime.UnlockOSThread()
		return func() {}, errno
	}
	return runtime.UnlockOSThread, nil
}
