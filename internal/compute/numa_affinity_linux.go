//go:build linux && amd64

package compute

import (
	"errors"
	"fmt"
	"runtime"
	"sync"
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

type cpuAffinityMask [cpuSetWords]uint64

// PinCurrentThreadToCPUs wires the CALLING goroutine to its OS thread and restricts that thread
// to cpus via sched_setaffinity(2). It must be called from inside the worker goroutine whose
// placement it enforces. The returned idempotent unpin restores the exact mask observed after
// LockOSThread and only then releases the OS thread; UnlockOSThread does not reset affinity.
//
// If restoring affinity fails, unpin panics without unlocking. Releasing a thread with a stale
// NUMA mask would make it unsafe for unrelated goroutines; the panic makes that poisoned-thread
// state visible while keeping the thread locked until its goroutine exits.
//
// A refusal is never fatal to decode: on an apply error the original mask is restored before the
// thread is unlocked. If that cleanup itself fails, the thread remains locked and the returned
// error reports both failures. In either case byte correctness is preserved and only the locality
// claim is forfeited.
func PinCurrentThreadToCPUs(cpus []int) (unpin func(), err error) {
	mask, err := affinityMaskForCPUs(cpus)
	if err != nil {
		return func() {}, err
	}

	runtime.LockOSThread()
	var prior cpuAffinityMask
	if err := getCurrentThreadAffinity(&prior); err != nil {
		runtime.UnlockOSThread()
		return func() {}, err
	}
	if err := setCurrentThreadAffinity(&mask); err != nil {
		if restoreErr := setCurrentThreadAffinity(&prior); restoreErr != nil {
			// Keep the modified or otherwise unknown thread locked. When this goroutine exits,
			// the runtime terminates the locked OS thread instead of returning it to the pool.
			return func() {}, errors.Join(err, fmt.Errorf("compute: restore affinity after apply failure: %w", restoreErr))
		}
		runtime.UnlockOSThread()
		return func() {}, err
	}

	var once sync.Once
	return func() {
		once.Do(func() {
			if err := setCurrentThreadAffinity(&prior); err != nil {
				panic(fmt.Sprintf("compute: restore thread affinity before unlock: %v", err))
			}
			runtime.UnlockOSThread()
		})
	}, nil
}

func affinityMaskForCPUs(cpus []int) (cpuAffinityMask, error) {
	if len(cpus) == 0 {
		return cpuAffinityMask{}, errors.New("compute: empty CPU set — nothing to pin to")
	}
	var mask cpuAffinityMask
	set := 0
	for _, c := range cpus {
		if c < 0 || c >= cpuSetWords*64 {
			continue // a CPU id outside the mask cannot be expressed; skip rather than corrupt it
		}
		mask[c/64] |= 1 << (uint(c) % 64)
		set++
	}
	if set == 0 {
		return cpuAffinityMask{}, errors.New("compute: no representable CPU ids in set")
	}
	return mask, nil
}

func getCurrentThreadAffinity(mask *cpuAffinityMask) error {
	return currentThreadAffinity(syscall.SYS_SCHED_GETAFFINITY, "read", mask)
}

func setCurrentThreadAffinity(mask *cpuAffinityMask) error {
	return currentThreadAffinity(syscall.SYS_SCHED_SETAFFINITY, "set", mask)
}

func currentThreadAffinity(operation uintptr, action string, mask *cpuAffinityMask) error {
	_, _, errno := syscall.RawSyscall(operation, 0, uintptr(len(mask)*8), uintptr(unsafe.Pointer(&mask[0])))
	if errno != 0 {
		return fmt.Errorf("compute: %s current thread affinity: %w", action, errno)
	}
	return nil
}
