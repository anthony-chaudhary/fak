package compute

import "unsafe"

func restoreCUDAKVAllocation(ptr *unsafe.Pointer, length, capacity *int, required int, allocate func() unsafe.Pointer, release func(unsafe.Pointer)) {
	if *ptr != nil && *capacity >= required {
		return
	}
	if *ptr != nil {
		old := *ptr
		// Retire ownership before release or allocation can panic: outer restore
		// cleanup must never retry a pointer already returned to the pool.
		*ptr, *length, *capacity = nil, 0, 0
		release(old)
	}
	*ptr = allocate()
	*capacity = required
}
