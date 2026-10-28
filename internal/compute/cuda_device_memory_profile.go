//go:build cuda && cgo

package compute

/*
#include <cuda_runtime_api.h>
*/
import "C"

// DeviceMemoryProfile reports only observed facts for the ordinary initialized
// backend. Its identity is device zero, not a calling thread's current device.
// Rank-capable builds refuse even before their process group is initialized.
func (c *cudaBackend) DeviceMemoryProfile() (DeviceMemoryProfile, bool) {
	if c == nil || c != cudaDev {
		return DeviceMemoryProfile{}, false
	}
	if _, rank := any(c).(RankUploader); rank {
		return DeviceMemoryProfile{}, false
	}
	if _, group := any(c).(ProcessGroupBackend); group {
		return DeviceMemoryProfile{}, false
	}
	// cudaMu is the plain device mutex. Do not read the timing profile or use
	// device/context selection, allocation, synchronization or kernel operations.
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return cudaDeviceMemoryProfileFromFacts(c.tier, cudaQueryDeviceMemoryFacts(0))
}

// Explicit-device getters can initialize CUDA internally or report a preceding
// asynchronous error. Each failed query therefore leaves its fact unknown.
func cudaQueryDeviceMemoryFacts(device int) cudaDeviceMemoryFacts {
	var cache, warp C.int
	cacheStatus := C.cudaDeviceGetAttribute(&cache, C.cudaDevAttrL2CacheSize, C.int(device))
	warpStatus := C.cudaDeviceGetAttribute(&warp, C.cudaDevAttrWarpSize, C.int(device))
	return cudaDeviceMemoryFacts{
		cacheBytes: int64(cache), cacheOK: cacheStatus == C.cudaSuccess,
		warpWidth: int64(warp), warpOK: warpStatus == C.cudaSuccess,
	}
}
