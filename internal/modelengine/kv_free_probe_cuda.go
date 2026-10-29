//go:build cuda && cgo

package modelengine

import "github.com/anthony-chaudhary/fak/internal/compute"

// The CUDA build supplies the paged-KV free-bytes probe from cudaMemGetInfo. It is
// consulted only under FAK_NATIVE_KV_AUTOSIZE (nativeKVFreeBytesProbeFromEnv), so
// a CUDA serve keeps preemption off by default.
func init() {
	nativeKVFreeBytesProbe = cudaKVFreeBytes
}

func cudaKVFreeBytes() (uint64, bool) {
	b, ok := compute.Lookup("cuda")
	if !ok {
		return 0, false
	}
	return kvFreeBytesFromBackend(b)
}
