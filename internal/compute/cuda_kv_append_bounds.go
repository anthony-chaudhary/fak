package compute

// The CUDA KV-write ABI takes signed 32-bit scalar offsets and counts. Keep the
// ordinary append path within the strict sibling's offset+n <= INT_MAX contract.
const cudaKVMaxCopyFloats = 1<<31 - 1

func cudaKVAppendBoundsError(site, msg string) error {
	return &CUDAOpError{Op: "AppendKV", Site: site, Msg: msg, Class: MemoryKVCache}
}

// cudaKVAppendCapacity checks arithmetic before any allocation, copy, or mutation.
// It also caps geometric growth before multiplying by the F32 byte width.
func cudaKVAppendCapacity(length, capacity, count int, site string) (int, error) {
	maxByteFloats := int(^uint(0)>>1) / 4
	maxFloats := min(cudaKVMaxCopyFloats, maxByteFloats)
	if length < 0 || capacity < length || count < 0 {
		return 0, cudaKVAppendBoundsError(site, "invalid KV length, capacity, or append count")
	}
	if capacity > maxByteFloats {
		return 0, cudaKVAppendBoundsError(site, "KV capacity byte size overflows host int")
	}
	if length > maxFloats || count > maxFloats-length {
		return 0, cudaKVAppendBoundsError(site, "KV append exceeds signed 32-bit copy or host byte-size limit")
	}
	if length+count <= capacity {
		return capacity, nil
	}
	// capacity*2+count can overflow even when the requested append fits. Saturate
	// spare capacity at the checked limit without changing the required length.
	if capacity > (maxFloats-count)/2 {
		return maxFloats, nil
	}
	return capacity*2 + count, nil
}
