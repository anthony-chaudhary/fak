package compute

import "strconv"

func singleResourceCapExceeded(nbytes int, capBytes int64) bool {
	return capBytes > 0 && int64(nbytes) > capBytes
}

func formatVulkanResourceCapError(what string, nbytes int, capBytes, storageRange, allocationSize int64) string {
	return "compute: vulkan " + what + " allocation of " + strconv.Itoa(nbytes) +
		" bytes exceeds device single-resource cap " + strconv.FormatInt(capBytes, 10) +
		" bytes (maxStorageBufferRange=" + strconv.FormatInt(storageRange, 10) +
		", maxMemoryAllocationSize=" + strconv.FormatInt(allocationSize, 10) +
		"); split/chunk this tensor before uploading"
}

func shapeText(shape []int) string {
	if len(shape) == 0 {
		return "[]"
	}
	s := "["
	for i, d := range shape {
		if i > 0 {
			s += ","
		}
		s += strconv.Itoa(d)
	}
	return s + "]"
}
